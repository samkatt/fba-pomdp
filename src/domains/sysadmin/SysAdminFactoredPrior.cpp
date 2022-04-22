#include "SysAdminFactoredPrior.hpp"

#include <cmath>
#include <string>

#include "easylogging++.h"

#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "configurations/FBAConf.hpp"
#include "environment/Action.hpp"
#include "environment/State.hpp"
#include "utils/index.hpp"
#include "utils/random.hpp"

namespace priors {

SysAdminFactoredPrior::SysAdminFactoredPrior(
    domains::SysAdmin const& d,
    configurations::FBAConf const& c) :
        FBAPOMDPPrior(c),
        _noise(c.noise),
        _noisy_total_counts(c.counts_total),
        _params(*d.params()),
        _network_topology(d.topology()),
        _domain_size({1 << c.domain_conf.size, (int)(2 * c.domain_conf.size), 2}),
        _domain_feature_size({std::vector<int>(c.domain_conf.size, 2), {2}}),
        _fbapomdp_step_size(
            indexing::stepSize(_domain_feature_size._S),
            indexing::stepSize(_domain_feature_size._O)),
        _action_distr(rnd::integerDistribution(0, _domain_size._A)),
        _comp_distr(rnd::integerDistribution(0, c.domain_conf.size))
{

    if (!c.structure_prior.empty())
    {
        throw("Structure noise is not enabled for the Sysadmin problem");
    }

    if (_network_topology == domains::SysAdmin::NETWORK_TOPOLOGY::UNINITIALIZED)
    {
        throw("Somehow network topology was not set for the sysadmin problem");
    }

    precomputeFactoredPrior(d);
}

bayes_adaptive::factored::BABNModel::Structure
    SysAdminFactoredPrior::mutate(bayes_adaptive::factored::BABNModel::Structure structure) const
{

    bayes_adaptive::factored::BABNModel::Structure::flip_random_edge(
        &structure.T[_action_distr(rnd::rng())][_comp_distr(rnd::rng())], _comp_distr.max() + 1);

    return structure;
}

FBAPOMDPState* SysAdminFactoredPrior::sampleFullyConnectedState(State const* domain_state) const
{
    assert(domain_state != nullptr && domain_state->index() < _domain_size._S);

    return new FBAPOMDPState(
        domain_state,
        bayes_adaptive::factored::BABNModel(
            &_domain_size,
            &_domain_feature_size,
            &_fbapomdp_step_size,
            _fully_connected_transition_nodes,
            _observation_nodes));
}

FBAPOMDPState* SysAdminFactoredPrior::sampleCorrectGraphState(State const* domain_state) const
{

    return new FBAPOMDPState(
        domain_state,
        bayes_adaptive::factored::BABNModel(
            &_domain_size,
            &_domain_feature_size,
            &_fbapomdp_step_size,
            _correct_prior_transition_nodes,
            _observation_nodes));
}

FBAPOMDPState* SysAdminFactoredPrior::sampleFBAPOMDPState(State const* domain_state) const
{
    assert(domain_state != nullptr && domain_state->index() < _domain_size._S);

    return new FBAPOMDPState(
        domain_state,
        bayes_adaptive::factored::BABNModel(
            &_domain_size,
            &_domain_feature_size,
            &_fbapomdp_step_size,
            _prior_transition_nodes,
            _observation_nodes));
}

bayes_adaptive::factored::BABNModel SysAdminFactoredPrior::computePriorModel(
    bayes_adaptive::factored::BABNModel::Structure const& structure) const
{
    bayes_adaptive::factored::BABNModel model(
        &_domain_size,
        &_domain_feature_size,
        &_fbapomdp_step_size,
        _empty_transition_nodes, // init empty transition model
        _observation_nodes);

    for (auto a = 0; a < _domain_size._A; ++a)
    {
        IndexAction action(a);

        for (size_t c = 0; c < _domain_feature_size._S.size(); ++c)
        {

            auto const& parents = structure.T[a][c];
            model.resetTransitionNode(&action, c, parents);

            // fill counts for <action,feature> <a,c>
            std::vector<int> parent_values(parents.size(), 0);
            std::vector<int> const parent_dimensions(parents.size(), 2);

            do {

                auto const fail_prob =
                    computeFailureProbability(&action, c, &parents, &parent_values);
                model.transitionNode(&action, c)
                    .setDirichletDistribution(
                        parent_values,
                        {_noisy_total_counts * fail_prob, _noisy_total_counts * 1 - fail_prob});

            } while (!indexing::increment(parent_values, parent_dimensions));
        }
    }

    return model;
}

void SysAdminFactoredPrior::precomputeFactoredPrior(domains::SysAdmin const& d)
{

    std::vector<int> const failing_parent_value{0};
    std::vector<int> const working_parent_value{1};

    if (_network_topology == domains::SysAdmin::NETWORK_TOPOLOGY::INDEPENDENT)
    {
        _prior_transition_nodes = disconnectedTransitions();
    } else if (_network_topology == domains::SysAdmin::NETWORK_TOPOLOGY::LINEAR)
    {
        _prior_transition_nodes = linearTransitions();
    }

    _correct_prior_transition_nodes   = _prior_transition_nodes;
    _fully_connected_transition_nodes = fullyConnectedT(d);

    // set factored O counts
    bayes_adaptive::factored::BABNModel model(
        &_domain_size, &_domain_feature_size, &_fbapomdp_step_size);

    _empty_transition_nodes = model.copyT();

    std::vector<float> const failing_computer_observation_counts = {
        _known_total_counts * _params._observe_prob,
        _known_total_counts * (1 - _params._observe_prob)};

    std::vector<float> const operating_computer_observation_counts = {
        _known_total_counts * (1 - _params._observe_prob),
        _known_total_counts * _params._observe_prob};

    for (auto a = 0; a < (int)_domain_feature_size._S.size(); ++a)
    {
        auto reboot  = IndexAction(a + (int)_domain_feature_size._S.size());
        auto observe = IndexAction(a);

        model.resetObservationNode(&reboot, 0, std::vector<int>{a});
        model.resetObservationNode(&observe, 0, std::vector<int>{a});

        auto& reboot_node  = model.observationNode(&reboot, 0);
        auto& observe_node = model.observationNode(&observe, 0);

        reboot_node.setDirichletDistribution(
            failing_parent_value, failing_computer_observation_counts);
        reboot_node.setDirichletDistribution(
            working_parent_value, operating_computer_observation_counts);

        observe_node.setDirichletDistribution(
            failing_parent_value, failing_computer_observation_counts);
        observe_node.setDirichletDistribution(
            working_parent_value, operating_computer_observation_counts);
    }

    _observation_nodes = model.copyO();
}

std::vector<DBNNode> SysAdminFactoredPrior::disconnectedTransitions()
{
    bayes_adaptive::factored::BABNModel model(
        &_domain_size, &_domain_feature_size, &_fbapomdp_step_size);

    std::vector<std::vector<int>> const parent_values{{0}, {1}};

    for (auto a = 0; a < _domain_size._A; ++a)
    {
        IndexAction const action(a);

        for (auto c = 0; c < static_cast<int>(_domain_feature_size._S.size()); ++c)
        {

            std::vector<int> const parent{c};

            model.resetTransitionNode(&action, c, parent);

            for (auto const& parent_value : parent_values)
            {
                auto const p = computeFailureProbability(&action, c, &parent, &parent_value);
                std::vector<float> counts{p * _known_total_counts, (1 - p) * _known_total_counts};

                model.transitionNode(&action, c)
                    .setDirichletDistribution(parent_value, std::move(counts));
            }
        }
    }

    return model.copyT();
}

std::vector<DBNNode> SysAdminFactoredPrior::linearTransitions()
{

    bayes_adaptive::factored::BABNModel model(
        &_domain_size, &_domain_feature_size, &_fbapomdp_step_size);

    // set factored T counts
    for (auto a = 0; a < _domain_size._A; ++a)
    {
        IndexAction const action(a);

        for (auto c = 0; c < static_cast<int>(_domain_feature_size._S.size()); ++c)
        {

            std::vector<int> parents{c};
            if (c > 0)
            {
                parents.insert(parents.begin(), c - 1);
            }

            if (c < static_cast<int>(_domain_feature_size._S.size() - 1))
            {
                parents.emplace_back(c + 1);
            }

            std::vector<int> parent_values(parents.size(), 0);
            std::vector<int> const parent_dimensions(parents.size(), 2);

            model.resetTransitionNode(&action, c, parents);
            do {
                auto const p = computeFailureProbability(&action, c, &parents, &parent_values);
                std::vector<float> counts{p * _known_total_counts, (1 - p) * _known_total_counts};

                model.transitionNode(&action, c)
                    .setDirichletDistribution(parent_values, std::move(counts));

            } while (!indexing::increment(parent_values, parent_dimensions));
        }
    }

    return model.copyT();
}

std::vector<DBNNode> SysAdminFactoredPrior::fullyConnectedT(domains::SysAdmin const& d)
{

    auto const n = d.size();
    bayes_adaptive::factored::BABNModel model(
        &_domain_size, &_domain_feature_size, &_fbapomdp_step_size);

    std::vector<int> all_parents(n);
    for (auto i = 0; i < n; ++i) { all_parents[i] = i; }

    for (auto a = 0; a < _domain_size._A; ++a)
    {
        IndexAction action(a);

        for (auto c = 0; c < n; ++c)
        {

            model.resetTransitionNode(&action, c, all_parents);

            std::vector<int> parent_values(n, 0);
            do {
                auto const fail_prob = d.failProbability(d.getState(parent_values), &action, c);
                model.transitionNode(&action, c)
                    .setDirichletDistribution(parent_values, {fail_prob, 1 - fail_prob});

            } while (!indexing::increment(parent_values, _domain_feature_size._S));
        }
    }

    return model.copyT();
}

float SysAdminFactoredPrior::computeFailureProbability(
    Action const* a,
    int computer,
    std::vector<int> const* parents,
    std::vector<int> const* parent_values) const
{
    assert(
        parents != nullptr && parent_values != nullptr && parents->size() == parent_values->size());
    assert(a != nullptr && a->index() >= 0 && a->index() < _domain_size._A);
    assert(computer >= 0 && computer < static_cast<int>(_domain_feature_size._S.size()));

    auto const it = std::find(parents->begin(), parents->end(), computer);
    auto const is_rebooting =
        a->index() == static_cast<int>(_domain_feature_size._S.size() + computer);

    // special case: computer is not working currently
    if (it != parents->end() && parent_values->at(it - parents->begin()) == 0)
    {
        return is_rebooting ? 1 - _params._reboot_success_rate : 1;
    }

    // calculate number of failing neighbours
    int num_failing_neighbours = 0;
    if (_network_topology == domains::SysAdmin::LINEAR)
    {
        for (size_t c = 0; c < parents->size(); ++c)
        {
            if ((parents->at(c) == computer - 1 || parents->at(c) == computer + 1)
                && parent_values->at(c) == 0)
            {
                num_failing_neighbours++;
            }
        }
    }

    auto fail_prob = 1
                     - (1 - _params._fail_prob)
                           * pow(1 - _params._fail_neighbour_factor, num_failing_neighbours);

    if (is_rebooting)
    {
        fail_prob *= (1 - _params._reboot_success_rate);
    }

    // special case where computer is not its own input
    if (it == parents->end())
    {
        fail_prob += is_rebooting ? (1 - _params._reboot_success_rate) : 1;

        fail_prob *= .5;
    }

    return fail_prob;
}

} // namespace priors
