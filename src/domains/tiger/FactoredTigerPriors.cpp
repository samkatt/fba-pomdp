#include "FactoredTigerPriors.hpp"

#include <algorithm>
#include <string>

#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/BAConf.hpp"
#include "configurations/FBAConf.hpp"
#include "domains/tiger/FactoredTiger.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"
#include "utils/index.hpp"

namespace priors {

FactoredTigerFlatPrior::FactoredTigerFlatPrior(configurations::BAConf const& c) :
        _domain_size(2 << c.domain_conf.size, 3, 2),
        _prior(
            std::make_shared<std::vector<float>>(
                _domain_size._S * _domain_size._A * _domain_size._S,
                _known_counts), // S * A * S counts in T
            std::make_shared<std::vector<float>>(
                _domain_size._A * _domain_size._S * _domain_size._O,
                _known_counts), // A * S * O counts in O
            &_domain_size),
        _uniform_count_prior(
            std::make_shared<std::vector<float>>(
                _domain_size._S * _domain_size._A * _domain_size._S,
                _known_counts), // S * A * S counts in T
            std::make_shared<std::vector<float>>(
                _domain_size._A * _domain_size._S * _domain_size._O,
                _known_counts), // A * S * O counts in O
            &_domain_size)
{

    if (c.noise <= -.15 || c.noise > .3)
    {
        throw "noise has to be between -.15 and .3";
    }

    IndexState const tmp_state(0);

    // listening does not alter the state
    // so we set the counts for listening to 0
    // unless the states match
    auto observe = IndexAction(domains::FactoredTiger::TigerAction::OBSERVE);
    for (auto s = 0; s < _domain_size._S; ++s)
    {
        for (auto new_s = 0; new_s < _domain_size._S; ++new_s)
        {
            if (s != new_s)
            {
                auto state = IndexState(s), new_state = IndexState(new_s);

                _prior.count(&state, &observe, &new_state) = 0;
            }
        }
    }

    float acc_o_prior   = (.85f - c.noise) * c.counts_total,
          inacc_o_prior = (.15f + c.noise) * c.counts_total, uniform_prior = .5f * c.counts_total;

    // initiate uniform count prior with known counts
    _uniform_count_prior = _prior;

    // set the observation function for listening according to noise
    // meaning that for each state we have to adjust the counts
    // for listening according to the tiger location
    for (auto s = 0; s < _domain_size._S; ++s)
    {
        auto const state         = IndexState(s);
        auto const tiger_is_left = s < _domain_size._S / 2;

        auto const acc_o = tiger_is_left ? IndexObservation(domains::FactoredTiger::LEFT)
                                         : IndexObservation(domains::FactoredTiger::RIGHT);

        auto const inacc_o = tiger_is_left ? IndexObservation(domains::FactoredTiger::RIGHT)
                                           : IndexObservation(domains::FactoredTiger::LEFT);

        _prior.count(&observe, &state, &acc_o)   = acc_o_prior;
        _prior.count(&observe, &state, &inacc_o) = inacc_o_prior;

        _uniform_count_prior.count(&observe, &state, &acc_o)   = uniform_prior;
        _uniform_count_prior.count(&observe, &state, &inacc_o) = uniform_prior;
    }
}

BAPOMDPState* FactoredTigerFlatPrior::sampleBAPOMDPState(State const* s) const
{
    return new BAPOMDPState(s, _prior);
}

FactoredTigerFactoredPrior::FactoredTigerFactoredPrior(configurations::FBAConf const& c) :
        FBAPOMDPPrior(c),
        _domain_size(2 << c.domain_conf.size, 3, 2),
        _domain_feature_size(std::vector<int>(c.domain_conf.size + 1, 2), {2}),
        _fbapomdp_step_size(
            indexing::stepSize(_domain_feature_size._S),
            indexing::stepSize(_domain_feature_size._O)),
        _acc_O_count((.85f - c.noise) * c.counts_total),
        _inacc_O_count((.15f + c.noise) * c.counts_total),
        _uniform_O_count(.5f * c.counts_total),
        _struct_noise(c.structure_prior)
{

    if (c.noise <= -.15 || c.noise > .3)
    {
        throw "noise must be between -.15 and .3";
    }

    if (_struct_noise == "match-counts") // match-counts is the same as no noise
    {
        _struct_noise = "";
    }

    std::vector<std::string> const legal_structure_noise_options = {
        "", "uniform", "match-uniform", "fully-connected"};

    if (std::find(
            legal_structure_noise_options.begin(),
            legal_structure_noise_options.end(),
            _struct_noise)
        == legal_structure_noise_options.end())
    {
        throw "FactoredTigerFactoredPrior is unfamiliar with provided structure prior '"
            + _struct_noise + "'";
    }

    auto model = bayes_adaptive::factored::BABNModel(
        &_domain_size, &_domain_feature_size, &_fbapomdp_step_size);

    auto const open_right = IndexAction(domains::FactoredTiger::TigerAction::OPEN_RIGHT);
    auto const open_left  = IndexAction(domains::FactoredTiger::TigerAction::OPEN_LEFT);
    auto const listen     = IndexAction(domains::FactoredTiger::TigerAction::OBSERVE);

    /**** correctly known part of the prior *****/
    // new location after listening depends on previous location
    model.resetTransitionNode(&listen, _tiger_loc_feature, std::vector<int>({_tiger_loc_feature}));
    // non-informative features after listening depends on previous non-informative feature
    for (auto f = 1; f < (int)_domain_feature_size._S.size(); ++f)
    { model.resetTransitionNode(&listen, f, std::vector<int>({f})); }

    /* listening */
    // T: listening deterministically keeps all state features the same
    for (auto feature = 0; feature < (int)_domain_feature_size._S.size(); ++feature)
    {
        for (auto feature_val = 0; feature_val < 2; ++feature_val)
        {
            model.transitionNode(&listen, feature)
                .count(std::vector<int>({feature_val}), feature_val) = _known_counts;
        }
    }

    /* opening door */
    // T: values are uniform for each feature & action
    for (auto a = 0; a < 2; ++a)
    {
        auto action = IndexAction(a);

        for (auto f = 0; f < (int)_domain_feature_size._S.size(); f++)
        {
            for (auto f_val = 0; f_val < 2; ++f_val)
            { model.transitionNode(&action, f).count({}, f_val) = _known_counts; }
        }
    }

    _transition_nodes = model.copyT();

    // O when opening doors: uniform
    model.observationNode(&open_right, _tiger_loc_feature).count({}, domains::FactoredTiger::LEFT) =
        _known_counts;
    model.observationNode(&open_left, _tiger_loc_feature).count({}, domains::FactoredTiger::LEFT) =
        _known_counts;
    model.observationNode(&open_right, _tiger_loc_feature)
        .count({}, domains::FactoredTiger::RIGHT) = _known_counts;
    model.observationNode(&open_left, _tiger_loc_feature).count({}, domains::FactoredTiger::RIGHT) =
        _known_counts;

    _unstructured_observation_nodes = model.copyO();

    setObservationModel(&model, {0});
    _correctly_structured_observation_nodes = model.copyO();

    std::vector<int> all_parents(_domain_feature_size._S.size());
    for (size_t i = 0; i < _domain_feature_size._S.size(); ++i) { all_parents[i] = i; }

    setObservationModel(&model, all_parents);
    _fully_connected_observation_nodes = model.copyO();
}

std::vector<int> FactoredTigerFactoredPrior::sampleNoisyObservationParents() const
{

    auto parents = std::vector<int>();

    // uniformly add any feature as parent
    for (auto f = 0; f < (int)_domain_feature_size._S.size(); ++f)
    {
        if (rnd::boolean())
        {
            parents.emplace_back(f);
        }
    }

    // if using match-uniform, we need feature 0 (tiger location)
    // to be part of the parents of the observation node
    if (_struct_noise == "match-uniform" && (parents.empty() || parents[0] != 0))
    {
        parents.insert(parents.begin(), 0);
    }

    return parents;
}

void FactoredTigerFactoredPrior::setObservationModel(
    bayes_adaptive::factored::BABNModel* model,
    std::vector<int> const& parents) const
{

    auto listen = IndexAction(domains::FactoredTiger::TigerAction::OBSERVE);
    model->resetObservationNode(&listen, 0, parents);

    /*** set count for each parent value ***/
    auto parent_values = std::vector<int>(parents.size(), 0);
    auto parent_ranges = std::vector<int>(parents.size(), 2);

    // base case: no parents means we just set uniform prior and return
    if (parents.empty())
    {
        model->observationNode(&listen, 0)
            .setDirichletDistribution(
                parent_values, std::vector<float>({_uniform_O_count, _uniform_O_count}));

        return;
    }

    // if we have tiger location as feature, we have informed counts
    if (parents[0] == 0)
    {
        do
        {
            model->observationNode(&listen, 0).count(parent_values, domains::FactoredTiger::LEFT) =
                parent_values[0] == domains::FactoredTiger::LEFT ? _acc_O_count : _inacc_O_count;

            model->observationNode(&listen, 0).count(parent_values, domains::FactoredTiger::RIGHT) =
                parent_values[0] == domains::FactoredTiger::RIGHT ? _acc_O_count : _inacc_O_count;

        } while (!indexing::increment(parent_values, parent_ranges));
    } else // we don't have informative counts as tiger location is not part of parents
    {
        do
        {
            model->observationNode(&listen, 0)
                .setDirichletDistribution(
                    parent_values, std::vector<float>({_uniform_O_count, _uniform_O_count}));

        } while (!indexing::increment(parent_values, parent_ranges));
    }
}

FBAPOMDPState* FactoredTigerFactoredPrior::sampleFBAPOMDPState(State const* domain_state) const
{

    if (_struct_noise.empty())
    {

        return new FBAPOMDPState(
            domain_state,
            bayes_adaptive::factored::BABNModel(
                &_domain_size,
                &_domain_feature_size,
                &_fbapomdp_step_size,
                _transition_nodes,
                _correctly_structured_observation_nodes));
    }

    auto model = bayes_adaptive::factored::BABNModel(
        &_domain_size,
        &_domain_feature_size,
        &_fbapomdp_step_size,
        _transition_nodes,
        _unstructured_observation_nodes);

    setObservationModel(&model, sampleNoisyObservationParents());

    return new FBAPOMDPState(domain_state, std::move(model));
}

bayes_adaptive::factored::BABNModel FactoredTigerFactoredPrior::computePriorModel(
    bayes_adaptive::factored::BABNModel::Structure const& structure) const
{

    assert(structure.O.size() == static_cast<size_t>(_domain_size._A));
    assert(structure.T.size() == static_cast<size_t>(_domain_size._A));

    for (auto action = 0; action < _domain_size._A; ++action)
    {
        assert(structure.O[action].size() == _domain_feature_size._O.size());
        for (auto const& parents : structure.O[action])
        { assert(std::is_sorted(parents.begin(), parents.end())); }

        assert(structure.T[action].size() == _domain_feature_size._S.size());
        for (auto const& parents : structure.T[action]) { assert(parents.size() <= 1); }
    }

    auto prior = bayes_adaptive::factored::BABNModel(
        &_domain_size,
        &_domain_feature_size,
        &_fbapomdp_step_size,
        _transition_nodes,
        _unstructured_observation_nodes);

    setObservationModel(&prior, structure.O[domains::FactoredTiger::TigerAction::OBSERVE][0]);

    return prior;
}

FBAPOMDPState*
    FactoredTigerFactoredPrior::sampleFullyConnectedState(State const* domain_state) const
{

    return new FBAPOMDPState(
        domain_state,
        bayes_adaptive::factored::BABNModel(
            &_domain_size,
            &_domain_feature_size,
            &_fbapomdp_step_size,
            _transition_nodes,
            _fully_connected_observation_nodes));
}

FBAPOMDPState* FactoredTigerFactoredPrior::sampleCorrectGraphState(State const* domain_state) const
{

    return new FBAPOMDPState(
        domain_state,
        bayes_adaptive::factored::BABNModel(
            &_domain_size,
            &_domain_feature_size,
            &_fbapomdp_step_size,
            _transition_nodes,
            _correctly_structured_observation_nodes));
}

bayes_adaptive::factored::BABNModel::Structure FactoredTigerFactoredPrior::mutate(
    bayes_adaptive::factored::BABNModel::Structure structure) const
{
    assert(structure.O.size() == static_cast<size_t>(_domain_size._A));
    assert(structure.T.size() == static_cast<size_t>(_domain_size._A));

    for (auto action = 0; action < _domain_size._A; ++action)
    {

        assert(structure.O[action].size() == _domain_feature_size._O.size());
        for (auto const& parents : structure.O[action])
        { assert(std::is_sorted(parents.begin(), parents.end())); }

        assert(structure.T[action].size() == _domain_feature_size._S.size());
        for (auto const& parents : structure.T[action]) { assert(parents.size() <= 1); }
    }

    // the unknown structure in this problem is which state features
    // influence the (single) observation (feature) when listening
    // So here we flip a single edge for that particle part of
    // the structure
    bayes_adaptive::factored::BABNModel::Structure::flip_random_edge(
        &structure.O[2][0], (int)_domain_feature_size._S.size());

    return structure;
}

} // namespace priors
