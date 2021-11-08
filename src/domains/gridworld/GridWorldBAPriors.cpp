#include "GridWorldBAPriors.hpp"

#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/BAConf.hpp"
#include "configurations/FBAConf.hpp"
#include "domains/gridworld/GridWorldBAExtension.hpp"
#include "domains/gridworld/GridWorldFBAExtension.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"
#include "utils/random.hpp"

using GridWorld            = domains::GridWorld;
using GridWorldAction      = GridWorld::GridWorldAction;
using GridWorldObservation = GridWorld::GridWorldObservation;
using GridWorldState       = GridWorld::GridWorldState;

namespace priors {

GridWorldFlatBAPrior::GridWorldFlatBAPrior(
    GridWorld const& domain,
    configurations::BAConf const& c) :
        _size(c.domain_conf.size),
        _noise(c.noise),
        _unknown_counts_total(c.counts_total),
        _domain_size(0, 0, 0), // initialized below
        _goal_locations(GridWorld::goalLocations(_size)),
        _prior_model()
{

    bayes_adaptive::domain_extensions::GridWorldBAExtension ba_ext(_size);

    _domain_size = ba_ext.domainSize();
    _prior_model = bayes_adaptive::table::BAFlatModel(&_domain_size);

    if (_noise < 0 || _noise > (1 - GridWorld::slow_move_prob))
    {
        throw "Gridworld expects noise in between 0 and "
            + std::to_string(1 - GridWorld::slow_move_prob) + " (received " + std::to_string(_noise)
            + ")";
    }

    // initiate the prior model
    for (auto a = 0; a < _domain_size._A; ++a)
    {
        GridWorldAction action(a);

        for (auto s = 0; s < _domain_size._S; ++s)
        {

            auto state = static_cast<GridWorldState const*>(ba_ext.getState(std::to_string(s)));

            setPriorTransitionProbabilities(state, &action, domain);

            // O for new_s & action
            setPriorObservationProbabilities(&action, state, domain);

            domain.releaseState(state);
        }
    }
}

void GridWorldFlatBAPrior::setPriorTransitionProbabilities(
    GridWorldState const* s,
    GridWorldAction const* a,
    GridWorld const& domain)
{

    double acc_prob = 0;

    float const success_prob = (domain.agentOnSlowLocation(s->_agent_position))
                                   ? domain.GridWorld::slow_move_prob + _noise
                                   : GridWorld::move_prob;

    auto const goal_prob = static_cast<float>(1) / static_cast<float>(_goal_locations.size());

    /*** fail move ***/
    if (s->_goal_position == s->_agent_position)
    {
        auto const prob = (1 - success_prob) * goal_prob;

        for (auto const& goal_loc : _goal_locations)
        {
            auto new_s = domain.getState(s->_agent_position, goal_loc);

            _prior_model.count(s, a, new_s) += prob * _unknown_counts_total;
            acc_prob += prob;

            domain.releaseState(new_s);
        }
    } else // not on top of goal
    {
        auto const prob = 1 - success_prob;

        _prior_model.count(s, a, s) += prob * _unknown_counts_total;

        acc_prob += prob;
    }

    /*** move succeeds ***/
    auto const new_agent_pos = domain.applyMove(s->_agent_position, a);
    if (s->_goal_position == s->_agent_position)
    {

        auto const prob = success_prob * goal_prob;

        for (auto const& goal_loc : _goal_locations)
        {
            auto new_s = domain.getState(new_agent_pos, goal_loc);

            _prior_model.count(s, a, new_s) += prob * _unknown_counts_total;
            acc_prob += prob;

            domain.releaseState(new_s);
        }
    } else // not on top of goal
    {
        auto new_s = domain.getState(new_agent_pos, s->_goal_position);
        _prior_model.count(s, a, new_s) += success_prob * _unknown_counts_total;

        acc_prob += success_prob;
    }

    assert(acc_prob > .99);
    assert(acc_prob < 1.01);
}

void GridWorldFlatBAPrior::setPriorObservationProbabilities(
    GridWorldAction const* a,
    GridWorldState const* new_s,
    GridWorld const& domain)
{

    double acc_prob = 0;
    for (unsigned int x = 0; x < _size; ++x)
    {
        for (unsigned int y = 0; y < _size; ++y)
        {
            auto const o = domain.getObservation({x, y}, new_s->_goal_position);

            auto const prob = domain.computeObservationProbability(o, a, new_s);
            acc_prob += prob;

            _prior_model.count(a, new_s, o) = prob * _static_total_count;
        }
    }

    assert(acc_prob > .999);
    assert(acc_prob < 1.001);
}

BAPOMDPState* GridWorldFlatBAPrior::sampleBAPOMDPState(State const* s) const
{
    return new BAPOMDPState(s, _prior_model);
}

GridWorldFactBAPrior::GridWorldFactBAPrior(
    GridWorld const& domain,
    configurations::FBAConf const& c) :
        FBAPOMDPPrior(c),
        _size(domain.size()),
        _noise(c.noise),
        _unknown_counts_total(c.counts_total),
        _only_know_loc_matters(c.structure_prior == "match-uniform"),
        _domain_size(0, 0, 0), // initialized below
        _domain_feature_size({}, {}), // initialized below
        _indexing_steps({}, {}), // initialized below
        _correct_struct_prior(), // initialized below
        _domain(domain)
{

    bayes_adaptive::domain_extensions::GridWorldBAExtension ba_ext(_size);
    bayes_adaptive::domain_extensions::GridWorldFBAExtension fba_ext(_size);

    _domain_size         = ba_ext.domainSize();
    _domain_feature_size = fba_ext.domainFeatureSize();

    _indexing_steps = {indexing::stepSize(_domain_feature_size._S),
                       indexing::stepSize(_domain_feature_size._O)};

    _correct_struct_prior = {&_domain_size, &_domain_feature_size, &_indexing_steps};

    if (_noise < 0 || _noise > (1 - GridWorld::slow_move_prob))
    {
        throw "Gridworld expects noise in between 0 and "
            + std::to_string(1 - GridWorld::slow_move_prob) + " (received " + std::to_string(_noise)
            + ")";
    }

    if (!c.structure_prior.empty() && c.structure_prior != "match-uniform"
        && c.structure_prior != "match-counts")
    {
        throw "Please enter a valid structure noise option for the GridWorld problem ('match-uniform' or 'match-counts')";
    }

    preComputePrior();
}

bayes_adaptive::factored::BABNModel::Structure
    GridWorldFactBAPrior::mutate(bayes_adaptive::factored::BABNModel::Structure structure) const
{

    auto const random_action  = rnd::slowRandomInt(0, _domain_size._A);
    auto const random_feature = rnd::slowRandomInt(0, 2); // x or y feature

    auto edges = &structure.T[random_action][random_feature];

    assert(edges->size() >= 2); // assuming we always have x,y dependence

    if (edges->size() == 2)
    {

        edges->push_back(_goal_feature);

    } else // assumes we now have 3 edges (meaning we remove the parent)
    {

        assert(edges->size() == 3);
        assert(edges->at(2) == _goal_feature);
        edges->pop_back();
    }

    return structure;
}

bayes_adaptive::factored::BABNModel GridWorldFactBAPrior::computePriorModel(
    bayes_adaptive::factored::BABNModel::Structure const& structure) const
{
    auto prior = _correct_struct_prior;

    auto const real_parents = std::vector<int>({_agent_x_feature, _agent_y_feature});

    for (auto a = 0; a < _domain_size._A; ++a)
    {
        IndexAction const action(std::to_string(a));

        auto const& agent_x_parents = structure.T[a][_agent_x_feature];
        auto const& agent_y_parents = structure.T[a][_agent_y_feature];

        if (agent_x_parents != real_parents)
        {
            setNoisyTransitionNode(
                &prior, action, _agent_x_feature, structure.T[a][_agent_x_feature]);
        }

        if (agent_y_parents != real_parents)
        {
            setNoisyTransitionNode(
                &prior, action, _agent_y_feature, structure.T[a][_agent_y_feature]);
        }
    }

    return prior;
}

void GridWorldFactBAPrior::setNoisyTransitionNode(
    bayes_adaptive::factored::BABNModel* model,
    Action const& action,
    int feature,
    std::vector<int> const& parents) const
{
    assert(parents.size() == 3);

    model->resetTransitionNode(&action, feature, parents);

    for (auto x = 0; x < _domain_feature_size._S[_agent_x_feature]; ++x)
    {
        for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y)
        {

            auto const loc = (feature == _agent_x_feature) ? x : y;

            auto const new_loc =
                (feature == _agent_x_feature)
                    ? _domain
                          .applyMove(
                              {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}, &action)
                          .x
                    : _domain
                          .applyMove(
                              {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}, &action)
                          .y;

            float const trans_prob = (_domain.agentOnSlowLocation({static_cast<unsigned int>(x),
                                                                   static_cast<unsigned int>(y)}))
                                         ? GridWorld::slow_move_prob + _noise
                                         : GridWorld::move_prob;

            for (auto g = 0; g < _domain_feature_size._S[_goal_feature]; ++g)
            {

                // fail move
                model->transitionNode(&action, feature).count({x, y, g}, loc) +=
                    (1 - trans_prob) * _unknown_counts_total;

                // success move
                model->transitionNode(&action, feature).count({x, y, g}, new_loc) +=
                    (trans_prob)*_unknown_counts_total;
            }
        }
    }
}

FBAPOMDPState* GridWorldFactBAPrior::sampleFullyConnectedState(State const* /*domain_state*/) const
{
    throw "GridWorldFactBAPrior::sampleFullyConnectedState nyi";
}

FBAPOMDPState const* GridWorldFactBAPrior::sampleCorrectGraphState(State const* domain_state) const
{
    return new FBAPOMDPState(domain_state, _correct_struct_prior);
}

void GridWorldFactBAPrior::preComputePrior()
{
    for (auto a = 0; a < _domain_size._A; ++a)
    {
        IndexAction const action(std::to_string(a));

        /*** O (known) ***/

        // observe agent location depends on the state feature and observation probabilities
        for (auto f = 0; f < 2; ++f)
        {
            _correct_struct_prior.resetObservationNode(&action, f, {f});

            for (auto agent_loc = 0; agent_loc < _domain_feature_size._S[f]; ++agent_loc)
            {

                for (auto observed_loc = 0; observed_loc < _domain_feature_size._S[f];
                     ++observed_loc)
                {
                    _correct_struct_prior.observationNode(&action, f)
                        .count({agent_loc}, observed_loc) =
                        _domain.obsDisplProb(agent_loc, observed_loc) * _static_total_count;
                }
            }
        }

        // observe the goal deterministically
        _correct_struct_prior.resetObservationNode(&action, _goal_feature, {_goal_feature});

        for (auto v = 0; v < _domain_feature_size._O[_goal_feature]; ++v)
        {
            _correct_struct_prior.observationNode(&action, _goal_feature).count({v}, v) =
                _static_total_count;
        }

        /**** T ****/

        // new location: depends on noise,
        // whether on a slow or regular cell
        // and on the transition probabilities
        // goal depends on all
        _correct_struct_prior.resetTransitionNode(
            &action, _agent_x_feature, {_agent_x_feature, _agent_y_feature});
        _correct_struct_prior.resetTransitionNode(
            &action, _agent_y_feature, {_agent_x_feature, _agent_y_feature});
        _correct_struct_prior.resetTransitionNode(
            &action, _goal_feature, {_agent_x_feature, _agent_y_feature, _goal_feature});

        for (auto x = 0; x < _domain_feature_size._S[_agent_x_feature]; ++x)
        {
            for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y)
            {

                float const trans_prob =
                    (_domain.agentOnSlowLocation(
                        {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}))
                        ? GridWorld::slow_move_prob + _noise
                        : GridWorld::move_prob;

                // fail move
                _correct_struct_prior.transitionNode(&action, _agent_x_feature).count({x, y}, x) +=
                    (1 - trans_prob) * _unknown_counts_total;
                _correct_struct_prior.transitionNode(&action, _agent_y_feature).count({x, y}, y) +=
                    (1 - trans_prob) * _unknown_counts_total;

                // success move
                auto const new_pos = _domain.applyMove(
                    {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}, &action);
                _correct_struct_prior.transitionNode(&action, _agent_x_feature)
                    .count({x, y}, new_pos.x) += (trans_prob)*_unknown_counts_total;
                _correct_struct_prior.transitionNode(&action, _agent_y_feature)
                    .count({x, y}, new_pos.y) += (trans_prob)*_unknown_counts_total;

                for (auto g = 0; g < _domain_feature_size._S[_goal_feature]; ++g)
                {

                    auto const goal_pos = _domain.goalLocation(g);

                    // not on goal
                    if (goal_pos->x != static_cast<unsigned int>(x)
                        || goal_pos->y != static_cast<unsigned int>(y))
                    {
                        _correct_struct_prior.transitionNode(&action, _goal_feature)
                            .count({x, y, g}, g) = _static_total_count;
                    } else // on goal: set new one
                    {
                        for (auto new_g = 0; new_g < _domain_feature_size._S[_goal_feature];
                             ++new_g)
                        {
                            _correct_struct_prior.transitionNode(&action, _goal_feature)
                                .count({x, y, g}, new_g) = _static_total_count;
                        }
                    }
                }
            }
        }
    }
}

FBAPOMDPState* GridWorldFactBAPrior::sampleFBAPOMDPState(State const* domain_state) const
{

    // base case: just return correct prior
    if (!_only_know_loc_matters)
    {
        return new FBAPOMDPState(domain_state, _correct_struct_prior);
    }

    /*** noisy struct prior, but we know that agent location matters ****/
    auto structure = _correct_struct_prior.structure();

    // randomly add goal location to parents for each action
    for (auto a = 0; a < _domain_size._A; ++a)
    {
        if (rnd::boolean())
        {
            structure.T[a][_agent_x_feature].emplace_back(_goal_feature);
        }
        if (rnd::boolean())
        {
            structure.T[a][_agent_y_feature].emplace_back(_goal_feature);
        }
    }

    return new FBAPOMDPState(domain_state, computePriorModel(structure));
}

} // namespace priors
