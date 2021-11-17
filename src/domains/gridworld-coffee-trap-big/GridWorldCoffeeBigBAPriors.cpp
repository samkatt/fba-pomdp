#include "GridWorldCoffeeBigBAPriors.hpp"

#include "bayes-adaptive/states/factored/AbstractFBAPOMDPState.hpp"
#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/BAConf.hpp"
#include "configurations/FBAConf.hpp"
#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBigBAExtension.hpp"
#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBigFBAExtension.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"
#include "utils/random.hpp"

using GridWorldCoffeeBig            = domains::GridWorldCoffeeBig;
using GridWorldCoffeeBigAction      = GridWorldCoffeeBig::GridWorldCoffeeBigAction;
using GridWorldCoffeeBigObservation = GridWorldCoffeeBig::GridWorldCoffeeBigObservation;
using GridWorldCoffeeBigState       = GridWorldCoffeeBig::GridWorldCoffeeBigState;

bool correctBigPrior = false;
bool correctStructurePrior = false;

namespace priors {

GridWorldCoffeeBigFlatBAPrior::GridWorldCoffeeBigFlatBAPrior(
    domains::GridWorldCoffeeBig const& domain,
    configurations::BAConf const& c) :
        _size(5),
        _abstraction(c.domain_conf.abstraction),
        _extra_features(c.domain_conf.size),
        _unknown_counts_total(c.counts_total),
        _domain_size(0, 0, 0), // initialized below
        _prior_model()
{

    bayes_adaptive::domain_extensions::GridWorldCoffeeBigBAExtension ba_ext(_extra_features, domain);

    _domain_size = ba_ext.domainSize();
    _prior_model = bayes_adaptive::table::BAFlatModel(&_domain_size);

    // initiate the prior model
    for (auto a = 0; a < _domain_size._A; ++a)
    {
        GridWorldCoffeeBigAction action(a);

        for (auto s = 0; s < _domain_size._S; ++s)
        {
            // TODO change...
            auto state = static_cast<GridWorldCoffeeBigState const*>(ba_ext.getState(std::to_string(s)));

            setPriorTransitionProbabilities(state, &action, domain);

            // O for new_s & action
            setPriorObservationProbabilities(&action, state, domain);

            domain.releaseState(state);
        }
    }
}

void GridWorldCoffeeBigFlatBAPrior::setPriorTransitionProbabilities(
    GridWorldCoffeeBigState const* s,
    GridWorldCoffeeBigAction const* a,
    domains::GridWorldCoffeeBig const& domain)
{

    double acc_prob = 0;

    float success_prob;
    if (correctBigPrior) {
        if ((s->_state_vector[_x_feature] == 0 && s->_state_vector[_y_feature] == 3)
        || (s->_state_vector[_x_feature] == 1 && s->_state_vector[_y_feature] == 3)
        || (s->_state_vector[_x_feature] == 2 && s->_state_vector[_y_feature] == 1)) {
            success_prob = GridWorldCoffeeBig::slow_move_prob;
        }
        else {
            success_prob = GridWorldCoffeeBig::move_prob;
        }
    }
    else {
        // TODO change
        success_prob = 0.9;
//        success_prob = domain.believedTransitionProb(domain.agentOnCarpet(
//                {static_cast<unsigned int>(s->_state_vector[_x_feature]),
//                 static_cast<unsigned int>(s->_state_vector[_y_feature])},
//                0), s->_state_vector[_rain_feature]);
    }

    /*** fail move ***/
    if (domain.foundGoal(s))
    {
        auto const prob = (1 - success_prob);
        for (auto const& rain : rain_values)
        {
            auto const rain_prob = (s->_state_vector[_rain_feature] == rain) ? GridWorldCoffeeBig::same_weather_prob : 1 - GridWorldCoffeeBig::same_weather_prob;
            auto state_vector = s->getFeatureValues();
            state_vector[_rain_feature] = rain;
            auto new_s = domain.getState(state_vector);

            _prior_model.count(s, a, new_s) += prob * rain_prob * _unknown_counts_total;
            acc_prob += prob * rain_prob;

            domain.releaseState(new_s);
        }
    } else // not on top of goal
    {
        auto const prob = 1 - success_prob;
        for (auto const& rain : rain_values)
        {
            auto const rain_prob = (s->_state_vector[_rain_feature] == rain) ? GridWorldCoffeeBig::same_weather_prob : 1 - GridWorldCoffeeBig::same_weather_prob;
            auto state_vector = s->getFeatureValues();
            state_vector[_rain_feature] = rain;
            auto new_s = domain.getState(state_vector);

            _prior_model.count(s, a, new_s) += prob * rain_prob * _unknown_counts_total;
            acc_prob += prob * rain_prob;

            domain.releaseState(new_s);
        }
    }

    /*** move succeeds ***/
    auto const new_agent_pos = domain.applyMove({static_cast<unsigned int>(s->_state_vector[_x_feature]),
                                                 static_cast<unsigned int>(s->_state_vector[_y_feature])}, a);
    if (domain.foundGoal(s))
    {
        for (auto const& rain : rain_values)
        {
            auto const rain_prob = (s->_state_vector[_rain_feature] == rain) ? GridWorldCoffeeBig::same_weather_prob : 1 - GridWorldCoffeeBig::same_weather_prob;
            auto state_vector = s->getFeatureValues();
            state_vector[_x_feature] = new_agent_pos.x;
            state_vector[_y_feature] = new_agent_pos.y;
            state_vector[_rain_feature] = rain;
            auto new_s = domain.getState(state_vector);

            _prior_model.count(s, a, new_s) += success_prob * rain_prob * _unknown_counts_total;
            acc_prob += success_prob * rain_prob;

            domain.releaseState(new_s);
        }
    } else // not on top of goal TODO what is the difference here?
    {
        for (auto const& rain : rain_values)
        {
            auto const rain_prob = (s->_state_vector[_rain_feature] == rain) ? GridWorldCoffeeBig::same_weather_prob : 1 - GridWorldCoffeeBig::same_weather_prob;
            auto state_vector = s->getFeatureValues();
            state_vector[_x_feature] = new_agent_pos.x;
            state_vector[_y_feature] = new_agent_pos.y;
            state_vector[_rain_feature] = rain;
            auto new_s = domain.getState(state_vector);

            _prior_model.count(s, a, new_s) += success_prob * rain_prob * _unknown_counts_total;
            acc_prob += success_prob * rain_prob;

            domain.releaseState(new_s);
        }
    }

    assert(acc_prob > .99);
    assert(acc_prob < 1.01);
}

void GridWorldCoffeeBigFlatBAPrior::setPriorObservationProbabilities(
    GridWorldCoffeeBigAction const* a,
    GridWorldCoffeeBigState const* new_s,
    domains::GridWorldCoffeeBig const& domain)
{
    double acc_prob = 0;
    for (unsigned int x = 0; x < _size; ++x)
    {
        for (unsigned int y = 0; y < _size; ++y)
        {
            auto const o = domain.getObservation({x, y});

            auto const prob = domain.computeObservationProbability(o, a, new_s);
            acc_prob += prob;

            _prior_model.count(a, new_s, o) = prob * _static_total_count;
        }
    }

    assert(acc_prob > .999);
    assert(acc_prob < 1.001);
}

BAPOMDPState* GridWorldCoffeeBigFlatBAPrior::sampleBAPOMDPState(State const* s) const
{
    return new BAPOMDPState(s, _prior_model);
}

/**
 * FACTORED STUFF
 */
GridWorldCoffeeBigFactBAPrior::GridWorldCoffeeBigFactBAPrior(
    domains::GridWorldCoffeeBig const& domain,
    configurations::FBAConf const& c) :
    FBAPOMDPPrior(c),
//    _num_abstractions(),
//    _minimum_abstraction(),
    _size(domain.size()),
    _abstraction(c.domain_conf.abstraction),
    _carpet_tiles(c.domain_conf.size),
    _unknown_counts_total(c.counts_total),
    _correct_prior(c.structure_prior == "match-uniform"),
    _domain_size(0, 0, 0), // initialized below
    _domain_feature_size({}, {}), // initialized below
    _indexing_steps({}, {}), // initialized below
    _correct_struct_prior(), // initialized below
    _domain(std::move(domain))
{

    bayes_adaptive::domain_extensions::GridWorldCoffeeBigBAExtension ba_ext(_carpet_tiles, domain);
    bayes_adaptive::domain_extensions::GridWorldCoffeeBigFBAExtension fba_ext(_carpet_tiles);

    _domain_size         = ba_ext.domainSize();
    _domain_feature_size = fba_ext.domainFeatureSize();

//    _num_abstractions   = fba_ext._num_abstractions;
//    _minimum_abstraction = fba_ext._minimum_abstraction;

    _indexing_steps = {indexing::stepSize(_domain_feature_size._S),
                       indexing::stepSize(_domain_feature_size._O)};

    _correct_struct_prior = {&_domain_size, &_domain_feature_size, &_indexing_steps};


    if (!c.structure_prior.empty() && c.structure_prior != "match-uniform"
        && c.structure_prior != "match-counts")
    {
        throw "Please enter a valid structure noise option for the GridWorldCoffeeBig problem ('match-uniform' or 'match-counts')";
    }

    preComputePrior();
}

bayes_adaptive::factored::BABNModel::Structure
GridWorldCoffeeBigFactBAPrior::mutate(bayes_adaptive::factored::BABNModel::Structure structure) const
{

    auto const random_action  = rnd::slowRandomInt(0, _domain_size._A);
    auto const random_feature = rnd::slowRandomInt(0, 2); // x or y feature

    auto edges = &structure.T[random_action][random_feature];

    assert(edges->size() >= 2); // assuming we always have x-y dependence (used to be: at least x->x or y->y dependence)

    // just need to add or remove a random one
    auto edge_to_flip = rnd::slowRandomInt(2, _num_features);

    // stealing code from flip_random_edge, since there are two edges we don't want to flip
    auto lower_bound  = std::lower_bound(edges->begin(), edges->end(), edge_to_flip);

    if (lower_bound != edges->end() && *lower_bound == edge_to_flip) // found the edge, remove
    {
        edges->erase(lower_bound);
    } else // edge was not there, add!
    {
       edges->insert(lower_bound, edge_to_flip);
    }

    return structure;
}

// this is the prior model for the agent?
bayes_adaptive::factored::BABNModel GridWorldCoffeeBigFactBAPrior::computePriorModel(
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

void GridWorldCoffeeBigFactBAPrior::setNoisyTransitionNode(
        bayes_adaptive::factored::BABNModel* model,
        Action const& action,
        int feature,
        std::vector<int> const& parents) const {

    /*** set count for each parent value ***/
    auto parent_values = std::vector<int>(parents.size(), 0);
    auto parent_ranges = std::vector<int>(parents.size(), 0);
    for (unsigned int i =0; i < parents.size(); ++i) {
        parent_ranges[i] = _domain_feature_size._S[parents[i]];
    }

    if (parents.size() == 2) { // only x and y as parents, this function shouldn't be called
        throw "this shouldn't be happening";
    }
    model->resetTransitionNode(&action, feature, parents);
    if (parents.size() >= 3) {

        do {
            float trans_prob;
            if (correctBigPrior || _correct_prior) {
                if ((parent_values[0] == 0 && parent_values[1] == 3)
                    || (parent_values[0] == 1 && parent_values[1] == 3)
                    || (parent_values[0] == 2 && parent_values[1] == 1)) {
                    trans_prob = 0.1;
                } else {
                    trans_prob = GridWorldCoffeeBig::move_prob;
                }
            } else {
                bool raining = false;
                unsigned int start = 2;
                if (parents[2] == _rain_feature) { // rain feature
                    start = 3;
                    if (parent_values[_rain_feature] == 1) {
                        raining = true;
                    }
                }
                int features_on = 0;
                for (unsigned int i = start; i < parents.size(); ++i) {
                    if (parent_values[i] == 1) {
                        features_on++;
                    }
                }

                if (!raining) { // no rain
                    if (parents.size() - start == 0) { // no other extra variables as parents
                        trans_prob = GridWorldCoffeeBig::move_prob; // TODO give wrong move prob...?
                    } else {
                        trans_prob = 0.5;
                    }
                } else { //rain
                    if (parents.size() - start == 0) { // no other extra variables as parents
                        trans_prob = GridWorldCoffeeBig::rain_move_prob;
                    } else {
                        trans_prob = 0.5; // TODO check
                    }
                }
            }

            // fail move
            model->transitionNode(&action, feature).count(parent_values, parent_values[feature]) +=
                    (1 - trans_prob) * _unknown_counts_total;

            // success move
            auto const new_pos = _domain.applyMove(
                    {static_cast<unsigned int>(parent_values[0]),
                     static_cast<unsigned int>(parent_values[1])}, &action);
            auto output = new_pos.y;
            if (feature == _agent_x_feature) {
                output = new_pos.x;
            }
            model->transitionNode(&action, feature).count(parent_values, output) +=
                    (trans_prob)*_unknown_counts_total;
        } while (!indexing::increment(parent_values, parent_ranges));
    }
}


FBAPOMDPState* GridWorldCoffeeBigFactBAPrior::sampleFullyConnectedState(State const* /*domain_state*/) const
{
    throw "GridWorldFactBAPrior::sampleFullyConnectedState nyi";
}

FBAPOMDPState const* GridWorldCoffeeBigFactBAPrior::sampleCorrectGraphState(State const* domain_state) const
{
    if (_abstraction) {
        return new AbstractFBAPOMDPState(domain_state, _correct_struct_prior);
    }
    return new FBAPOMDPState(domain_state, _correct_struct_prior);
}

// the model the agent beliefs is correct, so no knowledge about slow locations
void GridWorldCoffeeBigFactBAPrior::preComputePrior()
{
    for (auto a = 0; a < _domain_size._A; ++a)
    {
        IndexAction const action(std::to_string(a));

        /*** O (known) ***/
        // observe agent location, depends on state feature and observation probabilities
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


        /**** T ****/

        // new x and y: depends on whether on a slow or regular cell, i.e. both x and y
        // and on the transition probabilities
        // rain: depends on rain previous
        // carpet: depends on carpet
        _correct_struct_prior.resetTransitionNode(
            &action, _agent_x_feature, {_agent_x_feature, _agent_y_feature});
        _correct_struct_prior.resetTransitionNode(
            &action, _agent_y_feature, {_agent_x_feature, _agent_y_feature});
        _correct_struct_prior.resetTransitionNode(
                &action, _rain_feature, {_rain_feature});
        for (auto f = 3; f < (int)_domain_feature_size._S.size(); ++f)
        { _correct_struct_prior.resetTransitionNode(&action, f, {f}); }

        // agent is sure about the carpet, stays the same
        for (auto feature = 3; feature < (int)_domain_feature_size._S.size(); ++feature)
        {
            for (auto feature_val = 0; feature_val < 2; ++feature_val)
            {
                _correct_struct_prior.transitionNode(&action, feature)
                        .count({feature_val}, feature_val) += _static_total_count;
            }
        }

        // agent knows the chances of the rain changing
        for (auto r = 0; r < _domain_feature_size._S[_rain_feature]; ++r)
        {
            _correct_struct_prior.transitionNode(&action, _rain_feature).count({r}, r) +=
                    GridWorldCoffeeBig::same_weather_prob * _static_total_count;
            _correct_struct_prior.transitionNode(&action, _rain_feature).count({r}, (1-r)) +=
                    (1 - GridWorldCoffeeBig::same_weather_prob) * _static_total_count;
        }

        // X and Y
        for (auto x = 0; x < _domain_feature_size._S[_agent_x_feature]; ++x)
        {
            for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y)
            {
                float trans_prob = GridWorldCoffeeBig::move_prob;
                if (correctBigPrior || _correct_prior) {
                    if (_domain.agentOnSlowLocation({static_cast<unsigned int>(x),static_cast<unsigned int>(y)})) {
                        trans_prob = GridWorldCoffeeBig::slow_move_prob;
                    }
                }

                // fail move
                _correct_struct_prior.transitionNode(&action, _agent_x_feature).count({x, y}, x) +=
                    (1 - trans_prob) * _unknown_counts_total;
                _correct_struct_prior.transitionNode(&action, _agent_y_feature).count({x, y}, y) +=
                    (1 - trans_prob) * _unknown_counts_total;

                // success move
                auto const new_pos = _domain.applyMove(
                    {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}, &action);
                _correct_struct_prior.transitionNode(&action, _agent_x_feature).count({x, y}, new_pos.x) +=
                        (trans_prob)*_unknown_counts_total;
                _correct_struct_prior.transitionNode(&action, _agent_y_feature).count({x, y}, new_pos.y) +=
                        (trans_prob)*_unknown_counts_total;
            }
        }
    }
}

FBAPOMDPState* GridWorldCoffeeBigFactBAPrior::sampleFBAPOMDPState(State const* domain_state) const
{
    // TODO also add a base case?
    // base case: just return correct prior
//    if (!_only_know_loc_matters)
//    {
//        return new AbstractFBAPOMDPSTATE(domain_state, _correct_struct_prior);
//    }
    if (correctBigPrior || _correct_prior) {
        if (_abstraction) {
            return new AbstractFBAPOMDPState(domain_state, _correct_struct_prior);
        }
        return new FBAPOMDPState(domain_state, _correct_struct_prior);
    }
    if (correctStructurePrior) {
        if (_abstraction) {
            return new AbstractFBAPOMDPState(domain_state, _correct_struct_prior);
        }
        return new FBAPOMDPState(domain_state, _correct_struct_prior);
    }

    /*** noisy struct prior ****/
    auto structure = _correct_struct_prior.structure();

    if (rnd::slowRandomInt(1,100) <=50) // randomly add rain to parents of x and y
    {
        for (auto a = 0; a < _domain_size._A; ++a) {
            structure.T[a][_agent_x_feature].emplace_back(_rain_feature);
            structure.T[a][_agent_y_feature].emplace_back(_rain_feature);
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());

    int max_extra_parents = 3;
    auto extra_parents_to_add = std::vector<int> (max_extra_parents + 1);
    std::iota(std::begin(extra_parents_to_add), std::end(extra_parents_to_add), 0);
    std::shuffle(extra_parents_to_add.begin(), extra_parents_to_add.end(), g);

    auto random_to_add = std::vector<int> (_carpet_tiles);
    std::iota(std::begin(random_to_add), std::end(random_to_add), 3);
    std::shuffle(random_to_add.begin(), random_to_add.end(), g);

    // add 0 - 4 parents
    for (auto extra_feature = 0; extra_feature < extra_parents_to_add[0]; ++extra_feature) {
        // add the first entries from the random_to_add
        for (auto a = 0; a < _domain_size._A; ++a) {
            structure.T[a][_agent_x_feature].emplace_back(random_to_add[extra_feature]);
            structure.T[a][_agent_y_feature].emplace_back(random_to_add[extra_feature]);
        }
    }
    for (auto a = 0; a < _domain_size._A; ++a) {
        std::sort(structure.T[a][_agent_x_feature].begin(), structure.T[a][_agent_x_feature].end());
        std::sort(structure.T[a][_agent_y_feature].begin(), structure.T[a][_agent_y_feature].end());
    }

    // uniformly add any extra binary feature as parent
//    for (auto f = 3; f < (int)_domain_feature_size._S.size(); ++f)
//    {
//        if (rnd::slowRandomInt(1,100) <= 10) // randomly add binary feature to parents of x and y
//        {
//            for (auto a = 0; a < _domain_size._A; ++a) {
//                structure.T[a][_agent_x_feature].emplace_back(f);
//                structure.T[a][_agent_y_feature].emplace_back(f);
//            }
//        }
//    }

    if (_abstraction) {
        return new AbstractFBAPOMDPState(domain_state, computePriorModel(structure));
    }
    return new FBAPOMDPState(domain_state, computePriorModel(structure));
}

    Domain_Feature_Size *GridWorldCoffeeBigFactBAPrior::getDomainFeatureSize() {
        return &_domain_feature_size;
    }

} // namespace priors
