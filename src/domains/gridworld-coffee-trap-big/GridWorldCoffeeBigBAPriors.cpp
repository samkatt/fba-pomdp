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

// constexpr static double const move_prob      = .95;
// constexpr static double const slow_move_prob = .1;
// ((agent_pos.x == 0 && agent_pos.y == 3) || (agent_pos.x == 1 && agent_pos.y == 3) || (agent_pos.x == 2 && agent_pos.y == 1))
bool correctBigPrior = false;
bool useBigAbstraction = true;

namespace priors {

GridWorldCoffeeBigFlatBAPrior::GridWorldCoffeeBigFlatBAPrior(
    GridWorldCoffeeBig const& domain,
    configurations::BAConf const& c) :
        _size(5),
//        _carpet_configurations(2),
        _unknown_counts_total(c.counts_total),
        _domain_size(0, 0, 0), // initialized below
        _prior_model()
{

    bayes_adaptive::domain_extensions::GridWorldCoffeeBigBAExtension ba_ext;

    _domain_size = ba_ext.domainSize();
    _prior_model = bayes_adaptive::table::BAFlatModel(&_domain_size);

    // initiate the prior model
    for (auto a = 0; a < _domain_size._A; ++a)
    {
        GridWorldCoffeeBigAction action(a);

        for (auto s = 0; s < _domain_size._S; ++s)
        {

            auto state = static_cast<GridWorldCoffeeBigState const*>(ba_ext.getState(s));

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
    GridWorldCoffeeBig const& domain)
{

    double acc_prob = 0;

    float success_prob;
    // no carpet: rain: 0.8, no rain: 0.95
    // carpet: rain: 0.05, no rain: 0.15
    if (correctBigPrior) {
        if ((s->_agent_position.x == 0 && s->_agent_position.y == 3) || (s->_agent_position.x == 1 && s->_agent_position.y == 3)
        || (s->_agent_position.x == 2 && s->_agent_position.y == 1)) {
            success_prob = 0.1;
        }
        else {
            success_prob = 0.95;
        }
    }
    else {
        success_prob = (domain.agentOnCarpet(s->_agent_position))
                                   ? ((s->_rain) ? 0.05 : 0.1)
                                   : ((s->_rain) ? 0.9 : 0.95);
    }



    /*** fail move ***/
    if (GridWorldCoffeeBig::goal_location == s->_agent_position)
    {
        auto const prob = (1 - success_prob);
        for (auto const& rain : rain_values)
        {
            auto const rain_prob = (s->_rain == rain) ? GridWorldCoffeeBig::same_weather_prob : 1 - GridWorldCoffeeBig::same_weather_prob;

            auto new_s = domain.getState(s->_agent_position, rain, s->_carpet_config);

            _prior_model.count(s, a, new_s) += prob * rain_prob * _unknown_counts_total;
            acc_prob += prob * rain_prob;

            domain.releaseState(new_s);
        }
    } else // not on top of goal
    {
        auto const prob = 1 - success_prob;
        for (auto const& rain : rain_values)
        {
            auto const rain_prob = (s->_rain == rain) ? GridWorldCoffeeBig::same_weather_prob : 1 - GridWorldCoffeeBig::same_weather_prob;

            auto new_s = domain.getState(s->_agent_position, rain, s->_carpet_config);

            _prior_model.count(s, a, new_s) += prob * rain_prob * _unknown_counts_total;
            acc_prob += prob * rain_prob;

            domain.releaseState(new_s);
        }
    }

    /*** move succeeds ***/
    auto const new_agent_pos = domain.applyMove(s->_agent_position, a);
    if (GridWorldCoffeeBig::goal_location == s->_agent_position)
    {
        for (auto const& rain : rain_values)
        {
            auto const rain_prob = (s->_rain == rain) ? GridWorldCoffeeBig::same_weather_prob : 1 - GridWorldCoffeeBig::same_weather_prob;

            auto new_s = domain.getState(new_agent_pos, rain, s->_carpet_config);

            _prior_model.count(s, a, new_s) += success_prob * rain_prob * _unknown_counts_total;
            acc_prob += success_prob * rain_prob;

            domain.releaseState(new_s);
        }
    } else // not on top of goal
    {
        for (auto const& rain : rain_values)
        {
            auto const rain_prob = (s->_rain == rain) ? GridWorldCoffeeBig::same_weather_prob : 1 - GridWorldCoffeeBig::same_weather_prob;

            auto new_s = domain.getState(new_agent_pos, rain, s->_carpet_config);

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
    GridWorldCoffeeBig const& domain)
{
    double acc_prob = 0;
    for (unsigned int x = 0; x < _size; ++x)
    {
        for (unsigned int y = 0; y < _size; ++y)
        {
            auto const o = domain.getObservation({x, y}); //, new_s->_rain, new_s->_carpet_config);

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
    GridWorldCoffeeBig const& domain,
    configurations::FBAConf const& c) :
    FBAPOMDPPrior(c),
    _size(domain.size()),
    _carpet_features(15),
//    _noise(c.noise),
    _unknown_counts_total(c.counts_total),
//    _only_know_loc_matters(c.structure_prior == "match-uniform"),
    _domain_size(0, 0, 0), // initialized below
    _domain_feature_size({}, {}), // initialized below
    _indexing_steps({}, {}), // initialized below
    _correct_struct_prior(), // initialized below
    _domain(domain)
{

    bayes_adaptive::domain_extensions::GridWorldCoffeeBigBAExtension ba_ext;
    bayes_adaptive::domain_extensions::GridWorldCoffeeBigFBAExtension fba_ext;

    _domain_size         = ba_ext.domainSize();
    _domain_feature_size = fba_ext.domainFeatureSize();

    _indexing_steps = {indexing::stepSize(_domain_feature_size._S),
                       indexing::stepSize(_domain_feature_size._O)};

    _correct_struct_prior = {&_domain_size, &_domain_feature_size, &_indexing_steps};

//    if (_noise < 0 || _noise > (1 - GridWorldCoffeeBig::slow_move_prob))
//    {
//        throw "Gridworld expects noise in between 0 and "
//              + std::to_string(1 - GridWorldCoffeeBig::slow_move_prob) + " (received " + std::to_string(_noise)
//              + ")";
//    }

    if (!c.structure_prior.empty() && c.structure_prior != "match-uniform"
        && c.structure_prior != "match-counts")
    {
        throw "Please enter a valid structure noise option for the GridWorldCoffeeBig problem ('match-uniform' or 'match-counts')";
    }

    preComputePrior();
}

//std::vector<int> *addToOrderedFeatureVector(std::vector<int> edges, int featureToInclude) {
//    std::vector<int> toReturn;
//}


bayes_adaptive::factored::BABNModel::Structure
GridWorldCoffeeBigFactBAPrior::mutate(bayes_adaptive::factored::BABNModel::Structure structure) const
{

    auto const random_action  = rnd::slowRandomInt(0, _domain_size._A);
    auto const random_feature = rnd::slowRandomInt(0, 2); // x or y feature

    auto edges = &structure.T[random_action][random_feature];

    assert(edges->size() >= 2); // assuming we always have x-y dependence (used to be: at least x->x or y->y dependence)

    // just need to add or remove a random one
    auto edge_to_flip = rnd::slowRandomInt(2, _num_features);
//    if (random_feature == edge_to_flip) { // this only happens if both are "1", in which case we want to use feature 0
//        edge_to_flip = random_feature - 1;
//    }

    // stealing code from flip_random_edge, since there is one edge we don't want to flip
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
        IndexAction const action(a);

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

//Boolean agentOnCarpet(pos agentPos, int carpet_config)

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
            float trans_prob = GridWorldCoffeeBig::move_prob;
            if (parents[2] == _rain_feature) { // rain feature
                if (parent_values[_rain_feature] == 1) {
                    trans_prob = 0.8;
                }
            }
            if (correctBigPrior) {
                if ((parent_values[0] == 0 && parent_values[1] == 3)
                    || (parent_values[0] == 1 && parent_values[1] == 3)
                    || (parent_values[0] == 2 && parent_values[1] == 1)) {
                    trans_prob = 0.1;
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

//void GridWorldCoffeeBigFactBAPrior::setNoisyTransitionNode(
//    bayes_adaptive::factored::BABNModel* model,
//    Action const& action,
//    int feature,
//    std::vector<int> const& parents) const
//{
//    // so here we need to distinguish between quite a few cases.
//    bool rain_location = false;
//    bool carpet_location1 = false;
//    bool xy_location = false;
//
//    auto lower_bound_rain  = std::lower_bound(parents.begin(), parents.end(), _rain_feature);
//    if (lower_bound_rain != parents.end() && *lower_bound_rain == _rain_feature) // found the edge
//    {
//        rain_location = true;
//    }
//    auto lower_bound_carpet  = std::lower_bound(parents.begin(), parents.end(), _carpet_feature1);
//    if (lower_bound_carpet != parents.end() && *lower_bound_carpet == _carpet_feature1) // found the edge
//    {
//        carpet_location1 = true;
//    }
//
//    auto lower_bound_xy = std::lower_bound(parents.begin(), parents.end(), (1-feature));
//    if (lower_bound_xy != parents.end() && *lower_bound_xy == (1-feature)) // found the edge
//    {
//        xy_location = true;
//    }
//    assert(xy_location == true); // with recent change, this should always be true
//    // rest
//    model->resetTransitionNode(&action, feature, parents);
//    // TODO improve code?
//    if (xy_location) {
//        if (rain_location) {
//            if (carpet_location1) {
//                for (auto x = 0; x < _domain_feature_size._S[_agent_x_feature]; ++x) {
//                    for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y) {
//
//                        auto const loc = (feature == _agent_x_feature) ? x : y;
//
//                        auto const new_loc =
//                                (feature == _agent_x_feature)
//                                ? _domain
//                                        .applyMove(
//                                                {static_cast<unsigned int>(x),
//                                                 static_cast<unsigned int>(y)}, &action)
//                                        .x
//                                : _domain
//                                        .applyMove(
//                                                {static_cast<unsigned int>(x),
//                                                 static_cast<unsigned int>(y)}, &action)
//                                        .y;
//
//                        for (auto rain = 0; rain < _domain_feature_size._S[_rain_feature]; ++rain) {
//                            for (auto carpet_config1 = 0; carpet_config1 <
//                                                          _domain_feature_size._S[_carpet_feature1]; ++carpet_config1) {
//                                // no carpet: rain: 0.8, no rain: 0.95
//                                // carpet: rain: 0.05, no rain: 0.15
//                                float const success_prob = (_domain.agentOnCarpet(
//                                        {(unsigned int) x, (unsigned int) y}))
//                                                           ? (rain ? 0.05 : 0.1)
//                                                           : (rain ? 0.9 : 0.95);
//
//                                // fail move
//                                model->transitionNode(&action, feature).count({x, y, rain, carpet_config1},
//                                                                              loc) +=
//                                        (1 - success_prob) * _unknown_counts_total;
//
//                                // success move
//                                model->transitionNode(&action, feature).count({x, y, rain, carpet_config1},
//                                                                              new_loc) +=
//                                        (success_prob) * _unknown_counts_total;
//                            }
//                        }
//                    }
//                }
//            } else { // xy and rain
//                for (auto x = 0; x < _domain_feature_size._S[_agent_x_feature]; ++x) {
//                    for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y) {
//
//                        auto const loc = (feature == _agent_x_feature) ? x : y;
//
//                        auto const new_loc =
//                                (feature == _agent_x_feature)
//                                ? _domain
//                                        .applyMove(
//                                                {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}, &action)
//                                        .x
//                                : _domain
//                                        .applyMove(
//                                                {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}, &action)
//                                        .y;
//
//                        for (auto rain = 0; rain < _domain_feature_size._S[_rain_feature]; ++rain){
//                            // no carpet: rain: 0.8, no rain: 0.95
//                            float const success_prob = rain ? 0.9 : 0.95;
//
//                            // fail move
//                            model->transitionNode(&action, feature).count({x, y, rain}, loc) +=
//                                    (1 - success_prob) * _unknown_counts_total;
//
//                            // success move
//                            model->transitionNode(&action, feature).count({x, y, rain}, new_loc) +=
//                                    (success_prob) * _unknown_counts_total;
//                        }
//                    }
//                }
//            }
//        } else if (carpet_location1) { // xy and carpet
//            for (auto x = 0; x < _domain_feature_size._S[_agent_x_feature]; ++x) {
//                for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y) {
//
//                    auto const loc = (feature == _agent_x_feature) ? x : y;
//
//                    auto const new_loc =
//                            (feature == _agent_x_feature)
//                            ? _domain
//                                    .applyMove(
//                                            {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}, &action)
//                                    .x
//                            : _domain
//                                    .applyMove(
//                                            {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}, &action)
//                                    .y;
//
//                    for (auto carpet_config = 0; carpet_config < _domain_feature_size._S[_carpet_feature1]; ++carpet_config){
//                        // carpet: 0.15, no carpet: 0.95
//                        float const success_prob = (_domain.agentOnCarpet({(unsigned int) x,(unsigned int) y}))
//                                                   ? 0.1 : 0.95;
//
//                        // fail move
//                        model->transitionNode(&action, feature).count({x, y, carpet_config}, loc) +=
//                                (1 - success_prob) * _unknown_counts_total;
//
//                        // success move
//                        model->transitionNode(&action, feature).count({x, y, carpet_config}, new_loc) +=
//                                (success_prob) * _unknown_counts_total;
//                    }
//                }
//            }
//        } else { // xy
//            throw "this shouldn't be happening";
//        }
//    } else if(rain_location) {
//        if (carpet_location1) { // rain and carpet
//            for (auto loc = 0; loc < _domain_feature_size._S[feature]; ++loc) {
//                    auto const new_loc =
//                            (feature == _agent_x_feature)
//                            ? _domain
//                                    .applyMove(
//                                            {static_cast<unsigned int>(loc), static_cast<unsigned int>(loc)}, &action)
//                                    .x
//                            : _domain
//                                    .applyMove(
//                                            {static_cast<unsigned int>(loc), static_cast<unsigned int>(loc)}, &action)
//                                    .y;
//
//                    for (auto rain = 0; rain < _domain_feature_size._S[_rain_feature]; ++rain){
//                        for (auto carpet_config = 0; carpet_config < _domain_feature_size._S[_carpet_feature1]; ++carpet_config){
//                            // no carpet: rain: 0.8, no rain: 0.95
//                            // carpet: rain: 0.05, no rain: 0.15
//                            float const success_prob = rain ? 0.9 : 0.95;
//                            // can't really tell if you are on carpet if you do not take the other (x/y) into account
////                                        (GridWorldCoffeeBig::carpet_func({(unsigned int) x,(unsigned int) y}))
////                                                           ? (rain? 0.05 : 0.15)
////                                                           : (rain ? 0.8 : 0.95);
//
//                            // fail move
//                            model->transitionNode(&action, feature).count({loc, rain, carpet_config}, loc) +=
//                                    (1 - success_prob) * _unknown_counts_total;
//
//                            // success move
//                            model->transitionNode(&action, feature).count({loc, rain, carpet_config}, new_loc) +=
//                                    (success_prob) * _unknown_counts_total;
//                        }
////                        }
//                }
//            }
//        } else { // rain
//            for (auto loc = 0; loc < _domain_feature_size._S[feature]; ++loc) {
////                    for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y) {
//
////                        auto const loc = (feature == _agent_x_feature) ? x : y;
//
//                    auto const new_loc =
//                            (feature == _agent_x_feature)
//                            ? _domain
//                                    .applyMove(
//                                            {static_cast<unsigned int>(loc), static_cast<unsigned int>(loc)}, &action)
//                                    .x
//                            : _domain
//                                    .applyMove(
//                                            {static_cast<unsigned int>(loc), static_cast<unsigned int>(loc)}, &action)
//                                    .y;
//
//                    for (auto rain = 0; rain < _domain_feature_size._S[_rain_feature]; ++rain){
//                        // no carpet: rain: 0.8, no rain: 0.95
//                        // carpet: rain: 0.05, no rain: 0.15
//                        float const success_prob = rain? 0.9 : 0.95;
//
//                        // fail move
//                        model->transitionNode(&action, feature).count({loc, rain}, loc) +=
//                                (1 - success_prob) * _unknown_counts_total;
//
//                        // success move
//                        model->transitionNode(&action, feature).count({loc, rain}, new_loc) +=
//                                (success_prob) * _unknown_counts_total;
//                    }
////                    }
//            }
//        }
//    } else if (carpet_location1) { // carpet
//        for (auto loc = 0; loc < _domain_feature_size._S[feature]; ++loc) {
////                for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y) {
//
////                    auto const loc = (feature == _agent_x_feature) ? x : y;
//
//                auto const new_loc =
//                        (feature == _agent_x_feature)
//                        ? _domain
//                                .applyMove(
//                                        {static_cast<unsigned int>(loc), static_cast<unsigned int>(loc)}, &action)
//                                .x
//                        : _domain
//                                .applyMove(
//                                        {static_cast<unsigned int>(loc), static_cast<unsigned int>(loc)}, &action)
//                                .y;
//
//                for (auto carpet_config = 0; carpet_config < _domain_feature_size._S[_carpet_feature1]; ++carpet_config){
//                    // no carpet: rain: 0.8, no rain: 0.95
//                    // carpet: rain: 0.05, no rain: 0.15
//                    float const success_prob = GridWorldCoffeeBig::move_prob;
////                                (GridWorldCoffeeBig::carpet_func({(unsigned int) x,(unsigned int) y}))
////                                                   ? 0.15 : 0.95;
//
//                    // fail move
//                    model->transitionNode(&action, feature).count({loc, carpet_config}, loc) +=
//                            (1 - success_prob) * _unknown_counts_total;
//
//                    // success move
//                    model->transitionNode(&action, feature).count({loc, carpet_config}, new_loc) +=
//                            (success_prob) * _unknown_counts_total;
//                }
////                }
//        }
//    } else { // feature (x or y) only has itself as parent
//        for (auto loc = 0; loc < _domain_feature_size._S[feature]; ++loc) {
////                for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y) {
//
////                    auto const loc = (feature == _agent_x_feature) ? x : y;
//
//                auto const new_loc =
//                        (feature == _agent_x_feature)
//                        ? _domain
//                                .applyMove( // can use loc for both x and y since for applyMove it doesn't matter
//                                        {static_cast<unsigned int>(loc), static_cast<unsigned int>(loc)}, &action)
//                                .x
//                        : _domain
//                                .applyMove(
//                                        {static_cast<unsigned int>(loc), static_cast<unsigned int>(loc)}, &action)
//                                .y;
//
//                float const trans_prob = GridWorldCoffeeBig::move_prob;
//
//                // fail move
//                model->transitionNode(&action, feature).count({loc}, loc) +=
//                        (1 - trans_prob) * _unknown_counts_total;
//
//                // success move
//                model->transitionNode(&action, feature).count({loc}, new_loc) +=
//                        (trans_prob) * _unknown_counts_total;
//            }
////            }
//        }
//}

FBAPOMDPState* GridWorldCoffeeBigFactBAPrior::sampleFullyConnectedState(State const* /*domain_state*/) const
{
    throw "GridWorldFactBAPrior::sampleFullyConnectedState nyi";
}

    FBAPOMDPState const* GridWorldCoffeeBigFactBAPrior::sampleCorrectGraphState(State const* domain_state) const
{
    if (useBigAbstraction) {
        return new AbstractFBAPOMDPState(domain_state, _correct_struct_prior);
    }
    return new FBAPOMDPState(domain_state, _correct_struct_prior);
}

// TODO constructs the correct model? or the model that the agent beliefs is correct?
// changed to do the second one (the model the agent beliefs is correct, so no knowledge about slow locations)
void GridWorldCoffeeBigFactBAPrior::preComputePrior()
{
    for (auto a = 0; a < _domain_size._A; ++a)
    {
        IndexAction const action(a);

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

        // observe rain and carpet function deterministically
//        for (auto f = 2; f < (int) _domain_feature_size._O.size(); ++f)
//        {
//            _correct_struct_prior.resetObservationNode(&action, f, {f});
//
//            for (auto v = 0; v < _domain_feature_size._S[f]; ++v)
//            {
//                _correct_struct_prior.observationNode(&action, f).count({v}, v) =
//                        _static_total_count;
//            }
//        }

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

//        _correct_struct_prior.resetTransitionNode(
//                &action, _carpet_feature1, {_carpet_feature1});
//        _correct_struct_prior.resetTransitionNode(
//                &action, _carpet_feature2, {_carpet_feature2});
//        _correct_struct_prior.resetTransitionNode(
//                &action, _carpet_feature3, {_carpet_feature3});
//        _correct_struct_prior.resetTransitionNode(
//                &action, _carpet_feature4, {_carpet_feature4});

        // agent is sure about the carpet, stays the same
        for (auto feature = 3; feature < (int)_domain_feature_size._S.size(); ++feature)
        {
            for (auto feature_val = 0; feature_val < 2; ++feature_val)
            {
                _correct_struct_prior.transitionNode(&action, feature)
                        .count({feature_val}, feature_val) += _static_total_count;
            }
        }

        // rain
        for (auto r = 0; r < _domain_feature_size._S[_rain_feature]; ++r)
        {
            _correct_struct_prior.transitionNode(&action, _rain_feature).count({r}, r) +=
                    GridWorldCoffeeBig::same_weather_prob * _unknown_counts_total;
            _correct_struct_prior.transitionNode(&action, _rain_feature).count({r}, (1-r)) +=
                    (1 - GridWorldCoffeeBig::same_weather_prob) * _unknown_counts_total;
        }

        // X and Y
        for (auto x = 0; x < _domain_feature_size._S[_agent_x_feature]; ++x)
        {
            for (auto y = 0; y < _domain_feature_size._S[_agent_y_feature]; ++y)
            {
                float trans_prob = GridWorldCoffeeBig::move_prob;
                if (correctBigPrior) {
                    if ((x == 0 && y == 3) || (x == 1 && y == 3)
                        || (x == 2 && y == 1)) {
                        trans_prob = 0.1;
                    }
                }

//                    (_domain.agentOnSlowLocation(
//                        {static_cast<unsigned int>(x), static_cast<unsigned int>(y)}))
//                    ? GridWorldCoffeeBig::slow_move_prob
//                    : GridWorldCoffeeBig::move_prob;

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
    if (correctBigPrior) {
        if (useBigAbstraction) {
            return new AbstractFBAPOMDPState(domain_state, _correct_struct_prior);
        }
        return new FBAPOMDPState(domain_state, _correct_struct_prior);
    }

    /*** noisy struct prior ****/
    auto structure = _correct_struct_prior.structure();

    // what to do here
    // randomly remove y from parents of x for each action
    // randomly remove x from parents of y for each action
    // randomly add features to parents of x and y for each action
    for (auto a = 0; a < _domain_size._A; ++a)
    {
//        if (rnd::boolean()) // randomly remove y from parents of x for each action
//        {
//            auto lower_bound  = std::lower_bound(structure.T[a][_agent_x_feature].begin(), structure.T[a][_agent_x_feature].end(), _agent_y_feature);
//            structure.T[a][_agent_x_feature].erase(lower_bound);
//        }
//        if (rnd::boolean()) // randomly remove x from parents of y for each action
//        {
//            auto lower_bound  = std::lower_bound(structure.T[a][_agent_y_feature].begin(), structure.T[a][_agent_y_feature].end(), _agent_x_feature);
//            structure.T[a][_agent_y_feature].erase(lower_bound);
//        }
        if (rnd::boolean()) // randomly add rain to parents of x
        {
            structure.T[a][_agent_x_feature].emplace_back(_rain_feature);
        }
        if (rnd::boolean())  // randomly add rain to parents of y
        {
            structure.T[a][_agent_y_feature].emplace_back(_rain_feature);
        }
        // uniformly add any extra binary feature as parent
        for (auto f = 3; f < (int)_domain_feature_size._S.size(); ++f)
        {
            if (rnd::boolean()) // randomly add carpet to parents of x
            {
                structure.T[a][_agent_x_feature].emplace_back(f);
            }
            if (rnd::boolean()) // randomly add carpet to parents of y
            {
                structure.T[a][_agent_y_feature].emplace_back(f);
            }
        }
//        if (rnd::boolean()) // randomly add carpet to parents of x
//        {
//            structure.T[a][_agent_x_feature].emplace_back(_carpet_feature1);
//        }
//        if (rnd::boolean()) // randomly add carpet to parents of y
//        {
//            structure.T[a][_agent_y_feature].emplace_back(_carpet_feature1);
//        }

    }
    if (useBigAbstraction) {
        return new AbstractFBAPOMDPState(domain_state, computePriorModel(structure));
    }
    return new FBAPOMDPState(domain_state, computePriorModel(structure));
}

    Domain_Feature_Size *GridWorldCoffeeBigFactBAPrior::getDomainFeatureSize() {
        return &_domain_feature_size;
    }

} // namespace priors
