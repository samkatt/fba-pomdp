#include "GridWorldCoffeeBig.hpp"

#include <algorithm> // Why?
#include <cstdlib>

#include "easylogging++.h"

#include "environment/Reward.hpp"
#include "utils/index.hpp"
#include "utils/random.hpp"

namespace domains {

std::vector<std::string> const GridWorldCoffeeBig::GridWorldCoffeeBigAction::action_descriptions = {"UP",
                                                                                  "RIGHT",
                                                                                  "DOWN",
                                                                                  "LEFT"};

GridWorldCoffeeBig::GridWorldCoffeeBigState::pos const GridWorldCoffeeBig::start_location = {0, 0};
std::vector<GridWorldCoffeeBig::GridWorldCoffeeBigState::pos> const GridWorldCoffeeBig::goal_locations = {{2,4}, {3,4}, {4,3}};
std::vector<GridWorldCoffeeBig::GridWorldCoffeeBigState::pos> const GridWorldCoffeeBig::slow_locations = {{0, 3},{1, 3}}; //, {2, 1}};
std::vector<GridWorldCoffeeBig::GridWorldCoffeeBigState::pos> const GridWorldCoffeeBig::slippery_locations = {{2,3}}; //, {2,3}}; // {1, 0},{1, 1}, {2, 3}};



GridWorldCoffeeBig::GridWorldCoffeeBig(size_t extra_features, bool store_statespace ):
    _extra_features(extra_features),
    _store_statespace(store_statespace),
    _extrafeatures_space(std::vector<int>(_extra_features, 2)),
    _feature_space(std::vector<int>(3 + _extra_features, 2)), // further initiated below
    _S_size(0), // initiated below
    _O_size(0) // initiated below
{
    if (_extra_features > 1000) {
        throw "please enter a size between 0 and 15 to be able to run gridworldcoffeebig \
                    (you entered " + std::to_string(_extra_features) + ")";
    }
    _feature_space[0] = _size;
    _feature_space[1] = _size;

    // Number of states, 5x5 grid, rain/no rain, carpet on tile (0 or 1, 1 per tile).
    if (_store_statespace) {
        _S_size = (_size * _size * 2) * std::pow(2, _extra_features); //, _S.reserve(_S_size);
        _O_size = _size * _size; //

        _S.reserve(_S_size);
        _O.reserve(_O_size);

        // generate state space
        // TODO use indexing::project?
        int index = 0;
        for (unsigned int x_agent = 0; x_agent < _size; ++x_agent)
        {
            for (unsigned int y_agent = 0; y_agent < _size; ++y_agent)
            {

                GridWorldCoffeeBigState::pos const agent_pos{x_agent, y_agent};
                for (unsigned int rain = 0; rain < 2; ++rain)
                {
//                    std::vector<int> pos_and_rain = {static_cast<int>(x_agent), static_cast<int>(y_agent), static_cast<int>(rain)};
                    std::vector<int> binary_features(_extra_features);
//                    auto bin_i = 0;
                    do
                    {
//                        auto const i = positionsToIndex(agent_pos, rain, binary_features);
                        assert(static_cast<unsigned int>(index) == _S.size());

                        std::vector<int> state_vector = {static_cast<int>(x_agent), static_cast<int>(y_agent), static_cast<int>(rain)};
                        state_vector.insert(state_vector.end(), binary_features.begin(), binary_features.end());
//                        std::string index = positionsToIndex(state_vector);

                        _S.emplace_back(GridWorldCoffeeBigState(state_vector, std::to_string(index++)));
                    } while (!indexing::increment(binary_features, _extrafeatures_space));
                }
                auto const j = positionsToObservationIndex(agent_pos);

                assert(static_cast<unsigned int>(j) == _O.size());

                _O.emplace_back(GridWorldCoffeeBigObservation(agent_pos, j));

            }
        }
    } else { // do O anyway
        _O_size = _size * _size; //

        _O.reserve(_O_size);

        // generate state space
        // TODO use indexing::project?
        for (unsigned int x_agent = 0; x_agent < _size; ++x_agent)
        {
            for (unsigned int y_agent = 0; y_agent < _size; ++y_agent)
            {

                GridWorldCoffeeBigState::pos const agent_pos{x_agent, y_agent};

                auto const j = positionsToObservationIndex(agent_pos);

                assert(static_cast<unsigned int>(j) == _O.size());

                _O.emplace_back(GridWorldCoffeeBigObservation(agent_pos, j));
            }
        }
    }

    VLOG(1) << "initiated gridworld-coffee";
}


size_t GridWorldCoffeeBig::size() const
{
    return _size;
}

double GridWorldCoffeeBig::goalReward() const
{
    return goal_reward;
}

float GridWorldCoffeeBig::obsDisplProb(unsigned int loc, unsigned int observed_loc) const
{

    // compute displacement
    auto const displacement = std::abs(static_cast<int>(loc - observed_loc));

    // to be returned
    float res = (displacement == 0) ? 1 - wrong_obs_prob // special case: no displacement
                                    : _obs_displacement_probs[displacement] * .5; // per direction

    // special cases: if observed location is on the edge, then
    // the probability includes the accumulation of observing 'beyond' the edge
    if (observed_loc == _size - 1 || observed_loc == 0)
    {
        for (size_t i = displacement + 1; i < _obs_displacement_probs.size(); ++i)
        { res += _obs_displacement_probs[i] * .5; }
    }

    return res;
}

//float GridWorldCoffeeBig::believedTransitionProb(bool const& rain, int const& features_on) {
//    return rain ? rain_move_prob*pow(move_prob_reduction, features_on)  // rain
//    : move_prob*pow(move_prob_reduction, features_on); // no rain
//}

// Todo: change for the new carpet tiles? CHECK IF WORKS
bool GridWorldCoffeeBig::agentOnCarpet(GridWorldCoffeeBigState::pos const& agent_pos, unsigned int const& carpet_config) const {
    // _carpet_tiles between 0 and 15. Carpet can not be in the top or bottom row.
    // _carpet_config should be an int between 0 and 2^{_carpet_tiles}.
    // carpet states, rectangular area
    // TODO change
    if (agent_pos.y > 0 && agent_pos.y < 4)
    {
        // can use projectUsingStepSize? in indexing
        auto tilesWithCarpet = indexing::projectUsingStepSize(carpet_config, _stepSizes);
        unsigned int agentPos = (agent_pos.y - 1)*_size + agent_pos.x;
        if (agentPos > tilesWithCarpet.size()) {
            return false;
        } else if (tilesWithCarpet[agentPos] == 1) {
            return true; // carpet
        }
    }
    return false; // No carpet
}

bool GridWorldCoffeeBig::agentOnSlowLocation(GridWorldCoffeeBigState::pos const& agent_pos) const
{
    // On trap state
    return std::find(slow_locations.begin(), slow_locations.end(), agent_pos)
           != slow_locations.end();
}

bool GridWorldCoffeeBig::agentOnSlipperyLocation(GridWorldCoffeeBigState::pos const& agent_pos, int rain) const
{
    if (rain == 0) {
        return false;
    }
    // On slippery state
    return std::find(slippery_locations.begin(), slippery_locations.end(), agent_pos)
           != slippery_locations.end();
}

bool GridWorldCoffeeBig::foundGoal(GridWorldCoffeeBigState const* s) const
{
    return std::find(goal_locations.begin(), goal_locations.end(),
                     GridWorldCoffeeBigState::pos({static_cast<unsigned int>(s->_state_vector[0]),
                                                       static_cast<unsigned int>(s->_state_vector[1])}))
                        != goal_locations.end();
//    return (GridWorldCoffeeBigState::pos({static_cast<unsigned int>(s->_state_vector[0]),
//                                          static_cast<unsigned int>(s->_state_vector[1])}) == goal_location);
}

State const* GridWorldCoffeeBig::sampleRandomState() const
{
    auto randomState = std::vector<int>(3+_extra_features);
    for (auto i = 0; i < 2; i++) {
        randomState[i] = static_cast<unsigned int>(rnd::slowRandomInt(0, _size));
    }
    for (auto i = 2; i < (int) (3 + _extra_features); i++) {
        randomState[i] = static_cast<unsigned int>(rnd::slowRandomInt(0, 2));
    }
    return getState(randomState);
}

void GridWorldCoffeeBig::clearCache() const {
    _S_cache.clear();
//    _O_cache.clear();
}

GridWorldCoffeeBig::GridWorldCoffeeBigState const* GridWorldCoffeeBig::getState ( std::vector<int> const& state_vector) const
{
    if (_store_statespace) {
        return &_S[std::stoi(positionsToIndex(state_vector))];
    } else {
        auto index = positionsToIndex(state_vector);
        auto search = _S_cache.find(index);
        if (search != _S_cache.end()) { // found
            return &search->second;
        } else { // not found
            return &_S_cache.emplace(index, GridWorldCoffeeBigState(state_vector, index)).first->second;
        }
    }
}

GridWorldCoffeeBig::GridWorldCoffeeBigState const* GridWorldCoffeeBig::getState ( std::string index) const
{
    if (_store_statespace) {
        return &_S[std::stoi(index)];
    } else {
        auto search = _S_cache.find(index);
        if (search != _S_cache.end()) { // found
            return &search->second;
        } else { // not found
            return &_S_cache.emplace(index, GridWorldCoffeeBigState(getStateVectorFromIndex(index), index)).first->second;
        }
    }
}

GridWorldCoffeeBig::GridWorldCoffeeBigObservation const* GridWorldCoffeeBig::getObservation(
    GridWorldCoffeeBigState::pos const& agent_pos) const
{
    // try with storing O
    return &_O[positionsToObservationIndex(agent_pos)];
//    if (_store_statespace) {
//        return &_O[positionsToObservationIndex(agent_pos)];
//    } else {
//        auto index = positionsToObservationIndex(agent_pos);
//        auto search = _O_cache.find(std::to_string(index));
//        if (search != _O_cache.end()) { // found
//            return &search->second;
//        } else { // not found
//            return &_O_cache.emplace(std::to_string(index), GridWorldCoffeeBigObservation(agent_pos, index)).first->second;
////            auto inserted = _O_cache.emplace(std::to_string(index), GridWorldCoffeeBigObservation(agent_pos, index));
////            return &inserted.first->second;
//        }
////        auto const * observation = new GridWorldCoffeeBigObservation(agent_pos, positionsToObservationIndex(agent_pos));
////        return observation;
//    }
}

Action const* GridWorldCoffeeBig::generateRandomAction(State const* s) const
{
    assertLegal(s);
    return new GridWorldCoffeeBigAction(rnd::slowRandomInt(0, _A_size));
}

void GridWorldCoffeeBig::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assert(actions != nullptr && actions->empty());
    assertLegal(s);

    for (auto a = 0; a < _A_size; ++a) { actions->emplace_back(new GridWorldCoffeeBigAction(a)); }
}

double GridWorldCoffeeBig::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* new_s) const
{
    assertLegal(o);
    assertLegal(a);
    assertLegal(new_s);

//    auto const& agent_pos    = static_cast<GridWorldCoffeeBigState const*>(new_s)->_agent_position;
//    auto const& observed_pos = static_cast<GridWorldCoffeeBigObservation const*>(o)->_agent_pos;

    return obsDisplProb(static_cast<GridWorldCoffeeBigState const*>(new_s)->_state_vector[0],
                        static_cast<GridWorldCoffeeBigObservation const*>(o)->_observation_vector[0])
                        * obsDisplProb(static_cast<GridWorldCoffeeBigState const*>(new_s)->_state_vector[1],
                                       static_cast<GridWorldCoffeeBigObservation const*>(o)->_observation_vector[1]);
}

void GridWorldCoffeeBig::releaseAction(Action const* a) const
{
    assertLegal(a);
    if (a != nullptr) {
        delete static_cast<GridWorldCoffeeBigAction*>(const_cast<Action*>(a));
    }
}

Action const* GridWorldCoffeeBig::copyAction(Action const* a) const
{
    assertLegal(a);
    return new GridWorldCoffeeBigAction(std::stoi(a->index()));
}

State const* GridWorldCoffeeBig::sampleStartState() const
{
    // sample random start state, fixed location but random carpet configuration.
    auto randomState = std::vector<int>(3+_extra_features, 0);
    randomState[0] = start_location.x;
    randomState[1] = start_location.y;
    for (auto i = 2; i < (int) (3 + _extra_features); i++) {
        randomState[i] = static_cast<unsigned int>(rnd::slowRandomInt(0, 2));
    }
    return getState(randomState);
}

Terminal GridWorldCoffeeBig::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    assert(s != nullptr && o != nullptr && r != nullptr);
    assertLegal(*s);
    assertLegal(a);

    auto grid_state       = static_cast<GridWorldCoffeeBigState const*>(*s);
    auto const& agent_pos = GridWorldCoffeeBigState::pos({static_cast<unsigned int>(grid_state->_state_vector[_x_feature]),
                                                          static_cast<unsigned int>(grid_state->_state_vector[_y_feature])});

    /*** T ***/

    // if it's rains, 0.7 that it keeps raining
    // if it's dry, 0.7 that it stays dry
    bool const same_weather = rnd::uniform_rand01() < same_weather_prob;
    auto new_vector = grid_state->getFeatureValues();
    new_vector[_rain_feature] = (same_weather) ? grid_state->_state_vector[_rain_feature] :
                                            (1 - grid_state->_state_vector[_rain_feature]);

    // calculate whether move succeeds based on whether it is
    // on top of a slow surface
    // no carpet: rain: 0.8, no rain: 0.95
    // carpet: rain: 0.05, no rain: 0.15
    bool const move_succeeds = (agentOnSlowLocation(agent_pos))
            ? rnd::uniform_rand01() < slow_move_prob
            : (grid_state->_state_vector[_rain_feature]) ?
              ((agentOnSlipperyLocation(agent_pos, grid_state->_state_vector[_rain_feature]))
            ? rnd::uniform_rand01() < move_prob_slippery_rain : rnd::uniform_rand01() < move_prob_rain)
            : rnd::uniform_rand01() < move_prob;

    auto const new_agent_pos = (move_succeeds) ? applyMove(agent_pos, a) : agent_pos;
    new_vector[_x_feature] = new_agent_pos.x;
    new_vector[_y_feature] = new_agent_pos.y;

    auto const found_goal = foundGoal(grid_state);

//    auto const new_index = positionsToIndex(new_agent_pos, new_rain, grid_state->_feature_config);
    *s = getState(new_vector);

    /*** R & O ***/
    r->set(found_goal ? goal_reward : step_reward);
    *o = generateObservation(new_agent_pos);

    return Terminal(found_goal);
}

void GridWorldCoffeeBig::releaseObservation(Observation const* o) const
{
    assertLegal(o);
//    if (!_store_statespace) {
//        delete (o);
//    }
}

void GridWorldCoffeeBig::releaseState(State const* s) const
{
    assertLegal(s);
//    if (!_store_statespace) {
//        _S_cache.erase(positionsToIndex(s->getFeatureValues()));
//    }
}

Observation const* GridWorldCoffeeBig::copyObservation(Observation const* o) const
{
    assertLegal(o);
    return o;
}

State const* GridWorldCoffeeBig::copyState(State const* s) const
{
    assertLegal(s);
    return s;
}

//int GridWorldCoffeeBig::positionsToIndex(
//    GridWorldCoffeeBigState::pos const& agent_pos,
//    unsigned int const& rain, unsigned int const& feature_config) const
//{
//    // indexing: from 3 elements projecting to 1 dimension
//    // x*size*2 + y*2 + rain
//    return agent_pos.x * ((_size * 2) * std::pow(2, _extra_features)) + agent_pos.y * (2 * std::pow(2, _extra_features)) + rain * std::pow(2, _extra_features) + feature_config; // + velocity;
//}

std::string GridWorldCoffeeBig::positionsToIndex( std::vector<int> state_vector) const {
    // indexing: from 3 elements projecting to 1 dimension
    // x*size*2 + y*2 + rain
    if (_store_statespace) {
        int index = indexing::project(state_vector, _feature_space);
        return std::to_string(index);
//        return std::to_string(agent_pos.x * ((_size * 2) * std::pow(2, _extra_features)) + agent_pos.y * (2 * std::pow(2, _extra_features)) + rain * std::pow(2, _extra_features) + feature_config);
    }
    std::string index;
    for (int i=0; i < (int) state_vector.size() - 1; i++){
        index += std::to_string(state_vector[i]);
        index += '+';
    }
    index += std::to_string(state_vector[state_vector.size() - 1]);

    return index;
}

int GridWorldCoffeeBig::positionsToObservationIndex(
    GridWorldCoffeeBigState::pos const& agent_pos) const
{
    // indexing: from 2 elements projecting to 1 dimension
    return agent_pos.x * _size + agent_pos.y;
}


GridWorldCoffeeBig::GridWorldCoffeeBigState::pos
GridWorldCoffeeBig::applyMove(GridWorldCoffeeBigState::pos const& old_pos, Action const* a) const
{
    assertLegal(a);

    auto x = old_pos.x;
    auto y = old_pos.y;

    // move according to action
    switch (std::stoi(a->index()))
    {
        case GridWorldCoffeeBigAction::ACTION::UP:
            y = (old_pos.y == _size - 1) ? old_pos.y : old_pos.y + 1;
            break;
        case GridWorldCoffeeBigAction::ACTION::DOWN: y = (old_pos.y == 0) ? old_pos.y : old_pos.y - 1; break;
        case GridWorldCoffeeBigAction::ACTION::RIGHT:
            x = (old_pos.x == _size - 1) ? old_pos.x : old_pos.x + 1;
            break;
        case GridWorldCoffeeBigAction::ACTION::LEFT: x = (old_pos.x == 0) ? old_pos.x : old_pos.x - 1; break;
    }

    return {x, y};
}

Observation const* GridWorldCoffeeBig::generateObservation(
    GridWorldCoffeeBigState::pos const& agent_pos) const
{
    // sample x & y displacement
    auto const x_displacement = rnd::sample::Dir::sampleFromMult(
            _obs_displacement_probs.data(), _obs_displacement_probs.size(), 1);
    auto const y_displacement = rnd::sample::Dir::sampleFromMult(
            _obs_displacement_probs.data(), _obs_displacement_probs.size(), 1);

    // calculate observed positions
    auto const observed_x =
            (rnd::boolean())
            ? std::max(
                    0, static_cast<int>(agent_pos.x - x_displacement)) // x_displacement to the left
            : std::min(
                    agent_pos.x + x_displacement,
                    static_cast<unsigned int>(_size) - 1); // displacement to the right

    auto const observed_y =
            (rnd::boolean())
            ? std::max(
                    0, static_cast<int>(agent_pos.y - y_displacement)) // y_displacement to the left
            : std::min(
                    agent_pos.y + y_displacement,
                    static_cast<unsigned int>(_size) - 1); // displacement to the right

    // return corresponding indexed observation
    return getObservation({observed_x, observed_y});
}

void GridWorldCoffeeBig::assertLegal(Action const* a) const
{
    assert(a != nullptr);
    assert(std::stoi(a->index()) >= 0 && std::stoi(a->index()) < _A_size);
}

void GridWorldCoffeeBig::assertLegal(Observation const* o) const
{
    assert(o != nullptr);
//    assert(std::stoi(o->index()) >= 0 && std::stoi(o->index()) < _O_size);
    assertLegal({static_cast<unsigned int>(static_cast<GridWorldCoffeeBigObservation const*>(o)->_observation_vector[0]),
                                                                 static_cast<unsigned int>(static_cast<GridWorldCoffeeBigObservation const*>(o)->_observation_vector[1])});
}

void GridWorldCoffeeBig::assertLegal(State const* s) const
{
    assert(s != nullptr);
    // TODO fix
//    assert(std::stoi(s->index()) >= 0 &&std::stoi(s->index())< _S_size);
//    assertLegal(static_cast<GridWorldCoffeeBigState const*>(s)->_agent_position);
//    assert((static_cast<GridWorldCoffeeBigState const*>(s)->_state_vector[_rain_feature] == 0)
//            || (static_cast<GridWorldCoffeeBigState const*>(s)->_state_vector[_rain_feature] == 1));
//    assert(static_cast<GridWorldCoffeeBigState const*>(s)->_feature_config < (unsigned int) std::pow(2, _extra_features));
}

void GridWorldCoffeeBig::assertLegal(GridWorldCoffeeBigState::pos const& position) const
{
    assert(position.x < _size);
    assert(position.y < _size);
}

std::vector<int> GridWorldCoffeeBig::getStateVectorFromIndex(std::string index) const {
    auto state_vector = std::vector<int>(3 + _extra_features);
    std::string value;
    int entry = 0;
    for (char const &c: index) {
        if (c == '+') {
            state_vector[entry] = std::stoi(value);
            value.clear();
            entry++;
        } else {
            value += c;
        }
    }
    auto test = std::stoi(value);
    if (test > 1){

        return state_vector;
    }

    state_vector[state_vector.size()-1] = std::stoi(value);

    return state_vector;
}

std::vector<int> GridWorldCoffeeBig::GridWorldCoffeeBigState::getFeatureValues() const {
    return _state_vector;
}

std::vector<int> GridWorldCoffeeBig::GridWorldCoffeeBigObservation::getFeatureValues() const {

    return _observation_vector;
}
} // namespace domains
