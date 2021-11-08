#include "GridWorldCoffee.hpp"

#include <algorithm> // Why?
#include <cstdlib>

#include "easylogging++.h"

#include "environment/Reward.hpp"
#include "utils/index.hpp"
#include "utils/random.hpp"

namespace domains {

std::vector<std::string> const GridWorldCoffee::GridWorldCoffeeAction::action_descriptions = {"UP",
                                                                                  "RIGHT",
                                                                                  "DOWN",
                                                                                  "LEFT"};

GridWorldCoffee::GridWorldCoffeeState::pos const GridWorldCoffee::start_location = {0, 0};
GridWorldCoffee::GridWorldCoffeeState::pos const GridWorldCoffee::goal_location = {2,4};


GridWorldCoffee::GridWorldCoffee() :
    _S_size(0), // initiated below
    _O_size(0) // initiated below
{
    // Number of states, 5x5 grid, rain/no rain, carpet configuration (0 or 1).
    _S_size = _size * _size * 2 * _carpet_configurations, _S.reserve(_S_size);
    _O_size = _size * _size; //  * 2 * _carpet_configurations, _O.reserve(_O_size); // same number of observations

    // generate state space
    for (unsigned int x_agent = 0; x_agent < _size; ++x_agent)
    {
        for (unsigned int y_agent = 0; y_agent < _size; ++y_agent)
        {
            GridWorldCoffeeState::pos const agent_pos{x_agent, y_agent};
            for (unsigned int rain = 0; rain < 2; ++rain)
            {
                for (unsigned int carpet_config = 0; carpet_config < 2; ++carpet_config)
                {


                    auto const i = positionsToIndex(agent_pos, rain, carpet_config);

                    assert(static_cast<unsigned int>(i) == _S.size());

                    _S.emplace_back(GridWorldCoffeeState(agent_pos, rain, carpet_config, i));
                }
            }
            auto const j = positionsToObservationIndex(agent_pos);

            assert(static_cast<unsigned int>(j) == _O.size());

            _O.emplace_back(GridWorldCoffeeObservation(agent_pos, j));

        }
    }

    /** observation function **/
    _obs_displacement_probs.emplace_back(1 - wrong_obs_prob);

    // TODO pas aan
    // probability of generating location further and further away with diminishing return
//    auto prob = wrong_obs_prob;
//    auto prob_sum = .0;
//    for (size_t i = 1; i < _size - 2; ++i) // i < _size -3?
//    {
//        prob *= .55;
//        prob_sum += prob;
//        _obs_displacement_probs.emplace_back(prob);
//    }

    // fill up all probability of the end to ensure it sums up to 1
    auto prob = wrong_obs_prob;
    _obs_displacement_probs.emplace_back(prob);
//    _obs_displacement_probs.emplace_back(wrong_obs_prob - prob_sum);
    _obs_displacement_probs.emplace_back(0);
    _obs_displacement_probs.emplace_back(0);
    _obs_displacement_probs.emplace_back(0);

    // probability of generating location further and further away with diminishing return
//    auto prob = wrong_obs_prob;
//    for (size_t i = 1; i < _size - 1; ++i)
//    {
//        prob *= .5;
//        _obs_displacement_probs.emplace_back(prob);
//    }
//
//    // fill up all probability of the end to ensure it sums up to 1
//    _obs_displacement_probs.emplace_back(prob);

    VLOG(1) << "initiated gridworld-coffee";
}


size_t GridWorldCoffee::size() const
{
    return _size;
}

//size_t GridWorldCoffee::carpet_configurations() const
//{
//    return _carpet_configurations;
//}

double GridWorldCoffee::goalReward() const
{
    return goal_reward;
}

float GridWorldCoffee::obsDisplProb(unsigned int loc, unsigned int observed_loc) const
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

bool GridWorldCoffee::agentOnCarpet(GridWorldCoffeeState::pos const& agent_pos) const {
    // carpet states, rectangular area
    if (agent_pos.x < 3 && agent_pos.y > 0 && agent_pos.y < 4)
    {
        return true; // Carpet
    }
    return false; // No carpet
}

bool GridWorldCoffee::agentOnSlowLocation(GridWorldCoffeeState::pos const& agent_pos) const
{
    // On trap state
    if ((agent_pos.x == 0 && agent_pos.y == 3) || (agent_pos.x == 1 && agent_pos.y == 3)
        || (agent_pos.x == 2 && agent_pos.y == 1)) // || (agent_pos.x == 3 && agent_pos.y == 2))
    {
        return true;
    }
    return false;
}

bool GridWorldCoffee::foundGoal(GridWorldCoffeeState const* s) const
{
    return (s->_agent_position == goal_location);
}

State const* GridWorldCoffee::sampleRandomState() const
{
    return getState(
        {static_cast<unsigned int>(rnd::slowRandomInt(0, _size)),
         static_cast<unsigned int>(rnd::slowRandomInt(0, _size))},
        static_cast<unsigned int>(rnd::slowRandomInt(0, 2)),
        static_cast<unsigned int>(rnd::slowRandomInt(0, 2)));
}

GridWorldCoffee::GridWorldCoffeeState const* GridWorldCoffee::getState(
    GridWorldCoffeeState::pos const& agent_pos,
    unsigned int const& rain, unsigned int const& carpet_config) const
{
    return &_S[positionsToIndex(agent_pos, rain, carpet_config)];
}

GridWorldCoffee::GridWorldCoffeeObservation const* GridWorldCoffee::getObservation(
    GridWorldCoffeeState::pos const& agent_pos
//    , unsigned int const& rain, unsigned int const& carpet_config
    ) const
{
    return &_O[positionsToObservationIndex(agent_pos)]; // , rain, carpet_config)];
}

Action const* GridWorldCoffee::generateRandomAction(State const* s) const
{
    assertLegal(s);
    return new GridWorldCoffeeAction(rnd::slowRandomInt(0, _A_size));
}

void GridWorldCoffee::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assert(actions != nullptr && actions->empty());
    assertLegal(s);

    for (auto a = 0; a < _A_size; ++a) { actions->emplace_back(new GridWorldCoffeeAction(a)); }
}

double GridWorldCoffee::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* new_s) const
{
    assertLegal(o);
    assertLegal(a);
    assertLegal(new_s);

    auto const& agent_pos    = static_cast<GridWorldCoffeeState const*>(new_s)->_agent_position;
    auto const& observed_pos = static_cast<GridWorldCoffeeObservation const*>(o)->_agent_pos;
//    auto const& rain = static_cast<GridWorldCoffeeState const*>(new_s)->_rain;
//    auto const& observed_rain = static_cast<GridWorldCoffeeObservation const*>(o)->_rain;
//    auto const& carpet_config = static_cast<GridWorldCoffeeState const*>(new_s)->_carpet_config;
//    auto const& observed_carpet_config = static_cast<GridWorldCoffeeObservation const*>(o)->_carpet_config;

//    if ((rain == observed_rain) && (carpet_config == observed_carpet_config)){
    return obsDisplProb(agent_pos.x, observed_pos.x) * obsDisplProb(agent_pos.y, observed_pos.y);
//    }
    return 0;
}

void GridWorldCoffee::releaseAction(Action const* a) const
{
    assertLegal(a);
    if (a != nullptr) {
        delete static_cast<GridWorldCoffeeAction*>(const_cast<Action*>(a));
    }
}

Action const* GridWorldCoffee::copyAction(Action const* a) const
{
    assertLegal(a);
    return new GridWorldCoffeeAction(std::stoi(a->index()));
}

State const* GridWorldCoffee::sampleStartState() const
{
    // sample random agent position (there should be only 1, (0,0), no rain)
    auto const agent_pos = start_location; //s[rnd::slowRandomInt(0, start_locations.size())];

    return &_S[positionsToIndex(agent_pos, 0, 0)]; //, 1)];
}

Terminal GridWorldCoffee::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    assert(s != nullptr && o != nullptr && r != nullptr);
    assertLegal(*s);
    assertLegal(a);

    auto grid_state       = static_cast<GridWorldCoffeeState const*>(*s);
    auto const& agent_pos = grid_state->_agent_position;

    /*** T ***/

    // if it's rains, 0.7 that it keeps raining
    // if it's dry, 0.7 that it stays dry
    bool const same_weather = rnd::uniform_rand01() < same_weather_prob;
    auto const new_rain = (same_weather) ? grid_state->_rain : (1 - grid_state->_rain);

    // calculate whether move succeeds based on whether it is
    // on top of a slow surface
    // no carpet: rain: 0.8, no rain: 0.95
    // carpet: rain: 0.05, no rain: 0.15
    // trap states TODO: make a property of the domain
    bool const move_succeeds = (agentOnSlowLocation(grid_state->_agent_position))
            ? rnd::uniform_rand01() < slow_move_prob
            : rnd::uniform_rand01() < move_prob;

    auto const new_agent_pos = (move_succeeds) ? applyMove(agent_pos, a) : agent_pos;

    auto const found_goal = foundGoal(grid_state);

//    auto const new_carpet_config = grid_state->_carpet_config;

    auto const new_index = positionsToIndex(new_agent_pos, new_rain, grid_state->_carpet_config);
    *s                   = &_S[new_index];

    /*** R & O ***/
    r->set(found_goal ? goal_reward : step_reward);
    *o = generateObservation(new_agent_pos); // , new_rain, grid_state->_carpet_config);

    return Terminal(found_goal);
}

void GridWorldCoffee::releaseObservation(Observation const* o) const
{
    assertLegal(o);
}

void GridWorldCoffee::releaseState(State const* s) const
{
    assertLegal(s);
}

Observation const* GridWorldCoffee::copyObservation(Observation const* o) const
{
    assertLegal(o);
    return o;
}

State const* GridWorldCoffee::copyState(State const* s) const
{
    assertLegal(s);
    return s;
}

int GridWorldCoffee::positionsToIndex(
    GridWorldCoffeeState::pos const& agent_pos,
    unsigned int const& rain, unsigned int const& carpet_config) const
{
    // indexing: from 3 elements projecting to 1 dimension
    // x*size*2 + y*2 + rain
    return agent_pos.x * _size * 2 * _carpet_configurations + agent_pos.y * 2 * _carpet_configurations + rain * _carpet_configurations + carpet_config; // + velocity;
}

int GridWorldCoffee::positionsToObservationIndex(
    GridWorldCoffeeState::pos const& agent_pos
//    , unsigned int const& rain, unsigned int const& carpet_config
    ) const
{
    // indexing: from 3 elements projecting to 1 dimension
    // x*size*2 + y*2 + rain
//    return agent_pos.x * _size * 2 * _carpet_configurations + agent_pos.y * 2 * _carpet_configurations + rain * _carpet_configurations + carpet_config;
    return agent_pos.x * _size + agent_pos.y;
}


GridWorldCoffee::GridWorldCoffeeState::pos
GridWorldCoffee::applyMove(GridWorldCoffeeState::pos const& old_pos, Action const* a) const
{
    assertLegal(a);

    auto x = old_pos.x;
    auto y = old_pos.y;

    // move according to action
    switch (std::stoi(a->index()))
    {
        case GridWorldCoffeeAction::ACTION::UP:
            y = (old_pos.y == _size - 1) ? old_pos.y : old_pos.y + 1;
            break;
        case GridWorldCoffeeAction::ACTION::DOWN: y = (old_pos.y == 0) ? old_pos.y : old_pos.y - 1; break;
        case GridWorldCoffeeAction::ACTION::RIGHT:
            x = (old_pos.x == _size - 1) ? old_pos.x : old_pos.x + 1;
            break;
        case GridWorldCoffeeAction::ACTION::LEFT: x = (old_pos.x == 0) ? old_pos.x : old_pos.x - 1; break;
    }

    return {x, y};
}

Observation const* GridWorldCoffee::generateObservation(
    GridWorldCoffeeState::pos const& agent_pos
//    , unsigned int const& rain, unsigned int const& carpet_config
    ) const
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
    return &_O[positionsToObservationIndex({observed_x, observed_y})]; // , rain, carpet_config)];
}

void GridWorldCoffee::assertLegal(Action const* a) const
{
    assert(a != nullptr);
    assert(std::stoi(a->index()) >= 0 && std::stoi(a->index()) < _A_size);
}

void GridWorldCoffee::assertLegal(Observation const* o) const
{
    assert(o != nullptr);
    assert(std::stoi(o->index()) >= 0 && std::stoi(o->index()) < _O_size);
    assertLegal(static_cast<GridWorldCoffeeObservation const*>(o)->_agent_pos);
//    assert((static_cast<GridWorldCoffeeObservation const*>(o)->_rain == 0) || (static_cast<GridWorldCoffeeObservation const*>(o)->_rain == 1));
//    assert(static_cast<GridWorldCoffeeObservation const*>(o)->_carpet_config < _carpet_configurations);
}

void GridWorldCoffee::assertLegal(State const* s) const
{
    assert(s != nullptr);
    assert(std::stoi(s->index()) >= 0 &&std::stoi(s->index())< _S_size);
    assertLegal(static_cast<GridWorldCoffeeState const*>(s)->_agent_position);
    assert((static_cast<GridWorldCoffeeState const*>(s)->_rain == 0) || (static_cast<GridWorldCoffeeState const*>(s)->_rain == 1));
    assert(static_cast<GridWorldCoffeeState const*>(s)->_carpet_config < _carpet_configurations);
}

void GridWorldCoffee::assertLegal(GridWorldCoffeeState::pos const& position) const
{
    assert(position.x < _size);
    assert(position.y < _size);
}

void GridWorldCoffee::clearCache() const {

}

std::vector<int> GridWorldCoffee::GridWorldCoffeeObservation::getFeatureValues() const {
    // TODO implement
    return std::vector<int>();
}

std::vector<int> GridWorldCoffee::GridWorldCoffeeState::getFeatureValues() const {
    // TODO implement
    return std::vector<int>();
}
} // namespace domains
