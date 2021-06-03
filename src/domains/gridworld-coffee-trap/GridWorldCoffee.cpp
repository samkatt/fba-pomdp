//
// Created by rolf on 28-09-20.
//

#include "GridWorldCoffee.hpp"

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
    // Number of states, 5x5 grid, rain/no rain, carpet (is a function, returning 1 or 0). Carpet is fixed given x and y, and observable by the agent.
    // TODO: "velocity" beliefs should be in the prior? velocity now gone. Need Carpet somewhere?
    // There are only two types of velocity (normal and trap state) but the agent believes there are 4 (the 4 combinations of rain/no rain + carpet / no carpet)
//    const unsigned int carpet_configurations = 2; // just give 2 as the total number of possible configurations?
    //  pow(2, _size*_size);
    _S_size = _size * _size * 2 * _carpet_configurations, _S.reserve(_S_size);
    _O_size = _size * _size * 2 * _carpet_configurations;
    _O.reserve(_O_size); // same number of observations, just doesn't include the velocity (now velocity is gone)

    // generate state space
    for (unsigned int x_agent = 0; x_agent < _size; ++x_agent)
    {
        for (unsigned int y_agent = 0; y_agent < _size; ++y_agent)
        {
            GridWorldCoffee::GridWorldCoffeeState::pos const agent_pos{x_agent, y_agent};
            for (unsigned int rain = 0; rain < 2; ++rain)
            {
                for (unsigned int carpet_config = 0; carpet_config < 2; ++carpet_config)
                {
                    auto const i = positionsToIndex(agent_pos, rain, carpet_config);
                    assert(static_cast<unsigned int>(i) == _S.size());

                    _S.emplace_back(GridWorldCoffeeState(agent_pos, rain, carpet_config, i));
    //                auto const i = positionsToObservationIndex(agent_pos, rain);
                    assert(static_cast<unsigned int>(i) == _O.size());
                    _O.emplace_back(GridWorldCoffeeObservation(agent_pos, rain, carpet_config, i));
                }
            }
        }
    }
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

double GridWorldCoffee::goalReward()
{
    return goal_reward;
}

State const* GridWorldCoffee::sampleRandomState() const
{
    return getState(
        {static_cast<unsigned int>(rnd::slowRandomInt(0, _size)),
         static_cast<unsigned int>(rnd::slowRandomInt(0, _size))},
        static_cast<unsigned int>(rnd::slowRandomInt(0, 2)),
        static_cast<unsigned int>(rnd::slowRandomInt(0, 2)));
        //static_cast<unsigned int>(rnd::slowRandomInt(0, 3)));
}

GridWorldCoffee::GridWorldCoffeeState const* GridWorldCoffee::getState(
    GridWorldCoffee::GridWorldCoffeeState::pos const& agent_pos,
    unsigned int const& rain, unsigned int const& carpet_config) const
//    unsigned int const& velocity) const
{
    return &_S[positionsToIndex(agent_pos, rain, carpet_config)]; //, velocity)];
}

GridWorldCoffee::GridWorldCoffeeObservation const* GridWorldCoffee::getObservation(
    GridWorldCoffee::GridWorldCoffeeState::pos const& agent_pos,
    unsigned int const& rain,
    unsigned int const& carpet_config) const
{
    return &_O[positionsToObservationIndex(agent_pos, rain, carpet_config)];
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

    auto const& agent_pos    = dynamic_cast<GridWorldCoffeeState const*>(new_s)->_agent_position;
    auto const& observed_pos = dynamic_cast<GridWorldCoffeeObservation const*>(o)->_agent_pos;
    auto const& rain = dynamic_cast<GridWorldCoffeeState const*>(new_s)->_rain;
    auto const& observed_rain = dynamic_cast<GridWorldCoffeeObservation const*>(o)->_rain;
    auto const& carpet_config = dynamic_cast<GridWorldCoffeeState const*>(new_s)->_carpet_config;
    auto const& observed_carpet_config = dynamic_cast<GridWorldCoffeeObservation const*>(o)->_carpet_config;

    return ((agent_pos == observed_pos) && (rain == observed_rain) && (carpet_config == observed_carpet_config)) ? 1 : 0;
}

void GridWorldCoffee::releaseAction(Action const* a) const
{
    assertLegal(a);
    delete dynamic_cast<GridWorldCoffeeAction*>(const_cast<Action*>(a));
}

Action const* GridWorldCoffee::copyAction(Action const* a) const
{
    assertLegal(a);
    return new GridWorldCoffeeAction(a->index());
}

State const* GridWorldCoffee::sampleStartState() const
{
    // sample random agent position (there should be only 1, (0,0), no rain)
    auto const agent_pos = start_location; //s[rnd::slowRandomInt(0, start_locations.size())];

    return &_S[positionsToIndex(agent_pos, 0, 0)]; //, 1)];
}

bool GridWorldCoffee::agentOnSlowLocation(GridWorldCoffeeState::pos const& agent_pos) const
{
    // On trap state
    if ((agent_pos.x == 0 && agent_pos.y == 2) || (agent_pos.x == 1 && agent_pos.y == 3)
    || (agent_pos.x == 2 && agent_pos.y == 1) || (agent_pos.x == 3 && agent_pos.y == 2))
    {
        return true;
    }
    return false;
}

bool GridWorldCoffee::foundGoal(GridWorldCoffeeState const* s) const
{
    return (s->_agent_position == goal_location);
}

Terminal GridWorldCoffee::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    assert(s != nullptr && o != nullptr && r != nullptr);
    assertLegal(*s);
    assertLegal(a);

    auto grid_state       = dynamic_cast<GridWorldCoffeeState const*>(*s);
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
    bool move_succeeds;

    // trap states TODO: make a property of the domain
    // On trap state
    if (agentOnSlowLocation(agent_pos))
    {
        move_succeeds = rnd::uniform_rand01() < slow_move_prob;
    } else {
        move_succeeds = rnd::uniform_rand01() < move_prob;
    }

    auto const new_agent_pos = (move_succeeds) ? applyMove(agent_pos, a) : agent_pos;

    auto const found_goal = foundGoal(grid_state);

    auto const new_carpet_config = grid_state->_carpet_config;

//    auto const new_index = ; //, new_velocity);
    *s                   = &_S[positionsToIndex(new_agent_pos, new_rain, new_carpet_config)];

    /*** R & O ***/
    r->set(found_goal ? goalReward() : step_reward);
    *o = generateObservation(new_agent_pos, new_rain, new_carpet_config);

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
    GridWorldCoffee::GridWorldCoffeeState::pos const& agent_pos,
    unsigned int const& rain, unsigned int const& carpet_config) const // , unsigned int const& velocity) const
{
    // indexing: from 3 elements projecting to 1 dimension
    // x*size*2 + y*2 + rain
//    return agent_pos.x * _size * 2 * 4 + agent_pos.y * 2 * 4 + rain * 4; // + velocity;
    return agent_pos.x * _size * 2 * _carpet_configurations + agent_pos.y * 2 * _carpet_configurations + rain * _carpet_configurations + carpet_config; // + velocity;

}

int GridWorldCoffee::positionsToObservationIndex(
    GridWorldCoffee::GridWorldCoffeeState::pos const& agent_pos,
    unsigned int const& rain,
    unsigned int const& carpet_config) const
{
    // indexing: from 3 elements projecting to 1 dimension
    // x*size*2 + y*2 + rain
    return agent_pos.x * _size * 2 * _carpet_configurations + agent_pos.y * 2 * _carpet_configurations + rain * _carpet_configurations + carpet_config;
}


GridWorldCoffee::GridWorldCoffeeState::pos
GridWorldCoffee::applyMove(GridWorldCoffee::GridWorldCoffeeState::pos const& old_pos, Action const* a) const
{
    assertLegal(a);

    auto x = old_pos.x;
    auto y = old_pos.y;

    // move according to action
    switch (a->index())
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
    GridWorldCoffee::GridWorldCoffeeState::pos const& agent_pos,
    unsigned int const& rain,
    unsigned int const& carpet_config) const
{
    // return corresponding indexed observation
    return &_O[positionsToObservationIndex(agent_pos, rain, carpet_config)];
}

void GridWorldCoffee::assertLegal(Action const* a) const
{
    assert(a != nullptr);
    assert(a->index() >= 0 && a->index() < _A_size);
}

void GridWorldCoffee::assertLegal(Observation const* o) const
{
    assert(o != nullptr);
    assert(o->index() >= 0 && o->index() < _O_size);
    assertLegal(dynamic_cast<GridWorldCoffeeObservation const*>(o)->_agent_pos);
    assert((dynamic_cast<GridWorldCoffeeObservation const*>(o)->_rain == 0) || (dynamic_cast<GridWorldCoffeeObservation const*>(o)->_rain == 1));
    assert((dynamic_cast<GridWorldCoffeeObservation const*>(o)->_carpet_config < _carpet_configurations));
}

void GridWorldCoffee::assertLegal(State const* s) const
{
    assert(s != nullptr);
    assert(s->index() >= 0 && s->index() < _S_size);
    assertLegal(dynamic_cast<GridWorldCoffeeState const*>(s)->_agent_position);
    assert((dynamic_cast<GridWorldCoffeeState const*>(s)->_rain == 0) || (dynamic_cast<GridWorldCoffeeState const*>(s)->_rain == 1));
    assert((dynamic_cast<GridWorldCoffeeState const*>(s)->_carpet_config < _carpet_configurations));
}

void GridWorldCoffee::assertLegal(GridWorldCoffee::GridWorldCoffeeState::pos const& position) const
{
    assert(position.x < _size);
    assert(position.y < _size);
}

} // namespace domains
