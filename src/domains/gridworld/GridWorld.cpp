#include "GridWorld.hpp"

#include <algorithm>
#include <cstdlib>

#include "easylogging++.h"

#include "environment/Reward.hpp"
#include "utils/index.hpp"
#include "utils/random.hpp"

namespace domains {

std::vector<std::string> const GridWorld::GridWorldAction::action_descriptions = {"UP",
                                                                                  "RIGHT",
                                                                                  "DOWN",
                                                                                  "LEFT"};

std::vector<GridWorld::GridWorldState::pos> const GridWorld::start_locations = {{0, 0}};

GridWorld::GridWorld(size_t size) :
        _size(size),
        _slow_locations({}), // initialized below
        _goal_locations({}), // initialized below
        _goal_amount(0), // initialized below
        _S_size(0) // initiated below
{

    if (_size < 3)
    {
        throw "please enter a size larger than 3 to be able to run gridworld \
                (you entered " + std::to_string(_size) + ")";
    }

    generateSlowLocations();

    _goal_locations = goalLocations(_size);
    _goal_amount    = _goal_locations.size();

    _S_size = _size * _size * _goal_amount, _S.reserve(_S_size);
    _O.reserve(_S_size);

    // generate state space
    for (unsigned int x_agent = 0; x_agent < _size; ++x_agent)
    {
        for (unsigned int y_agent = 0; y_agent < _size; ++y_agent)
        {
            for (auto const& goal_pos : _goal_locations)
            {
                GridWorldState::pos const agent_pos{x_agent, y_agent};

                auto const i = positionsToIndex(agent_pos, goal_pos);

                assert(static_cast<unsigned int>(i) == _S.size());
                assert(static_cast<unsigned int>(i) == _O.size());

                _S.emplace_back(GridWorldState(agent_pos, goal_pos, i));
                _O.emplace_back(GridWorldObservation(agent_pos, goal_pos, i));
            }
        }
    }

    /** observation function **/
    _obs_displacement_probs.emplace_back(1 - wrong_obs_prob);

    // probability of generating location further and further away with diminishing return
    auto prob = wrong_obs_prob;
    for (size_t i = 1; i < _size - 1; ++i)
    {
        prob *= .5;
        _obs_displacement_probs.emplace_back(prob);
    }

    // fill up all probability of the end to ensure it sums up to 1
    _obs_displacement_probs.emplace_back(prob);

    VLOG(1) << "initiated gridworld";
}

void GridWorld::generateSlowLocations()
{
    unsigned int const edge = _size - 1;

    // bottom left side
    if (_size > 5)
    {
        _slow_locations.push_back({1, 1});
    }

    if (_size == 3)
    {
        _slow_locations.push_back({1, 1});
        return;
    }

    if (_size < 7)
    {
        _slow_locations.push_back({edge - 1, edge - 2});
        _slow_locations.push_back({edge - 2, edge - 1});
        return;
    }

    // 7 or bigger
    _slow_locations.push_back({edge - 1, edge - 3});
    _slow_locations.push_back({edge - 3, edge - 1});
    _slow_locations.push_back({edge - 2, edge - 2});
}

size_t GridWorld::size() const
{
    return _size;
}

double GridWorld::goalReward() const
{
    return goal_reward;
}

double GridWorld::correctObservationProb() const
{
    // wrong_obs_prob describes the probability of observing
    // the wrong location in 1 (out of 2) axis
    return (1 - wrong_obs_prob) * (1 - wrong_obs_prob);
}

std::vector<GridWorld::GridWorldState::pos> GridWorld::goalLocations(int size)
{
    auto const edge          = static_cast<unsigned int>(size - 1);
    unsigned int const start = (size < 5) ? size - 2 : (size < 7) ? size - 3 : size - 4;

    std::vector<GridWorldState::pos> locations;
    // fill in side line of goals (up and right)
    for (auto i = start; i < static_cast<unsigned int>(size - 1); ++i)
    {
        locations.push_back({i, edge});
        locations.push_back({edge, i});
    }

    locations.push_back({edge, edge});

    // fill in rest of goals
    if (size > 3)
    {
        locations.push_back({edge - 1, edge - 1});
    }

    if (size > 6)
    {
        locations.push_back({edge - 2, edge - 1});
        locations.push_back({edge - 1, edge - 2});
    }

    return locations;
}

std::vector<GridWorld::GridWorldState::pos> const* GridWorld::slowLocations() const
{
    return &_slow_locations;
}

GridWorld::GridWorldState::pos const* GridWorld::goalLocation(unsigned int goal_index) const
{
    return &_goal_locations[goal_index];
}

float GridWorld::obsDisplProb(unsigned int loc, unsigned int observed_loc) const
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

bool GridWorld::agentOnSlowLocation(GridWorldState::pos const& agent_pos) const
{
    return std::find(_slow_locations.begin(), _slow_locations.end(), agent_pos)
           != _slow_locations.end();
}

bool GridWorld::foundGoal(GridWorldState const* s) const
{
    return s->_goal_position == s->_agent_position;
}

State const* GridWorld::sampleRandomState() const
{
    return getState(
        {static_cast<unsigned int>(rnd::slowRandomInt(0, _size)),
         static_cast<unsigned int>(rnd::slowRandomInt(0, _size))},
        {_goal_locations[rnd::slowRandomInt(0, _goal_amount)]});
}

GridWorld::GridWorldState const* GridWorld::getState(
    GridWorldState::pos const& agent_pos,
    GridWorldState::pos const& goal_pos) const
{
    return &_S[positionsToIndex(agent_pos, goal_pos)];
}

GridWorld::GridWorldObservation const* GridWorld::getObservation(
    GridWorldState::pos const& agent_pos,
    GridWorldState::pos const& goal_pos) const
{
    return &_O[positionsToIndex(agent_pos, goal_pos)];
}

Action const* GridWorld::generateRandomAction(State const* s) const
{
    assertLegal(s);
    return new GridWorldAction(rnd::slowRandomInt(0, _A_size));
}

void GridWorld::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assert(actions != nullptr && actions->empty());
    assertLegal(s);

    for (auto a = 0; a < _A_size; ++a) { actions->emplace_back(new GridWorldAction(a)); }
}

double GridWorld::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* new_s) const
{
    assertLegal(o);
    assertLegal(a);
    assertLegal(new_s);

    auto const& agent_pos    = static_cast<GridWorldState const*>(new_s)->_agent_position;
    auto const& observed_pos = static_cast<GridWorldObservation const*>(o)->_agent_pos;

    return obsDisplProb(agent_pos.x, observed_pos.x) * obsDisplProb(agent_pos.y, observed_pos.y);
}

void GridWorld::releaseAction(Action const* a) const
{
    assertLegal(a);
    delete static_cast<GridWorldAction*>(const_cast<Action*>(a));
}

Action const* GridWorld::copyAction(Action const* a) const
{
    assertLegal(a);
    return new GridWorldAction(a->index());
}

State const* GridWorld::sampleStartState() const
{

    // sample random agent position
    auto const agent_pos = start_locations[rnd::slowRandomInt(0, start_locations.size())];

    auto const goal_pos = _goal_locations[rnd::slowRandomInt(0, _goal_locations.size())];

    return &_S[positionsToIndex(agent_pos, goal_pos)];
}

Terminal GridWorld::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    assert(s != nullptr && o != nullptr && r != nullptr);
    assertLegal(*s);
    assertLegal(a);

    auto grid_state       = static_cast<GridWorldState const*>(*s);
    auto const& agent_pos = grid_state->_agent_position;
    auto const& goal_pos  = grid_state->_goal_position;

    /*** T ***/

    // calculate whether move succeeds based on whether it is
    // on top of a slow surface
    bool const move_succeeds = (agentOnSlowLocation(grid_state->_agent_position))
                                   ? (rnd::uniform_rand01() < slow_move_prob)
                                   : (rnd::uniform_rand01() < move_prob);

    auto const new_agent_pos = (move_succeeds) ? applyMove(agent_pos, a) : agent_pos;

    auto const found_goal = foundGoal(grid_state);
    auto const new_goal_pos =
        (found_goal) ? _goal_locations[rnd::slowRandomInt(0, _goal_locations.size())] : goal_pos;

    auto const new_index = positionsToIndex(new_agent_pos, new_goal_pos);
    *s                   = &_S[new_index];

    /*** R & O ***/
    r->set(found_goal ? goal_reward : step_reward);
    *o = generateObservation(new_agent_pos, new_goal_pos);

    return Terminal(found_goal);
}

void GridWorld::releaseObservation(Observation const* o) const
{
    assertLegal(o);
}

void GridWorld::releaseState(State const* s) const
{
    assertLegal(s);
}

Observation const* GridWorld::copyObservation(Observation const* o) const
{

    assertLegal(o);
    return o;
}

State const* GridWorld::copyState(State const* s) const
{
    assertLegal(s);
    return s;
}

int GridWorld::positionsToIndex(
    GridWorldState::pos const& agent_pos,
    GridWorldState::pos const& goal_pos) const
{
    assertLegalGoal(goal_pos);

    auto const goal_index = std::find(_goal_locations.begin(), _goal_locations.end(), goal_pos)
                            - _goal_locations.begin();

    // indexing: from 3 elements projecting to 1 dimension
    return agent_pos.x * _size * _goal_amount + agent_pos.y * _goal_amount + goal_index;
}

GridWorld::GridWorldState::pos
    GridWorld::applyMove(GridWorldState::pos const& old_pos, Action const* a) const
{
    assertLegal(a);

    auto x = old_pos.x;
    auto y = old_pos.y;

    // move according to action
    switch (a->index())
    {
        case GridWorldAction::ACTION::UP:
            y = (old_pos.y == _size - 1) ? old_pos.y : old_pos.y + 1;
            break;
        case GridWorldAction::ACTION::DOWN: y = (old_pos.y == 0) ? old_pos.y : old_pos.y - 1; break;
        case GridWorldAction::ACTION::RIGHT:
            x = (old_pos.x == _size - 1) ? old_pos.x : old_pos.x + 1;
            break;
        case GridWorldAction::ACTION::LEFT: x = (old_pos.x == 0) ? old_pos.x : old_pos.x - 1; break;
    }

    return {x, y};
}

Observation const* GridWorld::generateObservation(
    GridWorldState::pos const& agent_pos,
    GridWorldState::pos const& goal_pos) const
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
    return &_O[positionsToIndex({observed_x, observed_y}, goal_pos)];
}

void GridWorld::assertLegal(Action const* a) const
{
    assert(a != nullptr);
    assert(a->index() >= 0 && a->index() < _A_size);
}

void GridWorld::assertLegal(Observation const* o) const
{
    assert(o != nullptr);
    assert(o->index() >= 0 && o->index() < _S_size);
    assertLegal(static_cast<GridWorldObservation const*>(o)->_agent_pos);
    assertLegal(static_cast<GridWorldObservation const*>(o)->_goal_pos);
}

void GridWorld::assertLegal(State const* s) const
{
    assert(s != nullptr);
    assert(s->index() >= 0 && s->index() < _S_size);
    assertLegal(static_cast<GridWorldState const*>(s)->_agent_position);
    assertLegal(static_cast<GridWorldState const*>(s)->_goal_position);
}

void GridWorld::assertLegal(GridWorldState::pos const& position) const
{
    assert(position.x < _size);
    assert(position.y < _size);
}

void GridWorld::assertLegalGoal(GridWorld::GridWorldState::pos const& position) const
{
    assert(
        std::find(_goal_locations.begin(), _goal_locations.end(), position)
        != _goal_locations.end());
}

} // namespace domains
