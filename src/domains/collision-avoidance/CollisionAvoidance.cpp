#include "CollisionAvoidance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include "easylogging++.h"

#include "utils/index.hpp"

namespace domains {

// NOTE: these re-declarations seem to be necessary for some (buggy?) compilers
constexpr int const CollisionAvoidance::NUM_ACTIONS;
constexpr double const CollisionAvoidance::MOVE_PENALTY;
constexpr double const CollisionAvoidance::COLLIDE_PENALTY;
constexpr double const CollisionAvoidance::BLOCK_MOVE_PROB;

/**
 * \brief An observation in the collision avoidance domain
 *
 * the location of obstacles
 **/
struct CollisionAvoidanceObservation : public Observation
{

    // cppcheck-suppress passedByValue
    CollisionAvoidanceObservation(int index, std::vector<int> pos) :
            _index(index), obstacles_pos(std::move(pos))
    {
    }

    /**** observation interface ****/

    void index(int /*i*/) final
    {
        throw "CollisionAvoidanceObservation::index(i) should not be called...?";
    }

    int index() const final { return _index; }

    std::string toString() const final
    {
        std::string s("{");

        for (unsigned int j = 0; j < obstacles_pos.size() - 1; j++)
        {
            s += std::to_string(obstacles_pos[j]) + ", ";
        }

        return s + std::to_string(obstacles_pos.back()) + "}";
    }

    int const _index;
    std::vector<int> const obstacles_pos;
};

CollisionAvoidance::CollisionAvoidance(
    int grid_width,
    int grid_height,
    int num_obstacles,
    VERSION version) :
        _grid_width(grid_width),
        _grid_height(grid_height),
        _num_obstacles(num_obstacles),
        _version(version)
{

    if (_grid_width < 1)
    {
        throw("Cannot initiate CollisionAvoidance with width " + std::to_string(_grid_width));
    }

    if (grid_height < 1 || grid_height % 2 != 1)
    {
        throw(
            "Cannot initiate CollisionAvoidance with height " + std::to_string(_grid_height)
            + ", must be uneven");
    }

    if (_num_obstacles > _grid_width)
    {
        throw "cannot initiate collision avoidance with more obstacles ("
            + std::to_string(_num_obstacles) + " ) than columns (" + std::to_string(_grid_width)
            + ")!";
    }

    // locally create & store all states
    auto index = 0;
    for (auto x = 0; x < grid_width; ++x)
    {
        for (auto y_agent = 0; y_agent < grid_height; ++y_agent)
        {

            std::vector<int> obstacles(_num_obstacles);
            auto obs_i = 0;
            do {
                _states[x][y_agent][obs_i++] =
                    new CollisionAvoidanceState(x, y_agent, obstacles, index++);
            } while (!indexing::increment(obstacles, _obstacles_space));
        }
    }

    // initialize all observations
    {
        int count = 0;
        std::vector<int> obstacles_pos(_num_obstacles);
        do {
            _observations[count] = new CollisionAvoidanceObservation(count, obstacles_pos);
            count++;
        } while (!indexing::increment(obstacles_pos, _obstacles_space));
    }

    // initiatlize the observation error probabilities
    for (auto d = 0; d < grid_height; ++d)
    {
        _observation_error_probability.emplace_back(
            rnd::normal::cdf(d + .5, 0, 1) - rnd::normal::cdf(d - .5, 0, 1));
    }

    if (version == INIT_RANDOM_POSITION)
    {

        auto const init_state_prob = 1.f / std::pow(_grid_height, _num_obstacles + 1);

        for (auto agent_y = 0; agent_y < _grid_height; ++agent_y)
        {
            auto count = 0;
            std::vector<int> obstacles(_num_obstacles);
            do {
                _state_prior.setRawValue(
                    _states[_grid_width - 1][agent_y][count++]->index(), init_state_prob);
            } while (!indexing::increment(obstacles, _obstacles_space));
        }

    } else // version is initialize centre
    {
        // this sets prior such that only 1 initial state, where obstacles and agent start in middle
        std::vector<int> obstacles(_num_obstacles, _grid_height / 2);

        _state_prior.setRawValue(
            _states[_grid_width - 1][_grid_height / 2]
                   [indexing::project(obstacles, _obstacles_space)]
                       ->index(),
            1);
    }

    VLOG(1) << "Initiated CA of size (w:" << _grid_width << ", h:" << _grid_height
            << ", n:" << _num_obstacles << ")";
}

CollisionAvoidance::~CollisionAvoidance()
{
    // clean up allocated states
    for (auto& outer : _states)
    {
        for (auto& inner : outer)
        {
            for (auto s : inner) { delete s; }
        }
    }

    for (auto& o : _observations) { delete o; }
}

CollisionAvoidance::VERSION CollisionAvoidance::type() const
{
    return _version;
}

int CollisionAvoidance::xAgent(State const* s) const
{
    return static_cast<CollisionAvoidanceState const*>(s)->x_agent;
}

int CollisionAvoidance::yAgent(State const* s) const
{
    return static_cast<CollisionAvoidanceState const*>(s)->y_agent;
}

std::vector<int> const& CollisionAvoidance::yObstacles(State const* s) const
{
    return static_cast<CollisionAvoidanceState const*>(s)->obstacles_pos;
}

State const* CollisionAvoidance::getState(int index) const
{
    auto features = indexing::projectUsingDimensions(
        index,
        {_grid_width, _grid_height, static_cast<int>(std::pow(_grid_height, _num_obstacles))});

    return _states[features[0]][features[1]][features[2]];
}

Reward CollisionAvoidance::reward(Action const* a, State const* new_s) const
{
    auto r = a->index() == STAY ? 0 : -MOVE_PENALTY;

    auto const x_agent = xAgent(new_s);
    if (x_agent < _num_obstacles && yAgent(new_s) == yObstacles(new_s)[x_agent])
    {
        return Reward(-COLLIDE_PENALTY);
    }

    return Reward(r);
}

void CollisionAvoidance::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assertLegal(s);
    assert(actions->empty());

    for (auto i = 0; i < NUM_ACTIONS; ++i) { actions->emplace_back(_actions.get(i)); }
}

double CollisionAvoidance::computeObservationProbability(
    Observation const* o,
    Action const* /*a*/,
    State const* new_s) const
{

    auto const& pos = static_cast<CollisionAvoidanceState const*>(new_s)->obstacles_pos;
    auto const& obs = static_cast<CollisionAvoidanceObservation const*>(o)->obstacles_pos;

    double p = 1;

    for (auto i = 0; i < _num_obstacles; ++i)
    {
        p *= _observation_error_probability[std::abs(pos[i] - obs[i])];
    }

    return p;
}

Terminal
    CollisionAvoidance::step(State const** s, Action const* a, Observation const** o, Reward* r)
        const
{
    assertLegal(*s);
    assertLegal(a);

    auto const* collision_state = static_cast<CollisionAvoidanceState const*>(*s);

    // move agent
    auto x = collision_state->x_agent - 1;
    auto y = keepInGrid(collision_state->y_agent + a->index() - 1);

    // move block
    auto blocks = collision_state->obstacles_pos;
    for (auto& b : blocks) { b = moveObstacle(b); }
    *s = _states[x][y][indexing::project(blocks, _obstacles_space)];

    // generate observation
    for (auto& b : blocks)
    {
        auto observation_noise = static_cast<int>(std::round(_observation_distr(rnd::rng())));
        b                      = keepInGrid(b + observation_noise);
    }
    *o = _observations[indexing::project(blocks, _obstacles_space)];

    assertLegal(*s);
    assertLegal(*o);

    r->set(reward(a, *s).toDouble());

    auto const crashed = xAgent(*s) < _num_obstacles && yAgent(*s) == yObstacles(*s)[xAgent(*s)];

    return Terminal(crashed || xAgent(*s) == 0);
}

State const* CollisionAvoidance::sampleStartState() const
{
    return getState(_state_prior.sample());
}

State const* CollisionAvoidance::getState(int x, int y, std::vector<int> const& obstacles) const
{
    assert(static_cast<unsigned int>(_num_obstacles) == obstacles.size());
    assert(x >= 0 && x <= _grid_width);
    assert(y >= 0 && y <= _grid_height);

    for (auto b : obstacles) { assert(b >= 0 && b <= _grid_height); }

    return _states[x][y][indexing::project(obstacles, _obstacles_space)];
}

Action const* CollisionAvoidance::getAction(move m) const
{
    return _actions.get(m);
}

Observation const* CollisionAvoidance::getObservation(int y) const
{
    return _observations[y];
}

Action const* CollisionAvoidance::generateRandomAction(State const* /*s*/) const
{
    // any action (up, stay, down) is possible in any state
    return getAction(move(_action_distr(rnd::rng())));
}

void CollisionAvoidance::releaseAction(Action const* a) const
{
    assertLegal(a);

    // all actions are stored locally and re-used,
    // so no need to deallocate them
}

void CollisionAvoidance::releaseObservation(Observation const* o) const
{
    assertLegal(o);

    // all observations are stored locally and re-used,
    // so no need to deallocate them
}

void CollisionAvoidance::releaseState(State const* s) const
{
    assertLegal(s);

    // all states are stored locally and re-used,
    // so no need to deallocate them
}

Action const* CollisionAvoidance::copyAction(Action const* a) const
{
    assertLegal(a);

    // all actions are stored locally and re-used
    return a;
}

Observation const* CollisionAvoidance::copyObservation(Observation const* o) const
{
    assertLegal(o);

    // all observations are stored locally and re-used,
    return o;
}

State const* CollisionAvoidance::copyState(State const* s) const
{
    assertLegal(s);

    // all observations are stored locally and re-used
    return s;
}

int CollisionAvoidance::keepInGrid(int y) const
{
    return std::max(0, std::min(_grid_height - 1, y));
}

int CollisionAvoidance::moveObstacle(int current_position) const
{
    assert(current_position >= 0 && current_position < _grid_height);

    auto prob = rnd::uniform_rand01();

    auto m = (prob < BLOCK_MOVE_PROB)              ? STAY
             : (prob > .5 * (1 + BLOCK_MOVE_PROB)) ? MOVE_UP
                                                   : MOVE_DOWN;

    // apply move but stay within bounds
    return keepInGrid(current_position + m - 1);
}

void CollisionAvoidance::assertLegal(State const* s) const
{
    assert(s != nullptr);

    auto const* collision_state = static_cast<CollisionAvoidanceState const*>(s);

    assert(collision_state->x_agent >= 0 && collision_state->x_agent < _grid_width);
    assert(collision_state->y_agent >= 0 && collision_state->y_agent < _grid_height);
    for (auto i = 0; i < _num_obstacles; ++i)
    {
        assert(
            collision_state->obstacles_pos[i] >= 0
            && collision_state->obstacles_pos[i] < _grid_height);
    }

    assert(
        s->index()
        == _states[collision_state->x_agent][collision_state->y_agent]
                  [indexing::project(collision_state->obstacles_pos, _obstacles_space)]
                      ->index());
}

void CollisionAvoidance::assertLegal(Action const* a) const
{
    assert(a != nullptr && a->index() >= 0 && a->index() < NUM_ACTIONS);
}

void CollisionAvoidance::assertLegal(Observation const* o) const
{
    assert(o != nullptr);

    for (auto f = 0; f < _num_obstacles; ++f)
    {
        assert(
            static_cast<CollisionAvoidanceObservation const*>(o)->obstacles_pos[f] < _grid_height);
        assert(static_cast<CollisionAvoidanceObservation const*>(o)->obstacles_pos[f] >= 0);
    }

    assert(o->index() < static_cast<int>(_observations.size()) && o->index() >= 0);
    assert(o == _observations[o->index()]);
}

} // namespace domains
