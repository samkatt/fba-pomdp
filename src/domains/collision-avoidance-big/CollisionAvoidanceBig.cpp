#include "CollisionAvoidanceBig.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include "easylogging++.h"

#include "utils/index.hpp"

namespace domains {

// NOTE: these re-declarations seem to be necessary for some (buggy?) compilers
constexpr int const CollisionAvoidanceBig::NUM_ACTIONS;
constexpr double const CollisionAvoidanceBig::MOVE_PENALTY;
constexpr double const CollisionAvoidanceBig::COLLIDE_PENALTY;
constexpr double const CollisionAvoidanceBig::BLOCK_MOVE_PROB;

/**
 * \brief An observation in the collision avoidance domain
 *
 * the location of obstacles
 **/
struct CollisionAvoidanceBigObservation : public Observation
{

    // cppcheck-suppress passedByValue
    CollisionAvoidanceBigObservation(int index, std::vector<int> pos) :
            _index(index),
            obstacles_pos(std::move(pos))
    {
    }

    /**** observation interface ****/

    void index(std::string /*i*/) final
    {
        throw "CollisionAvoidanceBigObservation::index(i) should not be called...?";
    }

    std::string index() const final { return std::to_string(_index); }

    std::string toString() const final
    {
        std::string s("{");

        for (unsigned int j = 0; j < obstacles_pos.size() - 1; j++)
        { s += std::to_string(obstacles_pos[j]) + ", "; }

        return s + std::to_string(obstacles_pos.back()) + "}";
    }

    int const _index;
    std::vector<int> const obstacles_pos;

    std::vector<int> getFeatureValues() const;
};

CollisionAvoidanceBig::CollisionAvoidanceBig(
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
        throw("Cannot initiate CollisionAvoidanceBig with width " + std::to_string(_grid_width));
    }

    if (grid_height < 1 || grid_height % 2 != 1)
    {
        throw(
            "Cannot initiate CollisionAvoidanceBig with height " + std::to_string(_grid_height)
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
    for (auto x = 0; x < grid_width; ++x) {
        for (auto y_agent = 0; y_agent < grid_height; ++y_agent) {
            for (auto speed = 0; speed < _num_speeds; ++speed) {
                for (auto traffic = 0; traffic < _num_traffics; ++traffic) {
                    for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) {
                        std::vector<int> obstacles(_num_obstacles);
                        auto obs_i = 0;
                        do {
                            _states[x][y_agent][speed][traffic][timeofday][obs_i++] =
                                    new CollisionAvoidanceBigState(x, y_agent, speed, traffic, timeofday, obstacles, index++);
                        } while (!indexing::increment(obstacles, _obstacles_space));
                    }
                }
            }
        }
    }

    // initialize all observations
    {
        int count = 0;
        std::vector<int> obstacles_pos(_num_obstacles);
        do
        {
            _observations[count] = new CollisionAvoidanceBigObservation(count, obstacles_pos);
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
        // gh^(#obs + 1) (grid height, number of obstacles, plane location) *  3^2 (speed, traffic) * 2 (time of day)
        auto const init_state_prob = 1.f / (std::pow(_grid_height, _num_obstacles + 1) * _num_speeds * _num_traffics * _num_timeofdays);

        for (auto agent_y = 0; agent_y < _grid_height; ++agent_y) {
            for (auto speed = 0; speed < _num_speeds; ++speed) {
                for (auto traffic = 0; traffic < _num_traffics; ++traffic) {
                    for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) {
                        auto count = 0;
                        std::vector<int> obstacles(_num_obstacles);
                        do {
                            _state_prior.setRawValue(std::stoi(
                                    _states[_grid_width - 1][agent_y][speed][traffic][timeofday][count++]->index()), init_state_prob);
                        } while (!indexing::increment(obstacles, _obstacles_space));
                    }
                }
            }
        }

    } else // version is initialize centre
    {
        // this sets prior such that only 1 initial state, where obstacles and agent start in middle
        // with medium speed, medium traffic and day time
        std::vector<int> obstacles(_num_obstacles, _grid_height / 2);

        _state_prior.setRawValue(
            std::stoi(_states[_grid_width - 1][_grid_height / 2][1][1][1]
                   [indexing::project(obstacles, _obstacles_space)]
                       ->index()),
            1);
    }

    VLOG(1) << "Initiated CA of size (w:" << _grid_width << ", h:" << _grid_height
            << ", n:" << _num_obstacles << ")";
}

CollisionAvoidanceBig::~CollisionAvoidanceBig()
{
    // clean up allocated states
    for (auto& outer : _states) {
        for (auto& inner_width : outer) {
            for (auto& speed : inner_width) {
                for (auto& traffic : speed) {
                    for (auto& timeofday : traffic) {
                        for (auto s : timeofday) { delete s; }
                    }
                }
            }
        }
    }

    for (auto& o : _observations) { delete o; }
}

CollisionAvoidanceBig::VERSION CollisionAvoidanceBig::type() const
{
    return _version;
}

int CollisionAvoidanceBig::xAgent(State const* s) const
{
    return static_cast<CollisionAvoidanceBigState const*>(s)->x_agent;
}

int CollisionAvoidanceBig::yAgent(State const* s) const
{
    return static_cast<CollisionAvoidanceBigState const*>(s)->y_agent;
}

int CollisionAvoidanceBig::speedRelative(const State *s) const {
    return static_cast<CollisionAvoidanceBigState const*>(s)->speed;
}

int CollisionAvoidanceBig::trafficStatus(const State *s) const {
    return static_cast<CollisionAvoidanceBigState const*>(s)->traffic;
}

int CollisionAvoidanceBig::timeofdayStatus(const State *s) const {
    return static_cast<CollisionAvoidanceBigState const*>(s)->timeofday;
}

std::vector<int> const& CollisionAvoidanceBig::yObstacles(State const* s) const
{
    return static_cast<CollisionAvoidanceBigState const*>(s)->obstacles_pos;
}

State const* CollisionAvoidanceBig::getState(int index) const
{
    auto features = indexing::projectUsingDimensions(
        index,
        {_grid_width, _grid_height, _num_speeds, _num_traffics, _num_timeofdays, static_cast<int>(std::pow(_grid_height, _num_obstacles))});

    return _states[features[0]][features[1]][features[2]][features[3]][features[4]][features[5]];
}

Reward CollisionAvoidanceBig::reward(Action const* a, State const* new_s) const
{
    auto r = std::stoi(a->index()) == STAY ? 0 : -MOVE_PENALTY;

    auto const x_agent = xAgent(new_s);
    if (x_agent < _num_obstacles && yAgent(new_s) == yObstacles(new_s)[x_agent])
    {
        return Reward(-COLLIDE_PENALTY);
    }

    return Reward(r);
}

void CollisionAvoidanceBig::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assertLegal(s);
    assert(actions->empty());

    for (auto i = 0; i < NUM_ACTIONS; ++i) { actions->emplace_back(_actions.get(i)); }
}

double CollisionAvoidanceBig::computeObservationProbability(
    Observation const* o,
    Action const* /*a*/,
    State const* new_s) const
{

    auto const& pos = static_cast<CollisionAvoidanceBigState const*>(new_s)->obstacles_pos;
    auto const& obs = static_cast<CollisionAvoidanceBigObservation const*>(o)->obstacles_pos;

    double p = 1;

    for (auto i = 0; i < _num_obstacles; ++i)
    { p *= _observation_error_probability[std::abs(pos[i] - obs[i])]; }

    return p;
}

int CollisionAvoidanceBig::keepInBounds(int value) const
{
    return std::max(0, std::min(2, value));
}

int CollisionAvoidanceBig::changeSpeed(int speed, int traffic) const {
    auto prob = rnd::uniform_rand01();
    int newSpeed;
    if (traffic == 2) {
        newSpeed = (prob < 0.5) ? speed : keepInBounds(speed - 1);
    } else if (traffic == 1) {
        newSpeed = (prob < 0.5) ? speed : (prob < 0.75) ? keepInBounds(speed + 1) : keepInBounds(speed - 1);
    } else {
        newSpeed = (prob < 0.5) ? speed : keepInBounds(speed + 1);
    }
    return newSpeed;
}

int CollisionAvoidanceBig::changeTraffic(int traffic, int timeofday) const {
    auto prob = rnd::uniform_rand01();
    int newTraffic;
    if (traffic == 2) {
        if (timeofday == 1) {
            newTraffic = (prob < 0.8) ? traffic : keepInBounds(traffic - 1);
        } else {
            newTraffic = (prob < 0.2) ? traffic : keepInBounds(traffic - 1);
        }
    } else if (traffic == 1) {
        if (timeofday == 1) {
            newTraffic = (prob < 0.6) ? traffic : (prob < 0.9) ? keepInBounds(traffic + 1) : keepInBounds(traffic - 1);
        } else {
            newTraffic = (prob < 0.6) ? traffic : (prob < 0.7) ? keepInBounds(traffic + 1) : keepInBounds(traffic - 1);
        }
    } else {
        if (timeofday == 1) {
            newTraffic = (prob < 0.2) ? traffic : keepInBounds(traffic + 1);
        } else {
            newTraffic = (prob < 0.8) ? traffic : keepInBounds(traffic + 1);
        }
    }
    return newTraffic;
}

Terminal
    CollisionAvoidanceBig::step(State const** s, Action const* a, Observation const** o, Reward* r)
        const
{
    assertLegal(*s);
    assertLegal(a);

    auto const* collision_state = static_cast<CollisionAvoidanceBigState const*>(*s);

    // move agent
    auto x = collision_state->x_agent - 1;
    auto y = keepInGrid(collision_state->y_agent + std::stoi(a->index()) - 1);
    auto speed = changeSpeed(collision_state->speed, collision_state->traffic);
    auto traffic = changeTraffic(collision_state->traffic, collision_state->timeofday);
    auto timeofday = collision_state->timeofday;

    // move block
    auto blocks = collision_state->obstacles_pos;
    for (auto& b : blocks) { b = moveObstacle(b, collision_state->speed); }
    *s = _states[x][y][speed][traffic][timeofday][indexing::project(blocks, _obstacles_space)];

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

State const* CollisionAvoidanceBig::sampleStartState() const
{
    return getState(_state_prior.sample());
}

State const* CollisionAvoidanceBig::getState(int x, int y, int speed, int traffic, int timeofday, std::vector<int> const& obstacles) const
{
    assert(static_cast<unsigned int>(_num_obstacles) == obstacles.size());
    assert(x >= 0 && x <= _grid_width);
    assert(y >= 0 && y <= _grid_height);

    for (auto b : obstacles) { assert(b >= 0 && b <= _grid_height); }

    return _states[x][y][speed][traffic][timeofday][indexing::project(obstacles, _obstacles_space)];
}

Action const* CollisionAvoidanceBig::getAction(move m) const
{
    return _actions.get(m);
}

Observation const* CollisionAvoidanceBig::getObservation(int y) const
{
    return _observations[y];
}

Action const* CollisionAvoidanceBig::generateRandomAction(State const* /*s*/) const
{
    // any action (up, stay, down) is possible in any state
    return getAction(move(_action_distr(rnd::rng())));
}

void CollisionAvoidanceBig::releaseAction(Action const* a) const
{
    assertLegal(a);

    // all actions are stored locally and re-used,
    // so no need to deallocate them
}

void CollisionAvoidanceBig::releaseObservation(Observation const* o) const
{
    assertLegal(o);

    // all observations are stored locally and re-used,
    // so no need to deallocate them
}

void CollisionAvoidanceBig::releaseState(State const* s) const
{
    assertLegal(s);

    // all states are stored locally and re-used,
    // so no need to deallocate them
}

Action const* CollisionAvoidanceBig::copyAction(Action const* a) const
{
    assertLegal(a);

    // all actions are stored locally and re-used
    return a;
}

Observation const* CollisionAvoidanceBig::copyObservation(Observation const* o) const
{
    assertLegal(o);

    // all observations are stored locally and re-used,
    return o;
}

State const* CollisionAvoidanceBig::copyState(State const* s) const
{
    assertLegal(s);

    // all observations are stored locally and re-used
    return s;
}

int CollisionAvoidanceBig::keepInGrid(int y) const
{
    return std::max(0, std::min(_grid_height - 1, y));
}

//    * Let's say: medium speed: 0.25 (4.5/18), 0.25 (4.5/18), 0.5 (9/18) (up, down, stay)
//    * high speed: 3/18, 3/18, 12/18 (0.1666,0.16666,0.6666)
//    * Low speed: 6/18, 6/18, 6/18 (0.333,0.333,0.333)
int CollisionAvoidanceBig::moveObstacle(int current_position, int speed) const
{
    assert(current_position >= 0 && current_position < _grid_height);

    auto prob = rnd::uniform_rand01();
    int m;
    if (speed == 2) {
        m = (prob > (1.0/3)) ? STAY : (prob > .5 * (1.0/3)) ? MOVE_UP : MOVE_DOWN;
    } else if (speed == 1) {
        m = (prob > BLOCK_MOVE_PROB) ? STAY : (prob > .5 * BLOCK_MOVE_PROB) ? MOVE_UP : MOVE_DOWN;
    } else {
        m = (prob > (2.0/3)) ? STAY : (prob > (1.0/3)) ? MOVE_UP : MOVE_DOWN;
    }

    // apply move but stay within bounds
    return keepInGrid(current_position + m - 1);
}

void CollisionAvoidanceBig::assertLegal(State const* s) const
{
    assert(s != nullptr);

    auto const* collision_state = static_cast<CollisionAvoidanceBigState const*>(s);

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
                  [collision_state->speed][collision_state->traffic][collision_state->timeofday]
                  [indexing::project(collision_state->obstacles_pos, _obstacles_space)]
                      ->index());
}

void CollisionAvoidanceBig::assertLegal(Action const* a) const
{
    assert(a != nullptr && std::stoi(a->index()) >= 0 && std::stoi(a->index()) < NUM_ACTIONS);
}

void CollisionAvoidanceBig::assertLegal(Observation const* o) const
{
    assert(o != nullptr);

    for (auto f = 0; f < _num_obstacles; ++f)
    {
        assert(
            static_cast<CollisionAvoidanceBigObservation const*>(o)->obstacles_pos[f] < _grid_height);
        assert(static_cast<CollisionAvoidanceBigObservation const*>(o)->obstacles_pos[f] >= 0);
    }

    assert(std::stoi(o->index()) < static_cast<int>(_observations.size()) && std::stoi(o->index()) >= 0);
    assert(o == _observations[std::stoi(o->index())]);
}

void CollisionAvoidanceBig::clearCache() const {

}

std::vector<int> CollisionAvoidanceBigObservation::getFeatureValues() const {
    // TODO implement
    return std::vector<int>();
}

std::vector<int> CollisionAvoidanceBigState::getFeatureValues() const {
    // TODO implement
    return std::vector<int>();
}
} // namespace domains
