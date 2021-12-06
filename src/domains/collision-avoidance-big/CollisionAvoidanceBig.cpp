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
    CollisionAvoidanceBigObservation(int index, int x, int tod, int obstype, std::vector<int> pos) :
            _index(index),
            x_pos(x),
            timeofday(tod),
            obstacletype(obstype),
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
        s += std::to_string(x_pos) + ", ";
        s += std::to_string(timeofday) + ", ";
        s += std::to_string(obstacletype) + ", ";
        for (unsigned int j = 0; j < obstacles_pos.size() - 1; j++)
        { s += std::to_string(obstacles_pos[j]) + ", "; }

        return s + std::to_string(obstacles_pos.back()) + "}";
    }

    int const _index;
    int x_pos;
    int timeofday;
    int obstacletype;
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
                        for (auto obstacletype = 0; obstacletype < _num_obstacletypes; ++obstacletype) {
                            std::vector<int> obstacles(_num_obstacles);
                            auto obs_i = 0;
                            do {
                                std::vector<int> vector1 = {x, y_agent, speed, traffic, timeofday, obstacletype};
                                vector1.insert(vector1.end(), obstacles.begin(), obstacles.end());
                                _states[x][y_agent][speed][traffic][timeofday][obstacletype][obs_i++] =
                                        new CollisionAvoidanceBigState(vector1, std::to_string(index++));
                            } while (!indexing::increment(obstacles, _obstacles_space));
                        }
                    }
                }
            }
        }
    }

    // initialize all observations
    {
        int count = 0;
        for (auto x = 0; x < grid_width; ++x) {
            for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) {
                for (auto obstacletype = 0; obstacletype < _num_obstacletypes; ++obstacletype) {
                    std::vector<int> obstacles_pos(_num_obstacles);
                    do {
                        _observations[count] = new CollisionAvoidanceBigObservation(count, x, timeofday, obstacletype, obstacles_pos);
                        count++;
                    } while (!indexing::increment(obstacles_pos, _obstacles_space));
                }
            }
        }
    }

    // initialize the observation error probabilities
    for (auto d = 0; d < grid_height; ++d)
    {
        _observation_error_probability.emplace_back(
            rnd::normal::cdf(d + .5, 0, 0.35) - rnd::normal::cdf(d - .5, 0, 0.35));
    }

    if (version == INIT_RANDOM_POSITION)
    {
        // gh^(#obs + 1) (grid height, number of obstacles, plane location) *  3^2 (speed, traffic) * 2 (time of day)
        auto const init_state_prob = 1.f / (std::pow(_grid_height, _num_obstacles + 1) * _num_speeds * _num_traffics * _num_timeofdays);

        for (auto agent_y = 0; agent_y < _grid_height; ++agent_y) {
            for (auto speed = 0; speed < _num_speeds; ++speed) {
                for (auto traffic = 0; traffic < _num_traffics; ++traffic) {
                    for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) {
                        for (auto obstacletype = 0; obstacletype < _num_obstacletypes; ++obstacletype) {
                            auto count = 0;
                            std::vector<int> obstacles(_num_obstacles);
                            do {
                                auto const state_prob = init_state_prob* _obstacletype_probs[obstacletype + _num_obstacletypes*timeofday];
                                auto testindex = std::stoi(
                                        _states[_grid_width - 1][agent_y][speed][traffic][timeofday][obstacletype][count++]->index());
                                _state_prior.setRawValue(testindex, state_prob);
                            } while (!indexing::increment(obstacles, _obstacles_space));
                        }
                    }
                }
            }
        }
    } else // version is initialize centre
    {
        // this sets prior such that only 1 initial state, where obstacles and agent start in middle
        // with medium speed, medium traffic and day time
        std::vector<int> obstacles(_num_obstacles, _grid_height / 2);

        // TODO let obstacle type probability depend on time of day
        auto const init_state_prob = 1.f / (_num_speeds * _num_traffics * _num_timeofdays);

        for (auto speed = 0; speed < _num_speeds; ++speed) {
            for (auto traffic = 0; traffic < _num_traffics; ++traffic) {
                for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) {
                    for (auto obstacletype = 0; obstacletype < _num_obstacletypes; ++obstacletype) {
                        auto const state_prob = init_state_prob* _obstacletype_probs[obstacletype + _num_obstacletypes*timeofday];
                        auto testindex = std::stoi(
                                _states[_grid_width - 1][_grid_height / 2][speed][traffic][timeofday][obstacletype][indexing::project(obstacles, _obstacles_space)]->index());
                        _state_prior.setRawValue(testindex, state_prob);
                    }
                }
            }
        }
    }

    VLOG(1) << "Initiated CA of size (w:" << _grid_width << ", h:" << _grid_height
            << ", n:" << _num_obstacles << ")";
}

CollisionAvoidanceBig::~CollisionAvoidanceBig()
{
    // clean up allocated states
    for (auto& width : _states) { // width
        for (auto& height : width) { // height
            for (auto& speed : height) { // speed
                for (auto& traffic : speed) { // traffic
                    for (auto& timeofday : traffic) { // timeofday
                        for (auto& obstacletype : timeofday) { // obstacletype
                            for (auto s: obstacletype) { delete s; } // obstacle
                        }
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
    return static_cast<CollisionAvoidanceBigState const*>(s)->_state_vector[x_agent_f];
}

int CollisionAvoidanceBig::yAgent(State const* s) const
{
    return static_cast<CollisionAvoidanceBigState const*>(s)->_state_vector[y_agent_f];
}

int CollisionAvoidanceBig::speedRelative(const State *s) const {
    return static_cast<CollisionAvoidanceBigState const*>(s)->_state_vector[speed_f];
}

int CollisionAvoidanceBig::trafficStatus(const State *s) const {
    return static_cast<CollisionAvoidanceBigState const*>(s)->_state_vector[traffic_f];
}

int CollisionAvoidanceBig::timeofdayStatus(const State *s) const {
    return static_cast<CollisionAvoidanceBigState const*>(s)->_state_vector[timeofday_f];
}

//std::vector<int> const& CollisionAvoidanceBig::yObstacles(State const* s) const
//{
//    s->getFeatureValues().begin() + 5
//
//    return static_cast<CollisionAvoidanceBigState const*>(s)->
//    { obstacles_pos };
//}

State const* CollisionAvoidanceBig::getState(int index) const
{
    auto features = indexing::projectUsingDimensions(
        index,
        {_grid_width, _grid_height, _num_speeds, _num_traffics, _num_timeofdays, _num_obstacletypes, static_cast<int>(std::pow(_grid_height, _num_obstacles))});

    return _states[features[0]][features[1]][features[2]][features[3]][features[4]][features[5]][features[6]];
}

Reward CollisionAvoidanceBig::reward(Action const* /*a*/, State const* new_s) const
{
    auto r = 0; // std::stoi(a->index()) == STAY ? 0 : -MOVE_PENALTY;

    auto const x_agent = xAgent(new_s);


    if (x_agent < _num_obstacles && yAgent(new_s) == new_s->getFeatureValues()[obstacle_start + x_agent]) // TODO now only works correctly for 1 obstacle
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

    auto const& feature_values  = static_cast<CollisionAvoidanceBigState const*>(new_s)->getFeatureValues();
    auto const& obs = static_cast<CollisionAvoidanceBigObservation const*>(o)->obstacles_pos;

    // check if timeofday and x position are correct
    if (feature_values[x_agent_f] != static_cast<CollisionAvoidanceBigObservation const*>(o)->x_pos) {
        return 0;
    }
    if (feature_values[timeofday_f] != static_cast<CollisionAvoidanceBigObservation const*>(o)->timeofday) {
        return 0;
    }
    if (feature_values[obstacle_type_f] != static_cast<CollisionAvoidanceBigObservation const*>(o)->obstacletype) {
        return 0;
    }

    double p = 1;

    for (auto i = 0; i < _num_obstacles; ++i)
    { p *= _observation_error_probability[std::abs(feature_values[obstacle_start + i] - obs[i])]; }

//    if (feature_values[obstacle_type_f] != static_cast<CollisionAvoidanceBigObservation const*>(o)->obstacletype) {
//        return p * 0.5 * (1 - CORRECT_TYPE);
//    }
    return p; // *CORRECT_TYPE;
}

int CollisionAvoidanceBig::keepInBounds(int value, int num_options) const
{
    return std::max(0, std::min(num_options-1, value));
}

int CollisionAvoidanceBig::changeSpeed(int speed, int traffic) const {
    auto prob = rnd::uniform_rand01();
    int newSpeed;
//    double diff = 0.05;
    if (speed == 1) {
        if (traffic == 1) {
            newSpeed = (prob < 0.1) ? speed : keepInBounds(speed - 1, _num_speeds);
//        } else if (traffic == 1) {
//            newSpeed = (prob < 0.8) ? speed : keepInBounds(speed - 1, _num_speeds);
        } else {
            newSpeed = (prob < 0.9) ? speed : keepInBounds(speed - 1, _num_speeds);
        }
    } else {
        if (traffic == 1) {
            newSpeed = (prob < 0.9) ? speed : keepInBounds(speed + 1, _num_speeds);
//        } else if (traffic == 1) {
//            newSpeed = (prob < 0.8) ? speed : keepInBounds(speed + 1, _num_speeds);
        } else {
            newSpeed = (prob < 0.1) ? speed : keepInBounds(speed + 1, _num_speeds);
        }
    }

    return newSpeed;
}

int CollisionAvoidanceBig::changeTraffic(int traffic, int timeofday) const {
    auto prob = rnd::uniform_rand01();
    int newTraffic;
//    double diff = 0.05;
    if (traffic == 1) {
        if (timeofday == 1) {
            newTraffic = (prob < 0.9) ? traffic : keepInBounds(traffic - 1, _num_traffics);
        } else {
            newTraffic = (prob < 0.1) ? traffic : keepInBounds(traffic - 1, _num_traffics);
        }
//    } else if (traffic == 1) {
//        if (timeofday == 1) {
//            newTraffic = (prob < 0.8) ? traffic : (prob < 0.95) ? keepInBounds(traffic + 1, _num_traffics) : keepInBounds(traffic - 1, _num_traffics);
//        } else {
//            newTraffic = (prob < 0.8) ? traffic : (prob < 0.85) ? keepInBounds(traffic + 1, _num_traffics) : keepInBounds(traffic - 1, _num_traffics);
//        }
    } else {
        if (timeofday == 1) {
            newTraffic = (prob < 0.1) ? traffic : keepInBounds(traffic + 1, _num_traffics);
        } else {
            newTraffic = (prob < 0.9) ? traffic : keepInBounds(traffic + 1, _num_traffics);
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
    auto x = moveAgent(collision_state->_state_vector[x_agent_f], collision_state->_state_vector[speed_f]);
    auto y = keepInGrid(collision_state->_state_vector[y_agent_f] + std::stoi(a->index()) - 1);
    auto speed = changeSpeed(collision_state->_state_vector[speed_f], collision_state->_state_vector[traffic_f]);
    auto traffic = changeTraffic(collision_state->_state_vector[traffic_f], collision_state->_state_vector[timeofday_f]);
    auto timeofday = collision_state->_state_vector[timeofday_f];
    auto obstacletype = collision_state->_state_vector[obstacle_type_f];

    // move block
    auto feature_values = collision_state->getFeatureValues();
    for (int i = obstacle_start; i < (int) feature_values.size(); i++) {
        feature_values[i] = moveObstacle(feature_values[i], collision_state->_state_vector[x_agent_f], collision_state->_state_vector[speed_f], collision_state->_state_vector[obstacle_type_f]);
    }
//    for (auto& b : blocks) { b = moveObstacle(b, collision_state->speed); }
    *s = _states[x][y][speed][traffic][timeofday][obstacletype][indexing::project({feature_values.begin() + obstacle_start, feature_values.end()}, _obstacles_space)];

    // generate observation
//    for (auto& b : blocks)
    for (int i = obstacle_start; i < (int) feature_values.size(); i++) {
        auto observation_noise = static_cast<int>(std::round(_observation_distr(rnd::rng())));
        feature_values[i] = keepInGrid(feature_values[i] + observation_noise);
    }

//    auto prob = rnd::uniform_rand01();
//    int obstacleTypeObservation = prob < CORRECT_TYPE ? obstacletype :
//            prob < (CORRECT_TYPE + 0.5 * (1 - CORRECT_TYPE)) ? ((obstacletype + 1) % _num_obstacletypes) : ((obstacletype + 2) % _num_obstacletypes);
    *o = _observations[getObservationIndex(x, timeofday, obstacletype, indexing::project({feature_values.begin() + obstacle_start, feature_values.end()}, _obstacles_space))];

    assertLegal(*s);
    assertLegal(*o);

    r->set(reward(a, *s).toDouble());

    auto const crashed = xAgent(*s) < _num_obstacles && yAgent(*s) == feature_values[obstacle_start +xAgent(*s)]; // TODO check?  //yObstacles(*s)[xAgent(*s)];

    return Terminal(crashed || xAgent(*s) <= 0);
}

int CollisionAvoidanceBig::getObservationIndex(int x, int timeofday, int obstacletype, int obstaclesIndex) const {
    return x * _num_timeofdays * _num_obstacletypes * static_cast<int>(std::pow(_grid_height, _num_obstacles))
    + timeofday * _num_obstacletypes * static_cast<int>(std::pow(_grid_height, _num_obstacles))
        + obstacletype * _grid_height + obstaclesIndex;
}

State const* CollisionAvoidanceBig::sampleStartState() const
{
    return getState(_state_prior.sample());
}

State const* CollisionAvoidanceBig::getState(int x, int y, int speed, int traffic, int timeofday, int obstacletype, std::vector<int> const& obstacles) const
{
    assert(static_cast<unsigned int>(_num_obstacles) == obstacles.size());
    assert(x >= 0 && x <= _grid_width);
    assert(y >= 0 && y <= _grid_height);

    for (auto b : obstacles) { assert(b >= 0 && b <= _grid_height); }

    return _states[x][y][speed][traffic][timeofday][obstacletype][indexing::project(obstacles, _obstacles_space)];
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

int CollisionAvoidanceBig::moveObstacle(int current_position, int /*x*/, int /*speed*/, int /*obstacletype*/) const
{
    assert(current_position >= 0 && current_position < _grid_height);
//    double obstacle_diff = 0.025;
//    double speed_diff = 0.025;
    auto endpointStayprob = 0.5;
    auto stayprob = 1 - endpointStayprob;

    auto prob = rnd::uniform_rand01();
    int m;
    auto adjusted_move_prob = 1 - stayprob; // + speed * 0; //  BLOCK_MOVE_PROB  +x*0 + obstacle_diff*(obstacletype+1)*speed + speed_diff*(obstacletype+1)*(speed - 1);

    m = (prob > adjusted_move_prob) ? STAY : (prob > .5 * adjusted_move_prob) ? MOVE_UP : MOVE_DOWN;

    // apply move but stay within bounds
    return keepInGrid(current_position + m - 1);


//    auto startpointStayprob = 0.05 + obstacletype * 0.1;
//    auto endpointStayprob = 0.9;
//    auto stayprob = startpointStayprob + (endpointStayprob - startpointStayprob)/(_grid_width-2) * (_grid_width-1 - x);
//
//    auto prob = rnd::uniform_rand01();
//    int m;
//    auto adjusted_move_prob = 1 - stayprob + speed * 0; //  BLOCK_MOVE_PROB  +x*0 + obstacle_diff*(obstacletype+1)*speed + speed_diff*(obstacletype+1)*(speed - 1);
//
//    m = (prob > adjusted_move_prob) ? STAY : (prob > .5 * adjusted_move_prob) ? MOVE_UP : MOVE_DOWN;
//
//    // apply move but stay within bounds
//    return keepInGrid(current_position + m - 1);
}

int CollisionAvoidanceBig::moveAgent(int current_position, int speed) const
{
    assert(current_position >= 0 && current_position < _grid_width);

    auto prob = rnd::uniform_rand01();
    int m;
    if (speed == 1) {
        m = (prob > MOVE_PROB_FAST) ? 2 : 1;
    } else {
        m = (prob > MOVE_PROB_SLOW) ? 0 : 1;
    }

    // apply move but stay within bounds
    return std::max(current_position - m, 0);
}

void CollisionAvoidanceBig::assertLegal(State const* s) const
{
    assert(s != nullptr);

    auto const* collision_state = static_cast<CollisionAvoidanceBigState const*>(s);
    auto block_vector = std::vector<int>(_num_obstacles);

    assert(collision_state->_state_vector[x_agent_f] >= 0 && collision_state->_state_vector[x_agent_f] < _grid_width);
    assert(collision_state->_state_vector[y_agent_f] >= 0 && collision_state->_state_vector[y_agent_f] < _grid_height);
    for (auto i = 0; i < _num_obstacles; ++i)
    {
        assert(
            collision_state->getFeatureValues()[obstacle_start + i] >= 0
            && collision_state->getFeatureValues()[obstacle_start + i] < _grid_height);
        block_vector[i] = collision_state->getFeatureValues()[obstacle_start + i];
    }
    assert(
        s->index()
        == _states[collision_state->_state_vector[x_agent_f]][collision_state->_state_vector[y_agent_f]]
                   [collision_state->_state_vector[speed_f]][collision_state->_state_vector[traffic_f]][collision_state->_state_vector[timeofday_f]]
                   [collision_state->_state_vector[obstacle_type_f]][indexing::project(block_vector, _obstacles_space)]
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
    throw "This shouldn't be called";
    return std::vector<int>();
}

std::vector<int> CollisionAvoidanceBigState::getFeatureValues() const {
    return _state_vector;
}
} // namespace domains
