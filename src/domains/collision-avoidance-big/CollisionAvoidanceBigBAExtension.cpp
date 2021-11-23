#include "CollisionAvoidanceBigBAExtension.hpp"

#include "utils/index.hpp"

namespace bayes_adaptive { namespace domain_extensions {

CollisionAvoidanceBigBAExtension::CollisionAvoidanceBigBAExtension(
    int grid_width,
    int grid_height,
    int num_obstacles,
    domains::CollisionAvoidanceBig const& problem_domain) :
        _grid_width(grid_width),
        _grid_height(grid_height),
        _num_obstacles(num_obstacles),
        collisionavoidancebig(problem_domain)
{
    // locally create & store all states
//    std::vector<int> const obstacles_range(_num_obstacles, _grid_height);
//
//    auto index = 0;
//    for (auto x = 0; x < grid_width; ++x) {
//        for (auto y_agent = 0; y_agent < grid_height; ++y_agent) {
//            for (auto speed = 0; speed < _num_speeds; ++speed) {
//                for (auto traffic = 0; traffic < _num_traffics; ++traffic) {
//                    for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) {
//                        for (auto obstacletype = 0; obstacletype < _num_obstacletypes; ++obstacletype) {
//                            std::vector<int> obstacles(_num_obstacles);
//                            auto obs_i = 0;
//                            do {
//                                std::vector<int> vector1 = {x, y_agent, speed, traffic, timeofday, obstacletype};
//                                vector1.insert(vector1.end(), obstacles.begin(), obstacles.end());
//                                _states[x][y_agent][speed][traffic][timeofday][obstacletype][obs_i++] =
//                                        new domains::CollisionAvoidanceBigState(vector1, index++);
//                            } while (!indexing::increment(obstacles, obstacles_range));
//                        }
//                    }
//                }
//            }
//        }
//    }
}

CollisionAvoidanceBigBAExtension::~CollisionAvoidanceBigBAExtension() {

}
//    // clean up allocated states
//    for (auto& width : _states) { // width
//        for (auto& height : width) { // height
//            for (auto& speed : height) { // speed
//                for (auto& traffic : speed) { // traffic
//                    for (auto& timeofday : traffic) { // timeofday
//                        for (auto& obstacletype : timeofday) { // obstacletype
//                            for (auto s: obstacletype) { delete s; } // obstacle
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

Domain_Size CollisionAvoidanceBigBAExtension::domainSize() const
{
    return _domain_size;
}

State const* CollisionAvoidanceBigBAExtension::getState(std::string index) const
{
    return collisionavoidancebig.getState(std::stoi(index));
//    auto features = indexing::projectUsingDimensions(
//        std::stoi(index),
//        {_grid_width, _grid_height, _num_speeds, _num_traffics, _num_timeofdays, _num_obstacletypes, static_cast<int>(std::pow(_grid_height, _num_obstacles))});
//
//    return _states[features[0]][features[1]][features[2]][features[3]][features[4]][features[5]][features[6]];
}

Terminal CollisionAvoidanceBigBAExtension::terminal(
    State const* /*s*/,
    Action const* /*a*/,
    State const* new_s) const
{
    auto const ca_state = static_cast<domains::CollisionAvoidanceBigState const*>(new_s);
    auto const crashed  = ca_state->_state_vector[x_agent_f] < _num_obstacles
                         && ca_state->_state_vector[y_agent_f] == ca_state->getFeatureValues()[obstacle_start + ca_state->_state_vector[x_agent_f]];

    return Terminal(crashed || ca_state->_state_vector[x_agent_f] == 0);
}

Reward CollisionAvoidanceBigBAExtension::reward(
    State const* /*s*/,
    Action const* a,
    State const* new_s) const
{
    auto r = std::stoi(a->index()) == domains::CollisionAvoidanceBig::STAY
                 ? 0
                 : -domains::CollisionAvoidanceBig::MOVE_PENALTY;

    auto const ca_state = static_cast<domains::CollisionAvoidanceBigState const*>(new_s);

    if (ca_state->_state_vector[x_agent_f] < _num_obstacles
        && ca_state->_state_vector[y_agent_f] == ca_state->getFeatureValues()[obstacle_start + ca_state->_state_vector[x_agent_f]])
    {
        return Reward(-domains::CollisionAvoidanceBig::COLLIDE_PENALTY);
    }

    return Reward(r);
}

}} // namespace bayes_adaptive::domain_extensions
