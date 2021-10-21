#include "CollisionAvoidanceBigBAExtension.hpp"

#include "utils/index.hpp"

namespace bayes_adaptive { namespace domain_extensions {

CollisionAvoidanceBigBAExtension::CollisionAvoidanceBigBAExtension(
    int grid_width,
    int grid_height,
    int num_obstacles) :
        _grid_width(grid_width),
        _grid_height(grid_height),
        _num_obstacles(num_obstacles)
{

    // locally create & store all states
    std::vector<int> const obstacles_range(_num_obstacles, _grid_height);

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
                                    new domains::CollisionAvoidanceBigState(x, y_agent, speed, traffic, timeofday, obstacles, index++);
                        } while (!indexing::increment(obstacles, obstacles_range));
                    }
                }
            }
        }
    }
}

CollisionAvoidanceBigBAExtension::~CollisionAvoidanceBigBAExtension()
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
}

Domain_Size CollisionAvoidanceBigBAExtension::domainSize() const
{
    return _domain_size;
}

State const* CollisionAvoidanceBigBAExtension::getState(int index) const
{
    auto features = indexing::projectUsingDimensions(
        index,
        {_grid_width, _grid_height, _num_speeds, _num_traffics, _num_timeofdays, static_cast<int>(std::pow(_grid_height, _num_obstacles))});

    return _states[features[0]][features[1]][features[2]][features[3]][features[4]][features[5]];
}

Terminal CollisionAvoidanceBigBAExtension::terminal(
    State const* /*s*/,
    Action const* /*a*/,
    State const* new_s) const
{
    auto const ca_state = static_cast<domains::CollisionAvoidanceBigState const*>(new_s);
    auto const crashed  = ca_state->x_agent < _num_obstacles
                         && ca_state->y_agent == ca_state->obstacles_pos[ca_state->x_agent];

    return Terminal(crashed || ca_state->x_agent == 0);
}

Reward CollisionAvoidanceBigBAExtension::reward(
    State const* /*s*/,
    Action const* a,
    State const* new_s) const
{
    auto r = a->index() == domains::CollisionAvoidanceBig::STAY
                 ? 0
                 : -domains::CollisionAvoidanceBig::MOVE_PENALTY;

    auto const ca_state = static_cast<domains::CollisionAvoidanceBigState const*>(new_s);

    if (ca_state->x_agent < _num_obstacles
        && ca_state->y_agent == ca_state->obstacles_pos[ca_state->x_agent])
    {
        return Reward(-domains::CollisionAvoidanceBig::COLLIDE_PENALTY);
    }

    return Reward(r);
}

}} // namespace bayes_adaptive::domain_extensions
