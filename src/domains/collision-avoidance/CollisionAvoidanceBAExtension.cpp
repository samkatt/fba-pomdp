#include "CollisionAvoidanceBAExtension.hpp"

#include "utils/index.hpp"

namespace bayes_adaptive { namespace domain_extensions {

CollisionAvoidanceBAExtension::CollisionAvoidanceBAExtension(
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
    for (auto x = 0; x < grid_width; ++x)
    {
        for (auto y_agent = 0; y_agent < grid_height; ++y_agent)
        {

            std::vector<int> obstacles(_num_obstacles);
            auto obs_i = 0;
            do
            {
                _states[x][y_agent][obs_i++] =
                    new domains::CollisionAvoidanceState(x, y_agent, obstacles, index++);
            } while (!indexing::increment(obstacles, obstacles_range));
        }
    }
}

CollisionAvoidanceBAExtension::~CollisionAvoidanceBAExtension()
{
    // clean up allocated states
    for (auto& outer : _states)
    {
        for (auto& inner : outer)
        {
            for (auto s : inner) { delete s; }
        }
    }
}

Domain_Size CollisionAvoidanceBAExtension::domainSize() const
{
    return _domain_size;
}

State const* CollisionAvoidanceBAExtension::getState(int index) const
{
    auto features = indexing::projectUsingDimensions(
        index,
        {_grid_width, _grid_height, static_cast<int>(std::pow(_grid_height, _num_obstacles))});

    return _states[features[0]][features[1]][features[2]];
}

Terminal CollisionAvoidanceBAExtension::terminal(
    State const* /*s*/,
    Action const* /*a*/,
    State const* new_s) const
{
    auto const ca_state = static_cast<domains::CollisionAvoidanceState const*>(new_s);
    auto const crashed  = ca_state->x_agent < _num_obstacles
                         && ca_state->y_agent == ca_state->obstacles_pos[ca_state->x_agent];

    return Terminal(crashed || ca_state->x_agent == 0);
}

Reward CollisionAvoidanceBAExtension::reward(
    State const* /*s*/,
    Action const* a,
    State const* new_s) const
{
    auto r = a->index() == domains::CollisionAvoidance::STAY
                 ? 0
                 : -domains::CollisionAvoidance::MOVE_PENALTY;

    auto const ca_state = static_cast<domains::CollisionAvoidanceState const*>(new_s);

    if (ca_state->x_agent < _num_obstacles
        && ca_state->y_agent == ca_state->obstacles_pos[ca_state->x_agent])
    {
        return Reward(-domains::CollisionAvoidance::COLLIDE_PENALTY);
    }

    return Reward(r);
}

}} // namespace bayes_adaptive::domain_extensions
