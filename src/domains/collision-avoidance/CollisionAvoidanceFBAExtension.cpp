#include "CollisionAvoidanceFBAExtension.hpp"

#include <vector>

#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "utils/index.hpp"

namespace bayes_adaptive { namespace domain_extensions {

CollisionAvoidanceFBAExtension::CollisionAvoidanceFBAExtension(
    int grid_width,
    int grid_height,
    int num_obstacles,
    domains::CollisionAvoidance::VERSION version) :
        _domain_feature_size(
            std::vector<int>(2 + num_obstacles, grid_height),
            std::vector<int>(num_obstacles, grid_height)),
        _state_prior(
            grid_width * grid_height * static_cast<int>(std::pow(grid_height, num_obstacles)))
{

    auto const domain =
        domains::CollisionAvoidance(grid_width, grid_height, num_obstacles, version);

    // set domain state prior
    if (version == domains::CollisionAvoidance::INIT_RANDOM_POSITION)
    {

        auto const init_state_prob = 1.f / std::pow(grid_height, num_obstacles + 1);

        for (auto agent_y = 0; agent_y < grid_height; ++agent_y)
        {
            std::vector<int> obstacles(num_obstacles);
            do
            {

                _state_prior.setRawValue(
                    std::stoi(domain.getState(grid_width - 1, agent_y, obstacles)->index()), init_state_prob);

            } while (!indexing::increment(obstacles, _domain_feature_size._O));
        }

    } else // version is initialize centre
    {
        // this sets prior such that only 1 initial state, where obstacles and agent start in middle
        std::vector<int> obstacles(num_obstacles, grid_height / 2);

        _state_prior.setRawValue(
                std::stoi(domain.getState(grid_width - 1, grid_height / 2, obstacles)->index()), 1);
    }
}

Domain_Feature_Size CollisionAvoidanceFBAExtension::domainFeatureSize() const
{
    return _domain_feature_size;
}

utils::categoricalDistr const* CollisionAvoidanceFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions
