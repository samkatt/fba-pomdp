#include "CollisionAvoidanceBigFBAExtension.hpp"

#include <vector>

#include "domains/collision-avoidance-big/CollisionAvoidanceBig.hpp"
#include "utils/index.hpp"

namespace bayes_adaptive { namespace domain_extensions {

CollisionAvoidanceBigFBAExtension::CollisionAvoidanceBigFBAExtension(
    int grid_width,
    int grid_height,
    int num_obstacles,
    domains::CollisionAvoidanceBig::VERSION version) :
        _domain_feature_size(
            std::vector<int>(5 + num_obstacles, grid_height),
            std::vector<int>(num_obstacles, grid_height)),
        _state_prior(
            grid_width * grid_height * 3 * 3 * 2 * static_cast<int>(std::pow(grid_height, num_obstacles)))
{

    auto const domain =
        domains::CollisionAvoidanceBig(grid_width, grid_height, num_obstacles, version);

    // set domain state prior
    if (version == domains::CollisionAvoidanceBig::INIT_RANDOM_POSITION) {
        auto const init_state_prob = 1.f / (std::pow(grid_height, num_obstacles + 1) * _num_speeds * _num_traffics * _num_timeofdays);

        for (auto agent_y = 0; agent_y < grid_height; ++agent_y) {
            for (auto speed = 0; speed < _num_speeds; ++speed) {
                for (auto traffic = 0; traffic < _num_traffics; ++traffic) {
                    for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) {
                        std::vector<int> obstacles(num_obstacles);
                        do {
                            _state_prior.setRawValue(
                                    std::stoi(domain.getState(grid_width - 1, agent_y, speed, traffic, timeofday, obstacles)->index()), init_state_prob);
                        } while (!indexing::increment(obstacles, _domain_feature_size._O));
                    }
                }
            }
        }
    } else { // version is initialize centre
        // this sets prior such that only 1 initial state, where obstacles and agent start in middle
        std::vector<int> obstacles(num_obstacles, grid_height / 2);

        auto const init_state_prob = 1.f / static_cast<float>(_num_speeds * _num_traffics * _num_timeofdays);

        for (auto speed = 0; speed < _num_speeds; ++speed) {
            for (auto traffic = 0; traffic < _num_traffics; ++traffic) {
                for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) {
                    _state_prior.setRawValue(
                            std::stoi(domain.getState(grid_width - 1, grid_height / 2, speed, traffic, timeofday, obstacles)->index()), init_state_prob);
                }
            }
        }
    }
}

Domain_Feature_Size CollisionAvoidanceBigFBAExtension::domainFeatureSize() const
{
    return _domain_feature_size;
}

utils::categoricalDistr const* CollisionAvoidanceBigFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions
