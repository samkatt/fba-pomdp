//
// Created by rolf on 25-01-21.
//

#include "GridWorldCoffeeBigFBAExtension.hpp"

#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBig.hpp"

namespace bayes_adaptive { namespace domain_extensions {

GridWorldCoffeeBigFBAExtension::GridWorldCoffeeBigFBAExtension(size_t carpet_tiles) :
    _size(5),
    _carpet_tiles(carpet_tiles),
    _state_prior(0) // initialized below
{

    auto const domain = domains::GridWorldCoffeeBig(_carpet_tiles);
//    auto const goals  = domains::GridWorldCoffeeBig::goalLocations(_size);

//    _goal_amount = 1;

    // generate _state_prior
    // TODO correct?
    _state_prior = utils::categoricalDistr(static_cast<int>((_size * _size * 2) << _carpet_tiles));

    float const start_prob =
            (static_cast<float>(1) / (1 << _carpet_tiles));

    for (unsigned int carpet_config = 0; carpet_config < (unsigned int) (1 << _carpet_tiles); ++carpet_config)
    {
        auto const* s = domain.getState(domains::GridWorldCoffeeBig::start_location, 0, carpet_config);

        _state_prior.setRawValue(s->index(), start_prob);

        domain.releaseState(s);
    }
}

Domain_Feature_Size GridWorldCoffeeBigFBAExtension::domainFeatureSize() const
{
    // For states, x, y, rain, velocity, carpet
    // For observation, x, y, rain, carpet
    auto feature_sizes = std::vector<int>(3 + _carpet_tiles, 2);
    feature_sizes[0] = _size;
    feature_sizes[1] = _size;
    return {feature_sizes, // 4 carpet tile binaries
            {static_cast<int>(_size), static_cast<int>(_size)}};
}

utils::categoricalDistr const* GridWorldCoffeeBigFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions