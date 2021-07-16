//
// Created by rolf on 25-01-21.
//

#include "GridWorldCoffeeBigFBAExtension.hpp"

#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBig.hpp"

namespace bayes_adaptive { namespace domain_extensions {

GridWorldCoffeeBigFBAExtension::GridWorldCoffeeBigFBAExtension() :
    _size(5),
    _carpet_configurations(1 << 15),
    _state_prior(0) // initialized below
{

    auto const domain = domains::GridWorldCoffeeBig();
//    auto const goals  = domains::GridWorldCoffeeBig::goalLocations(_size);

//    _goal_amount = 1;

    // generate _state_prior
    // TODO correct?
    _state_prior = utils::categoricalDistr(static_cast<int>(_size * _size * 2 * _carpet_configurations));

    auto const* s = domain.getState(domains::GridWorldCoffeeBig::start_location, 0, 0);
    _state_prior.setRawValue(s->index(), 1);

    domain.releaseState(s);
}

Domain_Feature_Size GridWorldCoffeeBigFBAExtension::domainFeatureSize() const
{
    // For states, x, y, rain, velocity, carpet
    // For observation, x, y, rain, carpet
    auto feature_sizes = std::vector<int>(3 + 15, 2);
    feature_sizes[0] = _size;
    feature_sizes[1] = _size;
    return {feature_sizes, // 4 carpet tile binaries
            {static_cast<int>(_size), static_cast<int>(_size)}};
//    return {{static_cast<int>(_size), static_cast<int>(_size), static_cast<int>(2),
//             static_cast<int>(2), static_cast<int>(2),static_cast<int>(2),static_cast<int>(2)}, // 4 carpet tile binaries
//            {static_cast<int>(_size), static_cast<int>(_size)}};
}

utils::categoricalDistr const* GridWorldCoffeeBigFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions