//
// Created by rolf on 25-01-21.
//

#include "GridWorldCoffeeFBAExtension.hpp"

#include "domains/gridworld-coffee-trap/GridWorldCoffee.hpp"

namespace bayes_adaptive { namespace domain_extensions {

GridWorldCoffeeFBAExtension::GridWorldCoffeeFBAExtension() :
    _size(5),
    _carpet_configurations(2),
    _state_prior(0) // initialized below
{

    auto const domain = domains::GridWorldCoffee();
//    auto const goals  = domains::GridWorldCoffee::goalLocations(_size);

//    _goal_amount = 1;

    // generate _state_prior
    // TODO correct?
    _state_prior = utils::categoricalDistr(static_cast<int>(_size * _size * 2 * _carpet_configurations));

    auto const* s = domain.getState(domains::GridWorldCoffee::start_location, 0, 0);
    _state_prior.setRawValue(s->index(), 1);

    domain.releaseState(s);
}

Domain_Feature_Size GridWorldCoffeeFBAExtension::domainFeatureSize() const
{
    // For states, x, y, rain, velocity, carpet
    // For observation, x, y, rain, carpet
    return {{static_cast<int>(_size), static_cast<int>(_size), static_cast<int>(2), static_cast<int>(_carpet_configurations)},
            {static_cast<int>(_size), static_cast<int>(_size)}};
}

utils::categoricalDistr const* GridWorldCoffeeFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions