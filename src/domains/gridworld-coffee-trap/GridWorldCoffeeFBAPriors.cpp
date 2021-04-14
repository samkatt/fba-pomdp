#include "GridWorldCoffeeFBAExtension.hpp"

#include "domains/gridworld-coffee-trap/GridWorldCoffee.hpp"

namespace bayes_adaptive { namespace domain_extensions {

GridWorldCoffeeFBAExtension::GridWorldCoffeeFBAExtension() :
    _size(5),
    _state_prior(0) // initialized below
{

    auto const domain = domains::GridWorldCoffee();
//    auto const goals  = domains::GridWorld::goalLocations(_size);

//    _goal_amount = goals.size();

    // generate _state_prior
    _state_prior = utils::categoricalDistr(static_cast<int>(_size * _size * 2 * 4 * 2));

    auto const* s = domain.getState(domains::GridWorldCoffee::start_location, 0, 1);

    _state_prior.setRawValue(s->index(), 1);

    domain.releaseState(s);
}

// State: x, y, rain, velocity, carpet. 5, 5, 2, 4?, 2
// Observation: x, y, rain.
Domain_Feature_Size GridWorldCoffeeFBAExtension::domainFeatureSize() const
{
    return {{static_cast<int>(_size), static_cast<int>(_size), static_cast<int>(2), static_cast<int>(4), static_cast<int>(2)},
            {static_cast<int>(_size), static_cast<int>(_size), static_cast<int>(2)}};
}

utils::categoricalDistr const* GridWorldCoffeeFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions
