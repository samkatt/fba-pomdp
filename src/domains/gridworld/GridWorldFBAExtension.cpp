#include "GridWorldFBAExtension.hpp"

#include "domains/gridworld/GridWorld.hpp"

namespace bayes_adaptive { namespace domain_extensions {

GridWorldFBAExtension::GridWorldFBAExtension(size_t size) :
        _size(size),
        _goal_amount(0), // initialized below
        _state_prior(0) // initialized below
{

    auto const domain = domains::GridWorld(_size);
    auto const goals  = domains::GridWorld::goalLocations(_size);

    _goal_amount = goals.size();

    // generate _state_prior
    _state_prior = utils::categoricalDistr(static_cast<int>(_size * _size * goals.size()));

    float const start_prob =
        (static_cast<float>(1) / (domains::GridWorld::start_locations.size() * goals.size()));

    for (auto const& agent : domains::GridWorld::start_locations)
    {
        for (auto const& goal : goals)
        {
            auto const* s = domain.getState(agent, goal);

            _state_prior.setRawValue(s->index(), start_prob);

            domain.releaseState(s);
        }
    }
}

Domain_Feature_Size GridWorldFBAExtension::domainFeatureSize() const
{
    return {
        {static_cast<int>(_size), static_cast<int>(_size), static_cast<int>(_goal_amount)},
        {static_cast<int>(_size), static_cast<int>(_size), static_cast<int>(_goal_amount)}};
}

utils::categoricalDistr const* GridWorldFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions
