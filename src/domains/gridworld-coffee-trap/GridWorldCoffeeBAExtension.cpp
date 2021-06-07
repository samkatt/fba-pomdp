//
// Created by rolf on 21-01-21.
//

#include "GridWorldCoffeeBAExtension.hpp"

#include "environment/Action.hpp"
#include "environment/State.hpp"

namespace bayes_adaptive { namespace domain_extensions {

void assertLegalCoffee(domains::GridWorldCoffee::GridWorldCoffeeState::pos const& position, size_t grid_size)
{
    assert(position.x < grid_size);
    assert(position.y < grid_size);
}

void assertLegalCoffee(State const* s, size_t grid_size, size_t state_space_size)
{
    assert(s != nullptr);
    assert(s->index() >= 0 && s->index() < static_cast<int>(state_space_size));
    assert(static_cast<domains::GridWorldCoffee::GridWorldCoffeeState const*>(s)->_carpet_config < 2);
    assert(static_cast<domains::GridWorldCoffee::GridWorldCoffeeState const*>(s)->_rain < 2 );
    assertLegalCoffee(
        static_cast<domains::GridWorldCoffee::GridWorldCoffeeState const*>(s)->_agent_position, grid_size);
}

void assertLegalCoffee(Action const* a, size_t action_space_size)
{
    assert(a != nullptr);
    assert(a->index() >= 0 && a->index() < static_cast<int>(action_space_size));
}

GridWorldCoffeeBAExtension::GridWorldCoffeeBAExtension() :
    _size(5),
    _carpet_configurations(2),
    _states(), // initiated below
    _domain_size(0, 0, 0) // initiated below
{

    _domain_size           = Domain_Size(
        static_cast<int>(_size * _size * 2 * _carpet_configurations), // * 4),
        4,
        static_cast<int>(_size * _size * 2 * _carpet_configurations));

    // generate state space
    _states.reserve(_domain_size._S);
    int i = 0;
    for (unsigned int x_agent = 0; x_agent < _size; ++x_agent)
    {
        for (unsigned int y_agent = 0; y_agent < _size; ++y_agent)
        {
            for (unsigned int rain = 0; rain < 2; ++rain)
            {
                for (unsigned int carpet_config = 0; carpet_config < _carpet_configurations; ++carpet_config)
                {
                    domains::GridWorldCoffee::GridWorldCoffeeState::pos const agent_pos{x_agent, y_agent};
                    assert(static_cast<unsigned int>(i) == _states.size());

                    _states.emplace_back(domains::GridWorldCoffee::GridWorldCoffeeState(agent_pos, rain, carpet_config, i)); //, velocity, i));
                    i++;
                }
            }
        }
    }
}

Domain_Size GridWorldCoffeeBAExtension::domainSize() const
{
    return _domain_size;
}

State const* GridWorldCoffeeBAExtension::getState(int index) const
{
    assert(index < _domain_size._S);
    return &_states[index];
}

Terminal GridWorldCoffeeBAExtension::terminal(State const* s, Action const* a, State const* new_s) const
{
    assertLegalCoffee(s, _size, _domain_size._S);
    assertLegalCoffee(a, _domain_size._A);
    assertLegalCoffee(new_s, _size, _domain_size._S);

    auto const gw_state = static_cast<domains::GridWorldCoffee::GridWorldCoffeeState const*>(s);

    return Terminal(domains::GridWorldCoffee::goal_location == gw_state->_agent_position);
}

Reward GridWorldCoffeeBAExtension::reward(State const* s, Action const* a, State const* new_s) const
{
    assertLegalCoffee(s, _size, _domain_size._S);
    assertLegalCoffee(a, _domain_size._A);
    assertLegalCoffee(new_s, _size, _domain_size._S);

    auto const gw_state = static_cast<domains::GridWorldCoffee::GridWorldCoffeeState const*>(s);

    if (domains::GridWorldCoffee::goal_location == gw_state->_agent_position) // found goal
    {
        return Reward(domains::GridWorldCoffee::goal_reward);
    } else
    {
        return Reward(domains::GridWorldCoffee::step_reward);
    }
}


}} // namespace bayes_adaptive::domain_extensions
