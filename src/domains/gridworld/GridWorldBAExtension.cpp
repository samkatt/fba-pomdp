#include "GridWorldBAExtension.hpp"

#include "environment/Action.hpp"
#include "environment/State.hpp"

namespace bayes_adaptive { namespace domain_extensions {

void assertLegal(domains::GridWorld::GridWorldState::pos const& position, size_t grid_size)
{
    assert(position.x < grid_size);
    assert(position.y < grid_size);
}

void assertLegal(State const* s, size_t grid_size, size_t state_space_size)
{
    assert(s != nullptr);
    assert(s->index() >= 0 && s->index() < static_cast<int>(state_space_size));
    assertLegal(
        static_cast<domains::GridWorld::GridWorldState const*>(s)->_agent_position, grid_size);
    assertLegal(
        static_cast<domains::GridWorld::GridWorldState const*>(s)->_goal_position, grid_size);
}

void assertLegal(Action const* a, size_t action_space_size)
{
    assert(a != nullptr);
    assert(a->index() >= 0 && a->index() < static_cast<int>(action_space_size));
}

GridWorldBAExtension::GridWorldBAExtension(size_t size) :
        _size(size),
        _states(), // initiated below
        _domain_size(0, 0, 0) // initiated below
{

    auto const goals       = domains::GridWorld::goalLocations(_size);
    auto const goal_amount = goals.size();
    _domain_size           = Domain_Size(
        static_cast<int>(_size * _size * goal_amount),
        4,
        static_cast<int>(_size * _size * goal_amount));

    // generate state space
    _states.reserve(_domain_size._S);
    int i = 0;
    for (unsigned int x_agent = 0; x_agent < _size; ++x_agent)
    {
        for (unsigned int y_agent = 0; y_agent < _size; ++y_agent)
        {
            for (auto const& goal_pos : goals)
            {
                domains::GridWorld::GridWorldState::pos const agent_pos{x_agent, y_agent};

                assert(static_cast<unsigned int>(i) == _states.size());

                _states.emplace_back(domains::GridWorld::GridWorldState(agent_pos, goal_pos, i));
                i++;
            }
        }
    }
}

Domain_Size GridWorldBAExtension::domainSize() const
{
    return _domain_size;
}

State const* GridWorldBAExtension::getState(int index) const
{
    assert(index < _domain_size._S);
    return &_states[index];
}

Terminal GridWorldBAExtension::terminal(State const* s, Action const* a, State const* new_s) const
{
    assertLegal(s, _size, _domain_size._S);
    assertLegal(a, _domain_size._A);
    assertLegal(new_s, _size, _domain_size._S);

    auto const gw_state = static_cast<domains::GridWorld::GridWorldState const*>(s);

    return Terminal(gw_state->_goal_position == gw_state->_agent_position);
}

Reward GridWorldBAExtension::reward(State const* s, Action const* a, State const* new_s) const
{
    assertLegal(s, _size, _domain_size._S);
    assertLegal(a, _domain_size._A);
    assertLegal(new_s, _size, _domain_size._S);

    auto const gw_state = static_cast<domains::GridWorld::GridWorldState const*>(s);

    if (gw_state->_goal_position == gw_state->_agent_position) // found goal
    {
        return Reward(domains::GridWorld::goal_reward);
    } else
    {
        return Reward(domains::GridWorld::step_reward);
    }
}

}} // namespace bayes_adaptive::domain_extensions
