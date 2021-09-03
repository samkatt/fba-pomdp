#include "GridWorldCoffeeBigBAExtension.hpp"

#include "environment/Action.hpp"
#include "environment/State.hpp"

namespace bayes_adaptive { namespace domain_extensions {

void assertLegalCoffeeBig(domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos const& position, size_t grid_size)
{
    assert(position.x < grid_size);
    assert(position.y < grid_size);
}

void assertLegalCoffeeBig(State const* s, size_t grid_size, size_t state_space_size)
{
    assert(s != nullptr);
    assert(s->index() >= 0 && s->index() < static_cast<int>(state_space_size));
    assert(static_cast<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const*>(s)->_rain < 2 );
    assertLegalCoffeeBig(
        static_cast<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const*>(s)->_agent_position, grid_size);
}

void assertLegalCoffeeBig(Action const* a, size_t action_space_size)
{
    assert(a != nullptr);
    assert(a->index() >= 0 && a->index() < static_cast<int>(action_space_size));
}

GridWorldCoffeeBigBAExtension::GridWorldCoffeeBigBAExtension(size_t extra_features) :
    _size(5),
    _extra_features(extra_features),
    _states(), // initiated below
    _domain_size(0, 0, 0) // initiated below
{

    _domain_size = Domain_Size(
        static_cast<int>((_size * _size * 2) << _extra_features),
        4,
        static_cast<int>(_size * _size));

    // generate state space
    _states.reserve(_domain_size._S);
    int i = 0;
    for (unsigned int x_agent = 0; x_agent < _size; ++x_agent)
    {
        for (unsigned int y_agent = 0; y_agent < _size; ++y_agent)
        {
            for (unsigned int rain = 0; rain < 2; ++rain)
            {
                for (unsigned int feature_config = 0; feature_config < (unsigned int) (1 << _extra_features); ++feature_config)
                {
                    domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos const agent_pos{x_agent, y_agent};
                    assert(static_cast<unsigned int>(i) == _states.size());

                    _states.emplace_back(domains::GridWorldCoffeeBig::GridWorldCoffeeBigState(agent_pos, rain, feature_config, i));
                    i++;
                }
            }
        }
    }
}

Domain_Size GridWorldCoffeeBigBAExtension::domainSize() const
{
    return _domain_size;
}

State const* GridWorldCoffeeBigBAExtension::getState(int index) const
{
    assert(index < _domain_size._S);
    return &_states[index];
}

Terminal GridWorldCoffeeBigBAExtension::terminal(State const* s, Action const* a, State const* new_s) const
{
    assertLegalCoffeeBig(s, _size, _domain_size._S);
    assertLegalCoffeeBig(a, _domain_size._A);
    assertLegalCoffeeBig(new_s, _size, _domain_size._S);

    auto const gw_state = static_cast<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const*>(s);

    return Terminal(domains::GridWorldCoffeeBig::goal_location == gw_state->_agent_position);
}

Reward GridWorldCoffeeBigBAExtension::reward(State const* s, Action const* a, State const* new_s) const
{
    assertLegalCoffeeBig(s, _size, _domain_size._S);
    assertLegalCoffeeBig(a, _domain_size._A);
    assertLegalCoffeeBig(new_s, _size, _domain_size._S);

    auto const gw_state = static_cast<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const*>(s);

    if (domains::GridWorldCoffeeBig::goal_location == gw_state->_agent_position) // found goal
    {
        return Reward(domains::GridWorldCoffeeBig::goal_reward);
    } else
    {
        return Reward(domains::GridWorldCoffeeBig::step_reward);
    }
}

}} // namespace bayes_adaptive::domain_extensions
