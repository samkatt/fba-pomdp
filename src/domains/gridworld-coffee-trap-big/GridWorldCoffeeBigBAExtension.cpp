#include <complex>
#include "GridWorldCoffeeBigBAExtension.hpp"

#include "environment/Action.hpp"
#include "environment/State.hpp"

#include "easylogging++.h"

namespace bayes_adaptive { namespace domain_extensions {

void assertLegalCoffeeBig(domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos const& position, size_t grid_size)
{
    assert(position.x < grid_size);
    assert(position.y < grid_size);
}

void assertLegalCoffeeBig(State const* s, size_t grid_size, size_t state_space_size)
{
    assert(s != nullptr);
    // TODO change this`
    assert(static_cast<int>(state_space_size) != -1);
    assert(grid_size != 0);
//    assert(std::stoi(s->index()) >= 0 &&std::stoi(s->index())< static_cast<int>(state_space_size)); // &&std::stoi(s->index())< static_cast<int>(state_space_size));
    assert(static_cast<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const*>(s)->_state_vector[2] < 2 );
//    assertLegalCoffeeBig(
//            {static_cast<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const*>(s)->_state_vector[0],
//             static_cast<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const*>(s)->_state_vector[1]},grid_size);
}

void assertLegalCoffeeBig(Action const* a, size_t action_space_size)
{
    assert(a != nullptr);
    assert(std::stoi(a->index()) >= 0 && std::stoi(a->index()) < static_cast<int>(action_space_size));
}

GridWorldCoffeeBigBAExtension::GridWorldCoffeeBigBAExtension(size_t extra_features, domains::GridWorldCoffeeBig const& problem_domain) :
    _size(5),
    _extra_features(extra_features),
//    _states(), // initiated below
    _domain_size(0, 0, 0), // initiated below
    gridworldcoffeebig(problem_domain)
    {
    _domain_size = Domain_Size(
        static_cast<int>((_size * _size * 2)* std::pow(2, _extra_features)),
        4,
        static_cast<int>(_size * _size));

//    _store_statespace = false;
//    if (_size < 1) {
//        _store_statespace = true;
//    }

    // generate state space
//    if (_store_statespace) {
//        _states.reserve(_domain_size._S);
//        int i = 0;
//        for (unsigned int x_agent = 0; x_agent < _size; ++x_agent)
//        {
//            for (unsigned int y_agent = 0; y_agent < _size; ++y_agent)
//            {
//                domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos const agent_pos{x_agent, y_agent};
//                for (unsigned int rain = 0; rain < 2; ++rain)
//                {
//                    for (unsigned int feature_config = 0; feature_config < (unsigned int) (1 << _extra_features); ++feature_config)
//                    {
//                        assert(static_cast<unsigned int>(i) == _states.size());
//                        if (i % 10000000 == 0) {
//                            VLOG(1) << "going strong " << i;
//                        }
//                        _states.emplace_back(domains::GridWorldCoffeeBig::GridWorldCoffeeBigState(agent_pos, rain, feature_config, i));
//                        i++;
//                    }
//                }
//            }
//        }
//    }
}

Domain_Size GridWorldCoffeeBigBAExtension::domainSize() const
{
    return _domain_size;
}

State const* GridWorldCoffeeBigBAExtension::getState(std::string index) const
{
//    assert(index < _domain_size._S);
    return gridworldcoffeebig.getState(index);
}

Terminal GridWorldCoffeeBigBAExtension::terminal(State const* s, Action const* a, State const* new_s) const
{
    assertLegalCoffeeBig(s, _size, _domain_size._S);
    assertLegalCoffeeBig(a, _domain_size._A);
    assertLegalCoffeeBig(new_s, _size, _domain_size._S);

    auto const gw_state = static_cast<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const*>(s);

    return Terminal(domains::GridWorldCoffeeBig::goal_locations[0] ==
                    domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos({static_cast<unsigned int>(gw_state->_state_vector[0]), static_cast<unsigned int>(gw_state->_state_vector[1])})
                    ||
                    domains::GridWorldCoffeeBig::goal_locations[1] ==
                    domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos({static_cast<unsigned int>(gw_state->_state_vector[0]), static_cast<unsigned int>(gw_state->_state_vector[1])}));
}

Reward GridWorldCoffeeBigBAExtension::reward(State const* s, Action const* a, State const* new_s) const
{
    assertLegalCoffeeBig(s, _size, _domain_size._S);
    assertLegalCoffeeBig(a, _domain_size._A);
    assertLegalCoffeeBig(new_s, _size, _domain_size._S);

    auto const gw_state = static_cast<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const*>(s);

    if (domains::GridWorldCoffeeBig::goal_locations[0] ==
            domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos({static_cast<unsigned int>(gw_state->_state_vector[0]), static_cast<unsigned int>(gw_state->_state_vector[1])})
            ||
            domains::GridWorldCoffeeBig::goal_locations[1] ==
            domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos({static_cast<unsigned int>(gw_state->_state_vector[0]), static_cast<unsigned int>(gw_state->_state_vector[1])})) // found goal
    {
        return Reward(domains::GridWorldCoffeeBig::goal_reward);
    } else
    {
        return Reward(domains::GridWorldCoffeeBig::step_reward);
    }
}

}} // namespace bayes_adaptive::domain_extensions
