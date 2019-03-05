#include "SysAdminBAExtension.hpp"

namespace bayes_adaptive { namespace domain_extensions {

SysAdminBAExtension::SysAdminBAExtension(int size) : _size(size), _states() // initialized below
{
    assert(size > 0);

    _states.reserve(_domain_size._S);
    for (auto i = 0; i < _domain_size._S; ++i)
    { _states.emplace_back(domains::SysAdminState(i, _size)); }
}

Domain_Size SysAdminBAExtension::domainSize() const
{
    return _domain_size;
}

State const* SysAdminBAExtension::getState(int index) const
{
    assert(index < _domain_size._S);
    return &_states[index];
}

Terminal SysAdminBAExtension::terminal(State const* s, Action const* a, State const* new_s) const
{
    assert(s != nullptr && s->index() >= 0 && s->index() < _domain_size._S);
    assert(new_s != nullptr && new_s->index() >= 0 && new_s->index() < _domain_size._S);
    assert(a != nullptr && a->index() >= 0 && a->index() < _domain_size._A);

    return Terminal(false);
}

Reward SysAdminBAExtension::reward(State const* s, Action const* a, State const* new_s) const
{
    assert(s != nullptr && s->index() >= 0 && s->index() < _domain_size._S);
    assert(new_s != nullptr && new_s->index() >= 0 && new_s->index() < _domain_size._S);
    assert(a != nullptr && a->index() >= 0 && a->index() < _domain_size._A);

    auto const sys_state = static_cast<domains::SysAdminState const*>(new_s);

    auto const rebooting = static_cast<unsigned>(a->index() >= static_cast<int>(_size));
    auto const reward    = sys_state->numOperationalComputers();

    return Reward(reward - domains::SysAdmin::param._reboot_cost * rebooting);
}

}} // namespace bayes_adaptive::domain_extensions
