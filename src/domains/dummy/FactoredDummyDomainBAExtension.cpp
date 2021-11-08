#include "FactoredDummyDomainBAExtension.hpp"

#include <vector>

#include "environment/Action.hpp"
#include "environment/State.hpp"

namespace bayes_adaptive { namespace domain_extensions {

FactoredDummyDomainBAExtension::FactoredDummyDomainBAExtension(int size) :
        _size(size),
        _num_states(size * size)
{
}

Domain_Size FactoredDummyDomainBAExtension::domainSize() const
{
    return {static_cast<int>(_num_states), 2, 1};
}

State const* FactoredDummyDomainBAExtension::getState(std::string index) const
{
    assert(std::stoi(index) < static_cast<int>(_num_states));
    return new IndexState(index);
}

Terminal FactoredDummyDomainBAExtension::terminal(
    State const* s,
    Action const* a,
    State const* new_s) const
{
    assert(s != nullptr &&std::stoi(s->index())< static_cast<int>(_num_states));
    assert(new_s != nullptr &&std::stoi(new_s->index()) < static_cast<int>(_num_states));
    assert(a != nullptr && std::stoi(a->index()) < 2);

    return Terminal(false); // this problem never terminates
}

Reward FactoredDummyDomainBAExtension::reward(State const* s, Action const* a, State const* new_s)
    const
{
    assert(s != nullptr &&std::stoi(s->index())< static_cast<int>(_num_states));
    assert(new_s != nullptr &&std::stoi(new_s->index()) < static_cast<int>(_num_states));
    assert(a != nullptr && std::stoi(a->index()) < 2);

    return (std::stoi(new_s->index()) < static_cast<int>(_num_states) - 1) ? Reward(-1) : Reward(0);
}

}} // namespace bayes_adaptive::domain_extensions
