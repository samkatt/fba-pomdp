#include "DummyDomainBAExtension.hpp"

#include <vector>

#include "environment/Action.hpp"
#include "environment/State.hpp"

namespace bayes_adaptive { namespace domain_extensions {

Domain_Size DummyDomainBAExtension::domainSize() const
{
    return {1, 1, 1};
}

State const* DummyDomainBAExtension::getState(int index) const
{
    assert(index == 0);
    return new IndexState(index);
}

Terminal DummyDomainBAExtension::terminal(State const* s, Action const* a, State const* new_s) const
{
    assert(s != nullptr && s->index() == 0);
    assert(a != nullptr && a->index() == 0);
    assert(new_s != nullptr && new_s->index() == 0);

    return Terminal(false);
}

Reward DummyDomainBAExtension::reward(State const* s, Action const* a, State const* new_s) const
{
    assert(s != nullptr && s->index() == 0);
    assert(a != nullptr && a->index() == 0);
    assert(new_s != nullptr && new_s->index() == 0);

    return Reward(1);
}

}} // namespace bayes_adaptive::domain_extensions
