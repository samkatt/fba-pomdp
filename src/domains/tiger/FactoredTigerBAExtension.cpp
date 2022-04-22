#include "FactoredTigerBAExtension.hpp"

#include "environment/Action.hpp"
#include "environment/State.hpp"

namespace bayes_adaptive { namespace domain_extensions {

FactoredTigerBAExtension::FactoredTigerBAExtension(
    domains::FactoredTiger::FactoredTigerDomainType type,
    size_t num_irrelevant_features) :
        _domain_size(2 << num_irrelevant_features, 3, 2), _type(type)
{
}

Domain_Size FactoredTigerBAExtension::domainSize() const
{
    return _domain_size;
}

State const* FactoredTigerBAExtension::getState(int index) const
{
    assert(index < _domain_size._S && index >= 0);

    return _states.get(index);
}

Terminal
    FactoredTigerBAExtension::terminal(State const* s, Action const* a, State const* new_s) const
{
    assert(s != nullptr && s->index() >= 0 && s->index() < _domain_size._S);
    assert(new_s != nullptr && new_s->index() >= 0 && new_s->index() < _domain_size._S);
    assert(a != nullptr && a->index() >= 0 && a->index() <= _domain_size._A);

    // its terminal if we are dealing with episodic and if action is to not observe
    return Terminal(
        _type == domains::FactoredTiger::FactoredTigerDomainType::EPISODIC
        && a->index() != domains::FactoredTiger::TigerAction::OBSERVE);
}

Reward FactoredTigerBAExtension::reward(State const* s, Action const* a, State const* new_s) const
{

    assert(s != nullptr && s->index() >= 0 && s->index() < _domain_size._S);
    assert(new_s != nullptr && new_s->index() >= 0 && new_s->index() < _domain_size._S);
    assert(a != nullptr && a->index() >= 0 && a->index() <= _domain_size._A);

    if (a->index() == domains::FactoredTiger::TigerAction::OBSERVE)
    {
        return Reward(-1);
    }

    auto const tiger_location = (s->index() < _domain_size._S / 2) ? domains::FactoredTiger::LEFT
                                                                   : domains::FactoredTiger::RIGHT;

    return (a->index() == tiger_location) ? Reward(10) : Reward(-100);
}

}} // namespace bayes_adaptive::domain_extensions
