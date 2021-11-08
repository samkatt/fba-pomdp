#include "TigerBAExtension.hpp"

#include "environment/Action.hpp"
#include "environment/State.hpp"

namespace bayes_adaptive { namespace domain_extensions {

TigerBAExtension::TigerBAExtension(domains::Tiger::TigerType type) : _type(type) {}

Domain_Size TigerBAExtension::domainSize() const
{
    return {2, 3, 2};
}

State const* TigerBAExtension::getState(std::string index) const
{
    assert(std::stoi(index) > -1 && std::stoi(index) < 2);
    return _states.get(std::stoi(index));
}

Terminal TigerBAExtension::terminal(State const* s, Action const* a, State const* new_s) const
{
    assert(s != nullptr &&std::stoi(s->index())>= 0 &&std::stoi(s->index())< 2);
    assert(new_s != nullptr &&std::stoi(new_s->index()) >= 0 &&std::stoi(new_s->index()) < 2);
    assert(a != nullptr && std::stoi(a->index()) >= 0 && std::stoi(a->index()) < 3);

    // episode stop only when we are opening  adoor in the
    // episodic scenario
    return Terminal(_type == domains::Tiger::EPISODIC && std::stoi(a->index()) != domains::Tiger::OBSERVE);
}

Reward TigerBAExtension::reward(State const* s, Action const* a, State const* new_s) const
{
    assert(s != nullptr &&std::stoi(s->index())>= 0 &&std::stoi(s->index())< 2);
    assert(new_s != nullptr &&std::stoi(new_s->index()) >= 0 &&std::stoi(new_s->index()) < 2);
    assert(a != nullptr && std::stoi(a->index()) >= 0 && std::stoi(a->index()) < 3);

    if (std::stoi(a->index()) == domains::Tiger::Literal::OBSERVE)
    {
        return Reward(-1);
    }

    return (a->index() == s->index() ? Reward(10) : Reward(-100));
}

}} // namespace bayes_adaptive::domain_extensions
