#include "DummyDomain.hpp"

#include "easylogging++.h"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"

namespace domains {

DummyDomain::DummyDomain()
{
    VLOG(1) << "initiated dummy domian";
}

State const* DummyDomain::sampleStartState() const
{
    return new IndexState("0");
}

Action const* DummyDomain::generateRandomAction(State const* /*s*/) const
{
    return new IndexAction(std::to_string(0));
}

void DummyDomain::addLegalActions(State const* /*s*/, std::vector<Action const*>* actions) const
{
    assert(actions->empty());
    actions->emplace_back(new IndexAction(std::to_string(0)));
}

double DummyDomain::computeObservationProbability(
    Observation const* /*o*/,
    Action const* /*a*/,
    State const* /*new_s*/
    ) const
{
    return 1;
}

Terminal
    DummyDomain::step(State const** /*s*/, Action const* /*a*/, Observation const** o, Reward* r)
        const
{
    *o = new IndexObservation("0");

    r->set(1);
    return Terminal(false);
}

void DummyDomain::releaseAction(Action const* a) const
{
    assert(a != nullptr);
    delete (a);
}

void DummyDomain::releaseObservation(Observation const* o) const
{
    assert(o != nullptr);
    delete (o);
}

void DummyDomain::releaseState(State const* s) const
{
    assert(s != nullptr);
    delete (s);
}

Action const* DummyDomain::copyAction(Action const* a) const
{
    assert(a != nullptr);
    return new IndexAction(a->index());
}

Observation const* DummyDomain::copyObservation(Observation const* o) const
{
    assert(o != nullptr);
    return new IndexObservation(o->index());
}

State const* DummyDomain::copyState(State const* s) const
{
    assert(s != nullptr);
    return new IndexState(s->index());
}

    void DummyDomain::clearCache() const {

    }

} // namespace domains
