#include "FactoredDummyDomain.hpp"

#include <vector>

#include "easylogging++.h"

#include "environment/State.hpp"
#include "utils/index.hpp"
#include "utils/random.hpp"

namespace domains {

FactoredDummyDomain::FactoredDummyDomain(size_t size) : _size(size), _num_states(size * size)
{
    VLOG(1) << "initiated factored dummy domain of size " << _size;
}

Action const* FactoredDummyDomain::generateRandomAction(State const* /*s*/) const
{
    return (rnd::boolean()) ? &_a_up : &_a_right;
}

void FactoredDummyDomain::addLegalActions(State const* /*s*/, std::vector<Action const*>* actions)
    const
{
    assert(actions->empty());
    actions->emplace_back(&_a_up);
    actions->emplace_back(&_a_right);
}

double FactoredDummyDomain::computeObservationProbability(
    Observation const* /*o*/,
    Action const* /*a*/,
    State const* /*new_s*/
    ) const
{
    return 1;
}

State const* FactoredDummyDomain::sampleStartState() const
{
    return new IndexState(0);
}

Terminal
    FactoredDummyDomain::step(State const** s, Action const* a, Observation const** o, Reward* r)
        const
{
    assertLegal(*s);
    assertLegal(a);

    auto state_index = (*s)->index();

    if (a->index() == UP)
    {
        // +1 unless on top edge already
        state_index += (((state_index + 1) % _size) != 0u) ? 1 : 0;
    }

    if (a->index() == RIGHT)
    {
        // + _size unless on right edge already
        state_index +=
            (state_index < static_cast<int>((_size - 1) * _size)) ? static_cast<int>(_size) : 0;
    }

    const_cast<State*>(*s)->index(state_index);

    *o = &_observation;
    r->set((state_index < static_cast<int>(_num_states) - 1) ? -1 : 0);
    return Terminal(false);
}

void FactoredDummyDomain::releaseAction(Action const* /*a*/) const
{
    // action locally stored as members
}

void FactoredDummyDomain::releaseObservation(Observation const* /*o*/) const
{
    // observation locally stored as members
}

void FactoredDummyDomain::releaseState(State const* s) const
{
    delete (s);
}

Action const* FactoredDummyDomain::copyAction(Action const* a) const
{
    // action locally stored as members
    return a;
}

Observation const* FactoredDummyDomain::copyObservation(Observation const* o) const
{
    // observation locally stored as members
    return o;
}

State const* FactoredDummyDomain::copyState(State const* s) const
{
    return new IndexState(s->index());
}

void FactoredDummyDomain::assertLegal(State const* s) const
{
    assert(s != nullptr && s->index() < static_cast<int>(_num_states));
}

void FactoredDummyDomain::assertLegal(Action const* a) const
{
    assert(a != nullptr && a->index() < 2);
}

void FactoredDummyDomain::assertLegal(Observation const* o) const
{
    assert(o == &_observation);
}

} // namespace domains
