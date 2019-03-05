#include "FactoredTiger.hpp"

#include "easylogging++.h"

#include <algorithm>
#include <cassert>

namespace domains {

FactoredTiger::FactoredTiger(FactoredTigerDomainType type, size_t num_irrelevant_features) :
        _type(type),
        _S_size(2 << num_irrelevant_features)
{

    if (num_irrelevant_features < 1)
    {
        throw "cannot initiate FactoredTiger with " + std::to_string(num_irrelevant_features)
            + " irrelevant features";
    }

    auto const type_string = (_type == EPISODIC) ? "episodic" : "continuous";

    VLOG(1) << "initiated " << type_string << " factored tiger problem "
            << " with " << num_irrelevant_features << " irrelevent features";
}

FactoredTiger::TigerLocation FactoredTiger::tigerLocation(State const* s) const
{
    return (s->index() < _S_size / 2) ? LEFT : RIGHT;
}

Action const* FactoredTiger::generateRandomAction(State const* /*s*/) const
{
    return _actions.get(_action_distr(rnd::rng()));
}

void FactoredTiger::addLegalActions(State const* /*s*/, std::vector<Action const*>* actions) const
{

    assert(actions->empty());
    for (auto i = 0; i < _A_size; ++i) { actions->emplace_back(_actions.get(i)); }
}

void FactoredTiger::releaseAction(Action const* a) const
{
    assertLegal(a);
    // actions are stored locally so not deleted
}

Action const* FactoredTiger::copyAction(Action const* a) const
{
    assertLegal(a);
    // actions are never modified locally we can simply return itself
    return a;
}

double FactoredTiger::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* new_s) const
{
    // when opening door, there is a 50/50 % for each observation
    if (a->index() != OBSERVE)
    {
        return .5;
    }

    // when listening: .85 probability for hearing correctly
    return (tigerLocation(new_s) == o->index()) ? .85 : .15;
}

State const* FactoredTiger::sampleStartState() const
{
    // any state is a legit start state
    return _states.get(_state_distr(rnd::rng()));
}

Terminal
    FactoredTiger::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    assertLegal(*s);
    assertLegal(a);
    assert(o != nullptr);
    assert(r != nullptr);

    if (a->index() == TigerAction::OBSERVE)
    // listening does not change the state
    // and return correct observation by some probability
    {
        r->set(-1);

        // agent hears correctly with 85%
        auto const correct_observation = rnd::uniform_rand01() < .85;

        // return listen right if:
        //  correct & right
        //  or
        //  incorrect & left
        //
        //  thus correct xor left will do the trick
        *o = ((correct_observation ^ (tigerLocation(*s) == TigerLocation::LEFT)) != 0)
                 ? &_hear_right
                 : &_hear_left;

    } else // opening door
           // produces a random state and observation
           // and reward associated with the state/action pair
    {
        // if action opens the same door as the state then good, otherwise bad
        r->set((a->index() == tigerLocation(*s)) ? 10 : -100);
        *o = (rnd::boolean()) ? &_observations[0] : &_observations[1];
        *s = sampleStartState();
    }

    assertLegal(a);
    assertLegal(*o);
    assertLegal(*s);
    assert(r->toDouble() == -1 || r->toDouble() == 10 || r->toDouble() == -100);

    // terminated if opening door && episodic type
    return Terminal(
        a->index() != TigerAction::OBSERVE && _type == FactoredTigerDomainType::EPISODIC);
}

void FactoredTiger::releaseObservation(Observation const* o) const
{
    assertLegal(o);
    // observation are stored locally so not deleted
}

void FactoredTiger::releaseState(State const* s) const
{
    assertLegal(s);
    // states are stored locally - so not actually deleted
}

Observation const* FactoredTiger::copyObservation(Observation const* o) const
{
    assertLegal(o);
    // observation are never modified locally we can simply return itself
    return o;
}

State const* FactoredTiger::copyState(State const* s) const
{
    assertLegal(s);
    // states are never modified locally we can simply return itself
    return s;
}

void FactoredTiger::assertLegal(Action const* a) const
{
    assert(a != nullptr && a->index() >= 0 && a->index() <= _A_size);
}

void FactoredTiger::assertLegal(State const* s) const
{
    assert(s != nullptr && s->index() >= 0 && s->index() < _S_size);
}

void FactoredTiger::assertLegal(Observation const* o) const
{
    assert(o != nullptr && o->index() >= 0 && o->index() < _O_size);
}

} // namespace domains
