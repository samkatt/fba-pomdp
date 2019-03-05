#include "Tiger.hpp"

#include "easylogging++.h"

#include "utils/random.hpp"

namespace domains {

Tiger::Tiger(TigerType type) : _type(type)
{
    auto const type_string = (_type == EPISODIC) ? "episodic" : "continuous";

    VLOG(1) << "initiated " << type_string << " tiger problem";
}

State const* Tiger::sampleStartState() const
{
    return rnd::boolean() ? &_tiger_left : &_tiger_right;
}

Action const* Tiger::generateRandomAction(State const* s) const
{
    legalStateCheck(s);
    return _actions.get(_action_distr(rnd::rng()));
}

double Tiger::computeObservationProbability(Observation const* o, Action const* a, State const* s)
    const
{
    // when opening door, there is a 50/50 % for each observation
    if (a->index() != OBSERVE)
    {
        return .5;
    }

    // when listening: .85 probability for hearing correctly
    return (s->index() == o->index()) ? .85 : .15;
}

Terminal Tiger::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    legalStateCheck(*s);
    legalActionCheck(a);
    assert(o != nullptr);
    assert(r != nullptr);

    auto const tiger_is_left = (*s)->index() == Literal::LEFT;

    if (a->index() == Literal::OBSERVE)
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
        *o = ((correct_observation ^ tiger_is_left) != 0) ? &_hear_right : &_hear_left;

    } else // opening door
           // produces a random state and observation
           // and reward associated with the state/action pair
    {
        // if action opens the same door as the state then good, otherwise bad
        r->set((a->index() == (*s)->index()) ? 10 : -100);
        *o = _observations.get(static_cast<int>(rnd::boolean()));
        *s = _states.get(static_cast<int>(rnd::boolean()));
    }

    legalActionCheck(a);
    legalObservationCheck(*o);
    legalStateCheck(*s);
    assert(r->toDouble() == -1 || r->toDouble() == 10 || r->toDouble() == -100);

    return Terminal(_type == EPISODIC && a->index() != OBSERVE);
}

void Tiger::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assert(actions->empty());
    legalStateCheck(s);

    actions->emplace_back(&_open_left);
    actions->emplace_back(&_open_right);
    actions->emplace_back(&_listen);
}

void Tiger::releaseAction(Action const* a) const
{
    legalActionCheck(a);
    // action is a member of the object
}

void Tiger::releaseObservation(Observation const* o) const
{
    legalObservationCheck(o);
    // observation is a member of the object
}

void Tiger::releaseState(State const* s) const
{
    legalStateCheck(s);
    // state is a member of the object
}

Action const* Tiger::copyAction(Action const* a) const
{
    legalActionCheck(a);
    return a;
}

Observation const* Tiger::copyObservation(Observation const* o) const
{
    legalObservationCheck(o);
    return o;
}

State const* Tiger::copyState(State const* s) const
{
    legalStateCheck(s);
    return s;
}

void Tiger::legalActionCheck(Action const* a) const
{
    assert(a != nullptr && a->index() >= 0 && a->index() < 3);
}

void Tiger::legalObservationCheck(Observation const* o) const
{
    assert(o != nullptr && o->index() >= 0 && o->index() < 2);
}

void Tiger::legalStateCheck(State const* s) const
{
    assert(s != nullptr && s->index() >= 0 && s->index() < 2);
}

} // namespace domains
