#include "CoffeeProblem.hpp"

#include "easylogging++.h"

#include "domains/coffee/CoffeeProblemIndices.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"

namespace domains {

CoffeeProblem::CoffeeProblem(std::string const& version) :
        _states(_S), _actions(_A), _observations(_O)
{
    // change probabilitie if boutilier's version
    if (version == "boutilier")
    {
        _fetch_coffee_lose_desire_chance = 0;
        _coffee_is_drinked               = 0;
        _acquire_coffee_desire           = 0;
    }
}

Action const* CoffeeProblem::generateRandomAction(State const* s) const
{
    assertLegal(s);

    // all actions are legal in every state in this problem,
    // so we can just return a random action
    return _actions.get(static_cast<int>(rnd::boolean()));
}

void CoffeeProblem::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assert(actions != nullptr && actions->empty());
    assertLegal(s);

    // all actions are legal in every state
    actions->emplace_back(_actions.get(0));
    actions->emplace_back(_actions.get(1));
}

double CoffeeProblem::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* s) const
{
    LOG(WARNING) << "CoffeeProblem::computeObservationProbability has not been tested";

    if (a->index() == GetCoffee)
    {
        return (o == _observations.get(Want_Coffee)) ? 1 : 0;
    }

    if (static_cast<CoffeeProblemState const*>(s)->wantsCoffee())
    {
        return (o == _observations.get(Want_Coffee)) ? _correctly_observes_desire
                                                     : 1 - _correctly_observes_desire;
    }

    // person does not want coffee
    return (o == _observations.get(Not_Want_Coffee)) ? _correctly_observes_lack_of_desire
                                                     : 1 - _correctly_observes_lack_of_desire;
}

State const* CoffeeProblem::sampleStartState() const
{
    return _states.get(_state_distr(rnd::rng()));
}

Terminal
    CoffeeProblem::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    assertLegal(*s);
    assertLegal(a);

    auto s_coffee  = static_cast<CoffeeProblemState const*>(*s);
    auto new_state = (*s)->index();

    double reward = -.5;
    // reward function
    if (a->index() == GetCoffee)
    {
        reward = -.5;
    }

    if (s_coffee->wet())
    {
        reward = -1;
    }

    if (s_coffee->wantsCoffee())
    {
        reward += (s_coffee->hasCoffee() ? 2 : -2);
    }

    // transition function
    if (a->index() == GetCoffee)
    {
        // robot becomes wet if raining & no umbrella
        if (s_coffee->rains() && !s_coffee->umbrella())
        {
            new_state = new_state | WET;
        }

        // agent has coffee if action succeeds
        if (rnd::uniform_rand01() < _fetch_coffee_success_rate)
        {
            new_state = new_state | HAS_COFEE;
        }

        // there's a change the desire for coffee stays
        if (rnd::uniform_rand01() < _fetch_coffee_lose_desire_chance)
        {
            new_state = new_state & ~WANTS_COFFEE;
        }

    } else // test coffee
    {
        if (rnd::uniform_rand01() < _coffee_is_drinked)
        {
            new_state = new_state & ~HAS_COFEE;
        }

        if (rnd::uniform_rand01() < _acquire_coffee_desire)
        {
            new_state = new_state | WANTS_COFFEE;
        }
    }

    assert(new_state < _S);

    *s       = _states.get(new_state);
    s_coffee = static_cast<CoffeeProblemState const*>(*s);

    // observation function
    if (a->index() == GetCoffee)
    {
        *o = _observations.get(Want_Coffee);
    } else
    {

        if (s_coffee->wantsCoffee())
        {
            *o = (rnd::uniform_rand01() < _correctly_observes_desire)
                     ? _observations.get(Want_Coffee)
                     : _observations.get(Not_Want_Coffee);
        } else
        {
            *o = (rnd::uniform_rand01() < _correctly_observes_lack_of_desire)
                     ? _observations.get(Not_Want_Coffee)
                     : _observations.get(Want_Coffee);
        }
    }

    r->set(reward);
    return Terminal(false);
}

void CoffeeProblem::releaseAction(Action const* a) const
{
    assertLegal(a); // both actions are member variables so not deleted
}

void CoffeeProblem::releaseObservation(Observation const* o) const
{
    assertLegal(o); // all observations are member variables so not deleted
}

void CoffeeProblem::releaseState(State const* s) const
{
    assertLegal(s); // all states are member variables so not deleted
}

Action const* CoffeeProblem::copyAction(Action const* a) const
{
    assertLegal(a);
    return a; // local const variable, so never modified and can return itself
}

Observation const* CoffeeProblem::copyObservation(Observation const* o) const
{
    assertLegal(o);
    return o; // local const variable, so never modified and can return itself
}

State const* CoffeeProblem::copyState(State const* s) const
{
    assertLegal(s);
    return s; // local const variable, so never modified and can return itself
}

void CoffeeProblem::assertLegal(State const* s) const
{
    assert(s != nullptr);
    assert(s->index() < _S);
}

void CoffeeProblem::assertLegal(Action const* a) const
{
    assert(a != nullptr);
    assert(a->index() < _A);
}

void CoffeeProblem::assertLegal(Observation const* o) const
{
    assert(o != nullptr);
    assert(o->index() < _O);
}

} // namespace domains
