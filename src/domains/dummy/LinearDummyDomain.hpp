#ifndef LINEARDUMMYDOMAIN_HPP
#define LINEARDUMMYDOMAIN_HPP

#include "domains/dummy/DummyDomain.hpp"

#include <cassert>

#include "environment/Terminal.hpp"
class Action;
class Observation;
class Reward;
class State;

namespace domains {

/**
 * @brief A dummy domain with deterministic linear state transitions
 *
 * A step in this domain will either increment (action = forward) or
 * decrement (action = backwards) the state index
 **/
class LinearDummyDomain : public DummyDomain
{
public:
    enum Actions { FORWARD, BACKWARD };

    LinearDummyDomain();

    /**** DummyDomain overrides ****/

    /**
     * @brief returns forward or backward
     **/
    Action const* generateRandomAction(State const* s) const final;

    /**
     * @brief returns both actions, unless in state == 0, then online 'forward'
     **/
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;

    /**
     * @brief increments if action == forward, decrements if otherwise
     */
    Terminal step(State const** s, Action const* a, Observation const** o, Reward* r) const final;
};

} // namespace domains

#endif // LINEARDUMMYDOMAIN_HPP
