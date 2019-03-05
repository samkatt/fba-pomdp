#ifndef DUMMYDOMAIN_HPP
#define DUMMYDOMAIN_HPP

#include "domains/POMDP.hpp"

#include <cassert>
#include <memory>

#include "environment/Terminal.hpp"
class Action;
class Observation;
class Reward;
class State;

namespace domains {

/**
 * @brief A domain for testing
 *
 * Has one state, action & observation.
 * Always returns a reward of 1 and never terminates
 *
 * Some dummy domains inherit from this very simple class
 **/
class DummyDomain : public POMDP
{
public:
    DummyDomain();
    ~DummyDomain() override = default;

    /*** environment implementation ***/
    State const* sampleStartState() const final;
    Action const* generateRandomAction(State const* s) const override;

    Terminal
        step(State const** s, Action const* a, Observation const** o, Reward* r) const override;

    double computeObservationProbability(Observation const* o, Action const* a, State const* s)
        const final;

    /**
     * @brief returns action(1)
     **/
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const override;

    void releaseAction(Action const* a) const final;
    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;

    Action const* copyAction(Action const* a) const final;
    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;
};

} // namespace domains

#endif // DUMMYDOMAIN_HPP
