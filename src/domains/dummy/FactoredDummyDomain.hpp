#ifndef FACTOREDDUMMYDOMAIN_HPP
#define FACTOREDDUMMYDOMAIN_HPP

#include "domains/POMDP.hpp"

#include <memory>

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
class State;
class vector;

namespace domains {

/**
 * @brief A dummy domain for the factored approaches
 *
 * A simple n*n grid where the agent starts in the down left corner (0,0)
 * and can only go 'up' or 'right' with a deterministic transition. The
 * observation function always returns observatio(0), and the reward for being in a cell
 * is -1 everywhere except for the upper right corner. This problem does
 * not terminate - ever.
 *
 * Moving 'into' a wall does nothing
 **/
class FactoredDummyDomain : public POMDP
{
public:
    enum ACTIONS { UP, RIGHT };

    explicit FactoredDummyDomain(size_t size);

    /*** POMDP interface ***/
    Action const* generateRandomAction(State const* s) const final;
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;

    double computeObservationProbability(Observation const* o, Action const* a, State const* s)
        const final;

    /*** environment interface ***/
    State const* sampleStartState() const final;
    Terminal step(State const** s, Action const* a, Observation const** o, Reward* r) const final;
    void releaseAction(Action const* a) const final;
    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;
    Action const* copyAction(Action const* a) const final;
    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;

private:
    size_t _size;
    size_t _num_states;

    IndexObservation const _observation = IndexObservation(0);
    IndexAction const _a_up = IndexAction(ACTIONS::UP), _a_right = IndexAction(ACTIONS::RIGHT);

    void assertLegal(State const* s) const;
    void assertLegal(Action const* a) const;
    void assertLegal(Observation const* o) const;
};

} // namespace domains

#endif // FACTOREDDUMMYDOMAIN_HPP
