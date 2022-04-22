#ifndef POMDP_HPP
#define POMDP_HPP

#include "environment/Environment.hpp"

#include <memory>
#include <vector>

class State;
class Action;
class Observation;

namespace configurations {
struct DomainConf;
}

/**
 * @brief An interface for a domain
 *
 * Adds function required by the planning methods.
 *
 * In addition to memory management of `State` and `Observation`, this class
 * forces any and all creation or de-allocation of `Action` to be done through
 * the interface provided here.
 *
 * @see `Environment` for more details on memory management (these are extended to actions here)
 **/
class POMDP : public Environment
{
public:
    ~POMDP() override = default;

    /**** functions required by planners.... ****/

    /**
     * @brief returns a random action that is legal in state s
     *
     * Do not forget to `releaseAction` on the result when done
     *
     * @return a const Action, because no-one else may modify it
     **/
    virtual Action const* generateRandomAction(State const* s) const = 0;

    /**
     * @brief fills actions with legal actions from s
     *
     * NOTE: this part cannot be optimized by making actions a **,
     * because eventually we need the actions individually, not
     * in a container, so the container is useless in that way
     *
     * Do not forget to `releaseAction` on elements in `actions`
     *
     * @param[in] s: the state for which to get the legal actions for
     * @param[out] actions: each (legal) action is added to this container
     *
     * @return void: `actions` is considered the result of this function call
     **/
    virtual void addLegalActions(State const* s, std::vector<Action const*>* actions) const = 0;

    /**** functions required by beliefs ****/

    /**
     * @brief returns the probability of generating observation o given <a,s'>
     *
     * p(o |  a, s')
     *
     * @param[in] o: for which to compute the probability of
     * @param[in] a: given action (probability is conditioned on this)
     * @param[in] new_s: given 'new' state (probability is conditioned on this)
     *
     * @return probability [0,1]
     **/
    virtual double computeObservationProbability(
        Observation const* o,
        Action const* a,
        State const* new_s) const = 0;

    /** action ownsership functionality **/
    virtual void releaseAction(Action const* a) const       = 0;
    virtual Action const* copyAction(Action const* a) const = 0;
};

namespace factory {

std::unique_ptr<POMDP> makePOMDP(configurations::DomainConf const& c);

} // namespace factory

#endif // POMDP_HPP
