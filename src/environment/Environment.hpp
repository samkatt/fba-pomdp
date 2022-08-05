#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

#include "environment/Terminal.hpp"
#include <memory>

class Action;
class Observation;
class Reward;
class State;

namespace configurations {
struct DomainConf;
}

/**
 * An environment in RL is the entity in which the agent interacts. This is in
 * charge of the real trajectories the agent goes through
 *
 * This class owns its own states & observations, so whenever you decide to
 * store any of them, please call the copy function, likewise call the release
 * function if you are done using them:
 *
 * # Memory management
 *
 * The memory management of `State`, and `Observation` in this project are the
 * responsibility of the environments. In particular, the allocation,
 * modification, and destruction is (can only be done) 'here'.
 *
 * To 'manage' memory, such as 'copying' or 'de-allocating', functions are
 * expected to be implemented to do so. In particular, here we force
 * implementations to provide a `copyState` and `copyObservation`, and
 * `releaseState` and `releaseObservation`. The whole point here is to
 * hopefully avoid as many allocations as possible (e.g. store all
 * actions/observations in a single table). For example, if a caller
 *
 * This also (to some degree) explains (justifies) the fact that *all* calls
 * return or expect `const` members (see `step`): outside of this (and derived)
 * classes, nobody is allowed to touch them, to avoid nasty bugs. This may
 * result in const-casts within these classes, because sometimes you do want
 * in-place modifications. This is probably a sub-optimal solution, but here we
 * are :(.
 *
 * In short, if you ever need to create, modify, copy, or release `State` or
 * `Observation`, look for functions here.
 */
class Environment
{
public:
    virtual ~Environment() = default;

    /**
     * @brief creates a state to start an episode
     *
     * Do not forget to `releaseState` the output of this function afterwards!
     *
     * @return is const because outside of this class, nobody should touch it
     **/
    virtual State const* sampleStartState() const = 0;

    /**
     * @brief performs a step in the environment
     *
     * Applies `a` on `s` and produces an `o` & `r` in the process Returns
     * whether the resulting state is `Terminal`
     *
     * Note that the arguments all are const. This is to allow for strict
     * memory management (no othe rparts of the code is allowed to change
     * them). To allow for 'simple' updates, where `s` is updated in place, it
     * is expected that the implementation will do dangerous `const_cast` with
     * caution.
     *
     * Do not forget to `releaseState` and `releaseObservation` afterwards!
     *
     * @param s: current -> next state (in/output)
     * @param[in] a: the taken action
     * @param[out] o: the resulting observation
     * @param[out] r: the reward associated with step
     *
     * @return whether or not the step was terminal
     **/
    virtual Terminal
        step(State const** s, Action const* a, Observation const** o, Reward* r) const = 0;

    /** state, observation ownership functionality **/
    virtual void releaseObservation(Observation const* o) const = 0;
    virtual void releaseState(State const* s) const             = 0;

    virtual Observation const* copyObservation(Observation const* o) const = 0;
    virtual State const* copyState(State const* s) const                   = 0;
};

namespace factory {

std::unique_ptr<Environment> makeEnvironment(configurations::DomainConf const& c);

} // namespace factory

#endif // ENVIRONMENT_HPP
