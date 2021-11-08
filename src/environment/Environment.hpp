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
 * An environment in RL is the entity in which
 * the agent interacts. This is in charge of
 * the real trajectories the agent goes through
 *
 * This class owns its own states, actions & observations,
 * so whenever you decide to store any of them, please call the
 * copy function, likewise call the release function if you are done
 * using them.
 */
class Environment
{
public:
    virtual ~Environment() = default;

    /**
     * @brief creates a state to start an episode
     **/
    virtual State const* sampleStartState() const = 0;

    /**
     * @brief performs a step in the environment
     *
     * Applies a on s and produces an o & r in the process
     * Returns whether the resulting state is terminal
     **/
    virtual Terminal
        step(State const** s, Action const* a, Observation const** o, Reward* r) const = 0;

    /** state, observation ownership functionality **/
    virtual void releaseObservation(Observation const* o) const = 0;
    virtual void releaseState(State const* s) const             = 0;

    virtual Observation const* copyObservation(Observation const* o) const = 0;
    virtual State const* copyState(State const* s) const                   = 0;

    virtual void clearCache() const = 0;
};

namespace factory {

std::unique_ptr<Environment> makeEnvironment(configurations::DomainConf const& c);

} // namespace factory

#endif // ENVIRONMENT_HPP
