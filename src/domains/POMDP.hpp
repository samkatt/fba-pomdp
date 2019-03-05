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
 * Adds function required by the planning methods
 **/
class POMDP : public Environment
{
public:
    ~POMDP() override = default;

    /**** functions required by planners.... ****/

    /**
     * @brief returns a random action that islegal in state s
     **/
    virtual Action const* generateRandomAction(State const* s) const = 0;

    /**
     * @brief fills actions with legal actions from s
     *
     * NOTE: this part cannot be optimized by making actions a **,
     * because eventually we need the actions individually, not
     * in a container, so the container is useless in that way
     **/
    virtual void addLegalActions(State const* s, std::vector<Action const*>* actions) const = 0;

    /**** functions required by beliefs ****/

    /**
     * @brief returns the probability of generating observation o given <a,s'>
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
