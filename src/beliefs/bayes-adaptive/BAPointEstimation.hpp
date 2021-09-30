#ifndef BAPOINTESTIMATION_HPP
#define BAPOINTESTIMATION_HPP

#include <bayes-adaptive/abstractions/Abstraction.hpp>
#include "beliefs/bayes-adaptive/BABelief.hpp"
class Action;
class BAPOMDP;
class Observation;
class POMDP;
class State;

namespace beliefs {

/**
 * @brief Estimates the state with a single point
 *
 * Estimates are maintained through rejection sampling:
 * Whenever a new observation has been perceived after performing action a,
 * this belief will sample transitions until the same observation is produced.
 * The state of this transition is used as the new point estimate.
 **/
class BAPointEstimation : public ::beliefs::BABelief
{
public:
    BAPointEstimation();
    explicit BAPointEstimation(State const* s);

    // deny shallow copies
    BAPointEstimation(BAPointEstimation const&) = delete;
    BAPointEstimation& operator=(BAPointEstimation const&) = delete;

    /*** implement Belief interface ***/
    State const* sample() const final;

    void initiate(POMDP const& simulator) final;
    void free(POMDP const& simulator) final;

    /**
     * @brief rejection sampling state based on observation
     **/
    void updateEstimation(Action const* a, Observation const* o, POMDP const& simulator) final;

    /*** implement BABelief ***/
    void resetDomainStateDistribution(BAPOMDP const& bapomdp) final;
    void resetDomainStateDistributionAndAddAbstraction(const BAPOMDP &bapomdp, Abstraction &abstraction, int i) final;

private:
    State const* _point_estimate = nullptr;
};

} // namespace beliefs

#endif // BAPOINTESTIMATION_HPP
