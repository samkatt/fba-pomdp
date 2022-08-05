#ifndef POINTESTIMATION_HPP
#define POINTESTIMATION_HPP

#include "beliefs/Belief.hpp"
class Action;
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
class PointEstimation : public Belief
{
public:
    PointEstimation();
    explicit PointEstimation(State const* s);

    // deny shallow copies
    PointEstimation(PointEstimation const&)            = delete;
    PointEstimation& operator=(PointEstimation const&) = delete;

    /*** implement Belief interface ***/
    State const* sample() const final;

    void initiate(POMDP const& simulator) final;
    void free(POMDP const& simulator) final;

    /**
     * @brief rejection sampling state based on observation
     **/
    void updateEstimation(Action const* a, Observation const* o, POMDP const& simulator) final;

private:
    State const* _point_estimate = nullptr;
};

} // namespace beliefs

#endif // POINTESTIMATION_HPP
