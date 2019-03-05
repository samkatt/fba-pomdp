#ifndef BAREJECTIONSAMPLING_HPP
#define BAREJECTIONSAMPLING_HPP

#include "beliefs/bayes-adaptive/BABelief.hpp"

#include "beliefs/particle_filters/FlatFilter.hpp"
#include "beliefs/particle_filters/RejectionSampling.hpp"
class Action;
class BAPOMDP;
class Observation;
class POMDP;
class State;

namespace beliefs {

/**
 * @brief <class description>
 **/
class BARejectionSampling : public ::beliefs::BABelief
{

public:
    explicit BARejectionSampling(size_t n);

    /**** Belief interface ****/
    void initiate(POMDP const& simulator) final;
    void free(POMDP const& simulator) final;
    State const* sample() const final;
    void updateEstimation(Action const* a, Observation const* o, POMDP const& d) final;

    /*** implement BABelief ***/
    void resetDomainStateDistribution(BAPOMDP const& bapomdp) final;

private:
    // number of particles
    size_t const _n;

    FlatFilter<State const*> _filter = {};
};

} // namespace beliefs

#endif // BAREJECTIONSAMPLING_HPP
