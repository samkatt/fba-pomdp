#ifndef BAIMPORTANCESAMPLER_HPP
#define BAIMPORTANCESAMPLER_HPP

#include "beliefs/bayes-adaptive/BABelief.hpp"
#include "beliefs/particle_filters/ImportanceSampler.hpp"

#include "beliefs/particle_filters/WeightedFilter.hpp"

class BAPOMDP;
class POMDP;
class State;
class Observation;

namespace beliefs {

/**
 * @brief <class description>
 **/
class BAImportanceSampling : public ::beliefs::BABelief
{

public:
    /**
     * @brief initiates importance sampling of n samples with an empty filter
     **/
    explicit BAImportanceSampling(size_t n);

    /**
     * @brief initiates importance sampling of n samples with provided filter
     *
     * Assumes the filter size is not larger than provided n
     *
     **/
    BAImportanceSampling(WeightedFilter<State const*> f, size_t n);

    /***** BABelief interface *****/
    void resetDomainStateDistribution(BAPOMDP const& bapomdp) final;

    /***** Belief interface *****/
    void initiate(POMDP const& d) final;
    void free(POMDP const& d) final;
    State const* sample() const final;
    void updateEstimation(Action const* a, Observation const* o, POMDP const& d) final;

private:
    WeightedFilter<State const*> _filter = {};

    // number of particles
    size_t _n;
};

} // namespace beliefs

#endif // BAIMPORTANCESAMPLER_HPP
