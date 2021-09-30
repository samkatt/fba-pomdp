#ifndef BAIMPORTANCESAMPLER_HPP
#define BAIMPORTANCESAMPLER_HPP

#include <bayes-adaptive/abstractions/Abstraction.hpp>
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
    explicit BAImportanceSampling(size_t n, bool abstraction, bool remake_abstraction, bool update_abstraction, bool update_abstract_model_normalized);

    /**
     * @brief initiates importance sampling of n samples with provided filter
     *
     * Assumes the filter size is not larger than provided n
     *
     **/
    BAImportanceSampling(WeightedFilter<State const*> f, size_t n, bool abstraction, bool remake_abstraction, bool update_abstraction, bool update_abstract_model_normalized);

    /***** BABelief interface *****/
    void resetDomainStateDistribution(const BAPOMDP &bapomdp) final;
    void resetDomainStateDistributionAndAddAbstraction(const BAPOMDP &bapomdp, Abstraction &abstraction, int i) final;

    /***** Belief interface *****/
    void initiate(POMDP const& d) final;
    void free(POMDP const& d) final;
    State const* sample() const final;
    void updateEstimation(Action const* a, Observation const* o, POMDP const& d) final;

private:
    WeightedFilter<State const*> _filter = {};

    // number of particles
    size_t _n;
    bool _abstraction;
    bool _remake_abstract_model;
    bool _update_abstract_model;
    bool _update_abstract_model_normalized;
};

} // namespace beliefs

#endif // BAIMPORTANCESAMPLER_HPP
