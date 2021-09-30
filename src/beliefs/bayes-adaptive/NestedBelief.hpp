#ifndef NESTEDBELIEF_HPP
#define NESTEDBELIEF_HPP

#include "beliefs/bayes-adaptive/BABelief.hpp"

#include "beliefs/particle_filters/FlatFilter.hpp"
#include "beliefs/particle_filters/WeightedFilter.hpp"

#include "bayes-adaptive/states/BAState.hpp"

namespace beliefs { namespace bayes_adaptive {

/**
 * @brief A bayes-adaptive belief with P(counts) * P(s|counts)
 *
 * Contains a particle set of particles. The 2nd layer consists
 * of domain states, which are conditioned on the counts. The upper
 * weighted filter, contains counts, which updated by taking the
 * averaged transitions of its underlying particle filter.
 **/
class NestedBelief : public BABelief
{
public:
    NestedBelief(size_t top_filter_size, size_t bottom_filter_size);

    /**** BABelief interface ****/
    void resetDomainStateDistribution(BAPOMDP const& bapomdp) final;
    void resetDomainStateDistributionAndAddAbstraction(const BAPOMDP &bapomdp, Abstraction &abstraction, int i) final;

    /**** Belief interface ****/
    void initiate(POMDP const& domain) final;
    void free(POMDP const& domain) final;
    State const* sample() const final;

    void updateEstimation(Action const* a, Observation const* o, POMDP const& d) final;

private:
    // nested belief
    WeightedFilter<std::pair<BAState const*, FlatFilter<State const*>>> _filter = {};

    size_t _top_filter_size, _bottom_filter_size;

    void assertValidBelief() const;
};

}} // namespace beliefs::bayes_adaptive

#endif // NESTEDBELIEF_HPP
