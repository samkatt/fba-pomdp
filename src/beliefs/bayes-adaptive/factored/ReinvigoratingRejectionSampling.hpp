#ifndef REINVIGORATINGREJECTIONSAMPLING_HPP
#define REINVIGORATINGREJECTIONSAMPLING_HPP

#include "beliefs/bayes-adaptive/BABelief.hpp"

#include <cstddef>

#include "bayes-adaptive/models/factored/FBAPOMDP.hpp"
#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "beliefs/particle_filters/FlatFilter.hpp"
class BAPOMDP;
class FBAPOMDPState;
class POMDP;

namespace beliefs { namespace bayes_adaptive { namespace factored {

/**
 * @brief breeds a new state based on the structure and counts of parents
 *
 * (1) takes structure of structure_state and applies 1 mutation
 * (2) generates CPTs by integrating out counts in counts_state according to structure in (1)
 * (3) returns state with structure from (1) with CPTs from (2)
 **/
FBAPOMDPState* breed(
    ::bayes_adaptive::factored::FBAPOMDP const& fbapomdp,
    FBAPOMDPState const* structure_state,
    FBAPOMDPState const* counts_state);

/**
 * @brief Performs rejection sampling with occasional reinvigoration of structures for the FBA-POMDP
 *
 * This approach towards FBA-POMDP state estimation is a particle filtering
 * approach using rejection sampling which keeps a fresh amount of structures.
 *
 * This is done by mutating structures in the belief. This is done
 * in three steps during each belief update:
 *
 * 1) sample a structure from the current belief
 * 2) mutate the sampled structure domain specifically
 * 3) sample counts for the fully connected belief from a separate maintained belief
 * 4) integrate out the sampled counts such that they correspond to the mutated structure
 * 5) add the new structure with the counts to the current belief
 *
 * The number of samples added this way is determined by the reinvigoration_amount parameter.
 **/
class ReinvigoratingRejectionSampling : public BABelief
{

public:
    /**
     * @brief sets up this belief tracking method intending to maintain <size> particles
     *
     * <reinvigoration_amount> of those samples will be sampled with fresh structures
     **/
    ReinvigoratingRejectionSampling(size_t size, size_t reinvigoration_amount);

    /***** BABelief interface *****/
    void resetDomainStateDistribution(BAPOMDP const& bapomdp) final;
    void resetDomainStateDistributionAndAddAbstraction(const BAPOMDP &bapomdp, Abstraction &abstraction, int i) final;

    /***** Belief interface *****/
    void initiate(POMDP const& domain) final;
    void free(POMDP const& domain) final;
    State const* sample() const final;
    void updateEstimation(Action const* a, Observation const* o, POMDP const& domain) final;

private:
    // number of desired particles
    size_t const _size, _reinvigoration_amount;

    FlatFilter<FBAPOMDPState const*> _belief = {}, _fully_connected_belief = {};

    /**
     * @brief reinvigorates the current _belief by combining its structures with counts over
     *compltee graph
     **/
    void reinvigorateParticles(POMDP const& domain);
};

}}} // namespace beliefs::bayes_adaptive::factored

#endif // REINVIGORATINGREJECTIONSAMPLING_HPP
