#ifndef STRUCTUREINCUBATORSAMPLING_HPP
#define STRUCTUREINCUBATORSAMPLING_HPP

#include <bayes-adaptive/abstractions/Abstraction.hpp>
#include "beliefs/bayes-adaptive/BABelief.hpp"

#include "bayes-adaptive/models/factored/FBAPOMDP.hpp"
#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"

#include "beliefs/particle_filters/FlatFilter.hpp"
#include "beliefs/particle_filters/WeightedFilter.hpp"

class BAPOMDP;
class POMDP;
class Action;
class Observation;

namespace beliefs { namespace bayes_adaptive { namespace factored {

/**
 * @brief Structure Reinvigoration of particlse that reach some weight
 *
 * Applies reinvigoration sampling by maintaining a second,
 * incubator belief. This belief directly receives reinvigorated
 * particles from combining structures in the true belief with
 * counts over the fully connected belief. When particles in the
 * incubator belief reach weights, then they are added to the
 * actual belief.
 *
 * There are two hand-set parameters in this class, one which
 * describes how many reinvigorated particles are added to the
 * incubator belief each step (10/size), and one which sets
 * the threshold of adding particles from the incubator to the
 * real belief.
 **/
class StructureIncubatorSampling : public BABelief
{
public:
    StructureIncubatorSampling(size_t size, size_t reinvigor_amount, double threshold);

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
    size_t const _size;
    size_t const _shadow_reinvigor_amount;
    double const _real_reinvigor_threshold;

    FlatFilter<FBAPOMDPState const*> _belief = {}, _fully_connected_belief = {};

    // belief that contains mutated/reinvigorated particles
    WeightedFilter<FBAPOMDPState const*> _shadow_belief = {};

    /**
     * @brief reinvigorates the shadow belief with mutations
     **/
    void reinvigorateShadowBelief(::bayes_adaptive::factored::FBAPOMDP const& fbapomdp);

    /**
     * @brief reinvigorates the belief with good particles from the shadow belief
     **/
    void reinvigorateBelief(::bayes_adaptive::factored::FBAPOMDP const& fbapomdp);
};

}}} // namespace beliefs::bayes_adaptive::factored

#endif // STRUCTUREINCUBATORSAMPLING_HPP
