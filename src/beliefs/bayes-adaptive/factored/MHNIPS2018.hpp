#ifndef MHNIPS_HPP
#define MHNIPS_HPP

#include "beliefs/bayes-adaptive/BABelief.hpp"

#include <cstddef>
#include <vector>
#include <bayes-adaptive/abstractions/Abstraction.hpp>

#include "beliefs/particle_filters/WeightedFilter.hpp"
#include "environment/History.hpp"
class Action;
class BAPOMDP;
class FBAPOMDP;
class FBAPOMDPState;
class Observation;
class POMDP;
class State;

namespace beliefs { namespace bayes_adaptive { namespace factored {

/**
 * @brief A belief over FBAPOMDPState using MH
 **/
class MHNIPS2018 : public BABelief
{
public:
    MHNIPS2018(size_t size, double ll_threshold);

    /*** BABelief interface ***/
    void resetDomainStateDistribution(BAPOMDP const& bapomdp) final;
    void resetDomainStateDistributionAndAddAbstraction(const BAPOMDP &bapomdp, Abstraction &abstraction, int i) final;

    /*** Belief interface ***/
    void initiate(POMDP const& domain) final;
    void free(POMDP const& domain) final;
    State const* sample() const final;
    void updateEstimation(Action const* a, Observation const* o, POMDP const& domain) final;

private:
    // params
    size_t const _size;
    double const _ll_threshold;

    // internal state
    double _log_likelihood = 0;

    std::vector<History> _history                = {};
    WeightedFilter<FBAPOMDPState const*> _belief = {};

    /**
     * @brief actually performs MH
     **/
    void MH(POMDP const& domain);
};

}}} // namespace beliefs::bayes_adaptive::factored

#endif // MHNIPS_HPP
