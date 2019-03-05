#ifndef MHWITHINGIBBS_HPP
#define MHWITHINGIBBS_HPP

#include "beliefs/bayes-adaptive/BABelief.hpp"

#include <cstddef>
#include <vector>

#include "beliefs/particle_filters/WeightedFilter.hpp"
#include "environment/History.hpp"
#include "environment/State.hpp"

class POMDP;
class FBAPOMDP;
class BAPOMDP;
class Action;
class Observation;
class FBAPOMDPState;

namespace bayes_adaptive { namespace factored {
class BABNModel;
}} // namespace bayes_adaptive::factored

namespace beliefs { namespace bayes_adaptive { namespace factored {

/**
 * @brief A belief over FBAPOMDPState using MH
 **/
class MHwithinGibbs : public BABelief
{
public:
    enum SAMPLE_STATE_HISTORY_TYPE { RS, MSG };

    MHwithinGibbs(
        size_t size,
        double ll_threshold,
        SAMPLE_STATE_HISTORY_TYPE state_history_sample_type);

    /*** BABelief interface ***/
    void resetDomainStateDistribution(BAPOMDP const& bapomdp) final;

    /*** Belief interface ***/
    void initiate(POMDP const& domain) final;
    void free(POMDP const& domain) final;
    State const* sample() const final;
    void updateEstimation(Action const* a, Observation const* o, POMDP const& domain) final;

private:
    // params
    size_t const _size;
    double const _ll_threshold;

    SAMPLE_STATE_HISTORY_TYPE const _state_history_sample_type;

    // internal state
    double _log_likelihood = 0;

    std::vector<History> _history                = {};
    WeightedFilter<FBAPOMDPState const*> _belief = {};

    /**
     * @brief reinvigorates belief using MH-within-gibbs
     **/
    void reinvigorate(POMDP const& domain);

    /**
     * @brief computes the BABNModel of going through the history given prior
     **/
    ::bayes_adaptive::factored::BABNModel computePosteriorCounts(
        ::bayes_adaptive::factored::BABNModel const& prior,
        std::vector<History> const& history,
        std::vector<IndexState> const& state_history) const;

    /**
     * @brief returns a sampled state sequence given history and model
     **/
    std::vector<IndexState> sampleStateHistory(
        ::bayes_adaptive::factored::BABNModel const& model,
        std::vector<History> const& history,
        BAPOMDP const& bapomdp) const;
};

}}} // namespace beliefs::bayes_adaptive::factored

#endif // MHWITHINGIBBS_HPP
