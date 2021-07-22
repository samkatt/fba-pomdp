#ifndef BAPOMDP_HPP
#define BAPOMDP_HPP

#include "domains/POMDP.hpp"

#include <memory>
#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/table/BADomainExtension.hpp"
#include "bayes-adaptive/priors/BAPrior.hpp"
#include "bayes-adaptive/states/BAState.hpp"
#include "environment/Observation.hpp"
#include "environment/Terminal.hpp"
#include "utils/DiscreteSpace.hpp"
#include "utils/random.hpp"
class Action;
class Reward;
class State;
namespace beliefs { namespace bayes_adaptive {
class NestedBelief;
}} // namespace beliefs::bayes_adaptive
namespace configurations {
struct BAConf;
}

/**
 * @brief The Bayes Adaptive POMDP
 **/
class BAPOMDP : public POMDP
{
public:
    enum StepType { UpdateCounts, KeepCounts };
    enum SampleType { Abstract, Normal};

    BAPOMDP(
        std::unique_ptr<POMDP> domain,
        std::unique_ptr<BADomainExtension> ba_domain_ext,
        std::unique_ptr<BAPrior> prior,
        rnd::sample::Dir::sampleMethod sample_method,
        rnd::sample::Dir::sampleMultinominal compute_mult_method,
        bool update_abstract_model);

    StepType mode() const;
    void mode(StepType new_mode) const;

    Domain_Size const* domainSize() const;

    /**
     * @brief returns an initial domain state
     **/
    State const* sampleDomainState() const;

    /**
     * @brief returns domain state of index i
     **/
    State const* domainState(int i) const;

    /**
     * @brief returns a copy of the provided domain_state
     **/
    State const* copyDomainState(State const* domain_state) const;

    /**
     * @brief releases the domain state
     **/
    void releaseDomainState(State const* s) const;

    /**
     * @brief updates the domain state to an initial config
     *
     * NOTE: retains counts in ba_s
     **/
    void resetDomainState(BAState const* s) const;

    /**
     * @brief performs a step of type t- which can either update counts or not
     *
     * This step will update the state, return an observation & reward
     * according to the dynamics describes by the counts in the state.
     *
     * Most importantly, however, this step may or may not update the
     * counts associated with the sampled transition, according to
     * its last parameter
     **/
    Terminal
        step(State const** s, Action const* a, Observation const** o, Reward* r, StepType step_type, SampleType sample_type)
            const;

    Terminal
        step(State const** s, Action const* a, Observation const** o, Reward* r, StepType step_type)
            const;

    Terminal
        step(State const** s, Action const* a, Observation const** o, Reward* r, SampleType sample_type)
            const;

    /**** POMDP interface *****/
    Action const* generateRandomAction(State const* s) const final;
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;

    double computeObservationProbability(Observation const* o, Action const* a, State const* s)
        const final;

    /**** environment interface ****/
    State const* sampleStartState() const final;
    Terminal
        step(State const** s, Action const* a, Observation const** o, Reward* r) const override;

    void releaseAction(Action const* a) const final;
    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;

    Action const* copyAction(Action const* a) const final;
    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;

protected:
    friend class beliefs::bayes_adaptive::NestedBelief;

    // provides BAPOMDP with domain knowledge
    std::unique_ptr<POMDP const> _domain;

    // provides bayes-adaptive domain functionality
    std::unique_ptr<BADomainExtension const> _ba_domain_ext;

    // sets the prior on the initial states
    std::unique_ptr<BAPrior const> _ba_prior;

private:
    // whether we're updating counts during steps
    mutable StepType _mode = UpdateCounts;

    mutable SampleType _sample = Normal;

    // store observations locally for performance boost
    utils::DiscreteSpace<IndexObservation> _observations;

    // size of the underlying domain (S,A,O)
    Domain_Size const _domain_size;

    // whether to use sampled or expected multinominal models when stepping
    rnd::sample::Dir::sampleMethod* _sample_method;

    // whether to use sampled or expected mult models when computing observation
    rnd::sample::Dir::sampleMultinominal* _compute_mult_method;

    bool _update_abstract_model;
};

namespace factory {

/**
 * @brief Constructor for a (flat, tabular) BA-POMDP instance according to the configurations
 *
 * @param c the configurations
 *
 * @return a flat, tabular BAPOMDP unique pointer
 */
std::unique_ptr<BAPOMDP> makeTBAPOMDP(configurations::BAConf const& c);

} // namespace factory

#endif // BAPOMDP_HPP
