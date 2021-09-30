#include "MHNIPS2018.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include "easylogging++.h"

#include "bayes-adaptive/models/factored/FBAPOMDP.hpp"
#include "bayes-adaptive/models/table/BAPOMDP.hpp"

#include "bayes-adaptive/states/factored/BABNModel.hpp"
#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"

#include "beliefs/particle_filters/ImportanceSampler.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "utils/random.hpp"

namespace {
std::string printStateIndex(FBAPOMDPState const* s)
{
    return std::to_string(s->index());
}
} // namespace

namespace beliefs { namespace bayes_adaptive { namespace factored {

namespace {

/**
 * @brief updates a given model w.r.t. history
 *
 * Returns the state index of the last step
 **/
int computePosterior(
    ::bayes_adaptive::factored::BABNModel* model,
    ::bayes_adaptive::factored::FBAPOMDP const& fbapomdp,
    std::vector<History> const& history)
{
    auto sampleMethod = rnd::sample::Dir::sampleFromExpectedMult;

    std::vector<std::pair<int, int>> state_transitions;

    auto s      = IndexState(0);
    auto temp_a = IndexAction(0);
    auto new_s  = IndexState(0);
    auto o      = IndexObservation(0);

    // go through whole history
    for (size_t episode = 0; episode < history.size(); ++episode)
    {

        // initiate start state
        auto sampled_state = fbapomdp.sampleDomainState();
        s.index(sampled_state->index());
        fbapomdp.releaseDomainState(sampled_state);

        // run episode
        auto const& steps = history[episode];
        for (auto const& step : steps)
        {
            // update step
            new_s.index(model->sampleStateIndex(&s, step.action, sampleMethod));

            o.index(model->sampleObservationIndex(step.action, &new_s, sampleMethod));

            // reject episode if observation not correct
            if (o.index() != step.observation->index())
            {
                break;
            }

            model->incrementCountsOf(&s, step.action, &o, &new_s);

            // register applied transitions
            state_transitions.emplace_back(std::pair<int, int>(s.index(), new_s.index()));
            s.index(new_s.index());
        }

        // failed episode: undo updates and redo episode
        if (state_transitions.size() != steps.length())
        {

            for (size_t t = 0; t < state_transitions.size(); ++t)
            {
                auto const& trans = state_transitions[t];

                s.index(trans.first);
                new_s.index(trans.second);

                model->incrementCountsOf(&s, steps[t].action, steps[t].observation, &new_s, -1);
            }

            // make sure we redo the episode
            episode--;
        }

        state_transitions.clear();
    }

    // return index of the last (current) state
    return new_s.index();
}

} // namespace

MHNIPS2018::MHNIPS2018(size_t size, double ll_threshold) : _size(size), _ll_threshold(ll_threshold)
{

    if (_size < 1)
    {
        throw("MHNIPS2018::cannot initiate MH with size " + std::to_string(size));
    }

    if (_ll_threshold >= 0)
    {
        throw(
            "MHNIPS2018::cannot initiate with threshold >= 0 (is:" + std::to_string(_ll_threshold)
            + ")");
    }

    VLOG(1) << "Initiated MH belief tracking of " << _size
            << " particles and log likelihood threshold " << _ll_threshold;
}

void MHNIPS2018::resetDomainStateDistribution(BAPOMDP const& bapomdp)
{
    assert(_belief.size() == _size);

    auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(bapomdp);

    for (size_t i = 0; i < _size; ++i) { fbapomdp.resetDomainState(_belief.particle(i)->particle); }

    VLOG(4) << "Reset domain state, current state belief:\n" << _belief.toString(printStateIndex);

    if (_history.back().length() != 0)
    {
        _history.emplace_back();
    }
}

void MHNIPS2018::initiate(POMDP const& domain)
{
    assert(_belief.empty());
    assert(_history.empty());

    auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(domain);

    for (size_t i = 0; i < _size; ++i)
    {
        _belief.add(
            static_cast<FBAPOMDPState const*>(fbapomdp.sampleStartState()),
            1.0 / static_cast<double>(_size));
    }

    VLOG(4) << "initiated with belief:\n" << _belief.toString(printStateIndex);

    _history.emplace_back();
}

void MHNIPS2018::free(POMDP const& domain)
{
    _belief.free([&domain](State const* s) { domain.releaseState(s); });

    for (auto& ep : _history)
    {
        for (auto& step : ep)
        {
            domain.releaseAction(step.action);
            domain.releaseObservation(step.observation);
        }
    }

    _history.clear();
}

State const* MHNIPS2018::sample() const
{
    return _belief.sample();
}

void MHNIPS2018::updateEstimation(Action const* a, Observation const* o, POMDP const& domain)
{

    _log_likelihood += log(::beliefs::importance_sampling::update(_belief, a, o, domain));

    beliefs::importance_sampling::resample(_belief, domain, _size);

    _history.back().add(domain.copyAction(a), domain.copyObservation(o));

    if (_log_likelihood < _ll_threshold)
    {
        MH(domain);
    }

    VLOG(3) << "Performed belief update, new log likelihood is: " << _log_likelihood;
    VLOG(4) << "belief is\n" << _belief.toString(printStateIndex);
}

void MHNIPS2018::MH(POMDP const& domain)
{
    VLOG(3) << "Initiating MH after ll dropped to " << _log_likelihood;

    auto old_belief      = std::move(_belief);
    auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(domain);

    _belief = WeightedFilter<FBAPOMDPState const*>();

    auto i = 0;
    while (_belief.size() < _size)
    {

        // sample graph from belief and find its prior
        auto const sampled_model        = old_belief.sample()->model();
        auto sampled_structure          = sampled_model->structure();
        auto const& sampled_prior_model = fbapomdp.prior()->computePriorModel(sampled_structure);

        // sample new structure and prior model
        // propose same structure 50% of the time
        auto new_structure = rnd::boolean() ? //
                                 sampled_structure
                                            : fbapomdp.mutate(std::move(sampled_structure));

        auto new_prior_model = fbapomdp.prior()->computePriorModel(new_structure);

        // find its posterior
        auto new_model = new_prior_model;
        auto s_index   = computePosterior(&new_model, fbapomdp, _history);

        // apply MH sampling strategy
        auto old_score = sampled_model->LogBDScore(sampled_prior_model);
        auto new_score = new_model.LogBDScore(new_prior_model);

        if (log(rnd::uniform_rand01()) < (new_score - old_score))
        {
            _belief.add(
                new FBAPOMDPState(fbapomdp.domainState(s_index), std::move(new_model)),
                1 / static_cast<double>(_size));

            VLOG(5) << "Sample " << i << " accepted";
        } else
            VLOG(5) << "Sample " << i << " rejected";

        i++;
    }

    old_belief.free([&domain](State const* s) { domain.releaseState(s); });
    _log_likelihood = 0;
}

            void MHNIPS2018::resetDomainStateDistributionAndAddAbstraction(const BAPOMDP &bapomdp,
                                                                           Abstraction &abstraction, int k) {
                assert(_belief.size() == _size);

                auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(bapomdp);

                for (size_t i = 0; i < _size; ++i) { fbapomdp.resetDomainState(_belief.particle(i)->particle); }

                VLOG(4) << "Reset domain state, current state belief:\n" << _belief.toString(printStateIndex);

                if (_history.back().length() != 0)
                {
                    _history.emplace_back();
                }
                VLOG( 3) << abstraction.printSomething();
                VLOG( 3) << k;

            }

        }}} // namespace beliefs::bayes_adaptive::factored
