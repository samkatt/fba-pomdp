#include "MHwithinGibbs.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <utility>
#include <bayes-adaptive/states/factored/AbstractFBAPOMDPState.hpp>
#include <boost/timer/timer.hpp>
#include <bayes-adaptive/abstractions/Abstraction.hpp>

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
    return s->index();
}
} // namespace

namespace beliefs { namespace bayes_adaptive { namespace factored {

namespace {

std::vector<IndexState> rejectionSampleStateHistory(
    ::bayes_adaptive::factored::BABNModel const& model,
    std::vector<History> const& history,
    ::bayes_adaptive::factored::FBAPOMDP const& fbapomdp,
    rnd::sample::Dir::sampleMethod sampleMethod)
{

    std::vector<IndexState> result;

    for (size_t e = 0; e < history.size(); ++e)
    {
        auto const ep = history[e];

        std::vector<IndexState> episode_state_sequence;
        episode_state_sequence.reserve(ep.length());

        // start state
        auto const init_state = fbapomdp.sampleDomainState();
        episode_state_sequence.emplace_back(IndexState(init_state->index()));
        fbapomdp.releaseDomainState(init_state);

        for (auto const& step : ep)
        {

            // attempt step
            auto new_s = IndexState(
                model.sampleStateIndex(&episode_state_sequence.back(), step.action, sampleMethod));

            auto const o =
                IndexObservation(std::to_string(model.sampleObservationIndex(step.action, &new_s, sampleMethod)));

            // reject episode if observation not correct
            if (o.index() != step.observation->index())
            {
                break;
            }

            // not reject, store state
            episode_state_sequence.emplace_back(new_s);
        }

        // if failed to sample a correct sequence, restart
        if (episode_state_sequence.size() != ep.length() + 1)
        {
            e--;
            continue;
        }

        // episode was successfull: save state sequence
        result.insert(
            result.end(),
            std::make_move_iterator(episode_state_sequence.begin()),
            std::make_move_iterator(episode_state_sequence.end()));
    }

    return result;
}

std::vector<IndexState> msgSampleStateHistory(
    ::bayes_adaptive::factored::BABNModel const& model,
    std::vector<History> const& history,
    ::bayes_adaptive::factored::FBAPOMDP const& fbapomdp)
{

    std::vector<IndexState> result;

    auto const s = fbapomdp.domainSize();

    auto const T = model.flattenT();
    auto const O = model.flattenO();
    auto const p = fbapomdp.domainStatePrior();

    // sample each episode separately
    for (auto const& episode : history)
    {

        // backward pass
        std::vector<std::vector<double>> message(
            episode.length() + 1, std::vector<double>(fbapomdp.domainSize()->_S));

        // last message: p( last_o | last_s )
        for (auto state = 0; state < s->_S; ++state)
        {
            message.back()[state] =
                O[std::stoi(episode.back().action->index())][state][std::stoi(episode.back().observation->index())];
        }

        double tot = 0;
        // message T-1 .. 0: p( o_t^last | s_t )
        for (int step = episode.length() - 1; step >= 0; --step)
        {

            auto const a = episode[step].action->index();

            tot = 0;
            // compute message for each state
            for (auto state = 0; state < s->_S; ++state)
            {

                // p(t_t+1^T | s_t)
                message[step][state] = std::inner_product(
                    T[state][std::stoi(a)].begin(), T[state][std::stoi(a)].end(), message[step + 1].begin(), 0.0);

                if (step != 0) // t != 0: multiply with probability of observation
                {

                    message[step][state] *= O[std::stoi(episode[step - 1].action->index())][state]
                                             [std::stoi(episode[step - 1].observation->index())];

                } else // p(s0) *= prior
                {

                    message[step][state] *= p->prob(state);
                }

                tot += message[step][state];
            }

            assert(tot > 0);

            // normalize
            std::transform(
                message[step].begin(),
                message[step].end(),
                message[step].begin(),
                std::bind(std::divides<double>(), std::placeholders::_1, tot));
        }

        // forward pass sample

        // s_0
        auto state = rnd::sample::Dir::sampleFromMult(message[0].data(), s->_S, 1);
        result.emplace_back(IndexState(std::to_string(state)));

        std::vector<double> probs(s->_S);
        // s=1..T
        for (size_t step = 0; step < episode.length(); ++step)
        {

            tot = 0;
            for (auto new_state = 0; new_state < s->_S; ++new_state)
            {

                probs[new_state] = T[state][std::stoi(episode[step].action->index())][new_state]
                                   * message[step + 1][new_state];

                tot += probs[new_state];
            }

            state = rnd::sample::Dir::sampleFromMult(probs.data(), s->_S, tot);
            result.emplace_back(IndexState(std::to_string(state)));
        }
    }

    if (VLOG_IS_ON(5))
    {

        VLOG(5) << "Conditional state sequence sampled:";

        auto state_counter = 0;
        for (auto const& ep : history)
        {

            VLOG(5) << "s0: " << result[state_counter++].toString();

            for (auto const& step : ep)
            {
                VLOG(5) << "a: " << step.action->toString()
                        << ", o: " << step.observation->toString()
                        << ", s: " << result[state_counter++].toString();
            }
        }
    }

    return result;
}

std::vector<IndexState> sampleStateHistory(
    ::bayes_adaptive::factored::BABNModel const& model,
    std::vector<History> const& history,
    ::bayes_adaptive::factored::FBAPOMDP const& fbapomdp,
    MHwithinGibbs::SAMPLE_STATE_HISTORY_TYPE type)
{
    if (type == MHwithinGibbs::RS)
    {
        return rejectionSampleStateHistory(
            model, history, fbapomdp, rnd::sample::Dir::sampleFromExpectedMult);
    }
    if (type == MHwithinGibbs::MSG)
    {
        return msgSampleStateHistory(model, history, fbapomdp);
    }

    throw "this could should not be reached, illegal type";
}

} // namespace

MHwithinGibbs::MHwithinGibbs(
    size_t size,
    double ll_threshold,
    bool abstraction,
    SAMPLE_STATE_HISTORY_TYPE state_history_sample_type) :
        _size(size),
        _ll_threshold(ll_threshold),
        _abstraction(abstraction),
        _state_history_sample_type(state_history_sample_type)
{

    if (_size < 1)
    {
        throw "MHwithinGibbs::cannot initiate MH with size 0";
    }

    if (_ll_threshold >= 0)
    {
        throw "MHwithinGibbs::cannot initiate with threshold >= 0 (is:"
            + std::to_string(_ll_threshold) + ")";
    }

    VLOG(1) << "Initiated MH belief tracking of " << _size
            << " particles and log likelihood threshold " << _ll_threshold;
}

void MHwithinGibbs::resetDomainStateDistribution(BAPOMDP const& bapomdp)
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

void MHwithinGibbs::initiate(POMDP const& domain)
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

void MHwithinGibbs::free(POMDP const& domain)
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

State const* MHwithinGibbs::sample() const
{
    return _belief.sample();
}

void MHwithinGibbs::updateEstimation(Action const* a, Observation const* o, POMDP const& domain)
{

    _log_likelihood += log(::beliefs::importance_sampling::update(_belief, a, o, domain));

    beliefs::importance_sampling::resample(_belief, domain, _size);

    _history.back().add(domain.copyAction(a), domain.copyObservation(o));

    if (_log_likelihood < _ll_threshold)
    {
        reinvigorate(domain);
    }

    VLOG(3) << "Performed belief update, new log likelihood is: " << _log_likelihood;
    VLOG(4) << "belief is\n" << _belief.toString(printStateIndex);
}

void MHwithinGibbs::reinvigorate(POMDP const& domain)
{
    VLOG(1) << "Initiating reinvigoration after ll dropped to " << _log_likelihood;
    auto old_belief      = std::move(_belief);
    auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(domain);

    _belief = WeightedFilter<FBAPOMDPState const*>();

    // init first complete sample
    auto model          = *old_belief.sample()->model();
    auto state_sequence = ::beliefs::bayes_adaptive::factored::sampleStateHistory(
        model, _history, fbapomdp, _state_history_sample_type);

    auto prior_model = fbapomdp.prior()->computePriorModel(model.structure());

    model = computePosteriorCounts(prior_model, _history, state_sequence);

    auto score = model.LogBDScore(prior_model);

    auto i = 0;
    auto numAccepted = 0;
    // gibbs loob
    while (_belief.size() < _size)
    {
        // 2: mh within gibbs to sample p(model | states)
        prior_model = fbapomdp.prior()->computePriorModel(fbapomdp.mutate(model.structure()));

        auto new_model = computePosteriorCounts(prior_model, _history, state_sequence);

        // apply MH sampling strategy
        auto new_score = new_model.LogBDScore(prior_model);

        if (log(rnd::uniform_rand01()) < (new_score - score))
        {
            if (_abstraction) {
                _belief.add(
                new AbstractFBAPOMDPState(
                        fbapomdp.domainState(state_sequence.back().index()), std::move(new_model)),
                1 / static_cast<double>(_size));
            } else {
                _belief.add(
                new FBAPOMDPState(
                    fbapomdp.domainState(state_sequence.back().index()), std::move(new_model)),
                        1 / static_cast<double>(_size));
            }

            // gibbs sample 1: sample p(states | struct)
            state_sequence = ::beliefs::bayes_adaptive::factored::sampleStateHistory(
                model, _history, fbapomdp, _state_history_sample_type);

            // setup for next iteration
            model = computePosteriorCounts(prior_model, _history, state_sequence);

            score = model.LogBDScore(prior_model);
            numAccepted++;
            VLOG(5) << "Sample " << i << " accepted";
            VLOG(1) << "Reinvigoration accepted: " << numAccepted;
        } else
        {
            VLOG(5) << "Sample " << i << " rejected";
        }
        i++;
    }

    old_belief.free([&domain](State const* s) { domain.releaseState(s); });
    _log_likelihood = 0;
}

::bayes_adaptive::factored::BABNModel MHwithinGibbs::computePosteriorCounts(
    ::bayes_adaptive::factored::BABNModel const& prior,
    std::vector<History> const& history,
    std::vector<IndexState> const& state_history) const
{
    assert(!history.empty());
    assert(!state_history.empty());

    // result starts with counts of prior
    auto result = prior;

    unsigned int state_index = 0;

    // increments counts in result for every
    // transition in history and state_history
    for (auto const& ep : history)
    {

        assert(ep.length() != 0);

        for (auto const& step : ep)
        {

            auto const a = step.action;
            auto const o = step.observation;

            auto const s     = state_history[state_index++];
            auto const new_s = state_history[state_index];

            result.incrementCountsOf(&s, a, o, &new_s);
        }

        // start next episode with initial state
        state_index++;
    }

    assert(state_index == state_history.size());

    return result;
}

void MHwithinGibbs::resetDomainStateDistributionAndAddAbstraction(const BAPOMDP &bapomdp,
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
