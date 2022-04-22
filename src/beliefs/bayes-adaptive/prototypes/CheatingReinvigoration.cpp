#include "CheatingReinvigoration.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "easylogging++.h"

#include "bayes-adaptive/models/factored/FBAPOMDP.hpp"
#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"

#include "beliefs/particle_filters/ImportanceSampler.hpp"
#include "beliefs/particle_filters/RejectionSampling.hpp"

#include "utils/random.hpp"

namespace {
std::string printStateIndex(FBAPOMDPState const* s)
{
    return std::to_string(s->index());
}
} // namespace

namespace beliefs { namespace bayes_adaptive { namespace prototypes {

CheatingReinvigoration::CheatingReinvigoration(
    size_t size,
    size_t cheat_amount,
    double resample_threshold) :
        _size(size), _cheat_amount(cheat_amount), _resample_threshold(resample_threshold)
{

    if (_size < 1 || _cheat_amount < 1)
    {
        throw "CheatingReinvigoration::cannot initiate belief of size < 1 (" + std::to_string(_size)
            + "), or resample size of < 1 (" + std::to_string(_cheat_amount) + ")";
    }

    if (_resample_threshold >= 0)
    {
        throw "CheatingReinvigoration::cannot initiate with resample_threshold >= 0 (is:"
            + std::to_string(_resample_threshold) + ")";
    }

    VLOG(1) << "Initiated cheating reinvigoration belief of size " << _size
            << " with cheating amount " << _cheat_amount << " and resampling threshold "
            << _resample_threshold;
}

void CheatingReinvigoration::resetDomainStateDistribution(BAPOMDP const& bapomdp)
{
    assert(_belief.size() == _size);
    assert(_correct_structured_belief.size() == _size);

    auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(bapomdp);

    for (auto& s : _correct_structured_belief.particles()) { fbapomdp.resetDomainState(s); }

    for (size_t i = 0; i < _size; ++i) { fbapomdp.resetDomainState(_belief.particle(i)->particle); }

    VLOG(3) << "Status of cheating filter after resetting domain states:\n"
            << _correct_structured_belief.toString()
            << "Status of actual belief after resetting domain states:\n"
            << _belief.toString(printStateIndex);
}

void CheatingReinvigoration::initiate(POMDP const& domain)
{
    assert(_belief.empty());
    assert(_correct_structured_belief.empty());

    auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(domain);

    std::vector<FBAPOMDPState const*> start_states;
    for (size_t i = 0; i < _size; ++i)
    {
        start_states.emplace_back(fbapomdp.sampleCorrectGraphState());
        ;
    }
    _correct_structured_belief = FlatFilter<FBAPOMDPState const*>(std::move(start_states));

    for (size_t i = 0; i < _size; ++i)
    {
        _belief.add(
            static_cast<FBAPOMDPState const*>(fbapomdp.sampleStartState()),
            1.0 / static_cast<double>(_size));
    }

    _likelihood = 1;

    VLOG(3) << "Status of cheating filter after initiation\n"
            << _correct_structured_belief.toString() << "Status of actual belief after initiation\n"
            << _belief.toString(printStateIndex);
}

void CheatingReinvigoration::free(POMDP const& domain)
{
    _belief.free([&domain](State const* s) { domain.releaseState(s); });
    _correct_structured_belief.free([&domain](State const* s) { domain.releaseState(s); });
}

State const* CheatingReinvigoration::sample() const
{
    return _belief.sample();
}

void CheatingReinvigoration::updateEstimation(
    Action const* a,
    Observation const* o,
    POMDP const& domain)
{

    beliefs::rejectSample(a, o, domain, _size, _correct_structured_belief);

    auto l = beliefs::importance_sampling::update(_belief, a, o, domain);

    beliefs::importance_sampling::resample(_belief, domain, _size);

    _likelihood *= l;

    if (log(_likelihood) < _resample_threshold)
    {

        VLOG(3) << "Cheating after log likelihood gotten to " << log(_likelihood);
        cheat(domain);
        _likelihood = 1;
    }

    VLOG(3) << "log likelihood after update: " << log(_likelihood)
            << "\nStatus of cheating filter after update\n"
            << _correct_structured_belief.toString() << "Status of actual belief after update\n"
            << _belief.toString(printStateIndex);
}

void CheatingReinvigoration::cheat(POMDP const& pomdp)
{

    // randomly replace particles in the belief with those in the correctly structured
    for (size_t i = 0; i < _cheat_amount; ++i)
    {
        _belief.replace(
            rnd::slowRandomInt(0, static_cast<int>(_size)),
            static_cast<FBAPOMDPState const*>(pomdp.copyState(_correct_structured_belief.sample())),
            [&pomdp](State const* s) { pomdp.releaseState(s); });
    }
}

}}} // namespace beliefs::bayes_adaptive::prototypes
