#include "StructureIncubatorSampling.hpp"

#include "easylogging++.h"

#include "beliefs/bayes-adaptive/factored/ReinvigoratingRejectionSampling.hpp"
#include "beliefs/particle_filters/ImportanceSampler.hpp"
#include "beliefs/particle_filters/RejectionSampling.hpp"

#include "bayes-adaptive/models/table/BAPOMDP.hpp"

#include "domains/POMDP.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"

using FBAPOMDP = ::bayes_adaptive::factored::FBAPOMDP;

namespace beliefs { namespace bayes_adaptive { namespace factored {

StructureIncubatorSampling::StructureIncubatorSampling(
    size_t size,
    size_t reinvigor_amount,
    double threshold) :
        _size(size),
        _shadow_reinvigor_amount(reinvigor_amount),
        _real_reinvigor_threshold(threshold)
{

    if (size < 1 || _shadow_reinvigor_amount < 1)
    {
        throw "StructureIncubatorSampling::Cannot initiate Incubator belief update with size < 1 ("
            + std::to_string(_size) + ") or resample size < 1 ("
            + std::to_string(_shadow_reinvigor_amount) + ")";
    }

    if (_real_reinvigor_threshold <= 0 || _real_reinvigor_threshold > 1)
    {
        throw "StructureIncubatorSampling::must initiate with 1 < threshold <= 0 (is:"
            + std::to_string(_real_reinvigor_threshold) + ")";
    }

    VLOG(1) << "Initiated StructureIncubatorSampling of size " << size << " with "
            << _shadow_reinvigor_amount << " shadow reinvigorations " << _real_reinvigor_threshold
            << " threshold for adding particles to the belief";
}

/***** BABelief interface *****/
void StructureIncubatorSampling::resetDomainStateDistribution(BAPOMDP const& bapomdp)
{
    assert(_belief.size() == _size);
    assert(_fully_connected_belief.size() == _size);
    assert(_shadow_belief.size() == _size);

    for (auto& p : _belief.particles()) { bapomdp.resetDomainState(p); }

    for (auto& p : _fully_connected_belief.particles()) { bapomdp.resetDomainState(p); }

    for (size_t i = 0; i < _size; ++i)
    { bapomdp.resetDomainState(_shadow_belief.particle(i)->particle); }
}

/***** Belief interface *****/
void StructureIncubatorSampling::initiate(POMDP const& domain)
{
    std::vector<FBAPOMDPState const*> tmp_states;
    tmp_states.reserve(_size);

    for (size_t i = 0; i < _size; ++i)
    { tmp_states.emplace_back(dynamic_cast<FBAPOMDPState const*>(domain.sampleStartState())); }

    _belief = FlatFilter<FBAPOMDPState const*>(tmp_states);

    tmp_states.clear();
    auto& fbapomdp = dynamic_cast<FBAPOMDP const&>(domain);
    for (size_t i = 0; i < _size; ++i)
    { tmp_states.emplace_back(fbapomdp.sampleFullyConnectedState()); }
    _fully_connected_belief = FlatFilter<FBAPOMDPState const*>(std::move(tmp_states));

    // start shadow belief with mutations
    for (size_t i = 0; i < _size; ++i)
    {
        _shadow_belief.add(
            breed(fbapomdp, _belief.sample(), _fully_connected_belief.sample()),
            1.0 / static_cast<double>(_size));
    }
}

void StructureIncubatorSampling::free(POMDP const& domain)
{
    _belief.free([&domain](State const* s) { domain.releaseState(s); });
    _fully_connected_belief.free([&domain](State const* s) { domain.releaseState(s); });
    _shadow_belief.free([&domain](State const* s) { domain.releaseState(s); });
}

State const* StructureIncubatorSampling::sample() const
{
    return _belief.sample();
}

void StructureIncubatorSampling::updateEstimation(
    Action const* a,
    Observation const* o,
    POMDP const& domain)
{
    assert(_belief.size() == _size);
    assert(_fully_connected_belief.size() == _size);
    assert(_shadow_belief.size() == _size);

    auto const& fbapomdp = dynamic_cast<FBAPOMDP const&>(domain);

    VLOG(3) << "Attempt to reinvigorate belief with particles from incubator";
    reinvigorateBelief(fbapomdp);
    VLOG(3) << "Reinvigorate incubator";
    reinvigorateShadowBelief(fbapomdp);

    VLOG(3) << "Performing Rejection Sampling on belief";
    ::beliefs::rejectSample(a, o, domain, _size, _belief);
    VLOG(3) << "Performing Rejection Sampling on fully connected belief";
    ::beliefs::rejectSample(a, o, domain, _size, _fully_connected_belief);
    VLOG(3) << "Performing Importance Sampling on shadow belief";
    ::beliefs::importance_sampling::update(_shadow_belief, a, o, domain);
    ::beliefs::importance_sampling::resample(_shadow_belief, domain, _size);

    assert(_belief.size() == _size);
    assert(_fully_connected_belief.size() == _size);
    assert(_shadow_belief.size() == _size);
}

void StructureIncubatorSampling::reinvigorateShadowBelief(FBAPOMDP const& fbapomdp)
{
    auto least_likely_n = _shadow_belief.leastLikely(_shadow_reinvigor_amount);

    for (auto i : least_likely_n)
    {
        VLOG(4) << "particle " << i << " with weight " << _shadow_belief.particle(i)->w
                << " in shadow was replace with a breeded state";

        _shadow_belief.replace(
            i,
            bayes_adaptive::factored::breed(
                fbapomdp, _belief.sample(), _fully_connected_belief.sample()),
            [&fbapomdp](State const* s) { fbapomdp.releaseState(s); });
    }
}

void StructureIncubatorSampling::reinvigorateBelief(FBAPOMDP const& fbapomdp)
{

    auto added_particle = false;

    for (size_t i = 0; i < _size; ++i)
    {
        auto p = _shadow_belief.particle(i);

        // add particles with high weight
        // and set their weight to 0 such that they will not
        // be picked again
        if (_shadow_belief.normalizedWeight(p->w) > _real_reinvigor_threshold)
        {
            added_particle = true;

            VLOG(4) << "reinvigoration found a shadow particle of weight " << p->w
                    << " and replaced a particle in the real belief";

            // transfer state from shadow to real belief
            _belief.replace(
                dynamic_cast<FBAPOMDPState const*>(fbapomdp.copyState(p->particle)),
                [&fbapomdp](State const* s) { fbapomdp.releaseState(s); });

            // by setting the weight of this particle in the
            // shadow belief to 0, we are effectively removing
            // it from the shadow belief
            p->w = 0;
        } else
            VLOG(5) << "reinvigoration REJECTED particles with weight" << p->w;
    }

    if (added_particle)
    {
        _shadow_belief.normalize();
    }
}

}}} // namespace beliefs::bayes_adaptive::factored
