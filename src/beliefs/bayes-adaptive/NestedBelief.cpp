#include "NestedBelief.hpp"

#include "easylogging++.h"

#include <string>

namespace {

std::string describeParticle(std::pair<BAState const*, FlatFilter<State const*>> const& particle)
{
    return "Top particle with nested filter:\n" + particle.second.toString();
}

} // namespace

namespace beliefs { namespace bayes_adaptive {

NestedBelief::NestedBelief(size_t top_filter_size, size_t bottom_filter_size) :
        _top_filter_size(top_filter_size), _bottom_filter_size(bottom_filter_size)
{

    if (top_filter_size < 1 || bottom_filter_size < 1)
    {
        throw "NestedBelief: cannot initiate with filter size < 1 (top: "
            + std::to_string(_top_filter_size) + ", bottom: " + std::to_string(_bottom_filter_size)
            + ")";
    }

    VLOG(1) << "initiated nested belief of top size " << _top_filter_size << " and bottom size "
            << _bottom_filter_size;
}

/**** BABelief interface ****/
void NestedBelief::resetDomainStateDistribution(BAPOMDP const& bapomdp)
{
    assertValidBelief();

    auto& domain = bapomdp._domain;

    for (size_t i = 0; i < _top_filter_size; ++i)
    {

        // construct domain state prior
        std::vector<State const*> domain_states;
        for (size_t j = 0; j < _bottom_filter_size; ++j)
        {
            domain_states.emplace_back(domain->sampleStartState());
        }

        // reconstruct each nested filter by freeing up current first

        _filter.particle(i)->particle.second.free(
            [&domain](State const* s) { domain->releaseState(s); });
        _filter.particle(i)->particle.second = FlatFilter<State const*>(std::move(domain_states));
    }

    VLOG(3) << "Status after resetting domain state to initial distribution\n"
            << _filter.toString(describeParticle);
}

/**** Belief interface ****/
void NestedBelief::initiate(POMDP const& domain)
{
    assert(_filter.empty());

    auto& bapomdp = dynamic_cast<BAPOMDP const&>(domain);

    auto bottom_filter_size = _bottom_filter_size;

    // determine how both the 1st and 2nd layer is constructed
    auto domain_state_alloc = [&bapomdp] { return bapomdp.sampleDomainState(); };
    auto top_particle_alloc = [&bapomdp, &domain_state_alloc, &bottom_filter_size] {
        return std::make_pair(
            static_cast<BAState const*>(bapomdp.sampleStartState()),
            FlatFilter<State const*>(bottom_filter_size, domain_state_alloc));
    };

    // actually construct the 2-level filter
    _filter = WeightedFilter<std::pair<BAState const*, FlatFilter<State const*>>>{
        _top_filter_size, top_particle_alloc};

    // free up the domain state in the top level filter, as those are usefull
    for (size_t i = 0; i < _top_filter_size; ++i)
    {
        bapomdp.releaseDomainState(_filter.particle(i)->particle.first->_domain_state);
    }

    assertValidBelief();
}

void NestedBelief::free(POMDP const& domain)
{
    assertValidBelief();

    auto& bapomdp = dynamic_cast<BAPOMDP const&>(domain);
    auto& pomdp   = *bapomdp._domain;

    for (size_t i = 0; i < _top_filter_size; ++i)
    {

        // free up domain states in nested filter
        _filter.particle(i)->particle.second.free(
            [&pomdp](State const* s) { pomdp.releaseState(s); });

        // free top counts (models) state
        // this particle contains a empty domain state (deleted and useless from the start)
        // and a model. We exploit our implementation knowledge and directly
        // delete pointer, knowing that the destructor will handle everything just fine
        delete _filter.particle(i)->particle.first;
    }

    // since we cannot use the typical 'free' function to clear our filter
    // we must redefine it, exploiting knowledge that the destructor will take
    // care of the remaining internal memory just fine
    _filter = WeightedFilter<std::pair<BAState const*, FlatFilter<State const*>>>();
}

State const* NestedBelief::sample() const
{

    auto const& top_particle = _filter.sample();

    // set the domian state of the BAState with counts
    const_cast<BAState*>(top_particle.first)->_domain_state = top_particle.second.sample();

    return top_particle.first;
}

void NestedBelief::updateEstimation(Action const* a, Observation const* o, POMDP const& d)
{
    assertValidBelief();

    auto const update_step = static_cast<float>(1.0 / static_cast<float>(_bottom_filter_size));

    auto const& bapomdp = static_cast<BAPOMDP const&>(d);
    auto const& domain  = bapomdp._domain;

    Reward r(0);
    Observation const* o_sample;

    for (size_t i = 0; i < _top_filter_size; ++i)
    {
        VLOG(4) << "updating particle " << i << " of top level filter of nested belief";

        // get top level particle and get its counts and domain state belief
        auto bastate = static_cast<State const*>(_filter.particle(i)->particle.first);
        auto counts  = const_cast<BAState*>(_filter.particle(i)->particle.first);
        auto& belief = _filter.particle(i)->particle.second;

        std::vector<State const*> new_states;
        new_states.reserve(_bottom_filter_size);

        // update domain state distribution by rejection sampling and keep statistics
        auto count = 0;
        while (new_states.size() < _bottom_filter_size)
        {

            auto const old_state  = belief.sample();
            counts->_domain_state = bapomdp._domain->copyState(old_state);

            bapomdp.step(&bastate, a, &o_sample, &r, BAPOMDP::StepType::KeepCounts);

            // accept or reject
            if (o_sample->index() == o->index())
            {
                VLOG(5) << "accepted state of index " << counts->_domain_state->index();
                new_states.emplace_back(counts->_domain_state);

                counts->incrementCountsOf(old_state, a, o, counts->_domain_state, update_step);
            } else
            {
                VLOG(5) << "rejected state of index " << counts->_domain_state->index();
                domain->releaseState(counts->_domain_state);
            }

            bapomdp.releaseObservation(o_sample);
            count++;
        }

        belief.free([&domain](State const* s) { domain->releaseState(s); });
        belief = FlatFilter<State const*>(std::move(new_states));

        // update weight of filter based on rejection success
        _filter.particle(i)->w *= 1.0 / static_cast<double>(count);

        VLOG(4) << "particle " << i << " updated  weight to " << _filter.particle(i)->w << " after "
                << count << " attempted rejection samples";
    }

    _filter.normalize();

    VLOG(3) << "Status of filter after update:" << _filter.toString(describeParticle);
}

void NestedBelief::assertValidBelief() const
{

    assert(_top_filter_size == _filter.size());

    for (size_t i = 0; i < _top_filter_size; ++i)
    {
        auto f = _filter.particle(i);

        assert(f->w > 0);
        assert(f->particle.second.size() == _bottom_filter_size);

        for (auto const& s : f->particle.second.particles()) { assert(s != nullptr); }
    }
}

}} // namespace beliefs::bayes_adaptive
