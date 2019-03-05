#ifndef REJECTIONSAMPLING_HPP
#define REJECTIONSAMPLING_HPP

#include "beliefs/Belief.hpp"

#include "easylogging++.h"

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

#include "beliefs/particle_filters/FlatFilter.hpp"

#include "domains/POMDP.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"

class Action;

namespace beliefs {

template<typename T>
void rejectSample(
    Action const* a,
    Observation const* o,
    POMDP const& simulator,
    size_t n,
    FlatFilter<T>& belief)
{
    assert(a != nullptr && o != nullptr);
    assert(belief.size() == n);

    // we will fill this with our updated particles/states
    auto new_states = std::vector<T>();
    new_states.reserve(n);

    // place holders
    Observation const* simulated_observation(nullptr);
    Reward r(0);

    auto count = 0;
    // apply rejection sampling
    while (new_states.size() < n)
    {
        // simulate step
        auto sample_state = simulator.copyState(belief.sample());
        simulator.step(&sample_state, a, &simulated_observation, &r);

        // accept or reject
        if (simulated_observation->index() == o->index())
        {
            VLOG(4) << "accepted state of index " << sample_state->index();
            new_states.emplace_back(dynamic_cast<T>(sample_state));
        } else
        {
            VLOG(4) << "rejected state of index " << sample_state->index();
            simulator.releaseState(sample_state);
        }

        simulator.releaseObservation(simulated_observation);

        count++;
    }

    VLOG(3) << "performed " << count << " loops for rejection sampling for " << n << " samples";

    belief.free([&simulator](State const* s) { simulator.releaseState(s); });
    belief = FlatFilter<T>(new_states);
}

/**
 * @brief particle belief updated through rejection sampling
 **/
class RejectionSampling : public Belief
{
public:
    explicit RejectionSampling(size_t n);

    /**** Belief interface ****/
    void initiate(POMDP const& simulator) final;
    void free(POMDP const& simulator) final;
    State const* sample() const final;
    void updateEstimation(Action const* a, Observation const* o, POMDP const& d) final;

private:
    // number of desired particles
    size_t const _n;

    FlatFilter<State const*> _filter = {};
};

} // namespace beliefs

#endif // REJECTIONSAMPLING_HPP
