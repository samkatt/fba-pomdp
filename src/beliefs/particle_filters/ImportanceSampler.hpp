#ifndef IMPORTANCESAMPLER_HPP
#define IMPORTANCESAMPLER_HPP

#include "beliefs/Belief.hpp"

#include <cstddef>
#include <string>

#include "easylogging++.h"

#include "beliefs/particle_filters/WeightedFilter.hpp"
#include "domains/POMDP.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
class Action;

namespace beliefs {

namespace importance_sampling {

/**
 * @brief apply importance sampling by updating all particles
 *
 * Will simulate a step with domain and multiply the particles weight
 * with the probability of generating the real observation
 *
 * PERF: more efficient way of building in resampling
 **/
template<typename T>
double update(WeightedFilter<T>& belief, Action const* a, Observation const* o, POMDP const& d)
{
    Observation const* tmp_observation(nullptr);
    Reward r(0);

    double total_weight = 0;
    for (size_t i = 0; i < belief.size(); ++i)
    {
        auto p = belief.particle(i);

        // required because ** cannot convert to each other
        // would be nice if there was some sort of solution here
        auto& s = reinterpret_cast<State const*&>(p->particle);

        d.step(&s, a, &tmp_observation, &r);
        p->w *= d.computeObservationProbability(o, a, p->particle);

        VLOG(4) << " sample " << i << " has now index " << p->particle->index()
                << " and was assigned weight " << p->w;

        total_weight += p->w;

        d.releaseObservation(tmp_observation);
    }

    VLOG(3) << "acquired total weight of " << total_weight << " after updating " << belief.size()
            << " particles";

    belief.normalize(total_weight);

    return total_weight;
}

/**
 * @brief apply importance sampling by sampling particles
 *
 * Will simulate a step with sampled particles and set the particles
 * weight to the probability of generating the real observation
 **/
template<typename T>
void resample(WeightedFilter<T>& belief, POMDP const& d, size_t n)
{
    auto const w = 1 / static_cast<double>(n);

    auto new_belief = WeightedFilter<T>();

    // create new (uniformly weighted) filter
    // by sampling particles from our current belief
    while (new_belief.size() != n)
    { new_belief.add(dynamic_cast<T>(d.copyState(belief.sample())), w); }

    // We do not need to normalize here, since the weights
    // assigned to the particles do not get multipled by their
    // precious weight (just calculated by computeObservationProbability
    // and thus do not run into problems of nearing 0
    // new_belief.normalize(total_weight);

    VLOG(4) << "Finished resampling " << n << " samples";

    belief.free([&d](State const* s) { d.releaseState(s); });
    belief = std::move(new_belief);
}

} // namespace importance_sampling

/**
 * @brief Estimates the states using importance sampling on weighted filters
 **/
class ImportanceSampler : public Belief
{
public:
    /**
     * @brief initiates importance sampling of n samples with an empty filter
     **/
    explicit ImportanceSampler(size_t n);

    /**
     * @brief initiates importance sampling of n samples with provided filter
     *
     * Assumes the filter size is not larger than provided n
     *
     **/
    ImportanceSampler(WeightedFilter<State const*> f, size_t n);

    /***** Belief interface *****/
    void initiate(POMDP const& d) final;
    void free(POMDP const& d) final;
    State const* sample() const final;
    void updateEstimation(Action const* a, Observation const* o, POMDP const& d) final;

private:
    WeightedFilter<State const*> _filter = {};

    // number of particles
    size_t _n;
};

} // namespace beliefs

#endif // IMPORTANCESAMPLER_HPP
