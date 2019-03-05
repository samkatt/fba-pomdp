#include "Belief.hpp"

#include "beliefs/particle_filters/ImportanceSampler.hpp"
#include "beliefs/particle_filters/RejectionSampling.hpp"
#include "beliefs/point_estimation/PointEstimation.hpp"
#include "configurations/Conf.hpp"

namespace factory {

std::unique_ptr<Belief> makeBelief(configurations::Conf const& c)
{

    if (c.belief == "point_estimate")
        return std::unique_ptr<Belief>(new beliefs::PointEstimation());
    if (c.belief == "rejection_sampling")
        return std::unique_ptr<Belief>(
            new beliefs::RejectionSampling(c.belief_conf.particle_amount));
    if (c.belief == "importance_sampling")
        return std::unique_ptr<Belief>(
            new beliefs::ImportanceSampler(c.belief_conf.particle_amount));

    throw "incorrect state estimator provided";
}

} // namespace factory
