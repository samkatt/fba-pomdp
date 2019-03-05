#include "TSPlanner.hpp"

#include "easylogging++.h"

#include "beliefs/Belief.hpp"
#include "beliefs/point_estimation/PointEstimation.hpp"
#include "domains/POMDP.hpp"
#include "environment/State.hpp"

namespace planners {

TSPlanner::TSPlanner(configurations::Conf const& c) : _planner(c)
{
    VLOG(1) << "initiated Thompson Sampling with internal planner";
}

Action const*
    TSPlanner::selectAction(POMDP const& simulator, Belief const& belief, History const& h) const
{

    // TS belief (borrows single sample)
    beliefs::PointEstimation sampled_estimation(belief.sample());

    VLOG(3) << "Thompson sampled and running internal planner on s="
            << sampled_estimation.sample()->index();

    // plan according to sampled belief
    auto const a = _planner.selectAction(simulator, sampled_estimation, h);

    return a;
}

} // namespace planners
