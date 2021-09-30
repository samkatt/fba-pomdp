#include "BATSPlanner.hpp"

#include "easylogging++.h"

#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "beliefs/bayes-adaptive/BAPointEstimation.hpp"
#include "configurations/Conf.hpp"

namespace planners {

BATSPlanner::BATSPlanner(configurations::Conf const& c) : _planner(c)
{
    VLOG(1) << "initiated Bayes Adaptive Thompson Sampling with internal PO-UCT planner";
}

Action const* BATSPlanner::selectAction(
    BAPOMDP const& bapomdp,
    ::beliefs::BABelief const& belief,
    History const& h,
    int& total_simulations) const
{

    // TS belief (borrows a single sample)
    beliefs::BAPointEstimation sampled_estimation(belief.sample());

    VLOG(3) << "BAThompson sampled and running internal planner on s="
            << sampled_estimation.sample()->toString();

    // plan according to sampled belief
    auto const a = _planner.selectAction(bapomdp, sampled_estimation, h, total_simulations);
    total_simulations += 1;
    return a;
}

} // namespace planners
