#include "RandomPlanner.hpp"

#include "easylogging++.h"

#include "beliefs/Belief.hpp"
#include "domains/POMDP.hpp"
#include "environment/Action.hpp"

namespace planners {

RandomPlanner::RandomPlanner()
{
    VLOG(1) << "initiated random planner";
}

Action const* RandomPlanner::selectAction(
    POMDP const& simulator,
    Belief const& belief,
    History const& /*h*/
) const
{
    auto random_action = simulator.generateRandomAction(belief.sample());

    VLOG(3) << "Action " << random_action->index() << " randomly picked by Random Planner";

    return random_action;
}

} // namespace planners
