#include "Planner.hpp"

#include "configurations/Conf.hpp"
#include "planners/mcts/POUCT.hpp"
#include "planners/random/RandomPlanner.hpp"
#include "planners/ts/TSPlanner.hpp"

namespace factory {

std::unique_ptr<Planner> makePlanner(configurations::Conf const& c)
{
    if (c.planner == "random")
        return std::unique_ptr<Planner>(new planners::RandomPlanner());
    if (c.planner == "ts")
        return std::unique_ptr<Planner>(new planners::TSPlanner(c));
    if (c.planner == "po-uct")
        return std::unique_ptr<Planner>(new planners::POUCT(c));

    throw "incorrect planner provided";
}

} // namespace factory
