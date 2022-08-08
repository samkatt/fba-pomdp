#include "PlanningExperiment.hpp"

#include "configurations/Conf.hpp"

#include "experiments/Episode.hpp"

#include "domains/POMDP.hpp"
#include "environment/Environment.hpp"

#include "beliefs/Belief.hpp"
#include "planners/Planner.hpp"

#include "environment/Discount.hpp"
#include "environment/Horizon.hpp"
#include "environment/Return.hpp"

namespace experiment { namespace planning {

void Result::log(el::base::type::ostream_t& os) const
{
    os << "# version 1:\n# return mean, return var, return count, return stder, step duration "
          "mean\n";
    os << episode_return.mean() << ", " << episode_return.var() << ", " << episode_return.count()
       << ", " << episode_return.stder() << ", " << episode_duration.mean();
}

Result run(configurations::Conf const& conf)
{
    auto planning_result = Result();

    auto const env       = factory::makeEnvironment(conf.domain_conf);
    auto const planner   = factory::makePlanner(conf);
    auto const simulator = factory::makePOMDP(conf.domain_conf);
    auto const belief    = factory::makeBelief(conf);
    auto const discount  = Discount(conf.discount);
    auto const h         = Horizon(conf.horizon);

    boost::timer timer;
    for (auto ep = 0; ep < conf.num_runs; ++ep)
    {
        VLOG(1) << "episode " << ep + 1 << "/" << conf.num_runs;

        belief->initiate(*simulator);

        timer.restart();
        auto const r = episode::run(*planner, *belief, *env, *simulator, h, discount);

        planning_result.episode_return.add(r.ret.toDouble());
        planning_result.episode_duration.add(timer.elapsed() / r.length);

        belief->free(*simulator);
    }

    return planning_result;
}

}} // namespace experiment::planning
