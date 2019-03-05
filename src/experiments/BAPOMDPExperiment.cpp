#include "BAPOMDPExperiment.hpp"

#include "configurations/BAConf.hpp"

#include "experiments/Episode.hpp"

#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "bayes-adaptive/states/BAState.hpp"

#include "beliefs/bayes-adaptive/BABelief.hpp"
#include "environment/Environment.hpp"
#include "planners/bayes-adaptive/BAPlanner.hpp"

#include "environment/Horizon.hpp"

namespace experiment { namespace bapomdp {

Result::Result(int size) : r(size) {}

void Result::log(el::base::type::ostream_t& os) const
{
    os << "# version 1:\n# return mean, return var, return count, return stder, step duration "
          "mean\n";

    for (auto const& i : r)
    {
        os << i.ret.mean() << ", " << i.ret.var() << ", " << i.ret.count() << ", " << i.ret.stder()
           << ", " << i.duration.mean() << "\n";
    }
}

Result run(BAPOMDP const* bapomdp, configurations::BAConf const& conf)
{
    auto learning_results = Result(conf.num_episodes);

    auto const env     = factory::makeEnvironment(conf.domain_conf);
    auto const planner = factory::makeBAPlanner(conf);

    auto const belief   = factory::makeBABelief(conf);
    auto const discount = Discount(conf.discount);
    auto const h        = Horizon(conf.horizon);

    boost::timer timer;
    for (auto run = 0; run < conf.num_runs; ++run)
    {
        belief->initiate(*bapomdp);

        if (VLOG_IS_ON(3))
        {
            VLOG(3) << "Example BA counts from prior in run " << run << ":";
            dynamic_cast<BAState const*>(belief->sample())->logCounts();
        }

        for (auto episode = 0; episode < conf.num_episodes; ++episode)
        {
            VLOG(1) << "run " << run + 1 << "/" << conf.num_runs << ", episode " << episode + 1
                    << "/" << conf.num_episodes;

            belief->resetDomainStateDistribution(*bapomdp);

            timer.restart();
            auto const r = episode::run(*planner, *belief, *env, *bapomdp, h, discount);

            learning_results.r[episode].ret.add(r.ret.toDouble());
            learning_results.r[episode].duration.add(timer.elapsed() / r.length);
        }

        if (VLOG_IS_ON(3))
        {
            VLOG(3) << "Example BA counts at end of run " << run << ":";
            dynamic_cast<BAState const*>(belief->sample())->logCounts();
        }

        belief->free(*bapomdp);
    }

    return learning_results;
}

}} // namespace experiment::bapomdp
