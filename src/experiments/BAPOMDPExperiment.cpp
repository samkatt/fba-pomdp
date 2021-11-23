#include <beliefs/bayes-adaptive/BAImportanceSampling.hpp>
#include "BAPOMDPExperiment.hpp"

#include "bayes-adaptive/models/factored/FBAPOMDP.hpp"
#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "bayes-adaptive/abstractions/Abstraction.hpp"

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
          "mean, step simulations mean\n";

    for (auto const& i : r)
    {
        os << i.ret.mean() << ", " << i.ret.var() << ", " << i.ret.count() << ", " << i.ret.stder()
           << ", " << i.duration.mean() << ", " << i.simulations.mean() << "\n";
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

    bool useAbstraction = conf.domain_conf.abstraction;
    int selectedAbstraction = conf.planner_conf.abstraction_k;

    auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(*bapomdp);
    auto const abstraction = factory::makeAbstraction(conf);
//    auto const num_abstractions = fbapomdp.domain()
    Domain_Feature_Size test = *fbapomdp.domainFeatureSize();
//    VLOG(1) << "Test " << test._S;

    boost::timer::cpu_timer timer;
    for (auto run = 0; run < conf.num_runs; ++run)
    {
//        env->clearCache();
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



            // time including resetDomainState
//            timer.restart();
//            belief->resetDomainStateDistribution(*bapomdp);
//            int selectedAbstraction = 1; // abstraction->selectAbstraction();
//            VLOG(1) << "Selected abstraction " << selectedAbstraction;
            // clear states used in previous episode, not needed for the resetDomainStateDistribution (I think?)
            if (!conf.domain_conf.store_statespace) {
                bapomdp->clearCache();
            }
            if (useAbstraction) {
                if (abstraction->isFullModel(selectedAbstraction)) {
                    belief->resetDomainStateDistribution(*bapomdp);
                } else {
                    belief->resetDomainStateDistributionAndAddAbstraction(*bapomdp, *abstraction, selectedAbstraction);
                }
            } else {
                belief->resetDomainStateDistribution(*bapomdp);
            }


            // TODO
            // Implement something like:
            // 1) Maintain Q-values or something for different abstractions
            // 2) choose abstraction to use for episode
            // 3) construct abstraction in particles
            // Implement: 3
            // Implement: in domain: number of abstractions, set of variables per abstraction?


            // time excluding resetDomainState
            timer.start();
            auto const r = episode::run(*planner, *belief, *env, *bapomdp, h, discount);
            learning_results.r[episode].simulations.add((float) r.simulations / (float) r.length);
            learning_results.r[episode].ret.add(r.ret.toDouble());
            learning_results.r[episode].duration.add(timer.elapsed().wall / r.length);
            if(useAbstraction) {
                abstraction->addReturn(selectedAbstraction, r.ret.toDouble());
            }
        }

        if (VLOG_IS_ON(3))
        {
            VLOG(3) << "Example BA counts at end of run " << run << ":";
            dynamic_cast<BAState const*>(belief->sample())->logCounts();
        }
        // two times states, in bapomdp and in env? where does the cache grow?
        belief->free(*bapomdp);
        if (!conf.domain_conf.store_statespace) {
            bapomdp->clearCache();
            env->clearCache();
        }
    }

    return learning_results;
}

}} // namespace experiment::bapomdp
