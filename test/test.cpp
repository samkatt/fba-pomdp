#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <ctime>
#include <string>

#include "easylogging++.h"

#include "configurations/BAConf.hpp"
#include "configurations/Conf.hpp"
#include "configurations/DomainConf.hpp"
#include "configurations/FBAConf.hpp"

#include "experiments/BAPOMDPExperiment.hpp"
#include "experiments/PlanningExperiment.hpp"

#include "bayes-adaptive/models/factored/FBAPOMDP.hpp"
#include "bayes-adaptive/models/table/BAPOMDP.hpp"

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"

#include "utils/random.hpp"

INITIALIZE_EASYLOGGINGPP

TEST_CASE("Main test", "[main][integration]")
{
    rnd::initiate();

    auto seed = std::to_string(time(nullptr));
    rnd::seed(seed);

    REQUIRE(true);
}

/**
 * \brief attempts to test all domains on all planners on all types of beliefs
 **/
SCENARIO("pomdp integration", "[hide][integration]")
{

    auto domains = {
        "dummy",
        "linear_dummy",
        "factored-dummy",
        "episodic-tiger",
        "continuous-tiger",
        "independent-sysadmin",
        "linear-sysadmin",
        "episodic-factored-tiger",
        "continuous-factored-tiger",
        "random-collision-avoidance",
        "centered-collision-avoidance"};
    auto planners         = {"random", "ts", "po-uct"};
    auto beliefs          = {"point_estimate", "rejection_sampling", "importance_sampling"};
    auto factored_domains = {
        "factored-dummy",
        "independent-sysadmin",
        "linear-sysadmin",
        "episodic-factored-tiger",
        "continuous-factored-tiger",
        "random-collision-avoidance",
        "centered-collision-avoidance",
        "gridworld"};

    configurations::Conf conf;

    conf.output_file = "planning_tests.res";

    conf.horizon  = 3;
    conf.num_runs = 2;

    conf.planner_conf.mcts_simulation_amount = 50;
    conf.planner_conf.mcts_max_depth         = conf.horizon;

    for (auto const& p : planners)
    {
        conf.planner = p;
        for (auto const& b : beliefs)
        {
            conf.belief = b;

            conf.domain_conf.size   = 0;
            conf.domain_conf.width  = 0;
            conf.domain_conf.height = 0;

            // these domains have not implemented the correct functions
            // to calculate the probability of an observation given a state
            // and action
            if (conf.belief != "importance_sampling")
            {

                // coffee problem
                for (std::string const v : {"", "boutilier-"})
                {
                    conf.domain_conf.domain = v + "coffee";

                    REQUIRE_NOTHROW(conf.validate());
                    REQUIRE_NOTHROW(experiment::planning::run(conf));
                }

            } else // gridworld problem only works with importance sampling really..
            {

                conf.domain_conf.domain = "gridworld";
                conf.domain_conf.size   = 3;

                REQUIRE_NOTHROW(conf.validate());
                REQUIRE_NOTHROW(experiment::planning::run(conf));
            }

            for (std::string const d : domains)
            {
                conf.domain_conf.domain = d;

                // set size for domains if necessary
                conf.domain_conf.size =
                    std::find(factored_domains.begin(), factored_domains.end(), d)
                            == factored_domains.end()
                        ? 0
                        : 3;

                if (d == "random-collision-avoidance" || d == "centered-collision-avoidance")
                {
                    conf.domain_conf.width  = 5;
                    conf.domain_conf.height = 3;
                } else
                {
                    conf.domain_conf.width  = 0;
                    conf.domain_conf.height = 0;
                }

                REQUIRE_NOTHROW(conf.validate());
                REQUIRE_NOTHROW(experiment::planning::run(conf));
            }
        }
    }
}

/**
 * \brief attempts to test all domains on all planners on all state estimators on BAPOMDPs
 **/
SCENARIO("bapomdp integration", "[hide][integration][flat][bayes-adaptive]")
{

    auto domains = {
        "dummy",
        "factored-dummy",
        "episodic-tiger",
        "continuous-tiger",
        "episodic-factored-tiger",
        "continuous-factored-tiger",
        "independent-sysadmin",
        "linear-sysadmin",
        "random-collision-avoidance",
        "centered-collision-avoidance"};
    auto planners = {"random", "ts", "po-uct"};
    auto beliefs  = {"point_estimate", "rejection_sampling", "importance_sampling", "nested"};

    auto factored_domains = {
        "factored-dummy",
        "independent-sysadmin",
        "linear-sysadmin",
        "episodic-factored-tiger",
        "continuous-factored-tiger",
        "random-collision-avoidance",
        "centered-collision-avoidance"};

    configurations::BAConf conf;

    conf.output_file = "bapomdp_tests.res";

    conf.horizon      = 3;
    conf.num_runs     = 2;
    conf.num_episodes = 3;

    conf.planner_conf.mcts_simulation_amount = 50;
    conf.planner_conf.mcts_max_depth         = conf.horizon;

    using rnd::sample::Dir::SAMPLETYPE;
    for (auto sampling_method : {SAMPLETYPE::Regular, SAMPLETYPE::Expected})
    {
        conf.bayes_sample_method = sampling_method;

        for (auto const p : planners)
        {
            conf.planner = p;
            for (auto const b : beliefs)
            {
                conf.belief = b;

                // regular 100 particles is way too much for that belief
                conf.belief_conf.particle_amount = (conf.belief == "nested") ? 5 : 100;

                for (std::string const d : domains)
                {
                    conf.domain_conf.domain = d;

                    if (d == "random-collision-avoidance" || d == "centered-collision-avoidance")
                    {
                        conf.domain_conf.width  = 5;
                        conf.domain_conf.height = 3;
                    } else
                    {
                        conf.domain_conf.width  = 0;
                        conf.domain_conf.height = 0;
                    }

                    conf.domain_conf.size =
                        std::find(factored_domains.begin(), factored_domains.end(), d)
                                == factored_domains.end()
                            ? 0
                            : 3;

                    REQUIRE_NOTHROW(conf.validate());

                    auto const bapomdp = factory::makeTBAPOMDP(conf);
                    REQUIRE_NOTHROW(experiment::bapomdp::run(bapomdp.get(), conf));
                }

                if (conf.belief == "importance_sampling") // gridworld fails on others
                {

                    conf.domain_conf.size   = 3;
                    conf.domain_conf.width  = 0;
                    conf.domain_conf.height = 0;

                    conf.domain_conf.domain = "gridworld";

                    REQUIRE_NOTHROW(conf.validate());

                    auto const bapomdp = factory::makeTBAPOMDP(conf);
                    REQUIRE_NOTHROW(experiment::bapomdp::run(bapomdp.get(), conf));
                }
            }
        }
    }
}

/**
 * \brief attempts to test all domains on all planners on all state estimators on FBAPOMDPs
 **/
SCENARIO("fbapomdp integration", "[hide][integration][bayes-adaptive][factored]")
{

    auto domains = {
        "factored-dummy",
        "episodic-factored-tiger",
        "continuous-factored-tiger",
        "independent-sysadmin",
        "linear-sysadmin",
        "random-collision-avoidance",
        "centered-collision-avoidance"};
    auto planners = {"random", "ts", "po-uct"};
    auto beliefs  = {
        "point_estimate",
        "rejection_sampling",
        "importance_sampling",
        "reinvigoration",
        "incubator",
        "nested"};
    auto resampling_beliefs = {"reinvigoration", "incubator"};
    auto factored_domains   = {
        "factored-dummy",
        "independent-sysadmin",
        "linear-sysadmin",
        "episodic-factored-tiger",
        "continuous-factored-tiger",
        "random-collision-avoidance",
        "centered-collision-avoidance"};

    configurations::FBAConf conf;

    conf.output_file = "fbapomdp_tests.res";

    conf.horizon      = 3;
    conf.num_runs     = 2;
    conf.num_episodes = 2;

    conf.planner_conf.mcts_simulation_amount = 50;
    conf.planner_conf.mcts_max_depth         = conf.horizon;

    conf.belief_conf.particle_amount = 100;

    using rnd::sample::Dir::SAMPLETYPE;
    for (auto sampling_method : {SAMPLETYPE::Regular, SAMPLETYPE::Expected})
    {
        conf.bayes_sample_method = sampling_method;
        for (auto const p : planners)
        {
            conf.planner = p;
            for (auto const b : beliefs)
            {
                conf.belief                = b;
                conf.belief_conf.threshold = .05;

                conf.belief_conf.resample_amount =
                    std::find(resampling_beliefs.begin(), resampling_beliefs.end(), b)
                            == resampling_beliefs.end()
                        ? 0
                        : 10;

                // regular 100 particles is way too much for that belief
                conf.belief_conf.particle_amount = (conf.belief == "nested") ? 5 : 100;

                for (std::string const d : domains)
                {
                    conf.domain_conf.domain = d;

                    if (d == "random-collision-avoidance" || d == "centered-collision-avoidance")
                    {
                        conf.domain_conf.width  = 5;
                        conf.domain_conf.height = 3;
                    } else
                    {
                        conf.domain_conf.width  = 0;
                        conf.domain_conf.height = 0;
                    }

                    conf.domain_conf.size =
                        std::find(factored_domains.begin(), factored_domains.end(), d)
                                == factored_domains.end()
                            ? 0
                            : 3;

                    REQUIRE_NOTHROW(conf.validate());

                    auto const fbapomdp = factory::makeFBAPOMDP(conf);
                    REQUIRE_NOTHROW(experiment::bapomdp::run(fbapomdp.get(), conf));
                }

                if (conf.belief == "importance_sampling") // gridworld fails on others
                {

                    conf.domain_conf.size   = 3;
                    conf.domain_conf.height = 0;
                    conf.domain_conf.width  = 0;

                    conf.domain_conf.domain = "gridworld";

                    REQUIRE_NOTHROW(conf.validate());

                    auto const fbapomdp = factory::makeFBAPOMDP(conf);
                    REQUIRE_NOTHROW(experiment::bapomdp::run(fbapomdp.get(), conf));
                }
            }

            // test cheating-reinvigoration specifically on the possible domains
            conf.belief                      = "cheating-reinvigoration";
            conf.belief_conf.threshold       = -3;
            conf.belief_conf.resample_amount = 10;
            conf.belief_conf.particle_amount = 100;

            conf.domain_conf.size = 4;

            for (auto const d : {"episodic-factored-tiger", "continuous-factored-tiger"})
            {

                conf.domain_conf.domain = d;
                conf.domain_conf.width  = 0;
                conf.domain_conf.height = 0;

                REQUIRE_NOTHROW(conf.validate());

                auto const fbapomdp = factory::makeFBAPOMDP(conf);
                REQUIRE_NOTHROW(experiment::bapomdp::run(fbapomdp.get(), conf));
            }

            for (auto const d : {"random-collision-avoidance", "centered-collision-avoidance"})
            {

                conf.domain_conf.domain = d;
                conf.domain_conf.width  = 5;
                conf.domain_conf.height = 3;

                REQUIRE_NOTHROW(conf.validate());

                auto const fbapomdp = factory::makeFBAPOMDP(conf);
                REQUIRE_NOTHROW(experiment::bapomdp::run(fbapomdp.get(), conf));
            }

            for (auto const mh : {"mh-nips", "mh-within-gibbs"})
            {
                // test mh specifically on the possible domains
                conf.belief                      = mh;
                conf.belief_conf.resample_amount = 0;

                conf.domain_conf.height = 0;
                conf.domain_conf.width  = 0;

                for (auto const d : {"episodic-factored-tiger", "continuous-factored-tiger"})
                {

                    conf.domain_conf.domain = d;
                    REQUIRE_NOTHROW(conf.validate());

                    auto const fbapomdp = factory::makeFBAPOMDP(conf);
                    REQUIRE_NOTHROW(experiment::bapomdp::run(fbapomdp.get(), conf));
                }
            }
        }
    }
}
