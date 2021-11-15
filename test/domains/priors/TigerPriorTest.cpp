#include "catch.hpp"

#include "easylogging++.h"

#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "beliefs/Belief.hpp"
#include "beliefs/bayes-adaptive/BABelief.hpp"
#include "configurations/BAConf.hpp"
#include "configurations/FBAConf.hpp"
#include "domains/tiger/FactoredTiger.hpp"
#include "domains/tiger/FactoredTigerBAExtension.hpp"
#include "domains/tiger/FactoredTigerPriors.hpp"
#include "domains/tiger/Tiger.hpp"
#include "domains/tiger/TigerBAExtension.hpp"
#include "domains/tiger/TigerPriors.hpp"
#include "environment/Discount.hpp"
#include "environment/Horizon.hpp"
#include "experiments/BAPOMDPExperiment.hpp"
#include "experiments/Episode.hpp"
#include "planners/Planner.hpp"

TEST_CASE("tiger bayes-adaptive prior", "[tiger][bayes-adaptive]")
{
    float known_counts = 5000;

    auto const d = domains::Tiger(domains::Tiger::TigerType::EPISODIC);
    auto const ext =
        bayes_adaptive::domain_extensions::TigerBAExtension(domains::Tiger::TigerType::EPISODIC);
    auto d_size = ext.domainSize();
    auto s      = d.sampleStartState();

    REQUIRE(d_size._A == 3);
    REQUIRE(d_size._S == 2);
    REQUIRE(d_size._O == 2);

    WHEN("the tiger sets bayes-adaptive prior counts")
    {
        configurations::FBAConf c;
        c.domain_conf.domain = "episodic-tiger";

        THEN("the noisy free prior is perfect")
        {
            c.counts_total = 100;

            auto const prior = priors::TigerBAPrior(c);
            auto ba_s        = static_cast<BAPOMDPState*>(prior.sample(s));

            IndexAction listen(std::to_string(domains::Tiger::Literal::OBSERVE));
            IndexAction open_left(std::to_string(domains::Tiger::Literal::LEFT));
            IndexAction open_right(std::to_string(domains::Tiger::Literal::RIGHT));

            IndexObservation correct_observation(ba_s->index());
            IndexObservation wrong_observation(std::to_string(1 - std::stoi(ba_s->index())));

            IndexState other_state(std::to_string(1 - std::stoi(ba_s->index())));

            // listening
            REQUIRE(ba_s->model()->count(ba_s, &listen, ba_s) == known_counts);
            REQUIRE(ba_s->model()->count(&other_state, &listen, ba_s) == 0);
            REQUIRE(
                ba_s->model()->count(&listen, ba_s, &correct_observation) == c.counts_total * .85f);
            REQUIRE(
                ba_s->model()->count(&listen, ba_s, &wrong_observation) == c.counts_total * .15f);

            // opening right
            REQUIRE(ba_s->model()->count(ba_s, &open_left, ba_s) == known_counts);
            REQUIRE(ba_s->model()->count(ba_s, &open_left, &other_state) == known_counts);
            REQUIRE(ba_s->model()->count(&open_left, ba_s, &correct_observation) == known_counts);
            REQUIRE(ba_s->model()->count(&open_left, ba_s, &wrong_observation) == known_counts);
            REQUIRE(
                ba_s->model()->count(&open_left, &other_state, &correct_observation)
                == known_counts);
            REQUIRE(
                ba_s->model()->count(&open_left, &other_state, &wrong_observation) == known_counts);

            // opening right
            REQUIRE(ba_s->model()->count(ba_s, &open_right, ba_s) == known_counts);
            REQUIRE(ba_s->model()->count(ba_s, &open_right, &other_state) == known_counts);
            REQUIRE(ba_s->model()->count(&open_right, ba_s, &correct_observation) == known_counts);
            REQUIRE(ba_s->model()->count(&open_right, ba_s, &wrong_observation) == known_counts);
            REQUIRE(
                ba_s->model()->count(&open_right, &other_state, &correct_observation)
                == known_counts);
            REQUIRE(
                ba_s->model()->count(&open_right, &other_state, &wrong_observation)
                == known_counts);

            delete (ba_s);
        }

        THEN("then noise of 1.5 with 10 counts is set correctly")
        {
            c.noise        = 0.15f;
            c.counts_total = 10;

            auto const prior = priors::TigerBAPrior(c);
            auto ba_s        = static_cast<BAPOMDPState*>(prior.sample(s));

            IndexAction listen(std::to_string(domains::Tiger::Literal::OBSERVE));
            IndexAction open_left(std::to_string(domains::Tiger::Literal::LEFT));
            IndexAction open_right(std::to_string(domains::Tiger::Literal::RIGHT));

            IndexObservation correct_observation(ba_s->index());
            IndexObservation wrong_observation(std::to_string(1 - std::stoi(ba_s->index())));

            IndexState other_state(std::to_string(1 - std::stoi(ba_s->index())));

            // listening
            REQUIRE(ba_s->model()->count(ba_s, &listen, ba_s) == known_counts);
            REQUIRE(ba_s->model()->count(&other_state, &listen, ba_s) == 0);
            REQUIRE(
                ba_s->model()->count(&listen, ba_s, &correct_observation)
                == Approx(c.counts_total * .7f));
            REQUIRE(
                ba_s->model()->count(&listen, ba_s, &wrong_observation)
                == Approx(c.counts_total * .3f));

            // opening right
            REQUIRE(ba_s->model()->count(ba_s, &open_left, ba_s) == known_counts);
            REQUIRE(ba_s->model()->count(ba_s, &open_left, &other_state) == known_counts);
            REQUIRE(ba_s->model()->count(&open_left, ba_s, &correct_observation) == known_counts);
            REQUIRE(ba_s->model()->count(&open_left, ba_s, &wrong_observation) == known_counts);
            REQUIRE(
                ba_s->model()->count(&open_left, &other_state, &correct_observation)
                == known_counts);
            REQUIRE(
                ba_s->model()->count(&open_left, &other_state, &wrong_observation) == known_counts);

            // opening right
            REQUIRE(ba_s->model()->count(ba_s, &open_right, ba_s) == known_counts);
            REQUIRE(ba_s->model()->count(ba_s, &open_right, &other_state) == known_counts);
            REQUIRE(ba_s->model()->count(&open_right, ba_s, &correct_observation) == known_counts);
            REQUIRE(ba_s->model()->count(&open_right, ba_s, &wrong_observation) == known_counts);
            REQUIRE(
                ba_s->model()->count(&open_right, &other_state, &correct_observation)
                == known_counts);
            REQUIRE(
                ba_s->model()->count(&open_right, &other_state, &wrong_observation)
                == known_counts);

            delete (ba_s);
        }

        d.releaseState(s);
    }
}

SCENARIO("tiger bapomdp prior test for real experiment", "[tiger][bayes-adaptive][hide][logging]")
{
    GIVEN("The configurations of a real experiment")
    {
        auto conf = configurations::FBAConf();

        conf.domain_conf.domain = "episodic-tiger";
        conf.num_episodes       = 200;

        conf.planner_conf.mcts_max_depth = 10;
        conf.num_runs                    = 1;
        conf.horizon                     = conf.planner_conf.mcts_max_depth;

        auto const env      = factory::makeEnvironment(conf.domain_conf);
        auto const planner  = factory::makePlanner(conf);
        auto const discount = Discount(conf.discount);
        auto const h        = Horizon(conf.horizon);

        WHEN("There is no noise, but 10 counts")
        {

            conf.noise        = 0;
            conf.counts_total = 10;
            REQUIRE_NOTHROW(conf.validate());

            auto const bapomdp = factory::makeTBAPOMDP(conf);
            auto const belief  = factory::makeBABelief(conf);

            belief->initiate(*bapomdp);
            belief->resetDomainStateDistribution(*bapomdp);

            LOG(INFO) << "This sample represents the count belief for no noise w/ 10 counts";
            static_cast<BAPOMDPState const*>(belief->sample())->logCounts();

            for (auto r = 0; r < conf.num_episodes; ++r)
            {
                belief->resetDomainStateDistribution(*bapomdp);
                episode::run(*planner, *belief, *env, *bapomdp, h, discount);
            }

            LOG(INFO) << "After no noise 10 counts run of 200 episodes, these 10 are sampled from "
                         "the belief:";
            for (auto i = 0; i < 10; ++i)
            { static_cast<BAPOMDPState const*>(belief->sample())->logCounts(); }
        }

        WHEN("There is .2 noise and 2 counts")
        {

            conf.noise        = .2f;
            conf.counts_total = 10;
            REQUIRE_NOTHROW(conf.validate());

            auto const bapomdp = factory::makeTBAPOMDP(conf);
            auto const belief  = factory::makeBABelief(conf);

            belief->initiate(*bapomdp);
            belief->resetDomainStateDistribution(*bapomdp);

            LOG(INFO) << "This sample represents the count belief for .2 noise noise w/ 2 counts";
            static_cast<BAPOMDPState const*>(belief->sample())->logCounts();

            for (auto r = 0; r < conf.num_episodes; ++r)
            {
                belief->resetDomainStateDistribution(*bapomdp);
                episode::run(*planner, *belief, *env, *bapomdp, h, discount);
            }

            LOG(INFO) << "After .2 noise 2 counts run of 200 episodes, these 10 are sampled from "
                         "the belief:";
            for (auto i = 0; i < 10; ++i)
            { static_cast<BAPOMDPState const*>(belief->sample())->logCounts(); }
        }
    }
}

SCENARIO("factored tiger BAPOMDPPrior", "[bayes-adaptive][factored][tiger][flat]")
{
    float known_counts = 5000;
    // tests should work for any size
    int num_features = rnd::slowRandomInt(1, 5);
    auto S           = 2 << num_features;

    auto tiger_types = std::vector<domains::FactoredTiger::FactoredTigerDomainType>(
        {domains::FactoredTiger::FactoredTigerDomainType::EPISODIC,
         domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS});

    for (auto type : tiger_types)
    {

        // test for any amount of noise
        for (auto n : std::vector<float>({0, .1, .2}))
        {
            GIVEN(
                "the BAPOMDPPrior for the factored tiger problem " + std::to_string(type)
                + " of size " + std::to_string(num_features) + " with noise " + std::to_string(n))
            {

                auto conf               = configurations::FBAConf();
                conf.domain_conf.domain = "episodic-factored-tiger";
                conf.counts_total       = 40;
                conf.noise              = n;
                conf.domain_conf.size   = num_features;

                auto const d = domains::FactoredTiger(type, num_features);
                auto const ext =
                    bayes_adaptive::domain_extensions::FactoredTigerBAExtension(type, num_features);
                auto const p = factory::makeTBAPOMDPPrior(d, conf);

                auto domain_state = d.sampleStartState();
                auto ba_s         = static_cast<BAPOMDPState*>(p->sample(domain_state));

                THEN("the observation prior counts for listening must be according to the noise")
                {
                    auto listen = IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OBSERVE));

                    auto hear_right =
                             IndexObservation(std::to_string(domains::FactoredTiger::TigerLocation::RIGHT)),
                         hear_left = IndexObservation(std::to_string(domains::FactoredTiger::TigerLocation::LEFT));

                    for (auto s = 0; s < S; ++s)
                    {
                        auto state = ext.getState(std::to_string(s));

                        auto hear_correct =
                            (d.tigerLocation(state) == domains::FactoredTiger::TigerLocation::LEFT)
                                ? &hear_left
                                : &hear_right;

                        auto hear_false =
                            (d.tigerLocation(state) == domains::FactoredTiger::TigerLocation::LEFT)
                                ? &hear_right
                                : &hear_left;

                        REQUIRE(
                            ba_s->model()->count(&listen, state, hear_correct)
                            == conf.counts_total * (.85f - n));
                        REQUIRE(
                            ba_s->model()->count(&listen, state, hear_false)
                            == conf.counts_total * (.15f + n));

                        d.releaseState(state);
                    }
                }

                THEN("the observation counts for opening doors is always 10000/10000")
                {
                    for (auto s = 0; s < S; ++s)
                    {
                        auto state = ext.getState(std::to_string(s));

                        for (auto o = 0; o < 2; ++o)
                        {
                            auto observation = IndexObservation(std::to_string(o));

                            auto action =
                                IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OPEN_LEFT));
                            REQUIRE(
                                ba_s->model()->count(&action, state, &observation) == known_counts);

                            action = IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OPEN_RIGHT));
                            REQUIRE(
                                ba_s->model()->count(&action, state, &observation) == known_counts);
                        }

                        d.releaseState(state);
                    }
                }

                THEN("the transition counts for listening must be deterministic")
                {

                    auto listen = IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OBSERVE));

                    for (auto s = 0; s < S; ++s)
                    {
                        auto state = ext.getState(std::to_string(s));

                        for (auto new_s = 0; new_s < S; ++new_s)
                        {
                            auto new_state = ext.getState(std::to_string(new_s));

                            auto expected_count =
                                (state->index() == new_state->index()) ? known_counts : 0;
                            REQUIRE(
                                ba_s->model()->count(state, &listen, new_state) == expected_count);

                            d.releaseState(new_state);
                        }

                        d.releaseState(state);
                    }
                }

                THEN("the transition counts for opening door is uniform")
                {
                    for (auto s = 0; s < S; ++s)
                    {
                        auto state = ext.getState(std::to_string(s));

                        for (auto new_s = 0; new_s < S; ++new_s)
                        {
                            auto new_state = ext.getState(std::to_string(new_s));

                            auto open_door =
                                IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OPEN_LEFT));
                            REQUIRE(
                                ba_s->model()->count(state, &open_door, new_state) == known_counts);

                            open_door =
                                IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OPEN_RIGHT));
                            REQUIRE(
                                ba_s->model()->count(state, &open_door, new_state) == known_counts);
                        }
                    }
                }

                d.releaseState(domain_state);
                delete (ba_s);
            }
        }
    }
}

SCENARIO("factored tiger FBAPOMDPPrior", "[bayes-adaptive][factored][tiger]")
{
    float known_counts = 5000;
    // test should hold for any number of features
    int num_features = rnd::slowRandomInt(1, 5);

    auto tiger_types = std::vector<domains::FactoredTiger::FactoredTigerDomainType>(
        {domains::FactoredTiger::FactoredTigerDomainType::EPISODIC,
         domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS});

    for (auto type : tiger_types)
    {
        for (std::string const& struct_noise_enabled : {"", "uniform"})
        {
            auto descr = struct_noise_enabled.empty() ? " without structure noise "
                                                      : " with structure noise ";

            for (auto n : std::vector<float>({0, .1, .2}))
            {
                GIVEN(
                    "the FBAPOMDPPrior for the factored tiger problem " + std::to_string(type)
                    + descr + " of size " + std::to_string(num_features) + " with noise "
                    + std::to_string(n))
                {
                    auto conf             = configurations::FBAConf();
                    conf.domain_conf.size = num_features;
                    conf.counts_total     = 40;
                    conf.noise            = n;
                    conf.structure_prior  = struct_noise_enabled;

                    conf.domain_conf.domain =
                        type == domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS
                            ? "continuous"
                            : "episodic";
                    conf.domain_conf.domain = conf.domain_conf.domain + "-factored-tiger";

                    auto const d = domains::FactoredTiger(type, conf.domain_conf.size);
                    auto const p = priors::FactoredTigerFactoredPrior(conf);

                    auto domain_state = d.sampleStartState();
                    auto ba_s         = static_cast<FBAPOMDPState*>(p.sample(domain_state));

                    auto tiger_left_state =
                        std::vector<int>({domains::FactoredTiger::TigerLocation::LEFT});
                    auto tiger_right_state =
                        std::vector<int>({domains::FactoredTiger::TigerLocation::RIGHT});

                    WHEN("we inspect the observation function")
                    {

                        for (auto a = 0; a < 2; ++a)
                        {
                            THEN(
                                "the observation node for opening door " + std::to_string(a)
                                + " has no parents")
                            {
                                auto open_door = IndexAction(std::to_string(a));

                                auto node = ba_s->model()->observationNode(&open_door, 0);

                                REQUIRE(node.range() == 2);
                                REQUIRE(node.numParams() == 2); // no parents:: range == numparams

                                AND_THEN("those counts are uniform")
                                {
                                    REQUIRE(
                                        node.count(
                                            tiger_left_state,
                                            domains::FactoredTiger::TigerLocation::LEFT)
                                        == known_counts);
                                    REQUIRE(
                                        node.count(
                                            tiger_left_state,
                                            domains::FactoredTiger::TigerLocation::RIGHT)
                                        == known_counts);
                                    REQUIRE(
                                        node.count(
                                            tiger_right_state,
                                            domains::FactoredTiger::TigerLocation::LEFT)
                                        == known_counts);
                                    REQUIRE(
                                        node.count(
                                            tiger_right_state,
                                            domains::FactoredTiger::TigerLocation::RIGHT)
                                        == known_counts);
                                }
                            }
                        }

                        THEN(
                            "the observation node for listening has only 1 parent: the tiger "
                            "location")
                        {
                            if (struct_noise_enabled.empty()) // these tests dont work when there
                                                              // might be different structures
                            {
                                auto listen =
                                    IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OBSERVE));

                                auto node = ba_s->model()->observationNode(&listen, 0);

                                REQUIRE(node.range() == 2);
                                REQUIRE(node.numParams() == 4); // no parents:: range == numparams

                                AND_THEN("the counts for hearing depends on the noise")
                                {
                                    REQUIRE(
                                        node.count(
                                            tiger_left_state,
                                            domains::FactoredTiger::TigerLocation::LEFT)
                                        == conf.counts_total * (.85f - n));

                                    REQUIRE(
                                        node.count(
                                            tiger_right_state,
                                            domains::FactoredTiger::TigerLocation::RIGHT)
                                        == conf.counts_total * (.85f - n));

                                    REQUIRE(
                                        node.count(
                                            tiger_left_state,
                                            domains::FactoredTiger::TigerLocation::RIGHT)
                                        == conf.counts_total * (.15f + n));

                                    REQUIRE(
                                        node.count(
                                            tiger_right_state,
                                            domains::FactoredTiger::TigerLocation::LEFT)
                                        == conf.counts_total * (.15f + n));
                                }
                            }
                        }
                    }

                    WHEN("we inspect the transition function")
                    {

                        for (auto a = 0; a < 2; ++a)
                        {

                            for (auto f = 0; f < num_features + 1; ++f)
                            {

                                THEN(
                                    "node for feature " + std::to_string(f) + " when opening door "
                                    + std::to_string(a) + " are independent of any parent")
                                {
                                    auto open_door = IndexAction(std::to_string(a));

                                    auto node = ba_s->model()->transitionNode(&open_door, f);

                                    REQUIRE(node.range() == 2);
                                    REQUIRE(node.numParams() == 2);

                                    AND_THEN("those counts are uniform")
                                    {
                                        REQUIRE(node.count(tiger_left_state, 0) == known_counts);
                                        REQUIRE(node.count(tiger_left_state, 1) == known_counts);
                                        REQUIRE(node.count(tiger_right_state, 0) == known_counts);
                                        REQUIRE(node.count(tiger_right_state, 1) == known_counts);
                                    }
                                }
                            }
                        }

                        for (auto f = 0; f < num_features + 1; ++f)
                        {
                            THEN(
                                "node " + std::to_string(f)
                                + " for listening are only dependent on itself")
                            {

                                auto listen =
                                    IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OBSERVE));

                                auto node = ba_s->model()->transitionNode(&listen, f);

                                REQUIRE(node.range() == 2);
                                REQUIRE(node.numParams() == 4);

                                AND_THEN(
                                    "those counts are deterministicly staying in the same state")
                                {
                                    REQUIRE(
                                        node.count(
                                            tiger_left_state,
                                            domains::FactoredTiger::TigerLocation::LEFT)
                                        == known_counts);
                                    REQUIRE(
                                        node.count(
                                            tiger_left_state,
                                            domains::FactoredTiger::TigerLocation::RIGHT)
                                        == 0);
                                    REQUIRE(
                                        node.count(
                                            tiger_right_state,
                                            domains::FactoredTiger::TigerLocation::LEFT)
                                        == 0);
                                    REQUIRE(
                                        node.count(
                                            tiger_right_state,
                                            domains::FactoredTiger::TigerLocation::RIGHT)
                                        == known_counts);
                                }
                            }
                        }
                    }
                    d.releaseState(domain_state);
                    delete (ba_s);
                }
            }
        }
    }
}

SCENARIO("fully connected factored tiger model", "[bayes-adaptive][domains][tiger][factored]")
{
    size_t const size(5);

    configurations::FBAConf c;

    c.domain_conf.domain = "continuous-factored-tiger";
    c.domain_conf.size   = size;

    auto const d =
        domains::FactoredTiger(domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS, size);

    GIVEN("a model")
    {
        auto const p            = factory::makeFBAPOMDPPrior(d, c);
        auto const domain_state = d.sampleStartState();
        auto const hyper_state  = p->sampleFullyConnectedState(domain_state);

        IndexAction random_action(std::to_string(domains::FactoredTiger::OBSERVE));
        REQUIRE(
            *hyper_state->model()->observationNode(&random_action, 0).parents()
            == std::vector<int>({0, 1, 2, 3, 4, 5}));

        d.releaseState(domain_state);
        delete (hyper_state);
    }
}

SCENARIO(
    "match-uniform structure prior for factored tiger",
    "[bayes-adaptive][domains][tiger][factored]")
{

    auto const size = rnd::slowRandomInt(1, 5);

    configurations::FBAConf c;

    c.domain_conf.domain = "episodic-factored-tiger";
    c.domain_conf.size   = size;
    c.structure_prior    = "match-uniform";

    auto const d =
        domains::FactoredTiger(domains::FactoredTiger::FactoredTigerDomainType::EPISODIC, size);
    auto const p = factory::makeFBAPOMDPPrior(d, c);

    GIVEN("a model sampled from the prior with match-uniform structure noise")
    {

        IndexAction a(std::to_string(domains::FactoredTiger::TigerAction::OBSERVE));

        for (auto i = 0; i < 10; ++i)
        {
            auto const domain_state = d.sampleStartState();
            auto const hyper_state  = static_cast<FBAPOMDPState const*>(p->sample(domain_state));
            auto const& model       = hyper_state->model();

            REQUIRE(model->observationNode(&a, 0).parents()->at(0) == 0);

            d.releaseState(domain_state);
            delete (hyper_state);
        }
    }
}
