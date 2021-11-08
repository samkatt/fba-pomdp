#include "catch.hpp"

#include "easylogging++.h"

#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/BAConf.hpp"
#include "configurations/FBAConf.hpp"
#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "domains/collision-avoidance/CollisionAvoidanceBAExtension.hpp"
#include "domains/collision-avoidance/CollisionAvoidanceFBAExtension.hpp"
#include "domains/collision-avoidance/CollisionAvoidancePriors.hpp"
#include "utils/random.hpp"

SCENARIO(
    "collision-avoidance bayes-adaptive table prior",
    "[domain][bayes-adaptive][collision-avoidance][flat]")
{

    auto conf       = configurations::BAConf();
    auto const w    = 5;
    auto const h    = 7;
    auto const size = 1;

    conf.domain_conf.domain = "random-collision-avoidance";
    conf.domain_conf.width  = w;
    conf.domain_conf.height = h;
    conf.domain_conf.size   = size;

    conf.noise        = 0;
    conf.counts_total = static_cast<float>(rnd::slowRandomInt(3, 20));

    auto const d = domains::CollisionAvoidance(
        conf.domain_conf.width, conf.domain_conf.height, conf.domain_conf.size);
    auto const ext = bayes_adaptive::domain_extensions::CollisionAvoidanceBAExtension(
        conf.domain_conf.width, conf.domain_conf.height, conf.domain_conf.size);
    auto const p = priors::CollisionAvoidanceTablePrior(d, conf);

    auto const S = ext.domainSize()._S;

    auto const s = d.sampleStartState();

    GIVEN("a sampled bayes-adaptive state from the collision-avoidance domain")
    {
        auto const ba_state = dynamic_cast<BAPOMDPState const*>(p.sample(s));

        WHEN("Testing prior on the behaviour of the block")
        {
            auto const stay_action = d.getAction(domains::CollisionAvoidance::STAY);

            AND_WHEN("Block is somewhere in the middle")
            {
                auto const x         = rnd::slowRandomInt(1, w);
                auto const y         = rnd::slowRandomInt(1, h);
                auto const block_pos = rnd::slowRandomInt(1, h - 1); // block in middle

                auto const state = d.getState(x, y, {block_pos});

                auto const stay_state = d.getState(x - 1, y, {block_pos});
                auto const up_state   = d.getState(x - 1, y, {block_pos + 1});
                auto const down_state = d.getState(x - 1, y, {block_pos - 1});

                REQUIRE(
                    ba_state->model()->transitionExpectation(
                        state, stay_action)[stay_state->index()]
                    == domains::CollisionAvoidance::BLOCK_MOVE_PROB);
                REQUIRE(
                    ba_state->model()->transitionExpectation(state, stay_action)[up_state->index()]
                    == ((1 - d.BLOCK_MOVE_PROB) / 2));
                REQUIRE(
                    ba_state->model()->transitionExpectation(
                        state, stay_action)[down_state->index()]
                    == ((1 - d.BLOCK_MOVE_PROB) / 2));

                d.releaseState(state);
                d.releaseState(stay_state);
                d.releaseState(up_state);
                d.releaseState(down_state);
            }

            AND_WHEN("Block is on the bottom")
            {
                auto const x         = rnd::slowRandomInt(1, w);
                auto const y         = rnd::slowRandomInt(1, h);
                auto const block_pos = 0; // block in on bottom

                auto const state = d.getState(x, y, {block_pos});

                auto const stay_state = d.getState(x - 1, y, {block_pos});
                auto const up_state   = d.getState(x - 1, y, {block_pos + 1});

                REQUIRE(
                    ba_state->model()->transitionExpectation(state, stay_action)[up_state->index()]
                    == ((1 - d.BLOCK_MOVE_PROB) / 2));
                REQUIRE(
                    ba_state->model()->transitionExpectation(
                        state, stay_action)[stay_state->index()]
                    == ((1 + d.BLOCK_MOVE_PROB) / 2));

                d.releaseState(state);
                d.releaseState(stay_state);
                d.releaseState(up_state);
            }

            AND_WHEN("Block is on the top")
            {

                auto const x         = rnd::slowRandomInt(1, w);
                auto const y         = rnd::slowRandomInt(1, h);
                auto const block_pos = h - 1; // block in on top

                auto const state = d.getState(x, y, {block_pos});

                auto const stay_state = d.getState(x - 1, y, {block_pos});
                auto const down_state = d.getState(x - 1, y, {block_pos - 1});

                REQUIRE(
                    ba_state->model()->transitionExpectation(
                        state, stay_action)[down_state->index()]
                    == ((1 - d.BLOCK_MOVE_PROB) / 2));
                REQUIRE(
                    ba_state->model()->transitionExpectation(
                        state, stay_action)[stay_state->index()]
                    == ((1 + d.BLOCK_MOVE_PROB) / 2));

                d.releaseState(state);
                d.releaseState(stay_state);
                d.releaseState(down_state);
            }

            d.releaseAction(stay_action);
        }

        WHEN("we sample steps using the counts")
        {

            for (auto i = 0; i < 10; ++i)
            {
                auto random_state  = ext.getState(rnd::slowRandomInt(0, S));
                auto random_action = d.generateRandomAction(random_state);

                // only care about those state than can actually step
                if (d.xAgent(random_state) == 0)
                {
                    continue;
                }

                auto new_state = ext.getState(ba_state->sampleStateIndex(
                    random_state, random_action, rnd::sample::Dir::sampleFromExpectedMult));

                THEN("the new x - position must be 1 less")
                REQUIRE(d.xAgent(new_state) == d.xAgent(random_state) - 1);

                THEN("the new y position should be within bounds")
                {
                    if (random_action->index() == d.MOVE_DOWN)
                    {
                        REQUIRE(d.yAgent(new_state) <= d.yAgent(random_state));
                    } else if (random_action->index() == d.MOVE_UP)
                    {
                        REQUIRE(d.yAgent(new_state) >= d.yAgent(random_state));
                    } else if (random_action->index() == d.STAY)
                    {
                        REQUIRE(d.yAgent(new_state) == d.yAgent(random_state));
                    }
                }

                THEN("the new y position of the block should be within bounds")
                REQUIRE(std::abs(d.yObstacles(new_state)[0] - d.yObstacles(random_state)[0]) <= 1);

                d.releaseAction(random_action);
                d.releaseState(random_state);
                d.releaseState(new_state);
            }
        }

        delete (ba_state);
    }

    d.releaseState(s);
}

SCENARIO(
    "print collision-avoidance prior",
    "[domain][bayes-adaptive][flat][collision-avoidance][hide][logging]")
{

    auto conf = configurations::BAConf();

    conf.domain_conf.domain = "random-collision-avoidance";
    conf.domain_conf.width  = 2;
    conf.domain_conf.height = 3;
    conf.domain_conf.size   = 2;

    conf.noise        = -.3f;
    conf.counts_total = 10;

    auto const d = domains::CollisionAvoidance(
        conf.domain_conf.width, conf.domain_conf.height, conf.domain_conf.size);
    auto const p = factory::makeTBAPOMDPPrior(d, conf);

    LOG(INFO) << "Printing a sample BAPOMDPState prior";

    auto const domain_state = d.sampleStartState();
    auto const ba_state     = p->sample(domain_state);

    ba_state->logCounts();
}

SCENARIO(
    "collision-avoidance bayes-adaptive factored prior",
    "[domain][bayes-adaptive][collision-avoidance][factored]")
{
    auto const w = 4;
    auto const h = 7;
    auto const s = 2;

    auto conf = configurations::FBAConf();

    conf.noise              = 0;
    conf.counts_total       = 1000;
    conf.domain_conf.width  = w;
    conf.domain_conf.height = h;
    conf.domain_conf.size   = s;
    conf.domain_conf.domain = "random-collision-avoidance";

    auto const d   = domains::CollisionAvoidance(w, h, s);
    auto const ext = bayes_adaptive::domain_extensions::CollisionAvoidanceBAExtension(w, h, s);

    auto const S = ext.domainSize()._S;

    GIVEN("a sampled bayes-adaptive state from the collision-avoidance domain with perfect prior")
    {

        auto const p        = factory::makeFBAPOMDPPrior(d, conf);
        auto const state    = d.sampleStartState();
        auto const ba_state = dynamic_cast<FBAPOMDPState const*>(p->sample(state));

        WHEN("Testing prior on the behaviour of the block")
        {
            auto const random_action = d.generateRandomAction(state);

            AND_WHEN("Block is somewhere in the middle")
            {
                auto const block_pos = rnd::slowRandomInt(1, h - 1); // block in middle

                REQUIRE(
                    ba_state->model()
                        ->transitionNode(random_action, 2)
                        .expectation({block_pos})[block_pos]
                    == Approx(d.BLOCK_MOVE_PROB));
                REQUIRE(
                    ba_state->model()
                        ->transitionNode(random_action, 2)
                        .expectation({block_pos})[block_pos + 1]
                    == Approx(d.BLOCK_MOVE_PROB * .5));
                REQUIRE(
                    ba_state->model()
                        ->transitionNode(random_action, 2)
                        .expectation({block_pos})[block_pos - 1]
                    == Approx(d.BLOCK_MOVE_PROB * .5));
            }

            AND_WHEN("Block is on the bottom")
            {
                auto const block_pos = 0; // block on bottom

                REQUIRE(
                    ba_state->model()
                        ->transitionNode(random_action, 2)
                        .expectation({block_pos})[block_pos]
                    == Approx(.5 * (1 + d.BLOCK_MOVE_PROB)));
                REQUIRE(
                    ba_state->model()
                        ->transitionNode(random_action, 2)
                        .expectation({block_pos})[block_pos + 1]
                    == Approx((.5 * (1 - d.BLOCK_MOVE_PROB))));
            }

            AND_WHEN("Block is on the top")
            {
                auto const block_pos = h - 1; // block on bottom

                REQUIRE(
                    ba_state->model()
                        ->transitionNode(random_action, 2)
                        .expectation({block_pos})[block_pos]
                    == Approx(.5 * (1 + d.BLOCK_MOVE_PROB)));
                REQUIRE(
                    ba_state->model()
                        ->transitionNode(random_action, 2)
                        .expectation({block_pos})[block_pos - 1]
                    == Approx((.5 * (1 - d.BLOCK_MOVE_PROB))));
            }

            d.releaseAction(random_action);
        }

        WHEN("Testing the prior on the agent's movement")
        {
            auto const random_action = d.generateRandomAction(state);
            auto const random_x      = rnd::slowRandomInt(1, w);
            auto const random_y      = rnd::slowRandomInt(1, h - 1);
            auto const top           = h - 1;
            auto const bottom        = 0;

            REQUIRE(
                ba_state->model()
                    ->transitionNode(random_action, 0)
                    .expectation({random_x})[random_x - 1]
                == 1.);

            d.releaseAction(random_action);

            auto const up_action   = d.getAction(domains::CollisionAvoidance::MOVE_UP);
            auto const down_action = d.getAction(domains::CollisionAvoidance::MOVE_DOWN);
            auto const stay_action = d.getAction(domains::CollisionAvoidance::STAY);

            REQUIRE(
                ba_state->model()
                    ->transitionNode(up_action, 1)
                    .expectation({random_y})[random_y + 1]
                == 1.);
            REQUIRE(
                ba_state->model()->transitionNode(stay_action, 1).expectation({random_y})[random_y]
                == 1.);
            REQUIRE(
                ba_state->model()
                    ->transitionNode(down_action, 1)
                    .expectation({random_y})[random_y - 1]
                == 1.);

            REQUIRE(ba_state->model()->transitionNode(up_action, 1).expectation({top})[top] == 1.);
            REQUIRE(
                ba_state->model()->transitionNode(down_action, 1).expectation({bottom})[bottom]
                == 1.);

            d.releaseAction(up_action);
            d.releaseAction(stay_action);
            d.releaseAction(down_action);
        }

        WHEN("we sample steps using the counts")
        {

            for (auto i = 0; i < 10; ++i)
            {
                auto random_state  = ext.getState(rnd::slowRandomInt(0, S));
                auto random_action = d.generateRandomAction(random_state);

                // only care about those state than can actually step
                if (d.xAgent(random_state) == 0)
                {
                    continue;
                }

                auto new_state = ext.getState(ba_state->sampleStateIndex(
                    random_state, random_action, rnd::sample::Dir::sampleFromExpectedMult));

                THEN("the new x - position must be 1 less")
                REQUIRE(d.xAgent(new_state) == d.xAgent(random_state) - 1);

                THEN("the new y position should be within bounds")
                {
                    if (random_action->index() == d.MOVE_DOWN)
                    {
                        REQUIRE(d.yAgent(new_state) <= d.yAgent(random_state));
                    } else if (random_action->index() == d.MOVE_UP)
                    {
                        REQUIRE(d.yAgent(new_state) >= d.yAgent(random_state));
                    } else if (random_action->index() == d.STAY)
                    {
                        REQUIRE(d.yAgent(new_state) == d.yAgent(random_state));
                    }
                }

                THEN("the new y position of the block should be within bounds")
                REQUIRE(std::abs(d.yObstacles(new_state)[0] - d.yObstacles(random_state)[0]) <= 1);

                d.releaseAction(random_action);
                d.releaseState(random_state);
                d.releaseState(new_state);
            }
        }
        d.releaseState(state);
        delete (ba_state);
    }

    GIVEN("a sampled bayes-adaptive state from the collision-avoidance domain with perfect prior")
    {
        auto const p        = factory::makeFBAPOMDPPrior(d, conf);
        auto const state    = d.sampleStartState();
        auto const ba_state = p->sample(state);

        WHEN("we sample steps using the counts")
        {

            for (auto i = 0; i < 10; ++i)
            {
                auto random_state  = ext.getState(rnd::slowRandomInt(0, S));
                auto random_action = d.generateRandomAction(random_state);

                // only care about those state than can actually step
                if (d.xAgent(random_state) == 0)
                {
                    continue;
                }

                auto new_state = ext.getState(ba_state->sampleStateIndex(
                    random_state, random_action, rnd::sample::Dir::sampleFromExpectedMult));

                THEN("the new x - position must be 1 less")
                REQUIRE(d.xAgent(new_state) == d.xAgent(random_state) - 1);

                THEN("the new y position should be within bounds")
                {
                    if (random_action->index() == d.MOVE_DOWN)
                    {
                        REQUIRE(d.yAgent(new_state) <= d.yAgent(random_state));
                    } else if (random_action->index() == d.MOVE_UP)
                    {
                        REQUIRE(d.yAgent(new_state) >= d.yAgent(random_state));
                    } else if (random_action->index() == d.STAY)
                    {
                        REQUIRE(d.yAgent(new_state) == d.yAgent(random_state));
                    }
                }

                d.releaseAction(random_action);
                d.releaseState(random_state);
                d.releaseState(new_state);
            }
        }

        d.releaseState(state);
        delete (ba_state);
    }
}

SCENARIO(
    "print factored collision-avoidance prior",
    "[domain][bayes-adaptive][factored][collision-avoidance][hide][logging]")
{
    auto seed = std::to_string(1);
    rnd::seed(seed);

    auto const w = 5;
    auto const h = 5;
    auto const n = 2;

    auto conf = configurations::FBAConf();

    conf.noise           = -.3f;
    conf.counts_total    = 10;
    conf.structure_prior = "uniform";

    conf.domain_conf.width  = w;
    conf.domain_conf.height = h;
    conf.domain_conf.size   = n;
    conf.domain_conf.domain = "random-collision-avoidance";

    auto const d = domains::CollisionAvoidance(w, h, n);
    auto const p = factory::makeFBAPOMDPPrior(d, conf);

    LOG(INFO) << "Printing a sample FBAPOMDPState prior";

    auto const domain_state = d.sampleStartState();
    auto const ba_state     = p->sample(domain_state);

    ba_state->logCounts();
}

SCENARIO(
    "sample correct graph from collision avoidance FBAPOMDP prior",
    "[domain][bayes-adaptive][factored][collision-avoidance]")
{
    GIVEN("collision avoidance knowing structure factored prior")
    {

        auto const h = 5, w = 5, n = 1;
        auto c = configurations::FBAConf();

        c.domain_conf.domain = "random-collision-avoidance";
        c.domain_conf.width  = w;
        c.domain_conf.height = h;
        c.domain_conf.size   = n;

        auto const d   = domains::CollisionAvoidance(w, h, n);
        auto const ext = bayes_adaptive::domain_extensions::CollisionAvoidanceBAExtension(w, h, n);
        auto const f_ext =
            bayes_adaptive::domain_extensions::CollisionAvoidanceFBAExtension(w, h, n, d.type());
        auto const p = priors::CollisionAvoidanceFactoredPrior(c);
        auto const s = IndexState(0);

        auto const fbapomdp_state = p.sampleCorrectGraphState(&s);
        REQUIRE(fbapomdp_state != 0);

        auto const& model = fbapomdp_state->model();

        THEN("The number of values of the features should be equal to the grid size")
        {
            auto const a = IndexAction(rnd::slowRandomInt(0, ext.domainSize()._A));

            REQUIRE(model->transitionNode(&a, 0).range() == 5);
            REQUIRE(model->transitionNode(&a, 1).range() == 5);
            REQUIRE(model->transitionNode(&a, 2).range() == 5);
        }

        THEN("The parents of the transition nodes should be themselves")
        {
            for (auto a = 0; a < ext.domainSize()._A; ++a)
            {
                auto const action = IndexAction(std::to_string(a));
                for (auto feature = 0;
                     feature < static_cast<int>(f_ext.domainFeatureSize()._S.size());
                     ++feature)
                {
                    REQUIRE(
                        *model->transitionNode(&action, feature).parents()
                        == std::vector<int>({feature}));
                }
            }
        }

        THEN("The parents of the observation feature should be the obstacle")
        {
            for (auto a = 0; a < ext.domainSize()._A; ++a)
            {
                auto const action = IndexAction(std::to_string(a));
                REQUIRE(*model->observationNode(&action, 0).parents() == std::vector<int>({2}));
            }
        }

        delete (fbapomdp_state);
    }
}

SCENARIO(
    "compute prior of specific BN model topology",
    "[domain][bayes-adaptive][factored][collision-avoidance]")
{

    GIVEN("the collision avoidance BA-POMDP")
    {

        auto const h = 5, w = 5, n = 1;
        auto c               = configurations::FBAConf();
        c.domain_conf.width  = w;
        c.domain_conf.height = h;
        c.domain_conf.size   = n;
        c.domain_conf.domain = "random-collision-avoidance";

        auto const d = domains::CollisionAvoidance(w, h, n);
        auto const p = factory::makeFBAPOMDPPrior(d, c);
        auto const s = IndexState(0);

        auto const fbapomdp_template_state = p->sampleCorrectGraphState(&s);

        REQUIRE_NOTHROW(p->computePriorModel(fbapomdp_template_state->model()->structure()));
        THEN(
            "we expect the computed prior of the correct graphs to be the same as sampled graph "
            "from the prior")
        {
            auto const correct_topologies = fbapomdp_template_state->model()->structure();
            auto const correct_prior      = p->computePriorModel(correct_topologies);

            auto const random_feature = rnd::slowRandomInt(0, 3);
            auto const random_action  = IndexAction(rnd::slowRandomInt(0, 3));

            // some silly /trivial tests to make sure at least something is okay
            REQUIRE(
                correct_prior.copyT().size() == fbapomdp_template_state->model()->copyT().size());
            REQUIRE(
                correct_prior.copyO().size() == fbapomdp_template_state->model()->copyO().size());

            REQUIRE(
                correct_prior.transitionNode(&random_action, random_feature).numParams()
                == fbapomdp_template_state->model()
                       ->transitionNode(&random_action, random_feature)
                       .numParams());

            REQUIRE(
                correct_prior.transitionNode(&random_action, random_feature).numParams()
                == fbapomdp_template_state->model()
                       ->transitionNode(&random_action, random_feature)
                       .numParams());

            REQUIRE(
                correct_prior.observationNode(&random_action, 0).numParams()
                == fbapomdp_template_state->model()
                       ->observationNode(&random_action, 0)
                       .numParams());
        }

        delete (fbapomdp_template_state);
    }
}
