//
// Created by rolf on 04-06-21.
//

#include "catch.hpp"

#include "domains/gridworld-coffee-trap/GridWorldCoffee.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffeeBAExtension.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffeeFBAExtension.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "utils/random.hpp"

SCENARIO("gridworldCoffee extensions", "[bayes-adaptive][gridworldCoffee]")
{

    using pos = domains::GridWorldCoffee::GridWorldCoffeeState::pos;

    domains::GridWorldCoffee d;
    bayes_adaptive::domain_extensions::GridWorldCoffeeBAExtension ext;

    WHEN("performing terminating moves for gridworldCoffee")
    {

        Reward r(0);
        Observation const* o(nullptr);

        for (auto i = 0; i < 5; ++i)
        {
            auto const p = domains::GridWorldCoffee::goal_location;

            auto s           = static_cast<State const*>(d.getState(p, 0, 0));
            auto const old_s = d.copyState(s);
            auto const a     = d.generateRandomAction(s);
            d.step(&s, a, &o, &r);

            REQUIRE(ext.terminal(old_s, a, s).terminated());
            REQUIRE(ext.reward(old_s, a, s).toDouble() == Approx(d.goalReward()));

            d.releaseState(s);
            d.releaseAction(a);
            d.releaseObservation(o);
            d.releaseState(old_s);
        }
    }

    WHEN("performing non-terminating moves for gridworldCoffee")
    {

        Reward r(0);
        Observation const* o(nullptr);

        for (auto i = 0; i < 5; ++i)
        {

            // generate different agent and goal location
            auto const goal_pos = domains::GridWorldCoffee::goal_location;

            unsigned int x = rnd::slowRandomInt(0, 5);
            unsigned int y = rnd::slowRandomInt(0, 5);

            while (goal_pos == pos{x, y})
            {
                x = rnd::slowRandomInt(0, 5);
                y = rnd::slowRandomInt(0, 5);
            }

            pos const agent_pos{x, y};

            // check that any step with any action will not terminate
            auto s           = static_cast<State const*>(d.getState(agent_pos, 0, 0));
            auto const old_s = d.copyState(s);
            auto const a     = d.generateRandomAction(s);
            d.step(&s, a, &o, &r);

            REQUIRE(!ext.terminal(old_s, a, s).terminated());
            REQUIRE(ext.reward(old_s, a, s).toDouble() == Approx(0));

            d.releaseAction(a);
            d.releaseState(s);
            d.releaseState(old_s);
            d.releaseObservation(o);
        }
    }
}

SCENARIO("gridworldCoffee domain sizes", "[domain][gridworldCoffee][bayes-adaptive]")
{

    domains::GridWorldCoffee d;
    bayes_adaptive::domain_extensions::GridWorldCoffeeBAExtension ext;
    bayes_adaptive::domain_extensions::GridWorldCoffeeFBAExtensionBig f_ext;

    WHEN("asking the size of the gridworldCoffee domain")
    {
        auto const size = ext.domainSize();

        REQUIRE(size._A == 4);
        REQUIRE(size._O == 5 * 5 * 2 * 2);
        REQUIRE(size._S == 5 * 5 * 2 * 2);
    }

    WHEN("asking the domain feature size")
    {
        auto const size = f_ext.domainFeatureSize();

        REQUIRE(size._O.size() == 4);
        REQUIRE(size._O[0] == 5);
        REQUIRE(size._O[1] == 5);
        REQUIRE(size._O[2] == 2);
        REQUIRE(size._O[3] == 2);

        REQUIRE(size._S.size() == 4);
        REQUIRE(size._S[0] == 5);
        REQUIRE(size._S[1] == 5);
        REQUIRE(size._S[2] == 2);
        REQUIRE(size._S[3] == 2);
    }
}

SCENARIO("gridworldCoffee state prior", "[bayes-adaptive][gridworldCoffee][domain][factored]")
{

    bayes_adaptive::domain_extensions::GridWorldCoffeeFBAExtensionBig const f_ext;

    // I know that the index of the state should be deterministic
    REQUIRE(f_ext.statePrior()->sample() == 0);

    // probability of whatever sampled state should be 1
    REQUIRE(f_ext.statePrior()->prob(f_ext.statePrior()->sample()) == Approx(1));
}
