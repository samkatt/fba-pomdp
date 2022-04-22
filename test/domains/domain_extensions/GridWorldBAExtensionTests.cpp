#include "catch.hpp"

#include "domains/gridworld/GridWorld.hpp"
#include "domains/gridworld/GridWorldBAExtension.hpp"
#include "domains/gridworld/GridWorldFBAExtension.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "utils/random.hpp"

SCENARIO("gridworld extensions", "[bayes-adaptive][gridworld]")
{

    using pos = domains::GridWorld::GridWorldState::pos;

    auto const size = 4;
    domains::GridWorld d(size);
    bayes_adaptive::domain_extensions::GridWorldBAExtension ext(size);

    WHEN("performing terminating moves for gridworld")
    {

        Reward r(0);
        Observation const* o(nullptr);

        for (auto i = 0; i < 5; ++i)
        {

            auto const p = *d.goalLocation(rnd::slowRandomInt(0, d.goalLocations(size).size()));

            auto s           = static_cast<State const*>(d.getState(p, p));
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

    WHEN("performing non-terminating moves for gridworld")
    {

        Reward r(0);
        Observation const* o(nullptr);

        for (auto i = 0; i < 5; ++i)
        {

            // generate different agent and goal location
            auto const goal_pos =
                *d.goalLocation(rnd::slowRandomInt(0, d.goalLocations(size).size()));

            unsigned int x = rnd::slowRandomInt(0, size);
            unsigned int y = rnd::slowRandomInt(0, size);

            while (goal_pos == pos{x, y})
            {
                x = rnd::slowRandomInt(0, size);
                y = rnd::slowRandomInt(0, size);
            }

            pos const agent_pos{x, y};

            // check that any step with any action will not terminate
            auto s           = static_cast<State const*>(d.getState(agent_pos, goal_pos));
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

SCENARIO("gridworld domain sizes", "[domain][gridworld][bayes-adaptive]")
{

    auto const n = 6;

    auto const num_locations = n * n;

    domains::GridWorld d(n);
    bayes_adaptive::domain_extensions::GridWorldBAExtension ext(n);
    bayes_adaptive::domain_extensions::GridWorldFBAExtension f_ext(n);

    WHEN("asking the size of the gridworld domain")
    {
        auto const size = ext.domainSize();

        REQUIRE(size._A == 4);
        REQUIRE(static_cast<std::size_t>(size._O) == num_locations * d.goalLocations(n).size());
        REQUIRE(static_cast<std::size_t>(size._S) == num_locations * d.goalLocations(n).size());
    }

    WHEN("asking the domain feature size")
    {
        auto const size = f_ext.domainFeatureSize();

        REQUIRE(size._O.size() == 3);
        REQUIRE(size._O[0] == n);
        REQUIRE(size._O[1] == n);
        REQUIRE(static_cast<std::size_t>(size._O[2]) == d.goalLocations(n).size());

        REQUIRE(size._S.size() == 3);
        REQUIRE(size._S[0] == n);
        REQUIRE(size._S[1] == n);
        REQUIRE(static_cast<std::size_t>(size._S[2]) == d.goalLocations(n).size());
    }
}

SCENARIO("gridworld state prior", "[bayes-adaptive][gridworld][domain][factored]")
{
    auto const size      = rnd::slowRandomInt(3, 10);
    auto const num_goals = domains::GridWorld::goalLocations(size).size();

    bayes_adaptive::domain_extensions::GridWorldFBAExtension const f_ext(size);

    // I kno that the index of the state only depends on the goal index, hence cannot be >= num
    // goals
    REQUIRE(f_ext.statePrior()->sample() < num_goals);

    // probability of whatever sampled state should be 1/#goals
    REQUIRE(f_ext.statePrior()->prob(f_ext.statePrior()->sample()) == Approx(1. / num_goals));
}
