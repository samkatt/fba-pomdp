#include "catch.hpp"

#include <cmath>
#include <string>

#include "easylogging++.h"

#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
#include "utils/random.hpp"

SCENARIO("sample avoidance collision start state", "[environment][collision-avoidance]")
{
    auto const width  = 4;
    auto const height = 3;

    auto const d = domains::CollisionAvoidance(width, height);

    auto const sample_state = d.sampleStartState();

    REQUIRE(d.xAgent(sample_state) == width - 1);
}

SCENARIO("collision avoidance step", "[environment][collision-avoidance]")
{
    GIVEN("the collision avoidance environment")
    {

        auto const width  = 6;
        auto const height = 3;

        auto const d = domains::CollisionAvoidance(width, height);

        Observation const* o = nullptr;
        Reward r(0);

        WHEN("we perform a random action")
        {
            auto const x = rnd::slowRandomInt(2, width);

            auto s = d.getState(x, rnd::slowRandomInt(0, height), {rnd::slowRandomInt(0, height)});
            auto const a = d.generateRandomAction(s);

            auto t = d.step(&s, a, &o, &r);

            THEN("we expect to be one step to the left")
            {
                REQUIRE(d.xAgent(s) == x - 1);
                REQUIRE(!t.terminated());
            }

            d.releaseState(s);
            d.releaseAction(a);
            d.releaseObservation(o);
        }

        WHEN("we reach the end of the grid during a step")
        {
            auto s = d.getState(1, rnd::slowRandomInt(0, height), {rnd::slowRandomInt(0, height)});

            auto const a = d.generateRandomAction(s);

            auto t = d.step(&s, a, &o, &r);

            THEN("We expect it to terminate")
            REQUIRE(t.terminated());

            d.releaseAction(a);
            d.releaseObservation(o);
            d.releaseState(s);
        }

        WHEN("the agent moves up ")
        {
            auto const y = rnd::slowRandomInt(0, height - 1);

            auto s = d.getState(rnd::slowRandomInt(1, width), y, {rnd::slowRandomInt(0, height)});
            auto const a = d.getAction(d.MOVE_UP);

            d.step(&s, a, &o, &r);

            THEN("this cost -1 and we end up 1 higher")
            {
                REQUIRE(d.yAgent(s) == y + 1);
                if (d.xAgent(s) != 0 || d.yObstacles(s)[0] != d.yAgent(s))
                {
                    REQUIRE(r.toDouble() == -1);
                } else
                {
                    REQUIRE(r.toDouble() == -1000);
                }
            }
        }

        WHEN("the agent moves down while at the bottom")
        {

            auto s = d.getState(rnd::slowRandomInt(1, width), 0, {rnd::slowRandomInt(2, height)});
            auto const a = d.getAction(d.MOVE_DOWN);

            d.step(&s, a, &o, &r);

            THEN("this cost -1 and we end up 1 higher")
            {
                REQUIRE(d.yAgent(s) == 0);
                if (d.yObstacles(s)[0] != d.yAgent(s) || d.xAgent(s) != 0)
                {
                    REQUIRE(r.toDouble() == -1);
                } else
                {
                    REQUIRE(r.toDouble() == -1000);
                }
            }

            d.releaseState(s);
            d.releaseObservation(o);
            d.releaseAction(a);
        }

        WHEN("the agent does not change its vertical position")
        {
            auto const y = rnd::slowRandomInt(0, height);

            auto s = d.getState(rnd::slowRandomInt(2, width), y, {rnd::slowRandomInt(0, height)});
            auto const a = d.getAction(d.STAY);

            d.step(&s, a, &o, &r);

            THEN("this cost 0 and we end up in the same spot")
            {
                REQUIRE(d.yAgent(s) == y);
                REQUIRE(r.toDouble() == 0);
            }

            d.releaseState(s);
            d.releaseObservation(o);
            d.releaseAction(a);
        }
    }
}

SCENARIO("when the agent runs into the object", "[environment][collision-avoidance]")
{
    GIVEN("a collision avoidance grid of height = 1")
    {

        auto const width  = 6;
        auto const height = 1;

        auto const d = domains::CollisionAvoidance(width, height);

        Observation const* o(nullptr);
        Reward r(0);

        WHEN("we perform a random action and do not reach the end")
        {
            auto const x = rnd::slowRandomInt(2, width);

            auto s = d.getState(x, rnd::slowRandomInt(0, height), {rnd::slowRandomInt(0, height)});
            auto const a = d.generateRandomAction(s);

            auto t = d.step(&s, a, &o, &r);

            THEN("we expect to be exactly 1 to the left")
            {
                REQUIRE(d.xAgent(s) == x - 1);
                REQUIRE(!t.terminated());
            }

            d.releaseState(s);
            d.releaseAction(a);
            d.releaseObservation(o);
        }

        WHEN("we reach the end of the grid during a step")
        {
            auto s = d.getState(1, rnd::slowRandomInt(0, height), {rnd::slowRandomInt(0, height)});
            auto const a = d.generateRandomAction(s);

            auto t = d.step(&s, a, &o, &r);

            THEN("We expect it to terminate with a huge penalty")
            {
                REQUIRE(r.toDouble() == -1000);
                REQUIRE(t.terminated());
            }

            d.releaseAction(a);
            d.releaseObservation(o);
            d.releaseState(s);
        }
    }
}

SCENARIO("generating random collision-avoidance actions", "[domain][collision-avoidance]")
{
    auto const d = domains::CollisionAvoidance(11, 11);
    auto const s = d.getState(
        rnd::slowRandomInt(0, 11), rnd::slowRandomInt(0, 11), {rnd::slowRandomInt(0, 11)});

    auto a = d.generateRandomAction(s);

    REQUIRE(std::stoi(a->index()) < 3);

    d.releaseState(s);
    d.releaseAction(a);
}

SCENARIO("adding legal actions in the collision avoidance domain", "[domain][collision-avoidance")
{
    auto const d = domains::CollisionAvoidance(4, 5);
    auto actions = std::vector<Action const*>();
    auto s       = d.sampleStartState();

    d.addLegalActions(s, &actions);

    REQUIRE(actions.size() == 3);
    for (auto i = 0; i < 3; i++) { REQUIRE(actions[i]->index() == i); }

    d.releaseState(s);
    for (auto const& a : actions) { d.releaseAction(a); }
}

SCENARIO("computing collision avoidance observation probability", "[domain][collision-avoidance]")
{
    auto const d = domains::CollisionAvoidance(11, 11);
    auto const s = d.getState(1, 5, {5});
    auto const a = d.generateRandomAction(s);

    auto o = d.getObservation(5);
    REQUIRE(d.computeObservationProbability(o, a, s) == Approx(.38292));

    o = d.getObservation(4);
    REQUIRE(d.computeObservationProbability(o, a, s) == Approx(.24173));

    o = d.getObservation(7);
    REQUIRE(d.computeObservationProbability(o, a, s) == Approx(.0606));

    o = d.getObservation(2);
    REQUIRE(d.computeObservationProbability(o, a, s) == Approx(.006).epsilon(.0002));

    o = d.getObservation(9);
    REQUIRE(d.computeObservationProbability(o, a, s) == Approx(.0003).epsilon(.0001));
}
