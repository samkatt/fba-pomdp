#include "catch.hpp"

#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "domains/collision-avoidance/CollisionAvoidanceBAExtension.hpp"
#include "domains/collision-avoidance/CollisionAvoidanceFBAExtension.hpp"
#include "utils/random.hpp"

SCENARIO("collision-avoidance state indexing", "[domain][bayes-adaptive][collision-avoidance]")
{
    WHEN("indexing collision-avoidance states")
    {
        auto const height = 7;
        auto const width  = rnd::slowRandomInt(1, 6);

        auto const d = domains::CollisionAvoidance(width, height, 1);
        auto const ext =
            bayes_adaptive::domain_extensions::CollisionAvoidanceBAExtension(width, height, 1);

        auto const x       = rnd::slowRandomInt(0, width);
        auto const y       = rnd::slowRandomInt(0, height);
        auto const y_block = rnd::slowRandomInt(0, height);

        auto const s = d.getState(x, y, {y_block});
        REQUIRE(std::stoi(s->index()) == x * height * height + y * height + y_block);

        auto const s_by_index = ext.getState(s->index());

        REQUIRE(d.xAgent(s_by_index) == x);
        REQUIRE(d.yAgent(s_by_index) == y);
        REQUIRE(d.yObstacles(s_by_index)[0] == y_block);

        d.releaseState(s);
        d.releaseState(s_by_index);
    }
}

SCENARIO(
    "collision-avoidance state prior",
    "[domain][bayes-adaptive][collision-avoidance][factored]")
{

    auto height = rnd::slowRandomInt(0, 5);

    if (height % 2 == 0) // ensure height is uneven
        height++;

    auto const width(height);
    auto const nr_obstacles = rnd::slowRandomInt(1, width);

    GIVEN("The centre collision avoidance state prior")
    {
        auto const version(domains::CollisionAvoidance::VERSION::INITIALIZE_CENTRE);
        auto const f_ext = bayes_adaptive::domain_extensions::CollisionAvoidanceFBAExtension(
            width, height, nr_obstacles, version);

        // there is only 1 possible state
        REQUIRE(f_ext.statePrior()->prob(f_ext.statePrior()->sample()) == 1);
    }

    GIVEN("The random collision avoidance state prior")
    {
        auto const version(domains::CollisionAvoidance::VERSION::INIT_RANDOM_POSITION);
        auto const f_ext = bayes_adaptive::domain_extensions::CollisionAvoidanceFBAExtension(
            width, height, nr_obstacles, version);

        // there is only 1 possible state
        REQUIRE(
            f_ext.statePrior()->prob(f_ext.statePrior()->sample())
            == Approx(1 / (std::pow(height, nr_obstacles + 1))));
    }
}

SCENARIO("collision avoidance reward", "[domain][bayes-adaptive][collision-avoidance]")
{
    auto const h = 5;
    auto const w = 5;
    auto const n = 2;
    auto const y = rnd::slowRandomInt(0, h);

    auto const d   = domains::CollisionAvoidance(w, h, n);
    auto const ext = bayes_adaptive::domain_extensions::CollisionAvoidanceBAExtension(w, h, n);

    WHEN("agent is colliding")
    {

        auto const collision_state_1 = d.getState(1, y, {rnd::slowRandomInt(0, h), y});
        auto const a                 = d.generateRandomAction(collision_state_1);

        REQUIRE(ext.reward(0, a, collision_state_1).toDouble() == Approx(-d.COLLIDE_PENALTY));
        REQUIRE(ext.terminal(0, a, collision_state_1).terminated());

        auto const collision_state_2 = d.getState(0, y, {y, rnd::slowRandomInt(0, h)});
        auto const a2                = d.generateRandomAction(collision_state_2);

        REQUIRE(ext.reward(0, a, collision_state_2).toDouble() == Approx(-d.COLLIDE_PENALTY));
        REQUIRE(ext.terminal(0, a, collision_state_2).terminated());

        d.releaseState(collision_state_1);
        d.releaseState(collision_state_2);
        d.releaseAction(a);
        d.releaseAction(a2);
    }
}
