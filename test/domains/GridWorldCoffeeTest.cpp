//
// Created by rolf on 04-06-21.
//

#include "catch.hpp"

#include <algorithm>
#include <string>

#include "configurations/DomainConf.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffee.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
#include "utils/random.hpp"

double const domains::GridWorldCoffee::step_reward;

SCENARIO("GridWorldCoffee actions, observations & states", "[domain][gridworldcoffee]")
{

    WHEN("Testing the position in a grid world coffee state")
    {
        using pos = domains::GridWorldCoffee::GridWorldCoffeeState::pos;

        unsigned int x(rnd::slowRandomInt(0, 5));
        unsigned int y(rnd::slowRandomInt(0, 5));
        unsigned int y2(9);

        REQUIRE((pos{x, y} == pos{x, y}));
        REQUIRE((pos{x, y} != pos{x, y2}));
    }

    WHEN("Calculating the observation probabilities")
    {

        domains::GridWorldCoffee d;

        REQUIRE(d.obsDisplProb(0, 0) == Approx(.95));
        REQUIRE(d.obsDisplProb(0, 1) == Approx(.05));
        REQUIRE(d.obsDisplProb(0, 2) == Approx(.0));

        REQUIRE(d.obsDisplProb(1, 1) == Approx(.9));
        REQUIRE(d.obsDisplProb(1, 0) == Approx(.05));

        REQUIRE(d.obsDisplProb(2, 3) == Approx(.05));

        REQUIRE(d.obsDisplProb(4, 4) == Approx(.95));
        REQUIRE(d.obsDisplProb(4, 2) == Approx(.0));
    }
}

SCENARIO("GridWorldCoffee dynamics", "[domain][gridworldcoffee]")
{

    using pos     = domains::GridWorldCoffee::GridWorldCoffeeState::pos;
    using actions = domains::GridWorldCoffee::GridWorldCoffeeAction::ACTION;

    auto const size = 4;
    domains::GridWorldCoffee d;

    domains::GridWorldCoffee::GridWorldCoffeeAction down(actions::DOWN);
    domains::GridWorldCoffee::GridWorldCoffeeAction left(actions::LEFT);
    domains::GridWorldCoffee::GridWorldCoffeeAction right(actions::RIGHT);
    domains::GridWorldCoffee::GridWorldCoffeeAction up(actions::UP);

    WHEN("applying moves in gridworldcoffee")
    {

        pos start     = {0, 0};
        pos right_end = {4, 0};
        pos top_end   = {2, 4};
        pos middle    = {2, 2};

        REQUIRE(d.applyMove(start, &down) == start);
        REQUIRE(d.applyMove(start, &left) == start);

        REQUIRE(d.applyMove({0, 1}, &down) == start);
        REQUIRE(d.applyMove({1, 0}, &left) == start);

        REQUIRE(d.applyMove(right_end, &right) == right_end);
        REQUIRE(d.applyMove({3, 0}, &right) == right_end);
        REQUIRE(d.applyMove({4, 1}, &down) == right_end);

        REQUIRE(d.applyMove(top_end, &up) == top_end);
        REQUIRE(d.applyMove({2, 3}, &up) == top_end);
        REQUIRE(d.applyMove({1, 4}, &right) == top_end);

        REQUIRE(d.applyMove({2, 1}, &up) == middle);
        REQUIRE(d.applyMove({1, 2}, &right) == middle);
        REQUIRE(d.applyMove({3, 2}, &left) == middle);
        REQUIRE(d.applyMove({2, 3}, &down) == middle);
    }

    WHEN("stepping through gridworldcoffees")
    {

        auto s = static_cast<State const*>(d.getState({0, 0}, 0, 0));
        Observation const* o(0);
        Reward r(0);

        auto const old_state_index = s->index();

        THEN("going moving into walls or corners should only change the rain")
        {
            auto next_state = static_cast<State const*>(d.getState({0, 0}, 1, 0));
            auto t = d.step(&s, &left, &o, &r);
            REQUIRE((s->index() == old_state_index ||s->index()== next_state->index()));
            REQUIRE(r.toDouble() == domains::GridWorldCoffee::step_reward);
            REQUIRE(!t.terminated());

            d.releaseObservation(o);

            d.step(&s, &down, &o, &r);
            REQUIRE((s->index() == old_state_index ||s->index()== next_state->index()));
            REQUIRE(r.toDouble() == domains::GridWorldCoffee::step_reward);
            REQUIRE(!t.terminated());

            d.releaseObservation(o);
            d.releaseState(next_state);
        }

        THEN("moving into the open has four possible next states")
        {
            d.step(&s, &right, &o, &r);

            auto next_state1 = static_cast<State const*>(d.getState({1, 0}, 0, 0));
            auto next_state2 = static_cast<State const*>(d.getState({0, 0}, 1, 0));
            auto next_state3 = static_cast<State const*>(d.getState({1, 0}, 1, 0));

            REQUIRE((s->index() == old_state_index ||s->index()== next_state1->index()
            ||s->index()== next_state2->index() ||s->index()== next_state3->index()));

            d.releaseObservation(o);
            d.releaseState(next_state1);
            d.releaseState(next_state2);
            d.releaseState(next_state3);
        }

        d.releaseState(s);
    }
}

SCENARIO("gridworldcoffee POMDP functions", "[domain][gridworldcoffee]")
{

    domains::GridWorldCoffee d;

    WHEN("Generating random actions for gridworldcoffee")
    {
        auto const s = d.sampleRandomState();

        auto const a = d.generateRandomAction(s);

        REQUIRE(std::stoi(a->index()) < 4); // hardcoded |A|
        REQUIRE(std::stoi(a->index()) >= 0);

        d.releaseAction(a);
        d.releaseState(s);
    }

    WHEN("generating all legal actions for gridworldCoffee")
    {

        std::vector<Action const*> actions;

        auto const s = d.sampleRandomState();
        d.addLegalActions(s, &actions);

        for (auto i = 0; i < 4; ++i) // hardcoded |A|
        {
            REQUIRE(actions[i]->index() == std::to_string(i));
            d.releaseAction(actions[i]);
        }
    }

    WHEN("sampling the start state for gridworld")
    {

        auto const s = static_cast<domains::GridWorldCoffee::GridWorldCoffeeState const*>(d.sampleStartState());

        REQUIRE((s->_agent_position == domains::GridWorldCoffee::GridWorldCoffeeState::pos{0, 0}));
    }
}

