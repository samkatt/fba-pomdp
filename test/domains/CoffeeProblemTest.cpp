#include "catch.hpp"

#include <vector>

#include "domains/coffee/CoffeeProblem.hpp"
#include "domains/coffee/CoffeeProblemAction.hpp"
#include "domains/coffee/CoffeeProblemIndices.hpp"
#include "domains/coffee/CoffeeProblemState.hpp"
#include "environment/Reward.hpp"
#include "utils/random.hpp"

using namespace domains;

SCENARIO("Coffee problem", "[domain][coffee]")
{
    auto d = CoffeeProblem("");
    GIVEN("A coffee state")
    {

        THEN("all possible states occur")
        {
            std::vector<bool> occurred_states(32);

            for (auto i = 0; i < 1000; ++i)
            {
                auto s                      = d.sampleStartState();
                occurred_states[s->index()] = true;

                d.releaseState(s);
            }

            for (auto s : occurred_states) { REQUIRE(s); }
        }

        auto s        = d.sampleStartState();
        auto s_coffee = static_cast<CoffeeProblemState const*>(s);

        THEN("legal actions should always be 2")
        {
            std::vector<Action const*> actions;
            d.addLegalActions(s, &actions);

            REQUIRE(actions.size() == 2);

            AND_THEN("those actions cannot exceed index of 2")
            {
                REQUIRE(actions[0]->index() < 3);
                REQUIRE(actions[0]->index() < 3);
                REQUIRE(actions[1]->index() > -1);
                REQUIRE(actions[1]->index() > -1);
            }

            for (auto& a : actions) { d.releaseAction(a); }
        }

        WHEN("Doing a random action")
        {
            auto old_rain     = s_coffee->rains();
            auto old_umbrella = s_coffee->umbrella();

            auto a = IndexAction(rnd::slowRandomInt(0, 2));

            Observation const* o;
            Reward r(0);

            d.step(&s, &a, &o, &r);

            s_coffee = static_cast<CoffeeProblemState const*>(s);

            THEN("raining and umbrella should not change")
            {
                REQUIRE(s_coffee->rains() == old_rain);
                REQUIRE(s_coffee->umbrella() == old_umbrella);
            }

            d.releaseObservation(o);
        }

        WHEN("The robot gets coffee")
        {
            auto a = CoffeeProblemAction(GetCoffee);

            Observation const* o;
            Reward r(0);

            auto was_wet = s_coffee->wet();
            d.step(&s, &a, &o, &r);

            s_coffee = static_cast<CoffeeProblemState const*>(s);
            THEN("The robot is wet if it is raining")
            if ((s_coffee->rains() && !s_coffee->umbrella()) || was_wet)
            {
                REQUIRE(s_coffee->wet());
            }

            THEN("The robot observes the person wants coffee")
            REQUIRE(o->index() == 0);

            d.releaseObservation(o);
            d.releaseState(s);
        }

        WHEN("The robot checks for coffee")
        {
            auto a = IndexAction(CheckCoffee);

            Observation const* o;
            Reward r(0);

            auto was_wet = s_coffee->wet();

            d.step(&s, &a, &o, &r);

            THEN("The robot is wet if it was wet")
            REQUIRE(static_cast<CoffeeProblemState const*>(s)->wet() == was_wet);

            d.releaseObservation(o);
            d.releaseState(s);
        }

        d.releaseState(s);
    }
}

SCENARIO("Boutilier coffee problem", "[domain][coffee]")
{
    GIVEN("The Boutilier coffee problem")
    {
        auto d = CoffeeProblem("boutilier");

        THEN("all possible states occur")
        {
            std::vector<bool> occurred_states(32);

            for (auto i = 0; i < 1000; ++i)
            {
                auto s                      = d.sampleStartState();
                occurred_states[s->index()] = true;

                d.releaseState(s);
            }

            for (auto s : occurred_states) { REQUIRE(s); }
        }

        auto s        = d.sampleStartState();
        auto s_coffee = static_cast<CoffeeProblemState const*>(s);

        THEN("legal actions should always be 2")
        {
            std::vector<Action const*> actions;
            d.addLegalActions(s, &actions);

            REQUIRE(actions.size() == 2);

            AND_THEN("those actions cannot exceed index of 2")
            {
                REQUIRE(actions[0]->index() < 3);
                REQUIRE(actions[0]->index() < 3);
                REQUIRE(actions[1]->index() > -1);
                REQUIRE(actions[1]->index() > -1);
            }

            for (auto& a : actions) { d.releaseAction(a); }
        }

        WHEN("Doing a random action")
        {
            auto rained        = s_coffee->rains();
            auto had_umbrella  = s_coffee->umbrella();
            auto wanted_coffee = s_coffee->wantsCoffee();

            auto a = IndexAction(rnd::slowRandomInt(0, 2));

            Observation const* o;
            Reward r(0);

            d.step(&s, &a, &o, &r);

            s_coffee = static_cast<CoffeeProblemState const*>(s);

            THEN("raining and umbrella should not change")
            {
                REQUIRE(s_coffee->rains() == rained);
                REQUIRE(s_coffee->umbrella() == had_umbrella);
            }

            THEN("The user should retains its coffee preference")
            REQUIRE(wanted_coffee == s_coffee->wantsCoffee());

            d.releaseObservation(o);
        }

        WHEN("The robot gets coffee")
        {
            auto a = IndexAction(GetCoffee);

            Observation const* o;
            Reward r(0);

            auto was_wet = s_coffee->wet();
            d.step(&s, &a, &o, &r);

            s_coffee = static_cast<CoffeeProblemState const*>(s);

            THEN("The robot is wet if it is raining")
            if ((s_coffee->rains() && !s_coffee->umbrella()) || was_wet)
            {
                REQUIRE(s_coffee->wet());
            }

            THEN("The robot observes the person wants coffee")
            REQUIRE(o->index() == 0);

            d.releaseObservation(o);
            d.releaseState(s);
        }

        WHEN("The robot checks for coffee")
        {
            auto a = IndexAction(CheckCoffee);

            Observation const* o;
            Reward r(0);

            auto was_wet    = s_coffee->wet();
            auto had_coffee = s_coffee->hasCoffee();

            d.step(&s, &a, &o, &r);

            s_coffee = static_cast<CoffeeProblemState const*>(s);

            THEN("The robot is wet if it was wet")
            REQUIRE(s_coffee->wet() == was_wet);
            THEN("Person will not magically get coffee or drink it")
            REQUIRE(had_coffee == s_coffee->hasCoffee());

            d.releaseObservation(o);
            d.releaseState(s);
        }

        d.releaseState(s);
    }
}
