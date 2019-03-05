#include "catch.hpp"

#include "domains/dummy/DummyDomain.hpp"
#include "domains/dummy/LinearDummyDomain.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"

TEST_CASE("dummy domain", "[domain][dummy]")
{
    auto d = domains::DummyDomain();

    Reward r(0);
    Observation const* o = nullptr;

    WHEN("A simple dummy domain exists")
    {
        THEN("all indices are 1")
        {
            auto s = d.sampleStartState();
            auto a = d.generateRandomAction(s);

            REQUIRE(s->index() == 0);
            REQUIRE(a->index() == 0);

            d.releaseState(s);
            d.releaseAction(a);
        }

        THEN("Step leads to the expected results")
        {
            auto s = d.sampleStartState();
            auto a = d.generateRandomAction(s);

            auto t = d.step(&s, a, &o, &r);

            REQUIRE(s->index() == 0);
            REQUIRE(o->index() == 0);
            REQUIRE(r.toDouble() == 1.0);
            REQUIRE(!t.terminated());

            d.releaseState(s);
            d.releaseAction(a);
            d.releaseObservation(o);
        }

        THEN("observation probability is always 1")
        {
            auto s = d.sampleStartState();
            auto a = d.generateRandomAction(s);

            auto o2 = IndexObservation(0);

            REQUIRE(d.computeObservationProbability(&o2, a, s) == 1);

            d.releaseState(s);
            d.releaseAction(a);
        }
    }
}

TEST_CASE("linear dummy domain", "[domain][dummy][linear]")
{
    GIVEN("The linear dummy domain")
    {
        auto d = domains::LinearDummyDomain();

        auto s = d.sampleStartState();
        auto a = IndexAction(domains::LinearDummyDomain::Actions::FORWARD);

        Reward r(0);
        Observation const* o = nullptr;

        THEN("Stepping will increment state index")
        {
            // step 1
            auto t = d.step(&s, &a, &o, &r);

            REQUIRE(s->index() == 1);
            REQUIRE(o->index() == 0);
            REQUIRE(r.toDouble() == 1.0);
            REQUIRE(!t.terminated());

            THEN("observation probability is 1")
            REQUIRE(d.computeObservationProbability(o, &a, s) == 1);

            d.releaseObservation(o);

            // step 2
            t = d.step(&s, &a, &o, &r);

            REQUIRE(s->index() == 2);
            REQUIRE(o->index() == 0);
            REQUIRE(r.toDouble() == 1.0);
            REQUIRE(!t.terminated());

            THEN("observation probability is 1")
            REQUIRE(d.computeObservationProbability(o, &a, s) == 1);

            d.releaseObservation(o);

            THEN("Stepping back will decrement the state index")
            {

                // step back
                a.index(domains::LinearDummyDomain::Actions::BACKWARD);
                t = d.step(&s, &a, &o, &r);

                REQUIRE(s->index() == 1);
                REQUIRE(o->index() == 0);
                REQUIRE(r.toDouble() == 1.0);
                REQUIRE(!t.terminated());

                THEN("observation probability is 1")
                REQUIRE(d.computeObservationProbability(o, &a, s) == 1);

                d.releaseObservation(o);
            }
        }
        d.releaseState(s);
    }
}
