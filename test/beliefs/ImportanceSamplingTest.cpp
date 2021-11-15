#include "catch.hpp"

#include "beliefs/particle_filters/ImportanceSampler.hpp"
#include "beliefs/particle_filters/WeightedFilter.hpp"

#include <cstddef>
#include <vector>

#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "domains/dummy/DummyDomain.hpp"
#include "domains/dummy/LinearDummyDomain.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "utils/random.hpp"

SCENARIO("weighted filter", "[state estimation][weighted filter]")
{
    GIVEN("a weighted filter of size 1")
    {
        auto filter = WeightedFilter<State const*>();
        auto s      = IndexState("0");

        THEN("sampling from 1 particle should return the same state")
        {
            filter.add(&s);
            REQUIRE(filter.sample() == &s);
        }

        THEN("sampling from 1 particle with high weight should return the same state")
        {
            filter.add(&s, 1000);
            REQUIRE(filter.sample() == &s);
        }

        THEN("sampling from 1 particle with low weight should return the same state")
        {
            filter.add(&s, .1);
            REQUIRE(filter.sample() == &s);
        }
    }

    GIVEN("A filter of multiple particles")
    {
        auto filter = WeightedFilter<State const*>();

        WHEN("the particles are the same state")
        {

            auto s = IndexState("0");
            for (auto i = 0; i < 4; ++i)
            { filter.add(&s, rnd::uniform_rand01() * rnd::slowRandomInt(1, 100)); }

            THEN("sampling should return that specific state")
            REQUIRE(filter.sample() == &s);
        }

        WHEN("the particles are states with the same index")
        {
            std::vector<IndexState> states;
            states.reserve(4);
            for (auto i = 0; i < 4; ++i) { states.emplace_back(IndexState("2")); }

            for (auto const& s : states)
            { filter.add(&s, rnd::uniform_rand01() * rnd::slowRandomInt(1, 100)); }

            THEN("sampling should return a state with that specific index")
            REQUIRE(filter.sample()->index() == "2");
        }
    }

    GIVEN("A filter of 3 particles, of which 1 with relative large weight")
    {
        auto b = WeightedFilter<State const*>();
        auto s = IndexState("0"), probable_s = IndexState("1");

        b.add(&s);
        b.add(&s);
        b.add(&probable_s, 10000);

        THEN("we ought to sample the one with a large weight")
        REQUIRE(b.sample() == &probable_s);
    }

    GIVEN("A filter of 3 particles, of which 1 with relative small weight")
    {
        auto b = WeightedFilter<State const*>();
        auto s = IndexState("0"), improbable_s = IndexState("1");

        b.add(&s);
        b.add(&s);
        b.add(&improbable_s, .00001);

        THEN("we ought to not sample the one with a large weight")
        REQUIRE(b.sample() != &improbable_s);
    }
}

SCENARIO("importance sampling", "[state estimation][weighted filter][importance sampling]")
{

    GIVEN("an importance sampler of 1 initial dummy domain state")
    {
        domains::DummyDomain d;

        auto s      = d.sampleStartState();
        auto filter = WeightedFilter<State const*>();

        filter.add(s, rnd::uniform_rand01() * rnd::slowRandomInt(1, 10));
        beliefs::ImportanceSampler belief(filter, 1);

        REQUIRE(belief.sample()->index() == "0");

        WHEN("we perform a sample update")
        {
            auto a = d.generateRandomAction(s);
            IndexObservation o(0);

            belief.updateEstimation(a, &o, d);
            REQUIRE(belief.sample()->index() == "0");
            d.releaseAction(a);
        }

        belief.free(d);

        belief = beliefs::ImportanceSampler(1);
        belief.initiate(d);

        REQUIRE(belief.sample()->index() == "0");

        WHEN("we perform a sample update")
        {
            auto a = d.generateRandomAction(s);
            IndexObservation o(0);

            belief.updateEstimation(a, &o, d);
            REQUIRE(belief.sample()->index() == "0");

            d.releaseAction(a);
        }

        belief.free(d);
    }

    GIVEN("a importance sampler of several initial linear dummy domain states")
    {
        domains::LinearDummyDomain d;

        beliefs::ImportanceSampler belief(10);
        belief.initiate(d);

        REQUIRE(belief.sample()->index() == "0");

        WHEN("we perform a sample update")
        {
            auto a = d.generateRandomAction(belief.sample());
            IndexObservation o(0);

            belief.updateEstimation(a, &o, d);
            REQUIRE(belief.sample()->index() == "1");

            d.releaseAction(a);
        }

        belief.free(d);
    }

    GIVEN(
        "a importance sampler in a domain that changes the state pointer instead of updating the "
        "state")
    {
        domains::CollisionAvoidance d(3, 3);

        beliefs::ImportanceSampler belief(10);
        belief.initiate(d);

        auto const init_state       = d.sampleStartState();
        auto const init_state_index = init_state->index();

        WHEN("we perform a sample update")
        {
            auto a = d.generateRandomAction(belief.sample());
            auto o = d.getObservation(0);

            belief.updateEstimation(a, o, d);

            REQUIRE(init_state_index == init_state->index());
            REQUIRE(belief.sample()->index() != init_state->index());

            d.releaseAction(a);
            d.releaseObservation(o);
        }

        belief.free(d);
        d.releaseState(init_state);
    }
}
