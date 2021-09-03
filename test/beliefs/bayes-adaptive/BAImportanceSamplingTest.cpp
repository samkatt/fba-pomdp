#include "catch.hpp"

#include "beliefs/bayes-adaptive/BAImportanceSampling.hpp"

#include <cstddef>
#include <vector>

#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "domains/dummy/DummyDomain.hpp"
#include "domains/dummy/LinearDummyDomain.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "utils/random.hpp"

SCENARIO(
    "bayes-adaptive importance sampling",
    "[state estimation][weighted filter][importance sampling][bayes-adaptive]")
{

    GIVEN("an importance sampler of 1 initial dummy domain state")
    {
        domains::DummyDomain d;

        auto s      = d.sampleStartState();
        auto filter = WeightedFilter<State const*>();

        filter.add(s, rnd::uniform_rand01() * rnd::slowRandomInt(1, 10));
        beliefs::BAImportanceSampling belief(filter, 1, false, false,false,false);

        REQUIRE(belief.sample()->index() == 0);

        WHEN("we perform a sample update")
        {
            auto a = d.generateRandomAction(s);
            IndexObservation o(0);

            belief.updateEstimation(a, &o, d);
            REQUIRE(belief.sample()->index() == 0);
            d.releaseAction(a);
        }

        belief.free(d);

        belief = beliefs::BAImportanceSampling(1, false, false, false,false);
        belief.initiate(d);

        REQUIRE(belief.sample()->index() == 0);

        WHEN("we perform a sample update")
        {
            auto a = d.generateRandomAction(s);
            IndexObservation o(0);

            belief.updateEstimation(a, &o, d);
            REQUIRE(belief.sample()->index() == 0);

            d.releaseAction(a);
        }

        belief.free(d);
    }

    GIVEN("a importance sampler of several initial linear dummy domain states")
    {
        domains::LinearDummyDomain d;

        beliefs::BAImportanceSampling belief(10, false, false, false,false);
        belief.initiate(d);

        REQUIRE(belief.sample()->index() == 0);

        WHEN("we perform a sample update")
        {
            auto a = d.generateRandomAction(belief.sample());
            IndexObservation o(0);

            belief.updateEstimation(a, &o, d);
            REQUIRE(belief.sample()->index() == 1);

            d.releaseAction(a);
        }

        belief.free(d);
    }

    GIVEN(
        "a importance sampler in a domain that changes the state pointer instead of updating the "
        "state")
    {
        domains::CollisionAvoidance d(3, 3);

        beliefs::BAImportanceSampling belief(10, false, false, false,false);
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
