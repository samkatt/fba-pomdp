#include "catch.hpp"

#include <vector>

#include "beliefs/bayes-adaptive/BARejectionSampling.hpp"

#include "domains/dummy/LinearDummyDomain.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

TEST_CASE("bayes-adaptive rejection sampling", "[state estimation][flat filter][bayes-adaptive]")
{
    auto size = 4;
    GIVEN("A new particle filter")
    {
        auto d = domains::LinearDummyDomain();
        auto b = beliefs::BARejectionSampling(size);
        auto s = d.sampleStartState();

        b.initiate(d);

        THEN("Samples should be correct initial state")
        {
            REQUIRE(b.sample()->index() == s->index());
        }

        THEN("Update should linear increment and decrement belief")
        {
            auto a = IndexAction(domains::LinearDummyDomain::Actions::FORWARD);
            auto o = IndexObservation(0); // generated observation SHOULD BE 0

            b.updateEstimation(&a, &o, d);
            REQUIRE(b.sample()->index() == s->index() + 1);

            b.updateEstimation(&a, &o, d);
            REQUIRE(b.sample()->index() == s->index() + 2);

            a = IndexAction(domains::LinearDummyDomain::Actions::BACKWARD);
            b.updateEstimation(&a, &o, d);
            REQUIRE(b.sample()->index() == s->index() + 1);
        }

        b.free(d);
        d.releaseState(s);
    }
}
