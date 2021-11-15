#include "catch.hpp"

#include "beliefs/point_estimation/PointEstimation.hpp"

#include <memory>

#include "domains/dummy/DummyDomain.hpp"
#include "domains/dummy/LinearDummyDomain.hpp"

#include "environment/Reward.hpp"
#include "environment/State.hpp"

TEST_CASE("update", "[belief][point estimator]")
{

    GIVEN("a point estimator with a linear state")
    {

        domains::LinearDummyDomain d;
        beliefs::PointEstimation b;
        b.initiate(d);

        THEN("an action will increase its index")
        {
            Observation const* o;
            Reward r(0);

            auto a = d.generateRandomAction(b.sample());
            auto s = d.sampleStartState();

            d.step(&s, a, &o, &r);

            b.updateEstimation(a, o, d);

            REQUIRE(b.sample()->index() == "1");

            d.releaseAction(a);
            d.releaseObservation(o);
            d.releaseState(s);
        }

        b.free(d);
    }
}
