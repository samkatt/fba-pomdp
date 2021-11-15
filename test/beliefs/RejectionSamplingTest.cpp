#include "catch.hpp"

#include <vector>

#include "beliefs/particle_filters/FlatFilter.hpp"
#include "beliefs/particle_filters/RejectionSampling.hpp"

#include "domains/dummy/LinearDummyDomain.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

TEST_CASE("sampling", "[state estimation][flat filter]")
{
    GIVEN("A filter of size 1")
    {
        auto s      = new IndexState("0");
        auto filter = FlatFilter<State const*>({s});

        REQUIRE(filter.sample() == s);
        delete s;
    }

    auto size = 4;

    std::vector<State const*> states;
    states.reserve(size);

    GIVEN("A filter of multiple particles of the same state")
    {
        auto s = new IndexState("0");
        for (auto i = 0; i < size; ++i) { states.push_back(s); }

        FlatFilter<State const*> filter(states);

        REQUIRE(filter.sample() == s);
        delete s;
    }

    GIVEN("A filter with multiple particles of the same index")
    {
        for (auto i = 0; i < size; i++) { states.emplace_back(new IndexState("10")); }

        FlatFilter<State const*> filter(states);

        REQUIRE(filter.sample()->index() == "10");

        for (auto s : states) { delete s; }
    }
}

TEST_CASE("Manager initiate and update tests", "[state estimation][flat filter]")
{
    auto size = 4;
    GIVEN("A new particle filter")
    {
        auto d = domains::LinearDummyDomain();
        auto b = beliefs::RejectionSampling(size);
        auto s = d.sampleStartState();

        b.initiate(d);

        THEN("Samples should be correct initial state")
        {
            REQUIRE(b.sample()->index() == s->index());
        }

        THEN("Update should linear increment and decrement belief")
        {
            auto a = IndexAction(std::to_string(domains::LinearDummyDomain::Actions::FORWARD));
            auto o = IndexObservation("0"); // generated observation SHOULD BE 0

            b.updateEstimation(&a, &o, d);
            REQUIRE(b.sample()->index() ==std::to_string(std::stoi(s->index())+ 1));

            b.updateEstimation(&a, &o, d);
            REQUIRE(b.sample()->index() ==std::to_string(std::stoi(s->index())+ 2));

            a = IndexAction(std::to_string(domains::LinearDummyDomain::Actions::BACKWARD));
            b.updateEstimation(&a, &o, d);
            REQUIRE(b.sample()->index() ==std::to_string(std::stoi(s->index())+ 1));
        }

        b.free(d);
        d.releaseState(s);
    }
}
