#include "catch.hpp"

#include <vector>

#include "domains/dummy/FactoredDummyDomain.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
#include "utils/random.hpp"

TEST_CASE("factored dummy environment", "[domain][dummy][factored]")
{
    auto const size = 3;

    auto const d = domains::FactoredDummyDomain(size);

    Reward r(0);

    WHEN("A simple factored dummy domain exists")
    {
        THEN("start state has index 0")
        {
            auto s = d.sampleStartState();
            REQUIRE(s->index() == 0);
            d.releaseState(s);
        }

        THEN("any random action is either indexed 0 or 1")
        {
            auto s = d.sampleStartState();
            auto a = d.generateRandomAction(s);

            REQUIRE(a->index() < 2);
            ;

            IndexObservation o(0);
            THEN("the observation probability to o=0 is 1")
            REQUIRE(d.computeObservationProbability(&o, a, s) == 1);

            d.releaseState(s);
            d.releaseAction(a);
        }

        THEN("Step up leads to (0,1)")
        {
            auto s       = d.sampleStartState();
            auto a_up    = IndexAction(domains::FactoredDummyDomain::ACTIONS::UP),
                 a_right = IndexAction(domains::FactoredDummyDomain::ACTIONS::RIGHT);

            Observation const* o = nullptr;
            auto t               = d.step(&s, &a_up, &o, &r);

            REQUIRE(s->index() == 1);
            REQUIRE(o->index() == 0);
            REQUIRE(r.toDouble() == -1.0);
            REQUIRE(!t.terminated());

            REQUIRE(d.computeObservationProbability(o, &a_up, s) == 1);
            REQUIRE(d.computeObservationProbability(o, &a_right, s) == 1);

            d.releaseObservation(o);

            AND_THEN("Step right leads to (1,1)")
            {
                t = d.step(&s, &a_right, &o, &r);

                REQUIRE(s->index() == 4);
                REQUIRE(o->index() == 0);
                REQUIRE(r.toDouble() == -1.0);
                REQUIRE(!t.terminated());

                REQUIRE(d.computeObservationProbability(o, &a_up, s) == 1);
                REQUIRE(d.computeObservationProbability(o, &a_right, s) == 1);
            }

            d.releaseState(s);
            d.releaseObservation(o);
        }

        THEN("Legal actions are always just <up> & <right>")
        {
            std::vector<Action const*> legal_actions;

            auto s = d.sampleStartState();
            d.addLegalActions(s, &legal_actions);

            REQUIRE(legal_actions.size() == 2);
            REQUIRE(legal_actions[0]->index() == domains::FactoredDummyDomain::ACTIONS::UP);
            REQUIRE(legal_actions[1]->index() == domains::FactoredDummyDomain::ACTIONS::RIGHT);

            d.releaseState(s);
            for (auto a : legal_actions) { d.releaseAction(a); }
        }
    }
}
