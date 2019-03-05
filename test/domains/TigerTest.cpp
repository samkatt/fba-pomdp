#include "catch.hpp"

#include "easylogging++.h"

#include <cmath>
#include <cstddef>
#include <ctime>
#include <memory>
#include <string>

#include "domains/tiger/FactoredTiger.hpp"
#include "domains/tiger/Tiger.hpp"
#include "environment/Action.hpp"
#include "environment/Discount.hpp"
#include "environment/Environment.hpp"
#include "environment/Horizon.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
#include "experiments/Episode.hpp"
#include "utils/random.hpp"

TEST_CASE("general tiger functionality", "[domain][tiger]")
{
    auto d = domains::Tiger(domains::Tiger::TigerType::EPISODIC);
    auto s = d.sampleStartState();
    auto a = IndexAction(0);

    Observation const* o;
    Reward r(0);

    auto initial_state_index = s->index();
    WHEN("a initial state is created")
    {
        THEN("the index does not exceed the legal range")
        {
            REQUIRE(initial_state_index < 2);
            REQUIRE(initial_state_index > -1);
        }
    }

    WHEN("agents observes")
    {
        a.index(domains::Tiger::Literal::OBSERVE);
        auto t = d.step(&s, &a, &o, &r);

        REQUIRE(s->index() == initial_state_index);
        REQUIRE(o->index() < 2);
        REQUIRE(o->index() > -1);
        REQUIRE(r.toDouble() == -1.0);
        REQUIRE(a.index() == domains::Tiger::Literal::OBSERVE);
        REQUIRE(!t.terminated());

        auto correct_observation = IndexObservation(s->index()),
             incorec_observation = IndexObservation(1 - s->index());
        REQUIRE(d.computeObservationProbability(&correct_observation, &a, s) == .85);
        REQUIRE(d.computeObservationProbability(&incorec_observation, &a, s) == .15);

        d.releaseState(s);
        d.releaseObservation(o);
    }

    WHEN("agent opens correct door")
    {
        a.index(s->index());
        d.step(&s, &a, &o, &r);

        REQUIRE(o->index() < 2);
        REQUIRE(o->index() > -1);
        REQUIRE(r.toDouble() == 10.0);

        auto ob1 = IndexObservation(0), ob2 = IndexObservation(1);
        REQUIRE(d.computeObservationProbability(&ob1, &a, s) == .5);
        REQUIRE(d.computeObservationProbability(&ob2, &a, s) == .5);

        d.releaseState(s);
        d.releaseObservation(o);
    }

    WHEN("agent opens wrong door")
    {
        a.index(1 - s->index());
        d.step(&s, &a, &o, &r);

        REQUIRE(o->index() < 2);
        REQUIRE(o->index() > -1);
        REQUIRE(r.toDouble() == -100.0);

        auto ob1 = IndexObservation(0), ob2 = IndexObservation(1);
        REQUIRE(d.computeObservationProbability(&ob1, &a, s) == .5);
        REQUIRE(d.computeObservationProbability(&ob2, &a, s) == .5);

        d.releaseState(s);
        d.releaseObservation(o);
    }

    WHEN("a random action is generated")
    {
        auto action = d.generateRandomAction(s);

        THEN("action should not exceed legal indices")
        {
            REQUIRE(action->index() < 3);
            REQUIRE(action->index() >= 0);
        }

        d.releaseAction(action);
    }
}

TEST_CASE("episodic tiger functionality", "[domain][tiger][episodic]")
{
    domains::Tiger d(domains::Tiger::TigerType::EPISODIC);
    Observation const* o;
    Reward r(0);

    // test opening both doors
    WHEN("a door is opened, the episode terminates")
    {
        for (auto i = 0; i < 2; ++i)
        {
            auto a = IndexAction(i);

            auto s = d.sampleStartState();
            auto t = d.step(&s, &a, &o, &r);

            REQUIRE(t.terminated());

            d.releaseState(s);
            d.releaseObservation(o);
        }
    }
}

TEST_CASE("continuous tiger functionality", "[domain][tiger][continuous]")
{
    domains::Tiger d(domains::Tiger::TigerType::CONTINUOUS);
    Observation const* o;
    Reward r(0);

    // test opening both doors
    WHEN("a door is opened, the episode terminates")
    {
        for (auto i = 0; i < 2; ++i)
        {
            auto a = IndexAction(i);

            auto s = d.sampleStartState();
            auto t = d.step(&s, &a, &o, &r);

            REQUIRE(!t.terminated());

            d.releaseState(s);
            d.releaseObservation(o);
        }
    }
}

TEST_CASE("factored tiger environment", "[environment][tiger][factored]")
{
    // the following tests work for any random size
    int num_features = rnd::slowRandomInt(1, 5);

    auto tiger_types = std::vector<domains::FactoredTiger::FactoredTigerDomainType>(
        {domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS,
         domains::FactoredTiger::FactoredTigerDomainType::EPISODIC});

    for (auto type : tiger_types)
    {
        GIVEN("the Factored Tiger environment " + std::to_string(type))
        {
            auto d = domains::FactoredTiger(type, num_features);

            WHEN("we sample a start state")
            {
                auto s = d.sampleStartState();

                THEN("The index is within the legal range")
                {
                    REQUIRE(s->index() < (2 << num_features));
                    REQUIRE(s->index() >= 0);
                }

                d.releaseState(s);
            }

            WHEN("the agent listens")
            {

                auto s           = d.sampleStartState();
                auto start_index = s->index();

                auto a = IndexAction(domains::Tiger::Literal::OBSERVE);
                auto r = Reward(0);

                Observation const* o;

                auto t = d.step(&s, &a, &o, &r);

                THEN("the observation has a legal index & reward is -1")
                {
                    REQUIRE(s->index() == start_index);
                    REQUIRE(o->index() < 2);
                    REQUIRE(o->index() >= 0);
                    REQUIRE(r.toDouble() == -1.0);
                    REQUIRE(!t.terminated());
                }

                d.releaseState(s);
                d.releaseObservation(o);
            }

            WHEN("the agent opens the correct door")
            {

                auto s = d.sampleStartState();
                auto a = IndexAction(d.tigerLocation(s));
                auto r = Reward(0);

                Observation const* o;

                auto t = d.step(&s, &a, &o, &r);

                THEN("the new state & observation are legal and reward is 10")
                {
                    REQUIRE(s->index() < 2 << num_features);
                    REQUIRE(s->index() >= 0);

                    REQUIRE(o->index() < 2);
                    REQUIRE(o->index() >= 0);

                    REQUIRE(r.toDouble() == 10.0);

                    if (type == domains::FactoredTiger::EPISODIC)
                    {
                        REQUIRE(t.terminated());
                    } else
                    {
                        REQUIRE(!t.terminated());
                    }
                }

                d.releaseState(s);
                d.releaseObservation(o);
            }

            WHEN("the agent opens the wrong door")
            {
                auto s = d.sampleStartState();

                auto a = IndexAction(static_cast<int>((d.tigerLocation(s) == 0u)));
                auto r = Reward(0);

                Observation const* o;

                auto t = d.step(&s, &a, &o, &r);

                THEN("the new state & observation are legal and reward is 10")
                {
                    REQUIRE(s->index() < 2 << num_features);
                    REQUIRE(s->index() >= 0);

                    REQUIRE(o->index() < 2);
                    REQUIRE(o->index() >= 0);

                    REQUIRE(r.toDouble() == -100.0);

                    if (type == domains::FactoredTiger::EPISODIC)
                    {
                        REQUIRE(t.terminated());
                    } else
                    {
                        REQUIRE(!t.terminated());
                    }
                }

                d.releaseState(s);
                d.releaseObservation(o);
            }
        }
    }
}

SCENARIO("FactoredTiger domain", "[domain][tiger][factored]")
{
    // tests should work for any random number of (irrelevant) features
    int num_features = rnd::slowRandomInt(1, 5);

    auto tiger_types = std::vector<domains::FactoredTiger::FactoredTigerDomainType>(
        {domains::FactoredTiger::FactoredTigerDomainType::EPISODIC,
         domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS});

    for (auto type : tiger_types)
    {

        GIVEN("The FactoredTiger domain and a random sampled state " + std::to_string(type))
        {

            auto d = domains::FactoredTiger(tiger_types[0], num_features);
            auto s = d.sampleStartState();

            WHEN("we generate a random action")
            {
                auto a = d.generateRandomAction(s);

                THEN("it is any of the actions 0,1 & 2")
                {
                    REQUIRE(a->index() < 3);
                    REQUIRE(a->index() >= 0);
                }

                d.releaseAction(a);
            }

            WHEN("adding legal actions for a random state")
            {
                std::vector<Action const*> actions = {};
                d.addLegalActions(s, &actions);

                THEN("it should return all three actions")
                {
                    REQUIRE(actions.size() == 3);

                    std::vector<bool> action_indices = {false, false, false};
                    for (auto const& a : actions) { action_indices[a->index()] = true; }

                    for (auto i : action_indices) { REQUIRE(i); }

                    REQUIRE(action_indices.size() == 3);
                }
            }

            WHEN("the agent opens a door")
            {
                auto a = IndexAction(rnd::slowRandomInt(0, 2));

                THEN("observation probability is always .5")
                {
                    auto o = IndexObservation(rnd::slowRandomInt(0, 2));

                    REQUIRE(d.computeObservationProbability(&o, &a, s) == .5);
                }
            }

            WHEN("the agent listens")
            {
                auto a = IndexAction(domains::FactoredTiger::OBSERVE);

                THEN(
                    "the observation probability of the correct & incorrect location is .85 and "
                    ".15 respectively")
                {
                    auto correct_observation = IndexObservation(d.tigerLocation(s)),
                         incorrt_observation = IndexObservation(1 - correct_observation.index());

                    REQUIRE(d.computeObservationProbability(&correct_observation, &a, s) == .85);
                    REQUIRE(d.computeObservationProbability(&incorrt_observation, &a, s) == .15);
                }
            }

            d.releaseState(s);
        }
    }
}
