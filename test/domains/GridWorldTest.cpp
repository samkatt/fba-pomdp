#include "catch.hpp"

#include <algorithm>
#include <string>

#include "configurations/DomainConf.hpp"
#include "domains/gridworld/GridWorld.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
#include "utils/random.hpp"

double const domains::GridWorld::step_reward;

SCENARIO("GridWorld actions, observations & states", "[domain][gridworld]")
{

    WHEN("Testing the position in a grid world state")
    {
        using pos = domains::GridWorld::GridWorldState::pos;

        unsigned int x(rnd::slowRandomInt(0, 8));
        unsigned int y(rnd::slowRandomInt(0, 8));
        unsigned int y2(9);

        REQUIRE((pos{x, y} == pos{x, y}));
        REQUIRE((pos{x, y} != pos{x, y2}));
    }

    WHEN("Calculating the observation probabilities")
    {

        domains::GridWorld d(5);

        REQUIRE(d.obsDisplProb(0, 0) == Approx(.9));
        REQUIRE(d.obsDisplProb(0, 1) == Approx(.05));
        REQUIRE(d.obsDisplProb(0, 2) == Approx(.025));

        REQUIRE(d.obsDisplProb(1, 1) == Approx(.8));
        REQUIRE(d.obsDisplProb(1, 0) == Approx(.1));

        REQUIRE(d.obsDisplProb(2, 3) == Approx(.05));

        REQUIRE(d.obsDisplProb(4, 4) == Approx(.9));
        REQUIRE(d.obsDisplProb(4, 2) == Approx(.025));

        domains::GridWorld d2(3);

        REQUIRE(d2.obsDisplProb(0, 0) == Approx(.9));
        REQUIRE(d2.obsDisplProb(0, 1) == Approx(.05));
        REQUIRE(d2.obsDisplProb(0, 2) == Approx(.05));

        REQUIRE(d2.obsDisplProb(1, 1) == Approx(sqrt(d2.correctObservationProb())));
        REQUIRE(d2.obsDisplProb(2, 1) == Approx(.05));
    }
}

SCENARIO("GridWorld dynamics", "[domain][gridworld]")
{

    using pos     = domains::GridWorld::GridWorldState::pos;
    using actions = domains::GridWorld::GridWorldAction::ACTION;

    auto const size = 4;
    domains::GridWorld d(size);

    domains::GridWorld::GridWorldAction down(actions::DOWN);
    domains::GridWorld::GridWorldAction left(actions::LEFT);
    domains::GridWorld::GridWorldAction right(actions::RIGHT);
    domains::GridWorld::GridWorldAction up(actions::UP);

    WHEN("applying moves in gridworld")
    {

        pos start     = {0, 0};
        pos right_end = {3, 0};
        pos top_end   = {2, 3};
        pos middle    = {2, 2};

        REQUIRE(d.applyMove(start, &down) == start);
        REQUIRE(d.applyMove(start, &left) == start);

        REQUIRE(d.applyMove({0, 1}, &down) == start);
        REQUIRE(d.applyMove({1, 0}, &left) == start);

        REQUIRE(d.applyMove(right_end, &right) == right_end);
        REQUIRE(d.applyMove({2, 0}, &right) == right_end);
        REQUIRE(d.applyMove({3, 1}, &down) == right_end);

        REQUIRE(d.applyMove(top_end, &up) == top_end);
        REQUIRE(d.applyMove({2, 2}, &up) == top_end);
        REQUIRE(d.applyMove({1, 3}, &right) == top_end);

        REQUIRE(d.applyMove({2, 1}, &up) == middle);
        REQUIRE(d.applyMove({1, 2}, &right) == middle);
        REQUIRE(d.applyMove({3, 2}, &left) == middle);
        REQUIRE(d.applyMove({2, 3}, &down) == middle);
    }

    WHEN("stepping through gridworlds")
    {

        auto s = static_cast<State const*>(d.getState({0, 0}, {size - 1, size - 1}));
        Observation const* o(0);
        Reward r(0);

        auto const old_state_index = s->index();

        THEN("going moving into walls or corners should not change the state")
        {

            auto t = d.step(&s, &left, &o, &r);
            REQUIRE(s->index() == old_state_index);
            REQUIRE(r.toDouble() == domains::GridWorld::step_reward);
            REQUIRE(!t.terminated());

            d.releaseObservation(o);

            d.step(&s, &down, &o, &r);
            REQUIRE(s->index() == old_state_index);
            REQUIRE(r.toDouble() == domains::GridWorld::step_reward);
            REQUIRE(!t.terminated());

            d.releaseObservation(o);
        }

        THEN("moving into the open has two possible next states")
        {
            d.step(&s, &right, &o, &r);

            auto next_state = static_cast<State const*>(d.getState({1, 0}, {size - 1, size - 1}));

            REQUIRE((s->index() == old_state_index ||std::stoi(s->index())== next_state->index()));

            d.releaseObservation(o);
            d.releaseState(next_state);
        }

        d.releaseState(s);
    }
}

SCENARIO("gridworld POMDP functions", "[domain][gridworld]")
{

    domains::GridWorld d(7);

    WHEN("Generating random actions for gridworld")
    {
        auto const s = d.sampleRandomState();

        auto const a = d.generateRandomAction(s);

        REQUIRE(std::stoi(a->index()) < 4); // hardcoded |A|
        REQUIRE(std::stoi(a->index()) >= 0);

        d.releaseAction(a);
        d.releaseState(s);
    }

    WHEN("generating all legal actions for gridworld")
    {

        std::vector<Action const*> actions;

        auto const s = d.sampleRandomState();
        d.addLegalActions(s, &actions);

        for (auto i = 0; i < 4; ++i) // hardcoded |A|
        {
            REQUIRE(actions[i]->index() == i);
            d.releaseAction(actions[i]);
        }
    }

    WHEN("sampling the start state for gridworld")
    {

        auto const s = static_cast<domains::GridWorld::GridWorldState const*>(d.sampleStartState());

        REQUIRE((s->_agent_position == domains::GridWorld::GridWorldState::pos{0, 0}));
    }
}

SCENARIO("size specific settings", "[domains][gridworld]")
{

    using pos = domains::GridWorld::GridWorldState::pos;

    GIVEN("gridworld of size 3")
    {
        domains::GridWorld const d(3);

        REQUIRE(d.size() == 3);

        WHEN("testing slow locations")
        {
            REQUIRE(d.slowLocations()->size() == 1);
            REQUIRE(d.slowLocations()->at(0) == pos({1, 1}));
            REQUIRE(d.agentOnSlowLocation(pos({1, 1})));

            REQUIRE(d.agentOnSlowLocation({1, 1}));
            REQUIRE(!d.agentOnSlowLocation({0, 1}));
        }

        WHEN("testing goal locations")
        {
            auto const& goals           = d.goalLocations(3);
            std::vector<pos> test_goals = {{1, 2}, {2, 2}, {2, 1}};

            REQUIRE(goals.size() == test_goals.size());
            for (auto const& g : test_goals)
            { REQUIRE(std::find(goals.begin(), goals.end(), g) != goals.end()); }
        }
    }

    GIVEN("gridworld of size 4")
    {
        domains::GridWorld const d(4);

        REQUIRE(d.size() == 4);

        WHEN("testing slow locations")
        {
            REQUIRE(d.slowLocations()->size() == 2);
            REQUIRE(d.slowLocations()->at(0) == pos({2, 1}));
            REQUIRE(d.slowLocations()->at(1) == pos({1, 2}));

            REQUIRE(d.agentOnSlowLocation({2, 1}));
            REQUIRE(d.agentOnSlowLocation({1, 2}));

            REQUIRE(!d.agentOnSlowLocation({1, 1}));
            REQUIRE(!d.agentOnSlowLocation({2, 2}));
            REQUIRE(!d.agentOnSlowLocation({1, 0}));
            REQUIRE(!d.agentOnSlowLocation({3, 0}));
            REQUIRE(!d.agentOnSlowLocation({3, 2}));
            REQUIRE(!d.agentOnSlowLocation({3, 3}));
        }

        WHEN("testing goal locations")
        {
            std::vector<pos> test_goals = {{2, 3}, {3, 3}, {2, 2}, {3, 2}};
            auto const& goals           = d.goalLocations(4);

            REQUIRE(goals.size() == test_goals.size());
            for (auto const& g : test_goals)
            { REQUIRE(std::find(goals.begin(), goals.end(), g) != goals.end()); }
        }
    }

    GIVEN("gridworld of size 5")
    {
        domains::GridWorld d(5);

        REQUIRE(d.size() == 5);

        WHEN("querying slow locations")
        {
            std::vector<pos> test_locations = {{2, 3}, {3, 2}};
            auto const& locs                = *d.slowLocations();

            REQUIRE(locs.size() == test_locations.size());
            for (auto const& l : test_locations)
            { REQUIRE(std::find(locs.begin(), locs.end(), l) != locs.end()); }
        }

        WHEN("querying goal locations")
        {
            std::vector<pos> test_locations = {{2, 4}, {3, 4}, {4, 4}, {4, 3}, {4, 2}, {3, 3}};
            auto const& goals               = d.goalLocations(5);

            REQUIRE(goals.size() == test_locations.size());
            for (auto const& g : test_locations)
            { REQUIRE(std::find(goals.begin(), goals.end(), g) != goals.end()); }
        }
    }

    GIVEN("gridworld of size 6")
    {
        domains::GridWorld d(6);

        REQUIRE(d.size() == 6);

        WHEN("querying slow locations")
        {
            std::vector<pos> test_locations = {{3, 4}, {4, 3}, {1, 1}};
            auto const& locs                = *d.slowLocations();

            REQUIRE(locs.size() == test_locations.size());
            for (auto const& l : test_locations)
            { REQUIRE(std::find(locs.begin(), locs.end(), l) != locs.end()); }
        }

        WHEN("querying goal locations")
        {
            std::vector<pos> test_locations = {{3, 5}, {4, 5}, {5, 5}, {5, 4}, {5, 3}, {4, 4}};
            auto const& goals               = d.goalLocations(6);

            REQUIRE(goals.size() == test_locations.size());
            for (auto const& g : test_locations)
            { REQUIRE(std::find(goals.begin(), goals.end(), g) != goals.end()); }
        }
    }

    GIVEN("gridworld of size 7")
    {
        domains::GridWorld d(7);

        REQUIRE(d.size() == 7);

        WHEN("querying slow locations")
        {
            std::vector<pos> test_locations = {{3, 5}, {5, 3}, {4, 4}, {1, 1}};
            auto const& locs                = *d.slowLocations();

            REQUIRE(locs.size() == test_locations.size());
            for (auto const& l : test_locations)
            { REQUIRE(std::find(locs.begin(), locs.end(), l) != locs.end()); }
        }

        WHEN("querying goal locations")
        {
            std::vector<pos> test_locations = {
                {3, 6}, {4, 6}, {5, 6}, {6, 6}, {6, 5}, {6, 4}, {6, 3}, {4, 5}, {5, 5}, {5, 4}};
            auto const& goals = d.goalLocations(7);

            REQUIRE(goals.size() == test_locations.size());
            for (auto const& g : test_locations)
            { REQUIRE(std::find(goals.begin(), goals.end(), g) != goals.end()); }
        }
    }
}
