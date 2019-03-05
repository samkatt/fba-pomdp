#include "catch.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "domains/sysadmin/SysAdmin.hpp"
#include "domains/sysadmin/SysAdminState.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
#include "utils/random.hpp"

using domains::SysAdminState;

std::vector<int> randomComputerConfig(int size)
{
    std::vector<int> config(size);

    for (auto& c : config)
        if (rnd::boolean())
            c = 1;

    return config;
}

SCENARIO("sysadmin state modifiers", "[domains][sysadmin]")
{
    auto size = rnd::slowRandomInt(2, 10);

    auto d = domains::SysAdmin(size, "independent");

    GIVEN("the sysadmin problem with initial state")
    {
        auto s = d.sampleStartState();
        WHEN("breaking a random computer")
        {
            auto broken_computer = rnd::slowRandomInt(0, size);
            auto s_break         = d.breakComputer(s, broken_computer);

            THEN("that computer is broken (the rest is not)")
            {
                for (auto i = 0; i < size; ++i)
                {
                    if (i == broken_computer)
                    {
                        REQUIRE(!static_cast<SysAdminState const*>(s_break)->isOperational(i));
                    } else
                    {
                        REQUIRE(static_cast<SysAdminState const*>(s_break)->isOperational(i));
                    }
                }

                AND_THEN("breaking the same computer again does not change the state")
                {
                    auto s_break_twice = d.breakComputer(s_break, broken_computer);
                    for (auto i = 0; i < size; ++i)
                    {
                        if (i == broken_computer)
                        {
                            REQUIRE(!static_cast<SysAdminState const*>(s_break)->isOperational(i));
                        } else
                        {
                            REQUIRE(static_cast<SysAdminState const*>(s_break)->isOperational(i));
                        }
                    }

                    d.releaseState(s_break_twice);
                }
            }

            AND_WHEN("we fix that computer")
            {
                auto s_fixed = d.fixComputer(s_break, broken_computer);

                THEN("the computer works again")
                {
                    for (auto i = 0; i < size; ++i)
                    { REQUIRE(static_cast<SysAdminState const*>(s_fixed)->isOperational(i)); }
                }

                d.releaseState(s_fixed);
            }

            AND_WHEN("we break another computer")
            {
                auto second_broken_computer = rnd::slowRandomInt(0, size);
                auto s_break_two            = d.breakComputer(s_break, second_broken_computer);
                THEN("both are broken")
                {
                    for (auto i = 0; i < size; ++i)
                    {
                        if (i == broken_computer || i == second_broken_computer)
                        {
                            REQUIRE(
                                !static_cast<SysAdminState const*>(s_break_two)->isOperational(i));
                        } else
                        {
                            REQUIRE(
                                static_cast<SysAdminState const*>(s_break_two)->isOperational(i));
                        }
                    }
                }
                d.releaseState(s_break_two);
            }

            d.releaseState(s_break);
        }

        d.releaseState(s);
    }

    GIVEN("sysadmin of random size")
    {
        THEN("we can easily (and correctly) generate the correct actions")
        {
            auto comp    = rnd::slowRandomInt(0, size);
            auto observe = d.observeAction(comp);
            auto reboot  = d.rebootAction(comp);

            REQUIRE(observe->index() == comp);
            REQUIRE(reboot->index() == comp + size);

            d.releaseAction(observe);
            d.releaseAction(reboot);
        }
    }
}

SCENARIO("sysadmin state initiation", "[environment][sysadmin][domains]")
{
    auto n = 3, num_states = 0x1 << n;

    GIVEN("A sysadmin state with " + std::to_string(n) + " computers")
    {
        THEN("initiating with index 0 means no computers are operational")
        {
            auto state = SysAdminState(0, n);

            REQUIRE(state.numOperationalComputers() == 0);

            for (auto c = 0; c < n; ++c) { REQUIRE(!state.isOperational(c)); }
        }

        THEN("initiating with max index means all computers are operational")
        {
            auto state = SysAdminState(num_states - 1, n);

            REQUIRE(state.numOperationalComputers() == n);

            for (auto c = 0; c < n; ++c) { REQUIRE(state.isOperational(c)); }
        }

        THEN("initiating with index 1 means only the first computer is operational")
        {
            auto state = SysAdminState(1, n);

            REQUIRE(state.numOperationalComputers() == 1);

            REQUIRE(state.isOperational(0));
            for (auto c = 1; c < n; ++c) { REQUIRE(!state.isOperational(c)); }
        }
    }
}

SCENARIO("sysadmin environment", "[environment][sysadmin]")
{
    int n = 5;
    GIVEN("A sysadmin environment of size " + std::to_string(n))
    {
        auto d = domains::SysAdmin(n, "independent");

        THEN("an initial state should have all computers on")
        {
            auto s = static_cast<SysAdminState const*>(d.sampleStartState());

            for (auto c = 0; c < n; ++c) { REQUIRE(s->isOperational(c)); }

            REQUIRE(s->numOperationalComputers() == n);

            d.releaseState(s);
        }

        THEN("A step is never terminal ")
        {

            AND_THEN("reward of a step when observing is 1 * the number of working computers")
            {
                auto s = d.sampleStartState();

                Observation const* o(nullptr);
                Reward r(0);

                for (auto step = 0; step < 10; ++step)
                {
                    auto a = d.observeAction(rnd::slowRandomInt(0, n)); // random ping action
                    auto t = d.step(&s, a, &o, &r);

                    REQUIRE(!t.terminated());
                    REQUIRE(
                        r.toDouble()
                        == static_cast<SysAdminState const*>(s)->numOperationalComputers());
                    REQUIRE(r.toDouble() <= n);

                    d.releaseAction(a);
                    d.releaseObservation(o);
                }

                d.releaseState(s);
            }

            AND_THEN("reward of a step when rebooting is 1 * the number of working computers - 1")
            {
                auto s = d.sampleStartState();
                Observation const* o(nullptr);
                Reward r(0);

                for (auto step = 0; step < 10; ++step)
                {
                    auto a = d.rebootAction(rnd::slowRandomInt(0, n)); // random boot action
                    auto t = d.step(&s, a, &o, &r);

                    REQUIRE(!t.terminated());
                    REQUIRE(
                        r.toDouble()
                        == (float)static_cast<SysAdminState const*>(s)->numOperationalComputers()
                               - d.params()->_reboot_cost);
                    REQUIRE(r.toDouble() <= n);

                    d.releaseAction(a);
                    d.releaseObservation(o);
                }
                d.releaseState(s);
            }
        }
    }
}

SCENARIO("sysadmin steps", "[domain][sysadmin]")
{
    auto n = 3;
    auto d = domains::SysAdmin(n, "independent");

    Observation const* o(nullptr);
    Reward r(0);

    GIVEN("sysadmin state with a failing computer")
    {
        auto broken_computer = 2;

        auto s = d.sampleStartState(), old_s = s;

        s            = d.breakComputer(s, broken_computer);
        auto old_s_2 = s;

        REQUIRE(!static_cast<SysAdminState const*>(s)->isOperational(broken_computer));

        THEN("a regular step (observe) will not make it work again")
        {
            auto a = d.observeAction(rnd::slowRandomInt(0, n));

            d.step(&s, a, &o, &r);
            REQUIRE(!static_cast<SysAdminState const*>(s)->isOperational(2));

            d.releaseAction(a);
            d.releaseObservation(o);
            d.releaseState(s);
        }

        d.releaseState(old_s_2);
        d.releaseState(old_s);
    }

    GIVEN("sysadmin state with no working computers")
    {
        State const *s = new SysAdminState(0, n), *old_s = s;

        auto computer = rnd::slowRandomInt(0, n);

        REQUIRE(!static_cast<SysAdminState const*>(s)->isOperational(computer));

        THEN("a regular step will not make any computer working")
        {
            auto a = d.observeAction(computer);

            d.step(&s, a, &o, &r);

            for (auto i = 0; i < 3; ++i)
            { REQUIRE(!static_cast<SysAdminState const*>(s)->isOperational(i)); }

            d.releaseAction(a);
            d.releaseState(s);
            d.releaseObservation(o);
        }

        delete (old_s);
    }
}

SCENARIO("sysadmin domain", "[domain][sysadmin]")
{
    auto size = rnd::slowRandomInt(1, 6);

    GIVEN("a sysadmin problem of random size")
    {
        auto d = domains::SysAdmin(size, "independent");

        THEN("generating random actions will return all")
        {
            // random legal sysadmin state
            auto state = IndexState((1 << rnd::slowRandomInt(0, size)));

            std::vector<bool> action_is_generated(2 * size, false);

            for (auto i = 0; i < 100; ++i)
            {
                auto a = d.generateRandomAction(&state);
                REQUIRE(a->index() < (int)action_is_generated.size());
                REQUIRE(a->index() >= 0);

                action_is_generated[a->index()] = true;
            }

            for (const auto& i : action_is_generated) { REQUIRE(i); }
        }

        THEN("legal actions will always be all actions and not too many")
        {
            auto state = IndexState((1 << rnd::slowRandomInt(0, size)));
            std::vector<Action const*> actions;

            d.addLegalActions(&state, &actions);

            REQUIRE(actions.size() == 2 * size);
            for (size_t i = 0; i < actions.size(); ++i) { REQUIRE(actions[i]->index() == i); }
        }

        WHEN("all computers are working & testing observation probabilities")
        {
            auto s        = d.sampleStartState();
            auto a        = d.generateRandomAction(s);
            auto work_obs = IndexObservation(domains::SysAdmin::OPERATIONAL),
                 fail_obs = IndexObservation(domains::SysAdmin::FAILING);

            // 0.95 is a parameter in sysadmin-- not public so this will
            // fail when those params are changed
            REQUIRE(d.computeObservationProbability(&work_obs, a, s) == Approx(.95));
            REQUIRE(d.computeObservationProbability(&fail_obs, a, s) == Approx(.05));

            d.releaseState(s);
            d.releaseAction(a);
        }

        WHEN("all computers are failing & testing observation probabilities")
        {

            auto s        = d.getState(std::vector<int>(size, 0));
            auto a        = d.generateRandomAction(s);
            auto work_obs = IndexObservation(domains::SysAdmin::OPERATIONAL),
                 fail_obs = IndexObservation(domains::SysAdmin::FAILING);

            // 0.95 is a parameter in sysadmin-- not public so this will
            // fail when those params are changed
            REQUIRE(d.computeObservationProbability(&work_obs, a, s) == Approx(.05));
            REQUIRE(d.computeObservationProbability(&fail_obs, a, s) == Approx(.95));

            d.releaseState(s);
            d.releaseAction(a);
        }

        WHEN("we have one failing computer")
        {
            auto breaking_computer = rnd::slowRandomInt(0, size);

            auto s = d.sampleStartState();
            s      = d.breakComputer(s, breaking_computer);

            auto rebooting = d.rebootAction(breaking_computer),
                 pinging   = d.observeAction(breaking_computer);

            auto work_obs = IndexObservation(domains::SysAdmin::OPERATIONAL),
                 fail_obs = IndexObservation(domains::SysAdmin::FAILING);

            REQUIRE(d.computeObservationProbability(&work_obs, rebooting, s) == Approx(.05));
            REQUIRE(d.computeObservationProbability(&work_obs, pinging, s) == Approx(.05));

            REQUIRE(d.computeObservationProbability(&fail_obs, rebooting, s) == Approx(.95));
            REQUIRE(d.computeObservationProbability(&fail_obs, pinging, s) == Approx(.95));
        }

        WHEN("we have one working computer")
        {
            auto working_computer = rnd::slowRandomInt(0, size);

            auto s = d.sampleStartState();

            // break computers except working computer
            for (auto i = 0; i < size; ++i)
            {
                if (i != working_computer)
                {
                    s = d.breakComputer(s, i);
                }
            }

            auto rebooting = d.rebootAction(working_computer),
                 pinging   = d.observeAction(working_computer);

            auto work_obs = IndexObservation(domains::SysAdmin::OPERATIONAL),
                 fail_obs = IndexObservation(domains::SysAdmin::FAILING);

            REQUIRE(d.computeObservationProbability(&fail_obs, rebooting, s) == Approx(.05));
            REQUIRE(d.computeObservationProbability(&fail_obs, pinging, s) == Approx(.05));

            REQUIRE(d.computeObservationProbability(&work_obs, rebooting, s) == Approx(.95));
            REQUIRE(d.computeObservationProbability(&work_obs, pinging, s) == Approx(.95));
        }
    }
}

SCENARIO("sysadmin failing neighbours", "[domain][sysadmin]")
{

    GIVEN("an independent network")
    {

        auto const size = rnd::slowRandomInt(1, 4);
        auto const d    = domains::SysAdmin(size, "independent");

        THEN("the number of failing neighbours in any state should be 0")
        {

            for (auto i = 0; i < 10; ++i)
            {
                auto s       = d.getState(randomComputerConfig(size));
                auto const c = rnd::slowRandomInt(0, size);

                REQUIRE(d.numFailingNeighbours(c, s) == 0);

                d.releaseState(s);
            }
        }
    }

    GIVEN("A linear network")
    {

        auto const size = 4;
        auto const d    = domains::SysAdmin(size, "linear");

        auto start_state = d.sampleStartState();
        for (auto c = 0; c < size; ++c) { REQUIRE(d.numFailingNeighbours(c, start_state) == 0); }

        auto broken_comp_state = d.breakComputer(start_state, 0);
        REQUIRE(d.numFailingNeighbours(0, broken_comp_state) == 0);
        REQUIRE(d.numFailingNeighbours(1, broken_comp_state) == 1);
        REQUIRE(d.numFailingNeighbours(2, broken_comp_state) == 0);
        REQUIRE(d.numFailingNeighbours(3, broken_comp_state) == 0);

        auto two_broken_comp_state = d.breakComputer(broken_comp_state, 2);
        REQUIRE(d.numFailingNeighbours(0, two_broken_comp_state) == 0);
        REQUIRE(d.numFailingNeighbours(1, two_broken_comp_state) == 2);
        REQUIRE(d.numFailingNeighbours(2, two_broken_comp_state) == 0);
        REQUIRE(d.numFailingNeighbours(3, two_broken_comp_state) == 1);

        d.releaseState(start_state);
        d.releaseState(broken_comp_state);
        d.releaseState(two_broken_comp_state);
    }
}

SCENARIO("sysadmin configuration to state", "[sysadmin][domain][util]")
{
    size_t size(5);
    domains::SysAdmin d(size, "independent");

    GIVEN("the domain with 5 computers")
    {
        THEN("requesting the state of specific configurations should give the right state")
        {

            auto state = d.getState({0, 0, 0, 0, 0});

            REQUIRE(!state->isOperational(3));
            REQUIRE(!state->isOperational(5));

            d.releaseState(state);
            state = d.getState({1, 0, 0, 1, 0});

            REQUIRE(state->isOperational(0));
            REQUIRE(state->isOperational(3));
            REQUIRE(!state->isOperational(4));

            d.releaseState(state);
            state = d.getState({0, 1, 0, 1, 1});

            REQUIRE(!state->isOperational(0));
            REQUIRE(state->isOperational(1));
            REQUIRE(state->isOperational(3));
            REQUIRE(state->isOperational(4));

            d.releaseState(state);
        }
    }
}

SCENARIO("sysadmin calculates probability of computer failing", "[domains][sysadmin][util]")
{
    size_t size(4);

    GIVEN("an independent network of size 4")
    {
        domains::SysAdmin d(size, "independent");

        THEN("a failing computer should keep on failing")
        {
            auto state  = d.getState({1, 0, 1, 1});
            auto action = d.observeAction(1);

            REQUIRE(d.failProbability(state, action, 1) == 1);

            d.releaseState(state);
            d.releaseAction(action);

            auto state_2  = d.breakComputer(state, 3);
            auto action_2 = d.rebootAction(0);
            REQUIRE(d.failProbability(state_2, action_2, 3) == 1);

            d.releaseState(state_2);
            d.releaseAction(action_2);
        }

        THEN("a working computer should always have d.params()->_fail_prob probability of failing")
        {

            auto const random_computer = rnd::slowRandomInt(0, size);

            auto state_config             = randomComputerConfig(size);
            state_config[random_computer] = 1;

            auto const state  = static_cast<SysAdminState const*>(d.getState(state_config));
            auto const action = d.observeAction(rnd::slowRandomInt(0, size));

            REQUIRE(
                d.failProbability(state, action, random_computer)
                == Approx(d.params()->_fail_prob));

            d.releaseState(state);
            d.releaseAction(action);
        }

        THEN("rebooting a failing computer always has 1-successrate as probability of failing")
        {

            auto state  = d.getState({1, 1, 0, 1});
            auto action = d.rebootAction(2);

            REQUIRE(
                d.failProbability(state, action, 2)
                == Approx(1 - d.params()->_reboot_success_rate));

            d.releaseState(state);
            d.releaseAction(action);
        }

        THEN("rebooting a failing computer increases its chances of staying working")
        {

            auto state  = d.getState({1, 1, 1, 1});
            auto reboot = d.rebootAction(1);

            REQUIRE(
                d.failProbability(state, reboot, 1)
                == Approx(d.params()->_fail_prob * (1 - d.params()->_reboot_success_rate)));

            auto state_2 = d.breakComputer(state, 2);
            d.releaseState(state);

            REQUIRE(
                d.failProbability(state_2, reboot, 1)
                == Approx(d.params()->_fail_prob * (1 - d.params()->_reboot_success_rate)));

            d.releaseAction(reboot);
            d.releaseState(state_2);
        }
    }

    GIVEN("a linear network of size 4")
    {
        domains::SysAdmin d(size, "linear");

        THEN("a failing computer should keep on failing")
        {
            auto state  = d.getState({1, 0, 1, 1});
            auto action = d.observeAction(1);

            REQUIRE(d.failProbability(state, action, 1) == 1);

            d.releaseState(state);
            d.releaseAction(action);

            auto state_2  = d.breakComputer(state, 3);
            auto action_2 = d.rebootAction(0);
            REQUIRE(d.failProbability(state_2, action_2, 3) == 1);

            d.releaseState(state_2);
            d.releaseAction(action_2);
        }

        THEN("a working computer failing rate depends on number of failing neighbours")
        {

            auto state  = d.getState({0, 1, 1, 1});
            auto action = d.observeAction(rnd::slowRandomInt(0, size));

            REQUIRE(d.failProbability(state, action, 2) == Approx(d.params()->_fail_prob));

            REQUIRE(
                d.failProbability(state, action, 1)
                == Approx(
                       1
                       - (1 - d.params()->_fail_prob) * (1 - d.params()->_fail_neighbour_factor)));

            auto state_2 = d.breakComputer(state, 2);
            d.releaseState(state);

            REQUIRE(
                d.failProbability(state_2, action, 3)
                == Approx(
                       1
                       - (1 - d.params()->_fail_prob) * (1 - d.params()->_fail_neighbour_factor)));

            REQUIRE(
                d.failProbability(state_2, action, 1)
                == Approx(
                       1
                       - (1 - d.params()->_fail_prob)
                             * pow(1 - d.params()->_fail_neighbour_factor, 2)));

            d.releaseState(state_2);
            d.releaseAction(action);
        }

        THEN("rebooting a failing computer always has 1-successrate as probability of failing")
        {

            auto state  = d.getState({1, 1, 0, 1});
            auto action = d.rebootAction(2);

            REQUIRE(
                d.failProbability(state, action, 2)
                == Approx(1 - d.params()->_reboot_success_rate));

            d.releaseState(state);
            d.releaseAction(action);
        }

        THEN("rebooting a failing computer increases its chances of staying working")
        {

            auto state  = d.getState({1, 1, 1, 1});
            auto reboot = d.rebootAction(1);

            REQUIRE(
                d.failProbability(state, reboot, 1)
                == Approx(d.params()->_fail_prob * (1 - d.params()->_reboot_success_rate)));

            auto state_2 = d.breakComputer(state, 2);
            d.releaseState(state);

            REQUIRE(
                d.failProbability(state_2, reboot, 1)
                == Approx(
                       (1 - (1 - d.params()->_fail_prob) * (1 - d.params()->_fail_neighbour_factor))
                       * (1 - d.params()->_reboot_success_rate)));

            d.releaseAction(reboot);
            d.releaseState(state_2);
        }
    }
}
