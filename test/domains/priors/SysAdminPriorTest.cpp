#include "catch.hpp"

#include "easylogging++.h"

#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/BAConf.hpp"
#include "configurations/FBAConf.hpp"
#include "domains/sysadmin/SysAdmin.hpp"
#include "domains/sysadmin/SysAdminBAExtension.hpp"
#include "domains/sysadmin/SysAdminFactoredPrior.hpp"
#include "domains/sysadmin/SysAdminFlatPrior.hpp"

using domains::SysAdminState;

SCENARIO("bapomdp independent sysadmin transition prior", "[bayes-adaptive][sysadmin][flat]")
{

    float total_counts = 10000;

    auto const size      = 3;
    auto c               = configurations::BAConf();
    c.domain_conf.domain = "independent-sysadmin";
    c.domain_conf.size   = size;
    auto const d         = domains::SysAdmin(size, "independent");
    auto const ext       = bayes_adaptive::domain_extensions::SysAdminBAExtension(size);

    auto s = d.sampleStartState();

    GIVEN("A noise free prior")
    {

        auto const prior   = priors::SysAdminFlatPrior(d, c);
        auto bapomdp_state = static_cast<BAPOMDPState*>(prior.sample(s));

        WHEN("performing an action in the initial state")
        {
            THEN("observing should leave it in the initial state with high probability")
            {
                for (auto comp = 0; comp < size; ++comp)
                {
                    auto action = d.observeAction(comp);
                    REQUIRE(
                        bapomdp_state->model()->count(bapomdp_state, action, bapomdp_state)
                        == Approx(total_counts * pow(1 - d.params()->_fail_prob, 3)));
                    d.releaseAction(action);
                }
            }

            THEN(
                "observing should have some small chance of leading to a state with a broken "
                "computer")
            {
                for (auto comp = 0; comp < size; ++comp)
                {

                    auto new_s =
                        d.breakComputer(bapomdp_state->_domain_state, rnd::slowRandomInt(0, size));
                    auto action = d.observeAction(comp);

                    REQUIRE(
                        bapomdp_state->model()->count(bapomdp_state, action, new_s)
                        == Approx(
                               total_counts * d.params()->_fail_prob
                               * pow(1 - d.params()->_fail_prob, 2)));

                    d.releaseAction(action);
                    d.releaseState(new_s);
                }
            }

            THEN(
                "observing should have some super small chance of leading to a state with two "
                "broken computers")
            {
                for (auto a = 0; a < size; ++a)
                {

                    auto tmp_s  = d.breakComputer(bapomdp_state->_domain_state, 1);
                    auto new_s  = d.breakComputer(tmp_s, 2);
                    auto action = IndexAction(std::to_string(a));

                    REQUIRE(
                        bapomdp_state->model()->count(bapomdp_state, &action, new_s)
                        == Approx(
                               total_counts * (1 - d.params()->_fail_prob)
                               * pow(d.params()->_fail_prob, 2)));

                    d.releaseState(tmp_s);
                    d.releaseState(new_s);
                }
            }

            THEN(
                "rebooting a computer should leave it in the initial state with high probability, "
                "especially the rebooted computer")
            {
                for (auto a = size; a < 2 * size; ++a)
                {
                    auto action = IndexAction(std::to_string(a));
                    REQUIRE(
                        bapomdp_state->model()->count(bapomdp_state, &action, bapomdp_state)
                        == Approx(
                               total_counts
                               * (pow(1 - d.params()->_fail_prob, size)
                                  + d.params()->_fail_prob
                                        * pow(1 - d.params()->_fail_prob, size - 1)
                                        * d.params()->_reboot_success_rate)));
                }
            }
        }

        WHEN("Performing an action in state(0)")
        {
            bapomdp_state->_domain_state = ext.getState(0);

            THEN("observing leave it in state(0)")
            {
                for (auto a = 0; a < size; ++a)
                {
                    auto action = IndexAction(std::to_string(a));
                    REQUIRE(
                        bapomdp_state->model()->count(bapomdp_state, &action, bapomdp_state)
                        == total_counts);
                }
            }

            THEN(
                "rebooting a computer should leave it in state(0) with low probability, and in "
                "rebooted with higher")
            {
                for (auto a = size; a < 2 * size; ++a)
                {
                    auto action = IndexAction(std::to_string(a));
                    REQUIRE(
                        bapomdp_state->model()->count(bapomdp_state, &action, bapomdp_state)
                        == Approx(total_counts * (1 - d.params()->_reboot_success_rate)));

                    auto new_s = d.copyState(bapomdp_state->_domain_state);
                    new_s      = d.fixComputer(new_s, a - size);

                    REQUIRE(
                        bapomdp_state->model()->count(bapomdp_state, &action, new_s)
                        == Approx(total_counts * d.params()->_reboot_success_rate));
                }
            }
        }

        WHEN("entering a state where 1 computer is working")
        {
            auto failing_state  = IndexState("0");
            auto fixed_computer = rnd::slowRandomInt(0, size);

            auto new_s = d.fixComputer(&failing_state, fixed_computer);

            THEN("the probability of getting there depends on the action")
            {
                std::vector<Action const*> legal_actions;
                d.addLegalActions(&failing_state, &legal_actions);

                for (auto const& a : legal_actions)
                {
                    if (std::stoi(a->index()) == size + fixed_computer)
                    {
                        REQUIRE(
                            bapomdp_state->model()->count(&failing_state, a, new_s)
                            == total_counts * d.params()->_reboot_success_rate);
                    } else
                    {
                        REQUIRE(bapomdp_state->model()->count(&failing_state, a, new_s) == 0);
                    }
                }
            }

            d.releaseState(new_s);
        }

        WHEN("taking a random state")
        {
            auto s_init = d.sampleStartState();
            auto state  = SysAdminState(rnd::slowRandomInt(0, ext.domainSize()._S), size);

            AND_WHEN("a random computer is working")
            {
                THEN(
                    "getting into that state (from initial) should be easier when rebooting than "
                    "otherwise")
                {
                    for (auto a = 0; a < size; ++a)
                    {
                        auto reboot = IndexAction(std::to_string(a + size));

                        for (auto a_2 = 0; a_2 < size; ++a_2)
                        {
                            auto observe = IndexAction(std::to_string(a_2));
                            if (state.isOperational(a))
                            {
                                REQUIRE(
                                    bapomdp_state->model()->count(s_init, &reboot, &state)
                                    > bapomdp_state->model()->count(s_init, &observe, &state));
                            }
                        }
                    }
                }
            }

            d.releaseState(s_init);
        }

        WHEN("Taking a complicated state")
        {
            auto s_init          = d.sampleStartState();
            auto broken_computer = 1;
            auto s_compl         = d.breakComputer(s_init, broken_computer);

            THEN("the transition into and out of are exactly right")
            {
                auto observe = IndexAction(std::to_string(0));
                REQUIRE(
                    bapomdp_state->model()->count(s_init, &observe, s_compl)
                    == Approx(
                           total_counts * pow(1 - d.params()->_fail_prob, 2)
                           * d.params()->_fail_prob));

                REQUIRE(
                    bapomdp_state->model()->count(s_compl, &observe, s_compl)
                    == Approx(total_counts * pow(1 - d.params()->_fail_prob, 2)));

                auto reboot = IndexAction(std::to_string(size));
                REQUIRE(
                    bapomdp_state->model()->count(s_init, &reboot, s_compl)
                    == Approx(
                           total_counts
                           * (pow(1 - d.params()->_fail_prob, 2) * d.params()->_fail_prob
                              + (1 - d.params()->_fail_prob) * pow(d.params()->_fail_prob, 2)
                                    * d.params()->_reboot_success_rate)));

                REQUIRE(
                    bapomdp_state->model()->count(s_compl, &reboot, s_compl)
                    == Approx(
                           total_counts
                           * (pow(1 - d.params()->_fail_prob, 2)
                              + d.params()->_fail_prob * (1 - d.params()->_fail_prob)
                                    * d.params()->_reboot_success_rate)));

                reboot = IndexAction(std::to_string(size + broken_computer));
                REQUIRE(
                    bapomdp_state->model()->count(s_compl, &reboot, s_compl)
                    == Approx(
                           total_counts * (1 - d.params()->_reboot_success_rate)
                           * pow(1 - d.params()->_fail_prob, 2)));
                REQUIRE(
                    bapomdp_state->model()->count(s_compl, &reboot, s_init)
                    == Approx(
                           total_counts * pow(1 - d.params()->_fail_prob, 2)
                           * d.params()->_reboot_success_rate));

                REQUIRE(
                    bapomdp_state->model()->count(s_compl, &reboot, s_init)
                    == Approx(
                           total_counts * pow(1 - d.params()->_fail_prob, 2)
                           * d.params()->_reboot_success_rate));

                auto broken_state = d.breakComputer(s_compl, 0);
                REQUIRE(bapomdp_state->model()->count(broken_state, &observe, s_init) == 0);
                REQUIRE(bapomdp_state->model()->count(broken_state, &reboot, s_init) == 0);
                REQUIRE(
                    bapomdp_state->model()->count(s_compl, &reboot, broken_state)
                    == Approx(
                           total_counts * (1 - d.params()->_fail_prob) * d.params()->_fail_prob
                           * (1 - d.params()->_reboot_success_rate)));
                REQUIRE(
                    bapomdp_state->model()->count(s_init, &observe, broken_state)
                    == Approx(
                           total_counts * pow(d.params()->_fail_prob, 2)
                           * (1 - d.params()->_fail_prob)));
                REQUIRE(
                    bapomdp_state->model()->count(s_init, &reboot, broken_state)
                    == Approx(
                           total_counts * pow(d.params()->_fail_prob, 2)
                           * (1 - d.params()->_fail_prob)
                           * (1 - d.params()->_reboot_success_rate)));

                d.releaseState(broken_state);
            }

            d.releaseState(s_init);
        }
        delete (bapomdp_state);
    }

    d.releaseState(s);
}

SCENARIO(
    "independent sysadmin bayes-adaptive transition prior for a larger network",
    "[bayes-adaptive][sysadmin][flat]")
{

    auto total_counts = 10000.0f;

    auto const size      = 6;
    auto const d         = domains::SysAdmin(size, "independent");
    auto c               = configurations::BAConf();
    c.domain_conf.domain = "independent-sysadmin";
    c.domain_conf.size   = size;
    c.counts_total       = total_counts;

    auto const prior = factory::makeTBAPOMDPPrior(d, c);

    GIVEN("Sysadmin problem of 6 computers")
    {

        auto bapomdp_state = static_cast<BAPOMDPState*>(prior->sample(d.sampleStartState()));

        THEN("the more computers failing the lower the chance")
        {
            auto computer = 4;
            auto observe  = d.observeAction(computer);
            auto reboot   = d.rebootAction(computer);

            auto same_state = d.copyState(bapomdp_state->_domain_state);
            REQUIRE(
                bapomdp_state->model()->count(same_state, observe, same_state)
                == Approx(total_counts * pow(1 - d.params()->_fail_prob, 6)));
            REQUIRE(
                bapomdp_state->model()->count(same_state, reboot, same_state)
                == Approx(
                       total_counts
                       * (pow(1 - d.params()->_fail_prob, 6)
                          + pow(1 - d.params()->_fail_prob, 5) * d.params()->_reboot_success_rate
                                * d.params()->_fail_prob)));

            auto other_failure_state = d.breakComputer(same_state, 1);
            REQUIRE(
                bapomdp_state->model()->count(same_state, observe, other_failure_state)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, 5) * d.params()->_fail_prob));
            REQUIRE(
                bapomdp_state->model()->count(same_state, reboot, other_failure_state)
                == Approx(
                       total_counts
                       * (pow(1 - d.params()->_fail_prob, 5) * d.params()->_fail_prob
                          + pow(1 - d.params()->_fail_prob, 4) * pow(d.params()->_fail_prob, 2)
                                * d.params()->_reboot_success_rate)));

            d.releaseState(same_state);
            d.releaseState(other_failure_state);
            d.releaseAction(observe);
            d.releaseAction(reboot);
        }

        THEN("Rebooting without success is a thing, but with low probability")
        {
            auto computer = rnd::slowRandomInt(1, size);
            auto a        = d.rebootAction(computer);

            auto broken_state = d.breakComputer(bapomdp_state->_domain_state, computer);
            auto broken_twice_state =
                d.breakComputer(broken_state, computer - 1); // randomly kill another computer

            REQUIRE(
                bapomdp_state->model()->count(bapomdp_state, a, broken_state)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, 5)
                       * (1 - d.params()->_reboot_success_rate) * d.params()->_fail_prob));

            REQUIRE(
                bapomdp_state->model()->count(broken_state, a, broken_state)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, 5)
                       * (1 - d.params()->_reboot_success_rate)));

            REQUIRE(
                bapomdp_state->model()->count(broken_state, a, broken_twice_state)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, 4)
                       * (1 - d.params()->_reboot_success_rate) * d.params()->_fail_prob));

            REQUIRE(
                bapomdp_state->model()->count(broken_state, a, bapomdp_state)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, 5)
                       * d.params()->_reboot_success_rate));

            d.releaseAction(a);
            d.releaseState(broken_state);
            d.releaseState(broken_twice_state);
        }

        d.releaseState(bapomdp_state->_domain_state);
        delete (bapomdp_state);
    }
}

SCENARIO("linear sysadmin ba-table prior", "[bayes-adaptive][sysadmin][flat]")
{

    auto const size         = 5;
    auto const total_counts = 10000.0f;

    auto const d         = domains::SysAdmin(size, "linear");
    auto c               = configurations::BAConf();
    c.domain_conf.domain = "linear-sysadmin";
    c.domain_conf.size   = size;
    c.counts_total       = total_counts;

    GIVEN("a model from the accurate prior")
    {

        auto const p           = factory::makeTBAPOMDPPrior(d, c);
        auto const start_state = d.sampleStartState();
        auto const ba_s        = static_cast<BAPOMDPState*>(p->sample(start_state));

        auto model = ba_s->model();

        THEN("probability of failing depends on failing neighbours")
        {

            auto const observe = d.observeAction(rnd::slowRandomInt(0, size));
            REQUIRE(
                model->count(start_state, observe, start_state)
                == Approx(total_counts * pow(1 - d.params()->_fail_prob, size)));

            auto const broken_computer_1 = 1;
            auto const broken_state      = d.breakComputer(start_state, broken_computer_1);
            REQUIRE(
                model->count(start_state, observe, broken_state)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, size - 1)
                       * d.params()->_fail_prob));
            REQUIRE(
                model->count(broken_state, observe, broken_state)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, size - 3)
                       * pow((1 - d.params()->_fail_prob)
                                 * (1 - d.params()->_fail_neighbour_factor),
                             2)));

            auto const broken_computer_2 = 3;
            auto const brokenst_state    = d.breakComputer(broken_state, broken_computer_2);
            REQUIRE(
                model->count(start_state, observe, brokenst_state)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, 3)
                       * pow(d.params()->_fail_prob, 2)));

            REQUIRE(
                model->count(broken_state, observe, brokenst_state)
                == Approx(
                       total_counts * (1 - d.params()->_fail_prob)
                       * pow(1 - d.params()->_fail_prob, 2)
                       * pow(1 - d.params()->_fail_neighbour_factor, 2) * d.params()->_fail_prob));

            auto const broken_computer_3 = 2;
            auto const brokenst_state_2  = d.breakComputer(broken_state, broken_computer_3);
            REQUIRE(
                model->count(broken_state, observe, brokenst_state_2)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, 2)
                       * (1 - d.params()->_fail_prob) * (1 - d.params()->_fail_neighbour_factor)
                       * (1
                          - (1 - d.params()->_fail_prob)
                                * (1 - d.params()->_fail_neighbour_factor))));

            auto const reboot_random = d.rebootAction(0);
            REQUIRE(
                model->count(broken_state, reboot_random, brokenst_state_2)
                == Approx(
                       total_counts * (1 - d.params()->_fail_prob)
                       * (1 - d.params()->_fail_prob
                          + d.params()->_fail_prob * d.params()->_reboot_success_rate)
                       * (1 - d.params()->_fail_prob) * (1 - d.params()->_fail_neighbour_factor)
                       * (1
                          - (1 - d.params()->_fail_prob)
                                * (1 - d.params()->_fail_neighbour_factor))));

            auto const reboot_broken = d.rebootAction(broken_computer_1);
            REQUIRE(
                model->count(start_state, reboot_broken, broken_state)
                == Approx(
                       total_counts * pow(1 - d.params()->_fail_prob, size - 1)
                       * d.params()->_fail_prob * (1 - d.params()->_reboot_success_rate)));

            REQUIRE(
                model->count(broken_state, reboot_broken, broken_state)
                == Approx(
                       total_counts * (1 - d.params()->_reboot_success_rate)
                       * pow(1 - d.params()->_fail_prob, size - 3)
                       * pow((1 - d.params()->_fail_prob)
                                 * (1 - d.params()->_fail_neighbour_factor),
                             2)));

            d.releaseState(broken_state);
            d.releaseState(brokenst_state);
            d.releaseState(brokenst_state_2);
            d.releaseAction(observe);
            d.releaseAction(reboot_random);
            d.releaseAction(reboot_broken);
        }

        d.releaseState(start_state);
        delete (ba_s);
    }
}

SCENARIO("sysadmin bayes-adaptive observation prior", "[bayes-adaptive][sysadmin][flat]")
{
    for (auto const& network_type : {"independent", "linear"})
    {

        GIVEN("a state in the " + network_type + " network")
        {
            auto const size = rnd::slowRandomInt(2, 10);
            auto const d    = domains::SysAdmin(size, network_type);
            auto const ext  = bayes_adaptive::domain_extensions::SysAdminBAExtension(size);

            auto c                   = configurations::BAConf();
            float const total_counts = 10000;
            c.counts_total           = total_counts;
            c.domain_conf.domain     = network_type + std::string("-sysadmin");
            c.domain_conf.size       = size;

            auto const p = factory::makeTBAPOMDPPrior(d, c);
            auto ba_s    = static_cast<BAPOMDPState*>(p->sample(d.sampleStartState()));

            THEN(
                "every observation count really just depends on whether the computer is "
                "operational")
            {
                auto working = IndexObservation(std::to_string(domains::SysAdmin::OPERATIONAL)),
                     failing = IndexObservation(std::to_string(domains::SysAdmin::FAILING));

                for (auto i = 0; i < 100; ++i)
                {
                    auto s        = ext.getState(std::to_string(rnd::slowRandomInt(0, ext.domainSize()._S)));
                    auto computer = rnd::slowRandomInt(0, size);

                    auto a = rnd::boolean() ? d.observeAction(computer) : d.rebootAction(computer);

                    REQUIRE(
                        ba_s->model()->count(a, s, &working)
                        == (static_cast<SysAdminState const*>(s)->isOperational(computer)
                                ? Approx(total_counts * d.params()->_observe_prob)
                                : Approx(total_counts * (1 - d.params()->_observe_prob))));

                    REQUIRE(
                        ba_s->model()->count(a, s, &failing)
                        == (static_cast<SysAdminState const*>(s)->isOperational(computer)
                                ? Approx(total_counts * (1 - d.params()->_observe_prob))
                                : Approx(total_counts * d.params()->_observe_prob)));

                    d.releaseAction(a);
                }
            }

            d.releaseState(ba_s->_domain_state);
            delete (ba_s);
        }
    }
}

SCENARIO(
    "sysadmin bayes-adaptive factored observation prior",
    "[bayes-adaptive][sysadmin][factored]")
{
    GIVEN("a sysadmin bapomdp state of random size")
    {

        auto const size = rnd::slowRandomInt(2, 10);
        auto const d    = domains::SysAdmin(size, "independent");

        auto const total_counts = 10000.f;

        auto c = configurations::FBAConf();

        c.domain_conf.domain = "independent-sysadmin";
        c.domain_conf.size   = size;
        c.counts_total       = total_counts;

        auto const p    = factory::makeFBAPOMDPPrior(d, c);
        auto const ba_s = static_cast<FBAPOMDPState*>(p->sample(d.sampleStartState()));

        std::vector<int> input_features(size);

        THEN("every observation count really just depends on whether the computer is operational")
        {
            auto working = IndexObservation(std::to_string(domains::SysAdmin::OPERATIONAL)),
                 failing = IndexObservation(std::to_string(domains::SysAdmin::FAILING));

            for (auto i = 0; i < 100; ++i)
            {
                auto computer = rnd::slowRandomInt(0, size);

                // create random input state features
                for (auto f = 0; f < size; ++f) { input_features[f] = rnd::slowRandomInt(0, 2); }

                auto a = rnd::boolean() ? d.observeAction(computer) : d.rebootAction(computer);

                REQUIRE(
                    ba_s->model()->observationNode(a, 0).count(input_features, std::stoi(working.index()))
                    == (input_features[computer]
                            ? Approx(total_counts * d.params()->_observe_prob)
                            : Approx(total_counts * (1 - d.params()->_observe_prob))));

                REQUIRE(
                    ba_s->model()->observationNode(a, 0).count(input_features, std::stoi(failing.index()))
                    == (input_features[computer]
                            ? Approx(total_counts * (1 - d.params()->_observe_prob))
                            : Approx(total_counts * d.params()->_observe_prob)));

                d.releaseAction(a);
            }
        }

        d.releaseState(ba_s->_domain_state);
        delete (ba_s);
    }
}

SCENARIO("fbapomdp independent sysadmin transition prior", "[bayes-adaptive][sysadmin][factored]")
{

    auto const size = rnd::slowRandomInt(3, 10);
    auto c          = configurations::FBAConf();

    c.domain_conf.domain = "independent-sysadmin";
    c.domain_conf.size   = size;

    auto const d = domains::SysAdmin(size, "independent");
    auto s       = d.sampleStartState();

    GIVEN("the sysadmin fbapomdp prior")
    {
        auto p              = factory::makeFBAPOMDPPrior(d, c);
        auto factored_state = static_cast<FBAPOMDPState*>(p->sample(s));

        WHEN("we have the initial (all working) state")
        {
            std::vector<int> initial_X(size, 1);
            THEN("each computer must have a high probability of working after observing")
            {
                for (auto comp = 0; comp < size; ++comp)
                {
                    auto action = d.observeAction(rnd::slowRandomInt(0, size));
                    REQUIRE(
                        factored_state->model()->transitionNode(action, comp).count(initial_X, 0)
                        == Approx(10000 * d.params()->_fail_prob));
                    REQUIRE(
                        factored_state->model()->transitionNode(action, comp).count(initial_X, 1)
                        == Approx(10000 * (1 - d.params()->_fail_prob)));
                    d.releaseAction(action);
                }
            }

            THEN(
                "rebooting should leave that particular computer working with higher probability "
                "than anything else")
            {
                for (auto comp = 0; comp < size; ++comp)
                {
                    auto action = d.rebootAction(comp);
                    REQUIRE(
                        factored_state->model()->transitionNode(action, comp).count(initial_X, 0)
                        == Approx(
                               10000 * d.params()->_fail_prob
                               * (1 - d.params()->_reboot_success_rate)));
                    REQUIRE(
                        factored_state->model()->transitionNode(action, comp).count(initial_X, 1)
                        == Approx(
                               10000
                               * (d.params()->_fail_prob * d.params()->_reboot_success_rate
                                  + (1 - d.params()->_fail_prob))));
                    d.releaseAction(action);
                }
            }

            THEN("rebooting should not affect any other computer")
            {
                auto feature = 0;
                for (auto comp = 0; comp < size; ++comp)
                {
                    // get a feature other than comp
                    do
                    {
                        feature = rnd::slowRandomInt(0, size);
                    } while (feature == comp);

                    auto action = d.rebootAction(comp);
                    REQUIRE(
                        factored_state->model()->transitionNode(action, feature).count(initial_X, 0)
                        == Approx(10000 * d.params()->_fail_prob));
                    REQUIRE(
                        factored_state->model()->transitionNode(action, feature).count(initial_X, 1)
                        == Approx(10000 * (1 - d.params()->_fail_prob)));
                    d.releaseAction(action);
                }
            }
        }

        WHEN("we have the 0 (all failing) state")
        {
            std::vector<int> initial_X(size, 0);
            THEN("each computer must be failing after observing")
            {
                for (auto comp = 0; comp < size; ++comp)
                {
                    auto action = d.observeAction(comp);
                    REQUIRE(
                        factored_state->model()->transitionNode(action, comp).count(initial_X, 0)
                        == Approx(10000));
                    REQUIRE(
                        factored_state->model()->transitionNode(action, comp).count(initial_X, 1)
                        == Approx(0));
                    d.releaseAction(action);
                }
            }

            THEN("rebooting should get that particular computer working with high probability")
            {
                for (auto comp = 0; comp < size; ++comp)
                {
                    auto action = d.rebootAction(comp);
                    REQUIRE(
                        factored_state->model()->transitionNode(action, comp).count(initial_X, 0)
                        == Approx(10000 * (1 - d.params()->_reboot_success_rate)));
                    REQUIRE(
                        factored_state->model()->transitionNode(action, comp).count(initial_X, 1)
                        == Approx(10000 * (d.params()->_reboot_success_rate)));
                    d.releaseAction(action);
                }
            }

            THEN("rebooting should leave other computers failing")
            {
                auto feature = 0;
                for (auto comp = 0; comp < size; ++comp)
                {
                    // get a feature other than comp
                    do
                    {
                        feature = rnd::slowRandomInt(0, size);
                    } while (feature == comp);

                    auto action = d.rebootAction(comp);
                    REQUIRE(
                        factored_state->model()->transitionNode(action, feature).count(initial_X, 0)
                        == Approx(10000));
                    REQUIRE(
                        factored_state->model()->transitionNode(action, feature).count(initial_X, 1)
                        == Approx(0));
                    d.releaseAction(action);
                }
            }
        }
        delete (factored_state);
    }

    d.releaseState(s);
}

SCENARIO("linear sysadmin factored prior", "[bayes-adaptive][sysadmin][factored]")
{

    size_t const size = 4;

    auto const d   = domains::SysAdmin(size, "linear");
    auto const ext = bayes_adaptive::domain_extensions::SysAdminBAExtension(size);
    configurations::FBAConf c;

    c.domain_conf.domain = "linear-sysadmin";
    c.domain_conf.size   = size;

    GIVEN("a model from the accurate prior")
    {

        auto const p           = factory::makeFBAPOMDPPrior(d, c);
        auto const start_state = d.sampleStartState();
        auto const ba_s        = static_cast<FBAPOMDPState*>(p->sample(start_state));

        auto const& model = ba_s->model();

        THEN("probability of failing depends on failing neighbours")
        {

            auto const observe = d.observeAction(rnd::slowRandomInt(0, size));

            std::vector<int> const all_working_computers(size, 1);

            auto expectations = model->transitionNode(observe, rnd::slowRandomInt(0, size))
                                    .expectation(all_working_computers);
            REQUIRE(Approx(expectations[0]) == d.params()->_fail_prob);
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            auto const reboot_2 = d.rebootAction(2);

            expectations = model->transitionNode(reboot_2, 2).expectation(all_working_computers);
            REQUIRE(
                Approx(expectations[0])
                == d.params()->_fail_prob * (1 - d.params()->_reboot_success_rate));
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            std::vector<int> broken_computer = all_working_computers;
            broken_computer[0]               = 0;

            expectations = model->transitionNode(reboot_2, 2).expectation(broken_computer);
            REQUIRE(
                Approx(expectations[0])
                == d.params()->_fail_prob * (1 - d.params()->_reboot_success_rate));
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            expectations = model->transitionNode(observe, 0).expectation(broken_computer);
            REQUIRE(Approx(expectations[0]) == 1);
            REQUIRE(Approx(expectations[1]) == 0);

            expectations = model->transitionNode(reboot_2, 0).expectation(broken_computer);
            REQUIRE(Approx(expectations[0]) == 1);
            REQUIRE(Approx(expectations[1]) == 0);

            expectations = model->transitionNode(observe, 1).expectation(broken_computer);
            REQUIRE(
                Approx(expectations[0])
                == (1 - (1 - d.params()->_fail_neighbour_factor) * (1 - d.params()->_fail_prob)));
            REQUIRE(
                Approx(expectations[0])
                == d.failProbability(d.getState(broken_computer), observe, 1));
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            expectations = model->transitionNode(reboot_2, 1).expectation(broken_computer);
            REQUIRE(
                Approx(expectations[0])
                == d.failProbability(d.getState(broken_computer), reboot_2, 1));
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            expectations = model->transitionNode(observe, 2).expectation(broken_computer);
            REQUIRE(Approx(expectations[0]) == d.params()->_fail_prob);
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            auto const reboot_broken = d.rebootAction(0);

            expectations = model->transitionNode(reboot_broken, 0).expectation(broken_computer);
            REQUIRE(Approx(expectations[0]) == 1 - d.params()->_reboot_success_rate);
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            d.releaseAction(observe);
            d.releaseAction(reboot_2);
            d.releaseAction(reboot_broken);
        }

        d.releaseState(start_state);
        delete (ba_s);
    }

    GIVEN("a model from the fully connected (correct) prior")
    {

        auto const p           = factory::makeFBAPOMDPPrior(d, c);
        auto const start_state = d.sampleStartState();
        auto const ba_s = static_cast<FBAPOMDPState*>(p->sampleFullyConnectedState(start_state));

        auto const model = ba_s->model();

        THEN("all transition nodes should have all parents")
        {
            IndexAction random_action(std::to_string(rnd::slowRandomInt(0, ext.domainSize()._A)));
            int random_computer(rnd::slowRandomInt(0, 4));

            REQUIRE(
                *model->transitionNode(&random_action, random_computer).parents()
                == std::vector<int>({0, 1, 2, 3}));
        }

        THEN("probability of failing depends on failing neighbours")
        {

            auto const observe = d.observeAction(rnd::slowRandomInt(0, size));

            std::vector<int> const all_working_computers(size, 1);

            auto expectations = model->transitionNode(observe, rnd::slowRandomInt(0, size))
                                    .expectation(all_working_computers);
            REQUIRE(Approx(expectations[0]) == d.params()->_fail_prob);
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            auto const reboot_2 = d.rebootAction(2);

            expectations = model->transitionNode(reboot_2, 2).expectation(all_working_computers);
            REQUIRE(
                Approx(expectations[0])
                == d.params()->_fail_prob * (1 - d.params()->_reboot_success_rate));
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            std::vector<int> broken_computer = all_working_computers;
            broken_computer[0]               = 0;

            expectations = model->transitionNode(reboot_2, 2).expectation(broken_computer);
            REQUIRE(
                Approx(expectations[0])
                == d.params()->_fail_prob * (1 - d.params()->_reboot_success_rate));
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            expectations = model->transitionNode(observe, 0).expectation(broken_computer);
            REQUIRE(Approx(expectations[0]) == 1);
            REQUIRE(Approx(expectations[1]) == 0);

            expectations = model->transitionNode(reboot_2, 0).expectation(broken_computer);
            REQUIRE(Approx(expectations[0]) == 1);
            REQUIRE(Approx(expectations[1]) == 0);

            expectations = model->transitionNode(observe, 1).expectation(broken_computer);
            REQUIRE(
                Approx(expectations[0])
                == (1 - (1 - d.params()->_fail_neighbour_factor) * (1 - d.params()->_fail_prob)));
            REQUIRE(
                Approx(expectations[0])
                == d.failProbability(d.getState(broken_computer), observe, 1));
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            expectations = model->transitionNode(reboot_2, 1).expectation(broken_computer);
            REQUIRE(
                Approx(expectations[0])
                == d.failProbability(d.getState(broken_computer), reboot_2, 1));
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            expectations = model->transitionNode(observe, 2).expectation(broken_computer);
            REQUIRE(Approx(expectations[0]) == d.params()->_fail_prob);
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            auto const reboot_broken = d.rebootAction(0);

            expectations = model->transitionNode(reboot_broken, 0).expectation(broken_computer);
            REQUIRE(Approx(expectations[0]) == 1 - d.params()->_reboot_success_rate);
            REQUIRE(Approx(expectations[1]) == 1 - expectations[0]);

            d.releaseAction(observe);
            d.releaseAction(reboot_2);
            d.releaseAction(reboot_broken);
        }

        d.releaseState(start_state);
        delete (ba_s);
    }
}

SCENARIO(
    "print sysadmin bayes-adaptive prior",
    "[sysadmin][bayes-adaptive][factored][hide][logging]")
{

    auto const size = 3;
    auto c          = configurations::FBAConf();

    c.counts_total     = 10;
    c.noise            = .2f;
    c.domain_conf.size = size;

    for (auto const version : {"independent", "linear"})
    {

        c.domain_conf.domain = version + std::string("-sysadmin");

        auto const d          = domains::SysAdmin(size, version);
        auto const p          = factory::makeTBAPOMDPPrior(d, c);
        auto const factored_p = factory::makeFBAPOMDPPrior(d, c);

        auto s = d.sampleStartState();

        auto ba_s  = static_cast<BAPOMDPState const*>(p->sample(s));
        auto fba_s = static_cast<FBAPOMDPState const*>(factored_p->sample(s));

        LOG(INFO) << version << "Sysadmin BAPOMDP Prior:";
        ba_s->logCounts();

        LOG(INFO) << version << "Sysadmin FBAPOMDP Prior:";
        fba_s->logCounts();
    }
}
