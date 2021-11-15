#include "catch.hpp"

#include "domains/tiger/FactoredTiger.hpp"
#include "domains/tiger/FactoredTigerBAExtension.hpp"
#include "domains/tiger/FactoredTigerFBAExtension.hpp"
#include "utils/random.hpp"

SCENARIO("factored tiger BAPOMDP", "[bayes-adaptive][factored][tiger]")
{
    // tests should work for any number of (irrelevant) features
    int num_features = rnd::slowRandomInt(1, 5);

    auto tiger_types = std::vector<domains::FactoredTiger::FactoredTigerDomainType>(
        {domains::FactoredTiger::FactoredTigerDomainType::EPISODIC,
         domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS});

    for (auto type : tiger_types)
    {

        GIVEN("the bayes-adaptive factored tiger problem " + std::to_string(type))
        {

            auto const d = domains::FactoredTiger(type, num_features);
            auto const ext =
                bayes_adaptive::domain_extensions::FactoredTigerBAExtension(type, num_features);

            THEN("the domain size should be specified correctly")
            {
                auto s = ext.domainSize();

                REQUIRE(s._S == 2 << num_features);
                REQUIRE(s._A == 3);
                REQUIRE(s._O == 2);
            }

            THEN("the correct state should be returned when requested")
            {
                int i  = rnd::slowRandomInt(0, 2 << num_features);
                auto s = ext.getState(std::to_string(i));

                REQUIRE(s->index() == std::to_string(i));

                d.releaseState(s);
            }

            WHEN("requesting the reward of an interaction " + std::to_string(type))
            {
                auto s = d.sampleStartState();

                THEN("observing should always produce -1")
                {
                    auto a = IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OBSERVE));
                    REQUIRE(ext.reward(s, &a, s).toDouble() == -1.0);
                }

                THEN("opening a door depends on whether is the correct one")
                {
                    auto new_s = d.sampleStartState();
                    auto a     = IndexAction(std::to_string(rnd::slowRandomInt(0, 2))); // open right or left

                    auto r = (a.index() == std::to_string(d.tigerLocation(s))) ? 10 : -100;
                    REQUIRE(ext.reward(s, &a, new_s).toDouble() == r);

                    d.releaseState(new_s);
                }

                d.releaseState(s);
            }
        }
    }

    GIVEN("the bayes-adaptive episodic factored tiger problem")
    {
        auto const d = domains::FactoredTiger(
            domains::FactoredTiger::FactoredTigerDomainType::EPISODIC, num_features);
        auto const ext = bayes_adaptive::domain_extensions::FactoredTigerBAExtension(
            domains::FactoredTiger::FactoredTigerDomainType::EPISODIC, num_features);

        auto s = d.sampleStartState();

        THEN("observing should never terminate an episode")
        {
            auto observe = IndexAction(std::to_string(domains::FactoredTiger::TigerAction::OBSERVE));
            REQUIRE(!ext.terminal(s, &observe, s).terminated());
        }

        THEN("opening a door should terminate an episode")
        {
            auto open_door = IndexAction(std::to_string(rnd::slowRandomInt(0, 2))); // open right or left

            auto new_s = d.sampleStartState();
            REQUIRE(ext.terminal(s, &open_door, s).terminated());

            d.releaseState(new_s);
        }

        d.releaseState(s);
    }

    GIVEN("the bayes-adaptive continuous factored tiger problem")
    {
        auto const d = domains::FactoredTiger(
            domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS, num_features);
        auto const ext = bayes_adaptive::domain_extensions::FactoredTigerBAExtension(
            domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS, num_features);

        THEN("no action should ever terminate an episode")
        {
            auto s     = d.sampleStartState();
            auto a     = IndexAction(std::to_string(rnd::slowRandomInt(0, 2)));
            auto new_s = s;

            // when opneing a door, we should test for any possible new state
            if (a.index() != std::to_string(domains::FactoredTiger::TigerAction::OBSERVE))
            {
                new_s = d.sampleStartState();
            }

            REQUIRE(!ext.terminal(s, &a, new_s).terminated());

            // if opening a door, we should also release the new state
            if (a.index() != std::to_string(domains::FactoredTiger::TigerAction::OBSERVE))
            {
                d.releaseState(new_s);
            }

            d.releaseState(s);
        }
    }
}

SCENARIO("factored tiger prior", "[bayes-adaptive][domain][factored][tiger]")
{
    auto const size = rnd::slowRandomInt(1, 5);

    auto const _s_size = bayes_adaptive::domain_extensions::FactoredTigerBAExtension(
                             domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS, size)
                             .domainSize()
                             ._S;
    bayes_adaptive::domain_extensions::FactoredTigerFBAExtension const f_ext(size);

    // all states have equal probability of being sampled
    REQUIRE(f_ext.statePrior()->prob(rnd::slowRandomInt(0, _s_size - 1)) == Approx(1. / _s_size));
}
