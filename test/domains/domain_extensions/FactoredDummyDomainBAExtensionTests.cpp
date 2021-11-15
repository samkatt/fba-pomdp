#include "catch.hpp"

#include "domains/dummy/FactoredDummyDomain.hpp"
#include "domains/dummy/FactoredDummyDomainBAExtension.hpp"
#include "domains/dummy/FactoredDummyDomainFBAExtension.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "utils/random.hpp"

SCENARIO("factored dummy extension test", "[bayes-adaptive][dummy][factored]")
{
    auto const size = 3;

    auto const d   = domains::FactoredDummyDomain(size);
    auto const ext = bayes_adaptive::domain_extensions::FactoredDummyDomainBAExtension(size);

    Reward r(0);

    WHEN("A simple factored dummy domain exists")
    {
        THEN("POMDP space size is {S:n*n,O:1,A:2}")
        {
            auto domain_size = ext.domainSize();

            REQUIRE(domain_size._A == 2);
            REQUIRE(domain_size._S == size * size);
            REQUIRE(domain_size._O == 1);
        }

        THEN("no steps are terminal")
        {
            Observation const* o(nullptr);

            for (auto i = 0; i < 10; ++i)
            {
                auto s = ext.getState(std::to_string(rnd::slowRandomInt(0, ext.domainSize()._S - 1)));
                auto a = d.generateRandomAction(s);

                auto s_old = d.copyState(s);

                d.step(&s, a, &o, &r);
                REQUIRE(ext.terminal(s_old, a, s).terminated() == false);

                REQUIRE(d.computeObservationProbability(o, a, s) == 1);

                d.releaseState(s);
                d.releaseState(s_old);
                d.releaseAction(a);
                d.releaseObservation(o);
            }
        }

        THEN("reward is always -1, except in corner")
        {
            Observation const* o(nullptr);

            for (auto i = 0; i < 10; ++i)
            {

                auto s = ext.getState(std::to_string(rnd::slowRandomInt(0, ext.domainSize()._S)));

                auto s_old = d.copyState(s);
                auto a     = d.generateRandomAction(s);

                d.step(&s, a, &o, &r);
                if (s->index() == std::to_string(ext.domainSize()._S - 1))
                {
                    REQUIRE(ext.reward(s_old, a, s).toDouble() == 0.0);
                } else
                {
                    REQUIRE(ext.reward(s_old, a, s).toDouble() == -1.0);
                }

                d.releaseState(s);
                d.releaseState(s_old);
                d.releaseAction(a);
                d.releaseObservation(o);
            }

            auto s     = ext.getState(std::to_string(ext.domainSize()._S - 1));
            auto a     = d.generateRandomAction(s);
            auto s_old = d.copyState(s);

            d.step(&s, a, &o, &r);
            REQUIRE(ext.reward(s_old, a, s).toDouble() == 0.0);

            d.releaseState(s);
            d.releaseState(s_old);
            d.releaseAction(a);
            d.releaseObservation(o);
        }
    }
}

SCENARIO("factored dummy domain prior", "[factored][bayes-adaptive][dummy][domain]")
{
    auto const size = rnd::slowRandomInt(1, 10);
    bayes_adaptive::domain_extensions::FactoredDummyDomainFBAExtension const f_ext(size);

    // initial state is always 0
    REQUIRE(f_ext.statePrior()->sample() == 0);
    REQUIRE(f_ext.statePrior()->prob(0) == 1);
}
