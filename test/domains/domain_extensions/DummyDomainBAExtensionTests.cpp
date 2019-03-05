#include "catch.hpp"

#include "domains/dummy/DummyDomain.hpp"
#include "domains/dummy/DummyDomainBAExtension.hpp"

SCENARIO("dummy domain bapomdp extension", "[bayes-adaptive][dummy]")
{

    auto const d   = domains::DummyDomain();
    auto const ext = bayes_adaptive::domain_extensions::DummyDomainBAExtension();

    THEN("no steps are terminal")
    {
        auto s = d.sampleStartState();
        auto a = d.generateRandomAction(s);

        REQUIRE(ext.terminal(s, a, s).terminated() == false);

        d.releaseState(s);
        d.releaseAction(a);
    }

    THEN("reward is always 1")
    {
        auto s = d.sampleStartState();
        auto a = d.generateRandomAction(s);

        REQUIRE(ext.reward(s, a, s).toDouble() == 1.0);

        d.releaseState(s);
        d.releaseAction(a);
    }

    THEN("POMDP space size is 1")
    {
        auto size = ext.domainSize();

        REQUIRE(size._A == 1);
        REQUIRE(size._S == 1);
        REQUIRE(size._O == 1);
    }
}
