#include "catch.hpp"

#include "configurations/BAConf.hpp"
#include "domains/dummy/FactoredDummyDomain.hpp"
#include "domains/dummy/FactoredDummyDomainBAExtension.hpp"
#include "domains/dummy/FactoredDummyDomainPriors.hpp"

SCENARIO("factored dummy table prior", "[factored][bayes-adaptive][dummy][flat]")
{

    auto const size      = 3;
    auto c               = configurations::BAConf();
    c.domain_conf.domain = "factored-dummy";
    c.domain_conf.size   = size;

    auto const d = domains::FactoredDummyDomain(c.domain_conf.size);
    auto const ext =
        bayes_adaptive::domain_extensions::FactoredDummyDomainBAExtension(c.domain_conf.size);
    auto const p = priors::FactoredDummyPrior(c.domain_conf.size);

    auto ba_state = static_cast<BAPOMDPPrior const&>(p).sample(d.sampleStartState());

    auto a_up    = IndexAction(domains::FactoredDummyDomain::ACTIONS::UP),
         a_right = IndexAction(domains::FactoredDummyDomain::ACTIONS::RIGHT);

    THEN("going up should sample a state higher")
    {
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_up, rnd::sample::Dir::sampleFromSampledMult)
            == 1);

        d.releaseState(ba_state->_domain_state);
        ba_state->_domain_state = ext.getState(1);

        AND_THEN("another step up should do the same")
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_up, rnd::sample::Dir::sampleFromSampledMult)
            == 2);

        AND_THEN("a step right here should increase by step size")
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_right, rnd::sample::Dir::sampleFromSampledMult)
            == 1 + size);
    }

    THEN("going right should sample a state to the right")
    {
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_right, rnd::sample::Dir::sampleFromSampledMult)
            == size);

        d.releaseState(ba_state->_domain_state);
        ba_state->_domain_state = ext.getState(size);
        AND_THEN("another step up should do the same")
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_right, rnd::sample::Dir::sampleFromSampledMult)
            == 2 * size);

        AND_THEN("a step up here should increase by 1")
        REQUIRE(
            ba_state->sampleStateIndex(
                ba_state->_domain_state, &a_up, rnd::sample::Dir::sampleFromSampledMult)
            == 1 + size);
    }

    d.releaseState(ba_state->_domain_state);
    delete (ba_state);
}
