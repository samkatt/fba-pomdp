#include "catch.hpp"

#include "domains/sysadmin/SysAdmin.hpp"
#include "domains/sysadmin/SysAdminBAExtension.hpp"
#include "domains/sysadmin/SysAdminFBAExtension.hpp"
#include "utils/random.hpp"

SCENARIO("bayes-adaptive sysadmin", "[domain][bayes-adaptive][sysadmin]")
{

    GIVEN("The sysadmin bayes-adaptive problem")
    {
        auto size      = rnd::slowRandomInt(1, 10);
        auto const d   = domains::SysAdmin(size, "independent");
        auto const ext = bayes_adaptive::domain_extensions::SysAdminBAExtension(size);

        THEN(
            "We expect the state size to be 2^n, observation space to be 2 and the action space to "
            "be 2*n")
        {
            auto domain_size = ext.domainSize();

            REQUIRE(domain_size._S == pow(2, size));
            REQUIRE(domain_size._O == 2);
            REQUIRE(domain_size._A == 2 * size);
        }

        THEN("updating a state (index) will do exactly that")
        {
            for (auto i = 0; i < 10; ++i)
            {
                auto r = rnd::slowRandomInt(0, ext.domainSize()._S);

                auto s = ext.getState(r);

                REQUIRE(s->index() == r);

                d.releaseState(s);
            }
        }

        THEN("no <s,a,s'> transition is terminal")
        {
            for (auto i = 0; i < 10; ++i)
            {
                auto s     = ext.getState(rnd::slowRandomInt(0, ext.domainSize()._S));
                auto a     = d.generateRandomAction(s);
                auto new_s = ext.getState(rnd::slowRandomInt(0, ext.domainSize()._S));

                REQUIRE(!ext.terminal(s, a, new_s).terminated());

                d.releaseState(s);
                d.releaseState(new_s);
                d.releaseAction(a);
            }
        }

        THEN("reward() should be the number of working computers - cost of action")
        {
            for (auto i = 0; i < 10; ++i)
            {

                auto s     = ext.getState(rnd::slowRandomInt(0, ext.domainSize()._S));
                auto new_s = ext.getState(rnd::slowRandomInt(0, ext.domainSize()._S));

                auto observe = d.observeAction(rnd::slowRandomInt(0, size));
                auto reboot  = d.rebootAction(rnd::slowRandomInt(0, size));

                REQUIRE(
                    ext.reward(s, observe, new_s).toDouble()
                    == (float)static_cast<domains::SysAdminState const*>(new_s)
                           ->numOperationalComputers());
                REQUIRE(
                    ext.reward(s, reboot, new_s).toDouble()
                    == (float)static_cast<domains::SysAdminState const*>(new_s)
                               ->numOperationalComputers()
                           - d.params()->_reboot_cost);
            }
        }
    }
}

SCENARIO("factored bayes-adaptive sysadmin", "[domain][bayes-adaptive][sysadmin][factored]")
{

    WHEN("We request the domain feature size of the sysadmin problem")
    {

        std::size_t size = rnd::slowRandomInt(2, 10);

        auto const f_ext = bayes_adaptive::domain_extensions::SysAdminFBAExtension(size);

        auto const f_size = f_ext.domainFeatureSize();
        REQUIRE(f_size._O.size() == 1);
        REQUIRE(f_size._S.size() == size);
        REQUIRE(f_size._O[0] == 2);

        for (auto const f : f_size._S) { REQUIRE(f == 2); }
    }
}

SCENARIO("sysadmin state prior", "[bayes-adaptive][domain][sysadmin][factored]")
{

    auto const num_comp(rnd::slowRandomInt(3, 7));
    bayes_adaptive::domain_extensions::SysAdminBAExtension const ext(num_comp);
    bayes_adaptive::domain_extensions::SysAdminFBAExtension const f_ext(num_comp);

    auto const _s_size = ext.domainSize()._S;

    REQUIRE(static_cast<int>(f_ext.statePrior()->sample()) == _s_size - 1);
    ;
    REQUIRE(static_cast<int>(f_ext.statePrior()->prob(_s_size - 1)) == 1);
}
