#include "catch.hpp"

#include "configurations/BAConf.hpp"
#include "configurations/FBAConf.hpp"

SCENARIO("config validation", "[config]")
{
    GIVEN("The default configurations")
    {
        auto c = configurations::BAConf();

        REQUIRE_THROWS(c.validate());

        c.planner_conf.mcts_max_depth = c.horizon;
        REQUIRE_THROWS(c.validate());

        c.domain_conf.domain = "episodic-tiger";
        REQUIRE_NOTHROW(c.validate());

        WHEN("The size of the domain is set to 4")
        {
            c.domain_conf.size = 4;

            THEN("It should complain about setting size of a static sized domain")
            REQUIRE_THROWS(c.validate());

            AND_WHEN("We set the domain to a sized domain")
            {
                c.domain_conf.domain = "independent-sysadmin";
                REQUIRE_NOTHROW(c.validate());
            }
        }
    }
}

SCENARIO("FBA-POMDP config validation", "[config][factored]")
{

    GIVEN("The default configurations")
    {
        auto c = configurations::FBAConf();

        REQUIRE_THROWS(c.validate());

        c.planner_conf.mcts_max_depth = c.horizon;
        REQUIRE_THROWS(c.validate());

        c.domain_conf.domain = "episodic-tiger";
        REQUIRE_THROWS(c.validate());

        c.domain_conf.domain = "episodic-factored-tiger";
        REQUIRE_THROWS(c.validate());

        WHEN("The size of the domain is set to 4")
        {
            c.domain_conf.size = 4;

            AND_WHEN("We set the domain to a sized domain")
            REQUIRE_NOTHROW(c.validate());
        }

        c.domain_conf.domain = "continuous-factored-tiger";
        c.domain_conf.size   = 2;
        REQUIRE_NOTHROW(c.validate());
    }
}
