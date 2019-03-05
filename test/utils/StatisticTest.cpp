#include "catch.hpp"

#include "utils/Statistic.hpp"

TEST_CASE("statistic", "[utils][statistics]")
{
    auto stat = utils::Statistic();
    WHEN("initiated statistic")
    {
        REQUIRE(stat.mean() == 0);
        REQUIRE(stat.count() == 0);
        REQUIRE(stat.var() == 0);
        REQUIRE(stat.stder() == 0);
    }

    WHEN("stat represents some values")
    {
        stat.add(5);
        stat.add(10);
        stat.add(100);
        stat.add(1);

        REQUIRE(stat.mean() == Approx(29.0));
        REQUIRE(stat.count() == 4);
        REQUIRE(stat.var() == Approx(2254));
        REQUIRE(stat.stder() == Approx(23.7381));
    }

    WHEN("stat represents negative values")
    {
        stat.add(-3);
        stat.add(-1000);

        REQUIRE(stat.mean() == -501.5);
        REQUIRE(stat.count() == 2);
        REQUIRE(stat.var() == 497004.5);
        REQUIRE(stat.stder() == Approx(498.5));
    }

    WHEN("stats has alot of values")
    {
        auto n = 10001;

        for (auto i = 1; i < n; ++i) { stat.add(i); }

        REQUIRE(stat.mean() == n / 2.0);
        REQUIRE(stat.count() == n - 1);
        REQUIRE(stat.var() == Approx(8334166.666666667));
        REQUIRE(stat.stder() == Approx(28.8689567991));
    }

    WHEN("stats has alot of small/big number variation")
    {
        stat.add(0.00001);
        stat.add(100000);
        stat.add(0.01);
        stat.add(58238);

        REQUIRE(stat.mean() == Approx(39559.5025025));
        REQUIRE(stat.count() == 4);
        REQUIRE(stat.var() == Approx(2377282563.673));
        REQUIRE(stat.stder() == Approx(24378.69370816793));
    }
}
