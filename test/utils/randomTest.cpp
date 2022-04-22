#include "catch.hpp"

#include <vector>

#include "utils/random.hpp"

TEST_CASE("random", "[utils][random]")
{
    WHEN("Sampling from a gamma distribution")
    {
        THEN("sample with shape different shapes results in predictable values")
        {
            REQUIRE(rnd::sample::gamma(0) == 0);

            REQUIRE(rnd::sample::gamma(1) > 0);
            REQUIRE(rnd::sample::gamma(1) < 10);

            REQUIRE(rnd::sample::gamma(2) > 0);
            REQUIRE(rnd::sample::gamma(2) < 10);

            REQUIRE(rnd::sample::gamma(5) > 0);
            REQUIRE(rnd::sample::gamma(5) < 25);

            REQUIRE(rnd::sample::gamma(20) > 5);
            REQUIRE(rnd::sample::gamma(20) < 45);

            REQUIRE(rnd::sample::gamma(40) > 15);
            REQUIRE(rnd::sample::gamma(40) < 65);

            REQUIRE(rnd::sample::gamma(70) > 35);
            REQUIRE(rnd::sample::gamma(70) < 105);

            REQUIRE(rnd::sample::gamma(100) > 65);
            REQUIRE(rnd::sample::gamma(100) < 135);
        }
    }

    auto size = 4;
    WHEN("Dirichlet counts are all 0 but one")
    {
        THEN("Sampling gives back that one")
        {
            for (auto i = 0; i < size; ++i)
            {
                std::vector<float> counts(size);
                counts[i] = (float)i + 1;

                REQUIRE(rnd::sample::Dir::sampleFromSampledMult(&counts[0], size) == i);
                REQUIRE(rnd::sample::Dir::sampleFromExpectedMult(&counts[0], size) == i);
            }
        }
    }

    WHEN("Dirichlet counts are all non-zero but one")
    {
        THEN("Sampling does not give that one back")
        {
            for (auto i = 0; i < size; ++i)
            {
                std::vector<float> counts(size);
                for (auto j = 0; j < size; ++j)
                {
                    if (i != j)
                    {
                        counts[j] = (float)j + 1;
                    }
                }

                REQUIRE(rnd::sample::Dir::sampleFromSampledMult(&counts[0], size) != i);
                REQUIRE(rnd::sample::Dir::sampleFromExpectedMult(&counts[0], size) != i);
                REQUIRE(rnd::sample::Dir::sampleFromSampledMult(&counts[0], size) < size);
                REQUIRE(rnd::sample::Dir::sampleFromExpectedMult(&counts[0], size) < size);
            }
        }
    }

    WHEN("One count is much higher than the other dirichlet counts")
    {
        for (auto i = 0; i < size; ++i)
        {
            std::vector<float> counts(size);

            for (auto j = 0; j < size; ++j) { counts[j] = (float)rnd::uniform_rand01() * 5; }

            counts[i] = 50000;

            REQUIRE(rnd::sample::Dir::sampleFromSampledMult(&counts[0], size) == i);
            REQUIRE(rnd::sample::Dir::sampleFromExpectedMult(&counts[0], size) == i);
        }
    }

    WHEN("computing expected multinomials of dirichlet distributions")
    {
        auto dir  = std::vector<float>({1, 2, 3, 4, 5});
        auto mult = rnd::sample::Dir::expectedMult(&dir[0], 5);

        REQUIRE(mult[0] == Approx(1. / 15));
        REQUIRE(mult[1] == Approx(2. / 15));
        REQUIRE(mult[2] == Approx(1. / 5));
        REQUIRE(mult[3] == Approx(4. / 15));
        REQUIRE(mult[4] == Approx(1. / 3));

        dir  = std::vector<float>({.3, .1, .2});
        mult = rnd::sample::Dir::expectedMult(&dir[0], 3);

        REQUIRE(mult[0] == Approx(.5));
        REQUIRE(mult[1] == Approx(1. / 6));
        REQUIRE(mult[2] == Approx(1. / 3));
    }

    WHEN("sampling multinominals from dirichlet distributions")
    {
        auto dir  = std::vector<float>({0, 0, 5.2});
        auto mult = rnd::sample::Dir::sampleMult(&dir[0], 3);

        REQUIRE(mult[0] == 0);
        REQUIRE(mult[1] == 0);
        REQUIRE(mult[2] == 1);

        dir  = std::vector<float>({2, 15});
        mult = rnd::sample::Dir::sampleMult(&dir[0], 2);
        REQUIRE(mult[0] < mult[1]);
        REQUIRE(mult[0] < 1);
        REQUIRE(mult[1] < 1);

        dir  = std::vector<float>({.1, .1, .1, .1, .1, .1});
        mult = rnd::sample::Dir::sampleMult(&dir[0], 6);

        for (auto& i : mult)
        {
            REQUIRE(i < 1);
            REQUIRE(i > 0);
        }

        dir  = std::vector<float>({10000000, 10000000, 10000000, 10000000, 10000000});
        mult = rnd::sample::Dir::sampleMult(&dir[0], 5);

        for (auto& i : mult)
        {
            for (auto& j : mult) { REQUIRE(i == Approx(j).epsilon(0.001)); }
        }
    }
}

SCENARIO("calculating normal cdf", "[random][utils]")
{
    REQUIRE(rnd::normal::cdf(0, 0, 1) == .5);
    REQUIRE(rnd::normal::cdf(-3, 0, 1) == Approx(0.00135).margin(0.00001));
    REQUIRE(rnd::normal::cdf(.5, 0, 1) == Approx(.69146).margin(0.00001));
    REQUIRE(rnd::normal::cdf(.5, 2, 1) == Approx(.06681).margin(0.00001));
    REQUIRE(rnd::normal::cdf(.5, 2, 1.857417562100671) == Approx(.20967).margin(0.00001));
}

SCENARIO("random ints", "[random][utils]")
{
    for (auto i = 0; i < 10; ++i)
    {
        REQUIRE(rnd::slowRandomInt(4, 6) < 6);
        REQUIRE(rnd::slowRandomInt(4, 6) >= 4);
    }

    auto tmp = rnd::slowRandomInt(0, 1000);

    REQUIRE(rnd::slowRandomInt(tmp, tmp) == tmp);

    for (auto i = 0; i < 20; ++i)
    {

        auto min = rnd::slowRandomInt(0, 10);

        REQUIRE(min >= 0);
        REQUIRE(min < 10);

        auto max = rnd::slowRandomInt(min + 1, 20);
        REQUIRE(max >= min);
        REQUIRE(max < 20);

        auto x = rnd::slowRandomInt(min, max);
        REQUIRE(x >= min);
        REQUIRE(x < max);
    }

    REQUIRE(rnd::slowRandomInt(0, 1) == 0);
    REQUIRE(rnd::slowRandomInt(107, 108) == 107);
}
