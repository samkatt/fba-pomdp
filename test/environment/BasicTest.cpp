#include "catch.hpp"

#include <cmath>
#include <limits>
#include <string>

#include "environment/Action.hpp"
#include "environment/Discount.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "utils/random.hpp"

// templated test, tests whether initiating and setting indices works as intended
template<class T, class GENERATOR>
void testIndexClass(std::string const& name, GENERATOR& distr)
{
    GIVEN("an " + name)
    {
        auto i       = distr(rnd::rng());
        auto to_test = T(std::to_string(i));

        THEN("index should be as initiated") { REQUIRE(to_test.index() == std::to_string(i)); }

        WHEN("changing index")
        {
            auto j = distr(rnd::rng());
            to_test.index(std::to_string(j));

            THEN("index should be as changed") { REQUIRE(to_test.index() == std::to_string(j)); }
        }
    }
}

TEST_CASE("Indices", "[environment]")
{
    rnd::initiate();
    auto int_distr = rnd::integerDistribution(0, std::numeric_limits<int>::max());

    testIndexClass<IndexAction, std::uniform_int_distribution<int>>("action", int_distr);
    testIndexClass<IndexObservation, std::uniform_int_distribution<int>>("observation", int_distr);
    testIndexClass<IndexState, std::uniform_int_distribution<int>>("state", int_distr);
}

TEST_CASE("Discount", "[environment][discount]")
{
    rnd::initiate();
    auto int_distr = rnd::integerDistribution(0, 100);

    GIVEN("a discount of .9")
    {
        auto d = Discount(.9);
        REQUIRE(d.toDouble() == 1);

        d.increment();
        REQUIRE(d.toDouble() == .9);
        d.increment();
        REQUIRE(d.toDouble() == .81);
        d.increment();
        REQUIRE(d.toDouble() == .7290000000000001);
    }

    GIVEN("a random discount")
    {
        auto random_double = rnd::uniform_rand01();
        auto random_int    = int_distr(rnd::rng());

        auto d = Discount(random_double);
        REQUIRE(d.toDouble() == 1);

        for (auto i = 0; i < random_int; i++) { d.increment(); }

        REQUIRE(d.toDouble() == Approx(pow(random_double, random_int)));
    }
}
