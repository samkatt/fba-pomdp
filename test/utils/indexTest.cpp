#include "catch.hpp"

#include "utils/index.hpp"
#include "utils/random.hpp"

SCENARIO("indexing", "[utils][index]")
{
    GIVEN("Three dimensions of size [*,3,8]")
    {
        auto size_y = 3, size_z = 8;

        using indexing::threeToOne;
        THEN("Projecting these into one dimension works as intended")
        {
            REQUIRE(threeToOne(0, 0, 0, size_y, size_z) == 0);
            REQUIRE(threeToOne(0, 0, 2, size_y, size_z) == 2);
            REQUIRE(threeToOne(0, 1, 0, size_y, size_z) == 8);
            REQUIRE(threeToOne(0, 2, 0, size_y, size_z) == 16);
            REQUIRE(threeToOne(0, 2, 7, size_y, size_z) == 23);
            REQUIRE(threeToOne(10, 0, 0, size_y, size_z) == 240);
            REQUIRE(threeToOne(1, 2, 7, size_y, size_z) == 47);
            REQUIRE(threeToOne(3, 0, 2, size_y, size_z) == 74);
        }
    }

    GIVEN("Two dimensions of size [*,13]")
    {
        auto size_y = 13;

        using indexing::twoToOne;
        THEN("Project values in those dimensions into a single works as inteded")
        {
            REQUIRE(twoToOne(0, 0, size_y) == 0);
            REQUIRE(twoToOne(0, 1, size_y) == 1);
            REQUIRE(twoToOne(0, 10, size_y) == 10);
            REQUIRE(twoToOne(1, 0, size_y) == 13);
            REQUIRE(twoToOne(2, 0, size_y) == 26);
            REQUIRE(twoToOne(4, 3, size_y) == 55);
        }
    }

    GIVEN("Some high dimensions with size [2,3,4]")
    {
        std::vector<int> dim_size = {5, 3, 4};

        using indexing::project;
        using indexing::projectUsingDimensions;
        THEN("Projecting values into that space works as intended")
        {
            REQUIRE(projectUsingDimensions(0, dim_size) == std::vector<int>({0, 0, 0}));
            REQUIRE(projectUsingDimensions(1, dim_size) == std::vector<int>({0, 0, 1}));
            REQUIRE(projectUsingDimensions(4, dim_size) == std::vector<int>({0, 1, 0}));
            REQUIRE(projectUsingDimensions(6, dim_size) == std::vector<int>({0, 1, 2}));
            REQUIRE(projectUsingDimensions(12, dim_size) == std::vector<int>({1, 0, 0}));
            REQUIRE(projectUsingDimensions(15, dim_size) == std::vector<int>({1, 0, 3}));
            REQUIRE(projectUsingDimensions(16, dim_size) == std::vector<int>({1, 1, 0}));
            REQUIRE(projectUsingDimensions(19, dim_size) == std::vector<int>({1, 1, 3}));
        }

        THEN("Project values from that space as a single value works as intended")
        {
            REQUIRE(project(std::vector<int>({0, 0, 0}), dim_size) == 0);
            REQUIRE(project(std::vector<int>({0, 0, 1}), dim_size) == 1);
            REQUIRE(project(std::vector<int>({0, 1, 1}), dim_size) == 5);
            REQUIRE(project(std::vector<int>({1, 0, 0}), dim_size) == 12);
            REQUIRE(project(std::vector<int>({2, 1, 1}), dim_size) == 29);
            REQUIRE(project(std::vector<int>({2, 2, 1}), dim_size) == 33);
            REQUIRE(project(std::vector<int>({2, 2, 0}), dim_size) == 32);
        }
    }
}
