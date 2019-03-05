#include "catch.hpp"

#include "bayes-adaptive/states/factored/BABNModel.hpp"

#include <algorithm>
#include <vector>

SCENARIO("flipping edges in graph structures", "[bayes-adaptive][factored][graph structure]")
{
    auto tests = 10;

    GIVEN("an empty set of parents")
    {
        std::vector<int> parents = {};

        WHEN("we flip an edge with range 1")
        {

            auto test_parents = parents;
            bayes_adaptive::factored::BABNModel::Structure::flip_random_edge(&test_parents, 1);

            THEN("we expect it become {0}")
            REQUIRE(test_parents == std::vector<int>(1, 0));
        }

        WHEN("we flip an edge with range 5")
        {

            auto added_1 = false;
            for (auto i = 0; i < 3 * tests; ++i)
            {
                auto test_parents = parents;
                bayes_adaptive::factored::BABNModel::Structure::flip_random_edge(&test_parents, 3);

                REQUIRE(test_parents.size() == 1);
                REQUIRE(test_parents[0] < 3);
                added_1 = added_1 || test_parents[0] == 1;
            }

            THEN("we expect it to have added parent 1 at some point")
            REQUIRE(added_1);
        }
    }

    GIVEN("a set of 1 parent {0}")
    {

        std::vector<int> parents = {0};

        WHEN("we flip an edge with range 1 ")
        {

            auto test_parents = parents;
            bayes_adaptive::factored::BABNModel::Structure::flip_random_edge(&test_parents, 1);

            THEN("we expect it to become empty")
            REQUIRE(test_parents.empty());
        }
    }

    GIVEN("a set of 2 parents {2,4}")
    {

        std::vector<int> parents = {2, 4};

        WHEN("we flip and edge with range 10")
        {

            for (auto i = 0; i < tests; ++i)
            {
                auto test_parents = parents;
                bayes_adaptive::factored::BABNModel::Structure::flip_random_edge(&test_parents, 1);

                THEN("we expect the parents to be in order and either 1 smaller or bigger")
                {
                    auto cor_size = (test_parents.size() == 1 || test_parents.size() == 3);
                    REQUIRE(cor_size);

                    if (test_parents.size() == 1)
                    {
                        auto cor_element = test_parents[0] == 2 || test_parents[0] == 4;
                        REQUIRE(cor_element);
                    } else
                    {
                        REQUIRE(test_parents[0] < test_parents[1]);
                        REQUIRE(test_parents[1] < test_parents[2]);
                        REQUIRE(
                            std::find(test_parents.begin(), test_parents.end(), 2)
                            != test_parents.end());
                        REQUIRE(
                            std::find(test_parents.begin(), test_parents.end(), 4)
                            != test_parents.end());
                    }
                }
            }
        }
    }
}
