#include "catch.hpp"

#include "bayes-adaptive/states/factored/DBNNode.hpp"

#include "utils/random.hpp"

using namespace rnd::sample::Dir;

SCENARIO("dbn getters", "[bayes-adaptive][factored][dbn]")
{
    std::vector<int> graph;

    auto node = DBNNode(&graph, {}, 5);
    REQUIRE(node.range() == 5);
    REQUIRE(node.numParams() == 5);

    node = DBNNode(&graph, {}, 4);
    REQUIRE(node.range() == 4);
    REQUIRE(node.numParams() == 4);

    graph = {2, 3};
    node  = DBNNode(&graph, {0, 1}, 3);
    REQUIRE(node.range() == 3);
    REQUIRE(node.numParams() == 18);
}

SCENARIO("dbn node sampling", "[bayes-adaptive][factored][dbn]")
{
    auto graph_range = std::vector<int>();

    // 1 of size 1 input output of size 1
    GIVEN("a DBNNode with 1 input and 1 output")
    {
        graph_range        = {1};
        auto node          = DBNNode(&graph_range, {0}, 1);
        node.count({0}, 0) = 1;

        THEN("sample must output that one value")
        {
            REQUIRE(node.sample({0}, sampleFromExpectedMult) == 0);
            REQUIRE(node.sample({0}, sampleFromSampledMult) == 0);
        }
    }

    // 1 of size > 1 input output of size 1
    GIVEN("a DBNNode with 1 input of multiple values and 1 output")
    {
        graph_range = {3};
        auto node   = DBNNode(&graph_range, {0}, 1);

        // random initiate cpt
        for (auto input = 0; input < 3; ++input)
        {
            node.count({input}, 0) = (float)rnd::slowRandomInt(1, 100);
        }

        THEN("all counts in 1 should output that 1")
        {
            // check all inputs produce same output
            for (auto input = 0; input < 3; ++input)
            {
                REQUIRE(node.sample({input}, sampleFromExpectedMult) == 0);
                REQUIRE(node.sample({input}, sampleFromSampledMult) == 0);
            }
        }
    }

    // 1 of size 1 input output of size > 1
    GIVEN("a DBNNode with 1 input and multiple outputs")
    {
        graph_range = {1};
        auto node   = DBNNode(&graph_range, {0}, 3);

        auto chosen_output = rnd::slowRandomInt(0, 3);

        THEN("all counts in 1 should output that 1")
        {
            node.count({0}, chosen_output) = 10;
            REQUIRE(node.sample({0}, sampleFromExpectedMult) == chosen_output);
            REQUIRE(node.sample({0}, sampleFromSampledMult) == chosen_output);
        }

        THEN("sampling should never return a value without counts")
        {
            for (auto output = 0; output < 3; ++output)
            {
                if (output != chosen_output)
                {
                    node.count({0}, output) = 10;
                }
            }

            REQUIRE(node.sample({0}, sampleFromExpectedMult) != chosen_output);
            REQUIRE(node.sample({0}, sampleFromSampledMult) != chosen_output);
        }
    }

    // 1 of size > 1 input output of size > 1
    GIVEN("a DBNNode with 1 input of multiple values and multiple outputs")
    {
        graph_range = {3};
        auto node   = DBNNode(&graph_range, {0}, 3);

        WHEN("a CPT that described input x output x")
        {

            for (auto i = 0; i < 3; ++i) { node.count({i}, i) = 10; }

            THEN("sampling with input x produces output x")
            {
                for (auto i = 0; i < 3; ++i)
                {
                    REQUIRE(node.sample({i}, sampleFromExpectedMult) == i);
                    REQUIRE(node.sample({i}, sampleFromSampledMult) == i);
                }
            }
        }

        WHEN("a uniform CPT")
        {
            for (auto i = 0; i < 3; ++i)
            {
                for (auto j = 0; j < 3; ++j) { node.count({i}, j) = 10; }
            }

            THEN("anything can be output, as long as it is less than its range")
            {
                for (auto i = 0; i < 3; ++i)
                {
                    REQUIRE(node.sample({i}, sampleFromExpectedMult) < 3);
                    REQUIRE(node.sample({i}, sampleFromSampledMult) < 3);
                }
            }
        }
    }

    // > 1 of size 1 input output of size 1
    GIVEN("A node with multiple parents of 1 value and 1 output")
    {
        graph_range = {1, 1, 1};
        auto node   = DBNNode(&graph_range, {0, 1, 2}, 1);

        node.count({0, 0, 0}, 0) = 10;

        THEN("a sample should output that value")
        {
            REQUIRE(node.sample({0, 0, 0}, sampleFromExpectedMult) == 0);
            REQUIRE(node.sample({0, 0, 0}, sampleFromSampledMult) == 0);
        }
    }

    // > 1 of size > 1 input output of size 1
    GIVEN("a node with multiple parents of multiple input and 1 output")
    {
        graph_range = {3, 3, 3};
        auto node   = DBNNode(&graph_range, {0, 1, 2}, 1);

        THEN("any CPT and sample input should output that one value")
        {

            // random counts
            for (auto in_1 = 0; in_1 < 3; ++in_1)
            {
                for (auto in_2 = 0; in_2 < 3; ++in_2)
                {
                    for (auto in_3 = 0; in_3 < 3; ++in_3)
                    {
                        node.count({in_1, in_2, in_3}, 0) = (float)rnd::slowRandomInt(1, 10);
                    }
                }
            }

            // check all samples == 0
            for (auto in_1 = 0; in_1 < 3; ++in_1)
            {
                for (auto in_2 = 0; in_2 < 3; ++in_2)
                {
                    for (auto in_3 = 0; in_3 < 3; ++in_3)
                    {
                        REQUIRE(node.sample({in_1, in_2, in_3}, sampleFromSampledMult) == 0);
                        REQUIRE(node.sample({in_1, in_2, in_3}, sampleFromExpectedMult) == 0);
                    }
                }
            }
        }
    }

    // > 1 of size 1 input output of size > 1
    GIVEN("a node with multiple parents of 1 value and multiple outputs")
    {
        graph_range = {1, 1, 1};
        auto node   = DBNNode(&graph_range, {0, 1, 2}, 3);

        WHEN("All counts in 1 output")
        {
            node.count({0, 0, 0}, 1) = 10;

            THEN("sampling will result in that outcome")
            {
                REQUIRE(node.sample({0, 0, 0}, sampleFromExpectedMult) == 1);
                REQUIRE(node.sample({0, 0, 0}, sampleFromSampledMult) == 1);
            }

            AND_WHEN("counts are added to different output")
            {
                node.count({0, 0, 0}, 0) = 10;

                THEN("sampling will output either of them")
                {
                    REQUIRE(node.sample({0, 0, 0}, sampleFromExpectedMult) != 2);
                    REQUIRE(node.sample({0, 0, 0}, sampleFromSampledMult) != 2);
                }
            }
        }
    }

    // > 1 of size > 1 input output of size > 1 (deterministic)
    GIVEN("a complex umbrella & rain => wet DBN node")
    {
        // {{no umbrella, broken umbrella, have umbrella}, {dry, little, much rain}}
        // -> {dry, unknown, wet}
        graph_range = {3, 3};
        auto node   = DBNNode(&graph_range, {0, 1}, 3);

        // no rain means dry
        for (auto u = 0; u < 3; ++u) { node.count({u, 0}, 0) = 10; }

        // umbrella means dry
        for (auto r = 0; r < 3; ++r) { node.count({2, r}, 0) = 10; }

        // no umbrella and not dry means wet
        node.count({0, 1}, 2) = 10;
        node.count({0, 2}, 2) = 10;

        // broken & little rain = unknown
        node.count({1, 1}, 1) = 10;

        // broken & much rain = wet
        node.count({1, 2}, 2) = 10;

        THEN("sampling matches our expectations")
        {
            REQUIRE(node.sample({0, 0}, sampleFromExpectedMult) == 0);
            REQUIRE(node.sample({0, 0}, sampleFromSampledMult) == 0);
            REQUIRE(node.sample({0, 1}, sampleFromExpectedMult) == 2);
            REQUIRE(node.sample({0, 1}, sampleFromSampledMult) == 2);
            REQUIRE(node.sample({0, 2}, sampleFromExpectedMult) == 2);
            REQUIRE(node.sample({0, 2}, sampleFromSampledMult) == 2);
            REQUIRE(node.sample({1, 0}, sampleFromExpectedMult) == 0);
            REQUIRE(node.sample({1, 0}, sampleFromSampledMult) == 0);
            REQUIRE(node.sample({1, 1}, sampleFromExpectedMult) == 1);
            REQUIRE(node.sample({1, 1}, sampleFromSampledMult) == 1);
            REQUIRE(node.sample({1, 2}, sampleFromExpectedMult) == 2);
            REQUIRE(node.sample({1, 2}, sampleFromSampledMult) == 2);
            REQUIRE(node.sample({2, 0}, sampleFromExpectedMult) == 0);
            REQUIRE(node.sample({2, 0}, sampleFromSampledMult) == 0);
            REQUIRE(node.sample({2, 1}, sampleFromExpectedMult) == 0);
            REQUIRE(node.sample({2, 1}, sampleFromSampledMult) == 0);
            REQUIRE(node.sample({2, 2}, sampleFromExpectedMult) == 0);
            REQUIRE(node.sample({2, 2}, sampleFromSampledMult) == 0);
        }
    }
}

SCENARIO("dbn node incrementing", "[bayes-adaptive][factored][dbn]")
{
    GIVEN("A DBNNode of random size")
    {
        THEN("Incremeting this shit should go well")
        {
            auto parents_num = 5;

            auto parents = std::vector<int>(parents_num);
            for (auto i = 0; i < parents_num; ++i) { parents[i] = i; }

            for (auto n = 0; n < 20; ++n)
            {

                int node_output_size = rnd::slowRandomInt(1, 5);

                std::vector<int> parent_sizes(parents_num);
                for (auto i = 0; i < parents_num; ++i)
                {
                    parent_sizes[i] = rnd::slowRandomInt(1, 10);
                }

                // test 5 random count increments
                for (auto m = 0; m < 5; ++m)
                {
                    auto node = DBNNode(&parent_sizes, parents, node_output_size);

                    auto output = rnd::slowRandomInt(0, node_output_size);

                    std::vector<int> values(parents_num);
                    for (auto i = 0; i < parents_num; ++i)
                    {
                        values[i] = rnd::slowRandomInt(0, parent_sizes[i]);
                    }

                    auto old_count = node.count(values, output);
                    node.increment(values, output);
                    REQUIRE(node.count(values, output) == old_count + 1);
                }
            }
        }
    }
}
