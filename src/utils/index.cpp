#include "index.hpp"

#include <cassert>
#include <cstddef>

namespace indexing {

int twoToOne(int x, int y, int size_y)
{
    return x * size_y + y;
}

int threeToOne(int x, int y, int z, int size_y, int size_z)
{
    return x * size_y * size_z + y * size_z + z;
}

std::vector<int> stepSize(std::vector<int> const& dimension_sizes)
{
    assert(!dimension_sizes.empty());

    auto const n = dimension_sizes.size();

    // base case
    if (dimension_sizes.size() == 1)
    {
        return {1};
    }

    auto result = std::vector<int>(n);

    // compute stepsize
    result[n - 1] = 1;
    auto i        = n - 2;

    while (true)
    {
        result[i] = result[i + 1] * dimension_sizes[i + 1];

        if (i == 0)
        {
            break;
        }

        --i;
    }

    return result;
}

int project(std::vector<int> const& high_dim_values, std::vector<int> const& high_dim_size)
{
    assert(high_dim_values.size() == high_dim_size.size());
    assert(!high_dim_values.empty());

    auto element_amount = high_dim_values.size();

    // base case
    if (element_amount == 1)
    {
        return high_dim_values[0];
    }

    int result    = high_dim_values[element_amount - 1];
    int step_size = high_dim_size[element_amount - 1];

    // backwards loop
    size_t i = element_amount - 2;
    while (true)
    {
        result += high_dim_values[i] * step_size;

        if (i == 0)
        {
            break;
        }

        step_size *= high_dim_size[i];
        i--;
    }

    return result;
}

std::vector<int> projectUsingDimensions(int v, std::vector<int> const& high_dim_size)
{
    assert(!high_dim_size.empty());

    // base case
    if (high_dim_size.size() == 1)
    {
        return {v};
    }

    return projectUsingStepSize(v, stepSize(high_dim_size));
}

std::vector<int> projectUsingStepSize(int v, std::vector<int> const& step_sizes)
{
    auto element_amount = step_sizes.size();

    // base case
    if (element_amount == 1)
    {
        return {v};
    }

    auto result = std::vector<int>();
    result.reserve(element_amount);

    // compute result using stepsize
    for (size_t i = 0; i < element_amount; ++i)
    {
        result.emplace_back(v / step_sizes[i]);
        v = v % step_sizes[i];
    }

    return result;
}

/**
 * @brief increment int i with max value range, sets 0 if range is reached
 *
 * returns whether range was reached (meaning it has a carry_over)
 **/
bool increment(int& i, int range)
{
    i = (i + 1) % range;

    return (i == 0); // would mean carry_over
}

bool increment(std::vector<int>& indices, std::vector<int> const& dimension_sizes)
{
    auto s = indices.size();

    assert(s == dimension_sizes.size());
    for (size_t i = 0; i < s; ++i) { assert(indices[i] < dimension_sizes[i]); }

    if (s == 0)
    {
        return true;
    }

    auto carry_over = true;

    // start at the last dimension
    auto i = s - 1;
    while (carry_over) // continue while we have carry_overs
    {
        // increment current dimension
        carry_over = increment(indices[i], dimension_sizes[i]);

        if (i == 0) // increment the first element, return
        {
            return carry_over;
        }

        // go to next dimension
        i--;
    }

    return carry_over;
}

} // namespace indexing
