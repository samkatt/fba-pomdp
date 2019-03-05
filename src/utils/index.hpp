#ifndef INDEX_HPP
#define INDEX_HPP

#include <vector>

namespace indexing {

/**
 * @brief projects (x,y) with domain size (*,size_y) into a single dimension
 **/
int twoToOne(int x, int y, int size_y);

/**
 * @brief projects (x,y,z) with domain size (*,size_y,size_z) into a single dimension
 **/
int threeToOne(int x, int y, int z, int size_y, int size_z);

/**
 * @brief returns the stepsizes given a set of dimensions
 **/
std::vector<int> stepSize(std::vector<int> const& dimension_sizes);

/**
 * @brief projects values (a,b,...,z) in higher dimension with ranges/size (A,B,...,Z) into a single
 *dimension value
 **/
int project(std::vector<int> const& high_dim_values, std::vector<int> const& high_dim_size);

/**
 * @brief projects a value v into a higher dimension of size [A,B,...,Z]
 * NOTE: it is faster if you store/precompute stepsize and
 * call "projectUsingStepSize" instead
 **/
std::vector<int> projectUsingDimensions(int v, std::vector<int> const& high_dim_size);

/**
 * @brief projects a value v into a higher dimension using step size
 **/
std::vector<int> projectUsingStepSize(int v, std::vector<int> const& step_sizes);

/**
 * @brief incrmeents indices with respect to its sizes
 *
 * indices gives for each element i the index in that dimension / feature
 * dimension_sizes provides the size/range of that specific dimension / feature
 *
 * will modify indices such that it represents the 'next' values
 *
 * Returns whether we have reached the last possible set of indices
 **/
bool increment(std::vector<int>& indices, std::vector<int> const& dimension_sizes);

} // namespace indexing

#endif // INDEX_HPP
