#ifndef PARTICLEFILTER_HPP
#define PARTICLEFILTER_HPP

#include <cassert>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

/**
 * @brief A probability distribution approximation through particle filters
 **/
template<typename T>
class FlatFilter
{

public:
    FlatFilter() = default;
    explicit FlatFilter(std::vector<T> particles);

    template<typename Allocator>
    FlatFilter(size_t size, Allocator alloc);

    /**
     * @brief replaces a particle with the provided one
     *
     * replacement is taken randomly
     **/
    template<typename Deallocator>
    void replace(T replacement, Deallocator const& dealloc);

    template<typename Deallocator>
    void free(Deallocator const& dealloc);

    size_t size() const;
    bool empty() const;

    std::vector<T> const& particles() const;

    std::string toString() const;

    /**
     * @brief samples a state uniformly
     **/
    T sample() const;

private:
    std::vector<T> _particles = {};

    // random number generator used to sample
    mutable std::uniform_int_distribution<int> _distr = std::uniform_int_distribution<int>(0, 0);
};

#include "FlatFilter.cpp"
#endif // PARTICLEFILTER_HPP
