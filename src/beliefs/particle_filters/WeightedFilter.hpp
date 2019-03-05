#ifndef WEIGHTEDFILTER_HPP
#define WEIGHTEDFILTER_HPP

#include "beliefs/particle_filters/WeightedParticle.hpp"

#include <cstddef>
#include <string>
#include <vector>

/**
 * @brief Estimation of a type T in the form of (a bag of) particles w/ weights
 **/
template<typename T>
class WeightedFilter
{
public:
    WeightedFilter();

    template<typename Allocator>
    WeightedFilter(size_t size, Allocator alloc);

    bool empty() const;
    size_t size() const;

    /**
     * @brief adds sample s to filter with weight 1
     **/
    void add(T s);

    /**
     * @brief adds sample s to filter with weight w
     **/
    void add(T s, double w);

    /**
     * @brief replaces element i with provided s with weight 1/total_weight
     **/
    template<typename Deallocator>
    void replace(int i, T s, Deallocator const& d);

    /**
     * @brief replaces element i with provided s with weight w
     **/
    template<typename Deallocator>
    void replace(int i, T s, Deallocator const& d, double w);

    /**
     * @brief clears out the filter and frees the particles
     **/
    template<typename Deallocator>
    void free(Deallocator const& d);

    /**
     * @brief returns particle i
     **/
    WeightedParticle<T>* particle(size_t i);

    /**
     * @brief returns particle i
     **/
    WeightedParticle<T> const* particle(size_t i) const;

    /**
     * @brief returns the relative weight of any given, unnormalized, weight
     **/
    double normalizedWeight(double w) const;

    /**
     * @brief returns the indices of the n least likely
     **/
    std::vector<int> leastLikely(size_t n) const;

    /**
     * @brief normalizes the weighted particles
     **/
    void normalize();

    /**
     * @brief normalizes the weighted particle to weight w
     **/
    void normalize(double total);

    /**
     * @brief returns a string description
     *
     * Assumes descr is some sort of object that will be able
     * to print out a description for the particles, it's assumed signature is this:
     * std::string (T const&);
     **/
    template<typename particleDescriptor>
    std::string toString(particleDescriptor const& partDescr) const;

    /**
     * @brief samples an element according to their weights
     **/
    T sample() const;

private:
    double _total_weight = 0;
    std::vector<WeightedParticle<T>> _particles;
};

#include "WeightedFilter.cpp"

#endif // WEIGHTEDFILTER_HPP
