#ifndef WEIGHTEDPARTICLE_HPP
#define WEIGHTEDPARTICLE_HPP

#include "environment/State.hpp"

/**
 * @brief A weight in the WeightedFilter
 *
 * Just a double under the hood
 **/
template<typename T>
struct WeightedParticle
{
    WeightedParticle(T p, double weight) : particle(std::move(p)), w(weight) { assert(w >= 0); };

    T particle;
    double w;
};

#endif // WEIGHTEDPARTICLE_HPP
