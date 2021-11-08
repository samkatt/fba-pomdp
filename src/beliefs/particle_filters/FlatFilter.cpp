#ifndef FLATFILTER_CPP
#define FLATFILTER_CPP

#include "FlatFilter.hpp"

#include "easylogging++.h"

#include "environment/State.hpp"
#include <map>

#include "utils/random.hpp"

template<typename T>
template<typename Allocator>
FlatFilter<T>::FlatFilter(size_t size, Allocator alloc) :
        _distr(rnd::integerDistribution(0, static_cast<int>(size)))
{

    if (size == 0)
    {
        throw "FlatFilter cannot be initiated with size 0";
    }

    _particles.reserve(size);

    for (size_t i = 0; i < size; ++i) { _particles.emplace_back(alloc()); }
}

template<typename T>
// cppcheck-suppress passedByValue
FlatFilter<T>::FlatFilter(std::vector<T> particles) :
        _particles(std::move(particles)),
        _distr(rnd::integerDistribution(0, static_cast<int>(_particles.size())))
{
}

template<typename T>
template<typename Deallocator>
void FlatFilter<T>::replace(T replacement, Deallocator const& dealloc)
{
    // we want to replace randomly
    auto i = _distr(rnd::rng());

    dealloc(_particles[i]);
    _particles[i] = replacement;
}

template<typename T>
template<typename Deallocator>
void FlatFilter<T>::free(Deallocator const& dealloc)
{

    for (auto s : _particles) { dealloc(s); }

    _particles.clear();
}

template<typename T>
size_t FlatFilter<T>::size() const
{
    return _particles.size();
}

template<typename T>
bool FlatFilter<T>::empty() const
{
    return _particles.empty();
}

template<typename T>
std::string FlatFilter<T>::toString() const
{
    auto n = _particles.size();

    std::map<std::string, int> state_count;
    std::map<std::string, T> state_map;

    // count how many states there are of each type, and save their strings
    for (auto const s : _particles)
    {
        state_count[s->index()]++;
        state_map[s->index()] = s;
    }

    std::string res = "Particle filter contains:\n";
    for (auto const& el : state_count)
    {
        res += "\t(" + state_map[el.first]->toString() + ": "
               + std::to_string(el.second / static_cast<double>(n)) + "("
               + std::to_string(el.second) + "))\n";
    }

    return res;
}

template<typename T>
T FlatFilter<T>::sample() const
{
    auto const particle = _particles[_distr(rnd::rng())];

    return particle;
}

template<typename T>
std::vector<T> const& FlatFilter<T>::particles() const
{
    return _particles;
}

#endif // FLATFILTER_CPP
