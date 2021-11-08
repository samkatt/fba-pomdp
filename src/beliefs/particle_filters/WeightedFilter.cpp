#ifndef WEIGHTEDFILTER_CPP
#define WEIGHTEDFILTER_CPP

#include "WeightedFilter.hpp"

#include <functional>
#include <map>
#include <queue>

#include "environment/State.hpp"

#include "utils/random.hpp"

template<typename T>
WeightedFilter<T>::WeightedFilter() : _particles()
{
}

template<typename T>
template<typename Allocator>
WeightedFilter<T>::WeightedFilter(size_t size, Allocator alloc) : _total_weight(1), _particles()
{
    assert(size > 0);

    _particles.reserve(size);

    for (size_t i = 0; i < size; ++i)
    { _particles.emplace_back(WeightedParticle<T>(alloc(), 1 / static_cast<double>(size))); }
}

template<typename T>
bool WeightedFilter<T>::empty() const
{
    return _particles.empty();
}

template<typename T>
size_t WeightedFilter<T>::size() const
{
    return _particles.size();
}

template<typename T>
void WeightedFilter<T>::add(T s)
{
    // base case, first element:
    if (_total_weight == 0 && _particles.empty())
    {
        add(s, 1);
    }

    assert(!_particles.empty() && _total_weight != 0);

    add(s, _total_weight / static_cast<double>(size()));
}

template<typename T>
void WeightedFilter<T>::add(T s, double w)
{
    assert(w >= 0);

    _total_weight += w;
    _particles.emplace_back(WeightedParticle<T>(s, w));
}

template<typename T>
template<typename Deallocator>
void WeightedFilter<T>::replace(int i, T s, Deallocator const& d)
{
    replace(i, s, d, _total_weight / static_cast<double>(size()));
}

template<typename T>
template<typename Deallocator>
void WeightedFilter<T>::replace(int i, T s, Deallocator const& d, double w)
{
    assert(w >= 0);

    auto const weight_diff = w - _particles[i].w;

    d(_particles[i].particle);

    _particles[i] = {s, w};
    _total_weight += weight_diff;
}

template<typename T>
template<typename Deallocator>
void WeightedFilter<T>::free(Deallocator const& d)
{

    for (auto& p : _particles) { d(p.particle); }

    _particles.clear();
    _total_weight = 0;
}

template<typename T>
template<typename Deallocator>
void WeightedFilter<T>::free(Deallocator const& d, int index)
{
    d(_particles[index].particle);
    _particles[index].particle = nullptr;
}

template<typename T>
void WeightedFilter<T>::free()
{
    _particles.clear();
    _total_weight = 0;
}

template<typename T>
WeightedParticle<T>* WeightedFilter<T>::particle(size_t i)
{
    return &_particles[i];
}

template<typename T>
WeightedParticle<T> const* WeightedFilter<T>::particle(size_t i) const
{
    return &_particles[i];
}

template<typename T>
void WeightedFilter<T>::normalize()
{
    double w = 0;

    for (auto const& p : _particles) { w += p.w; }

    normalize(w);
}

template<typename T>
double WeightedFilter<T>::normalizedWeight(double w) const
{
    assert(w >= 0 && w <= _total_weight);
    return w / _total_weight;
}

template<typename T>
void WeightedFilter<T>::normalize(double total)
{
    assert(total > 0);

    double accumulated_weight = 0;

    for (auto& p : _particles)
    {
        p.w /= total;
        accumulated_weight += p.w;
    }

    _total_weight = accumulated_weight;
}

template<typename T>
template<typename particleDescriptor>
std::string WeightedFilter<T>::toString(particleDescriptor const& partDescr) const
{

    std::string descr = "{";

    for (auto const& p : _particles)
    { descr += std::to_string(p.w) + ": " + partDescr(p.particle) + "\n"; }

    descr.back() = '}';

    return descr;
}


template<typename T>
int WeightedFilter<T>::sampleIndex() const {
    assert(_total_weight > 0);
    assert(!_particles.empty());

    auto sample_threshold = rnd::uniform_rand01() * _total_weight;

    auto sample           = _particles.size() - 1;
    auto remaining_weight = _total_weight;
    for (; sample > 0; --sample)
    {
        assert(remaining_weight > 0);
        remaining_weight -= _particles[sample].w;

        // found sample if our random threshold has been
        // reached.
        if (sample_threshold > remaining_weight)
        {
            break;
        }
    }

    // make sure we either found a sample, or that
    // the last element would have reached the threshold
    // (i.e. returning last sample (= 0) is correct)
    assert(sample || sample_threshold > remaining_weight - _particles[0].w);

    return sample;
}

template<typename T>
T WeightedFilter<T>::sample() const
{
    return _particles[sampleIndex()].particle;
}

using queue_elements = std::pair<double, int>;

/**
 * @brief Lesser comparator
 **/
struct Less
{
public:
    bool operator()(queue_elements l, queue_elements r) const { return l.first < r.first; }
};

template<typename T>
std::vector<int> WeightedFilter<T>::leastLikely(size_t n) const
{
    assert(n < size());

    static std::priority_queue<queue_elements, std::vector<queue_elements>, Less> q;

    size_t i = 0;

    // populate queue with first n elements
    for (; i < n; ++i) { q.push({_particles[i].w, i}); }

    // for the rest of the elements,
    // add if better than current worst
    for (i = 0; i < _particles.size(); ++i)
    {
        if (_particles[i].w < q.top().first)
        {
            q.pop();
            q.push({_particles[i].w, i});
        }
    }

    // move n least likely into a vector
    std::vector<int> res;
    res.reserve(n);

    for (i = 0; i < n; ++i)
    {
        res.emplace_back(q.top().second);
        q.pop();
    }

    return res;
}

#endif // WEIGHTEDFILTER_CPP
