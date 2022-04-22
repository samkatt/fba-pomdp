#include "distributions.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>

#include "utils/random.hpp"

namespace utils {

categoricalDistr::categoricalDistr(size_t size, float init) :
        _values(size, init), _total(init * size)
{
    assert(init >= 0);
}

categoricalDistr::categoricalDistr(std::vector<float> const& distr) :
        _values(), _total(std::accumulate(distr.begin(), distr.end(), 0.0))
{

    assert(_total > 1e-10);

    _values.reserve(distr.size());
    std::transform(
        distr.begin(),
        distr.end(),
        std::back_inserter(_values),
        std::bind(std::divides<float>(), std::placeholders::_1, _total));
}

float categoricalDistr::prob(size_t i) const
{
    return _values[i] / _total;
}

void categoricalDistr::setRawValue(size_t i, float v)
{
    assert(v >= 0);

    _total += v - _values[i];
    _values[i] = v;

    assert(_total >= 0);
}

unsigned int categoricalDistr::sample() const
{
    assert(std::fabs(std::accumulate(_values.begin(), _values.end(), 0.0) - _total) < 0.0001);
    return rnd::sample::Dir::sampleFromMult(_values.data(), _values.size(), _total);
}

} // namespace utils
