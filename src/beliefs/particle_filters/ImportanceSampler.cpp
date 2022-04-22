#include "ImportanceSampler.hpp"

namespace {

std::string stateToString(State const* s)
{
    return std::to_string(s->index());
}

} // namespace

namespace beliefs {

ImportanceSampler::ImportanceSampler(size_t n) : _n(n)
{

    if (_n < 1)
    {
        throw("cannot initiate ImportanceSampler with n " + std::to_string(_n));
    }

    VLOG(1) << "Initiated Importance Sampling belief of size " << n;
}

ImportanceSampler::ImportanceSampler(WeightedFilter<State const*> f, size_t n) :
        _filter(std::move(f)), _n(n)
{

    if (_n < 1)
    {
        throw("cannot initiate ImportanceSampler with n " + std::to_string(_n));
    }

    if (n < _filter.size())
    {
        throw "cannot initiate ImportanceSampler with n (" + std::to_string(n) + ") < filter size ("
            + std::to_string(_filter.size()) + ")";
    }

    VLOG(1) << "Initiated Importance Sampling belief of size " << _n
            << " with initial weighted filter of size " << f.size();
}

void ImportanceSampler::initiate(POMDP const& d)
{
    assert(_filter.empty());

    for (size_t i = 0; i < _n; ++i)
    {
        _filter.add(d.sampleStartState(), 1.0 / static_cast<double>(_n));
    }

    VLOG(3) << "Status of importance sampling filter after initiating:\n"
            << _filter.toString(stateToString);
}

void ImportanceSampler::free(POMDP const& d)
{
    _filter.free([&d](State const* s) { d.releaseState(s); });

    assert(_filter.empty());
}

State const* ImportanceSampler::sample() const
{
    return _filter.sample();
}

void ImportanceSampler::updateEstimation(Action const* a, Observation const* o, POMDP const& d)
{
    assert(a != nullptr);
    assert(o != nullptr);
    assert(_n == _filter.size());

    beliefs::importance_sampling::update(_filter, a, o, d);

    beliefs::importance_sampling::resample(_filter, d, _n);

    VLOG(3) << "weight filter after importance sampling update contains:\n"
            << _filter.toString(stateToString);

    assert(_n == _filter.size());
}

} // namespace beliefs
