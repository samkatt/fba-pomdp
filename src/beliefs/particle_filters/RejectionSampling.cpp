#include "RejectionSampling.hpp"

namespace beliefs {

RejectionSampling::RejectionSampling(size_t n) : _n(n)
{
    if (_n < 1)
    {
        throw("cannot initiate RejectionSampling with n = " + std::to_string(_n));
    }

    VLOG(1) << "initiated rejection sampling belief of size " << n;
}

void RejectionSampling::initiate(POMDP const& simulator)
{
    assert(_filter.empty());

    auto allocator = [&simulator] { return simulator.sampleStartState(); };
    _filter        = FlatFilter<State const*>(_n, allocator);

    VLOG(3) << "Status of rejection sampling filter after initiating:" << _filter.toString();
}

void RejectionSampling::free(POMDP const& simulator)
{
    _filter.free([&simulator](State const* s) { simulator.releaseState(s); });
}

State const* RejectionSampling::sample() const
{
    return _filter.sample();
}

void RejectionSampling::updateEstimation(Action const* a, Observation const* o, POMDP const& d)
{
    beliefs::rejectSample(a, o, d, _n, _filter);

    VLOG(3) << "Status of rejection sampling filter after update:" << _filter.toString();
}

} // namespace beliefs
