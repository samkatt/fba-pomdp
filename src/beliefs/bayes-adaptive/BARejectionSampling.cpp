#include "BARejectionSampling.hpp"

#include "easylogging++.h"

#include "bayes-adaptive/states/BAState.hpp"
#include "domains/POMDP.hpp"

namespace beliefs {

BARejectionSampling::BARejectionSampling(size_t n) : _n(n)
{

    if (_n < 1)
    {
        throw("cannot initiate RejectionSampling with n = " + std::to_string(_n));
    }

    VLOG(1) << "initiated rejection sampling belief of size " << n;
}

void BARejectionSampling::initiate(POMDP const& simulator)
{

    assert(_filter.empty());

    auto allocator = [&simulator] { return simulator.sampleStartState(); };
    _filter        = FlatFilter<State const*>(_n, allocator);

    VLOG(3) << "Status of rejection sampling filter after initiating:" << _filter.toString();
}

void BARejectionSampling::free(POMDP const& simulator)
{
    _filter.free([&simulator](State const* s) { simulator.releaseState(s); });
}

State const* BARejectionSampling::sample() const
{
    return _filter.sample();
}

void BARejectionSampling::updateEstimation(Action const* a, Observation const* o, POMDP const& d)
{
    ::beliefs::rejectSample(a, o, d, _n, _filter);

    VLOG(3) << "Status of rejection sampling filter after update:" << _filter.toString();
}

void BARejectionSampling::resetDomainStateDistribution(BAPOMDP const& bapomdp)
{
    assert(_filter.size() == _n);

    for (auto& s : _filter.particles())
    { bapomdp.resetDomainState(dynamic_cast<BAState const*>(s)); }

    VLOG(3) << "Status of rejection sampling filter after initiating while keeping counts:"
            << _filter.toString();
}

} // namespace beliefs
