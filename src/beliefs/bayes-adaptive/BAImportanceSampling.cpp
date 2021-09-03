#include "BAImportanceSampling.hpp"

#include "easylogging++.h"

#include <string>
#include <bayes-adaptive/states/factored/AbstractFBAPOMDPState.hpp>

#include "bayes-adaptive/models/table/BAPOMDP.hpp"

#include "domains/POMDP.hpp"
#include "environment/State.hpp"

namespace beliefs {

std::string stateToString(State const* s)
{
    return std::to_string(s->index());
}

BAImportanceSampling::BAImportanceSampling(size_t n, bool abstraction, bool remake_abstract_model, bool update_abstract_model, bool update_abstract_model_normalized) :
    _n(n),
    _abstraction(abstraction),
    _remake_abstract_model(remake_abstract_model),
    _update_abstract_model(update_abstract_model),
    _update_abstract_model_normalized(update_abstract_model_normalized)
{

    if (_n < 1)
    {
        throw("cannot initiate BAImportanceSampling with n " + std::to_string(_n));
    }

    VLOG(1) << "Initiated Importance Sampling belief of size " << n;
}

BAImportanceSampling::BAImportanceSampling(WeightedFilter<State const*> f, size_t n, bool abstraction, bool remake_abstract_model, bool update_abstract_model, bool update_abstract_model_normalized) :
        _filter(std::move(f)),
        _n(n),
        _abstraction(abstraction),
        _remake_abstract_model(remake_abstract_model),
        _update_abstract_model(update_abstract_model),
        _update_abstract_model_normalized(update_abstract_model_normalized)
{

    if (_n < 1)
    {
        throw("cannot initiate ImportanceSampling with n " + std::to_string(_n));
    }

    if (n < _filter.size())
    {
        throw "cannot initiate ImportanceSampling with n (" + std::to_string(n)
            + ") < filter size (" + std::to_string(_filter.size()) + ")";
    }

    VLOG(1) << "Initiated Importance Sampling belief of size " << _n
            << " with initial weighted filter of size " << f.size();
}

void BAImportanceSampling::initiate(POMDP const& d)
{
    assert(_filter.empty());

    for (size_t i = 0; i < _n; ++i)
    {
        auto s = (AbstractFBAPOMDPState *) dynamic_cast<BAState const*>(d.sampleStartState());
        _filter.add(s, 1.0 / static_cast<double>(_n));
        if (_abstraction) {
            if (_update_abstract_model_normalized) {
                // Update the abstraction
                static_cast<AbstractFBAPOMDPState *>(s)->setAbstractionNormalized(0);
            }
            if (_update_abstract_model) {
                // Update the abstraction
                static_cast<AbstractFBAPOMDPState *>(s)->setAbstraction(0);
            }
        }
    }

    VLOG(3) << "Status of importance sampling filter after initiating:\n"
            << _filter.toString(stateToString);
}

void BAImportanceSampling::free(POMDP const& d)
{
    _filter.free([&d](State const* s) { d.releaseState(s); });

    assert(_filter.empty());
}

State const* BAImportanceSampling::sample() const
{
    return _filter.sample();
}

void BAImportanceSampling::updateEstimation(Action const* a, Observation const* o, POMDP const& d)
{
    assert(a != nullptr);
    assert(o != nullptr);
    assert(_n == _filter.size());

    ::beliefs::importance_sampling::update(_filter, a, o, d);

    ::beliefs::importance_sampling::resample(_filter, d, _n);

    VLOG(3) << "weight filter after importance sampling update contains:\n"
            << _filter.toString(stateToString);

    assert(_n == _filter.size());
}

void BAImportanceSampling::resetDomainStateDistribution(BAPOMDP const& bapomdp)
{
    assert(_filter.size() == _n);

    auto new_filter = WeightedFilter<State const*>();

    // fill up the new filter
    // by sampling models from our current belief
    while (new_filter.size() != _n)
    {
        auto s = (AbstractFBAPOMDPState *) dynamic_cast<BAState const*>(bapomdp.copyState(_filter.sample()));
        bapomdp.resetDomainState(s);
        if (_abstraction) {
            if(_remake_abstract_model) {
                // Update the abstraction if it is already there. If not, it can be created later when needed.
                // TODO, actually, better to just make it here? Since this is before the episode,
                //  more time for simulations if we already have the abstract model ready.
//                if (*static_cast<AbstractFBAPOMDPState*>(s)->getAbstraction() == 0) {
                static_cast<AbstractFBAPOMDPState*>(s)->setAbstraction(0);
//                }
            }
        }

        new_filter.add(s, 1.0 / static_cast<double>(_n));
    }

    _filter.free([&bapomdp](State const* s) { bapomdp.releaseState(s); });
    _filter = std::move(new_filter);

    VLOG(3) << "Status of importance sampling filter after initiating while keeping counts:\n"
            << _filter.toString(stateToString);
}

} // namespace beliefs
