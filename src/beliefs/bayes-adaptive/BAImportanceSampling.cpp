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
    return s->index();
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

void BAImportanceSampling::resetDomainStateDistribution(const BAPOMDP &bapomdp)
{
    assert(_filter.size() == _n);

    auto new_filter = WeightedFilter<State const*>();

    // more memory efficient way
    auto sampled_samples = std::vector<int>(_n, 0);

    for (auto i = 0; i < (int) _n; i++) {
        sampled_samples[_filter.sampleIndex()] += 1;
    }
    // First free unused particles from memory
    for (int i = 0; i < (int) _n; i++) {
        if (sampled_samples[i] == 0) {
            _filter.free([&bapomdp](State const* s) { bapomdp.releaseState(s); }, i);
        }
    }
    // Then build new belief, and free remaining particles while doing so
    for (int i = 0; i < (int) _n; i++) {
        if (sampled_samples[i] > 0) {
            for (int j = 0; j < sampled_samples[i]; j++) {
                auto s = dynamic_cast<BAState const*>(bapomdp.copyState(_filter.particle(i)->particle));
                bapomdp.resetDomainState(s);
                new_filter.add(s, 1.0 / static_cast<double>(_n));
            }
            _filter.free([&bapomdp](State const* s) { bapomdp.releaseState(s); }, i);
        }
    }

    _filter.free();
    _filter = std::move(new_filter);

    VLOG(3) << "Status of importance sampling filter after initiating while keeping counts:\n"
            << _filter.toString(stateToString);
}

void BAImportanceSampling::resetDomainStateDistributionAndAddAbstraction(const BAPOMDP &bapomdp, Abstraction &abstraction, int i)
{

    assert(_filter.size() == _n);

    auto new_filter = WeightedFilter<State const*>();

    // more memory efficient way
    auto sampled_samples = std::vector<int>(_n, 0);

    for (auto j = 0; j < (int) _n; j++) {
        sampled_samples[_filter.sampleIndex()] += 1;
    }
    // First free unused particles from memory
    for (int j = 0; j < (int) _n; j++) {
        if (sampled_samples[j] == 0) {
            _filter.free([&bapomdp](State const* s) { bapomdp.releaseState(s); }, j);
        }
    }
    // Then build new belief, and free remaining particles while doing so
    for (int j = 0; j < (int) _n; j++) {
        if (sampled_samples[j] > 0) {
            for (int k = 0; k < sampled_samples[j]; k++) {
                auto s = (AbstractFBAPOMDPState *) dynamic_cast<BAState const*>(bapomdp.copyState(_filter.particle(j)->particle));
                bapomdp.resetDomainState(s);

                if (_abstraction) {
                    if(_remake_abstract_model) {
                        static_cast<AbstractFBAPOMDPState*>(s)->setAbstraction(abstraction.constructAbstractModel(
                                s->model_real(), i, bapomdp, &static_cast<AbstractFBAPOMDPState *>(s)->feature_set));
                    } else {
                        if (*static_cast<AbstractFBAPOMDPState*>(s)->getAbstraction() != 0) {
                            static_cast<AbstractFBAPOMDPState*>(s)->setAbstraction(abstraction.constructAbstractModel(
                                    s->model_real(), i, bapomdp, &static_cast<AbstractFBAPOMDPState *>(s)->feature_set));
                        }
                    }
                }
                new_filter.add(s, 1.0 / static_cast<double>(_n));
            }
            _filter.free([&bapomdp](State const* s) { bapomdp.releaseState(s); }, j);
        }
    }

    _filter.free();
    _filter = std::move(new_filter);

    VLOG(3) << "Status of importance sampling filter after initiating while keeping counts:\n"
            << _filter.toString(stateToString);
}

} // namespace beliefs
