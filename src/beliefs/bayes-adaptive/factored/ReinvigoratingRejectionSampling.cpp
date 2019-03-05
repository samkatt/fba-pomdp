#include "ReinvigoratingRejectionSampling.hpp"

#include "easylogging++.h"

#include "beliefs/particle_filters/RejectionSampling.hpp"

#include "bayes-adaptive/models/factored/FBAPOMDP.hpp"
#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "bayes-adaptive/states/factored/BABNModel.hpp"
#include "domains/POMDP.hpp"

#include "bayes-adaptive/states/factored/DBNNode.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "environment/Reward.hpp"

using FBAPOMDP = ::bayes_adaptive::factored::FBAPOMDP;

namespace beliefs { namespace bayes_adaptive { namespace factored {

FBAPOMDPState* breed(
    FBAPOMDP const& fbapomdp,
    FBAPOMDPState const* structure_state,
    FBAPOMDPState const* counts_state)
{

    auto mutated_model_structure = fbapomdp.mutate(structure_state->model()->structure());
    auto mutated_model = counts_state->model()->marginalizeOut(std::move(mutated_model_structure));

    return new FBAPOMDPState(
        fbapomdp.copyDomainState(structure_state->_domain_state), std::move(mutated_model));
}

ReinvigoratingRejectionSampling::ReinvigoratingRejectionSampling(
    size_t size,
    size_t reinvigoration_amount) :
        _size(size),
        _reinvigoration_amount(reinvigoration_amount)
{

    if (_size < 1 || _reinvigoration_amount < 1)
    {
        throw "ReinvigoratingRejectionSampling::cannot initiate belief of size < 1 ("
            + std::to_string(_size) + "), or resample size of < 1 ("
            + std::to_string(_reinvigoration_amount) + ")";
    }

    VLOG(1) << "initiated reinvigorating rejectiong sampling belief  of size " << _size
            << " and reinvigoration amount " << _reinvigoration_amount;
}

void ReinvigoratingRejectionSampling::initiate(POMDP const& domain)
{
    assert(_belief.empty());
    assert(_fully_connected_belief.empty());

    auto& fbapomdp = dynamic_cast<FBAPOMDP const&>(domain);

    auto tmp_states = std::vector<FBAPOMDPState const*>();
    for (size_t i = 0; i < _size; ++i)
    { tmp_states.emplace_back(dynamic_cast<FBAPOMDPState const*>(domain.sampleStartState())); }
    _belief = FlatFilter<FBAPOMDPState const*>(tmp_states);

    tmp_states.clear();
    for (size_t i = 0; i < _size; ++i)
    { tmp_states.emplace_back(fbapomdp.sampleFullyConnectedState()); }
    _fully_connected_belief = FlatFilter<FBAPOMDPState const*>(std::move(tmp_states));

    VLOG(3) << "Status of rejection sampling filter after initiating:\n" << _belief.toString();
}

void ReinvigoratingRejectionSampling::free(POMDP const& domain)
{
    _belief.free([&domain](State const* s) { domain.releaseState(s); });
    _fully_connected_belief.free([&domain](State const* s) { domain.releaseState(s); });
}

State const* ReinvigoratingRejectionSampling::sample() const
{
    return _belief.sample();
}

void ReinvigoratingRejectionSampling::updateEstimation(
    Action const* a,
    Observation const* o,
    POMDP const& domain)
{
    assert(a != nullptr && o != nullptr);
    assert(_belief.size() == _size);
    assert(_fully_connected_belief.size() == _size);

    reinvigorateParticles(domain);

    ::beliefs::rejectSample(a, o, domain, _size, _belief);
    ::beliefs::rejectSample(a, o, domain, _size, _fully_connected_belief);

    assert(_belief.size() == _size);

    VLOG(3) << "Status of rejection sampling filter after updating:\n" << _belief.toString();
}

void ReinvigoratingRejectionSampling::resetDomainStateDistribution(BAPOMDP const& bapomdp)
{
    assert(_belief.size() == _size);
    assert(_fully_connected_belief.size() == _size);

    for (auto& p : _belief.particles()) { bapomdp.resetDomainState(p); }

    for (auto& p : _fully_connected_belief.particles()) { bapomdp.resetDomainState(p); }

    VLOG(3) << "Status of rejection sampling filter after initiating while keeping counts:\n"
            << _belief.toString();
}

void ReinvigoratingRejectionSampling::reinvigorateParticles(POMDP const& domain)
{
    auto& fbapomdp = dynamic_cast<FBAPOMDP const&>(domain);

    for (size_t i = 0; i < _reinvigoration_amount; ++i)
    {
        _belief.replace(
            breed(fbapomdp, _belief.sample(), _fully_connected_belief.sample()),
            [&domain](State const* s) { domain.releaseState(s); });
    }
}

}}} // namespace beliefs::bayes_adaptive::factored
