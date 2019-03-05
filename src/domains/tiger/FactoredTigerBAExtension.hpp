#ifndef FACTOREDTIGERBAEXTENSION_HPP
#define FACTOREDTIGERBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "domains/tiger/FactoredTiger.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
class Action;
class State;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief Bayes-adaptive extensions for the FactoredTiger domain
 **/
class FactoredTigerBAExtension : public BADomainExtension
{

public:
    FactoredTigerBAExtension(
        domains::FactoredTiger::FactoredTigerDomainType type,
        size_t num_irrelevant_features);

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(int index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    Domain_Size _domain_size;
    domains::FactoredTiger::FactoredTigerDomainType _type;

    utils::DiscreteSpace<IndexState> _states{_domain_size._S};
};

}} // namespace bayes_adaptive::domain_extensions

#endif // FACTOREDTIGERBAEXTENSION_HPP
