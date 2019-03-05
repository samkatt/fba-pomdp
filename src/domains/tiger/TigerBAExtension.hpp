#ifndef TIGERBAEXTENSION_HPP
#define TIGERBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "domains/tiger/Tiger.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
#include "utils/DiscreteSpace.hpp"
class Action;
class State;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief Bayes-adaptive extensions for the tiger domain
 **/
class TigerBAExtension : public BADomainExtension
{

public:
    explicit TigerBAExtension(domains::Tiger::TigerType type);

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(int index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    domains::Tiger::TigerType _type;
    utils::DiscreteSpace<IndexState> _states{2};
};

}} // namespace bayes_adaptive::domain_extensions

#endif // TIGERBAEXTENSION_HPP
