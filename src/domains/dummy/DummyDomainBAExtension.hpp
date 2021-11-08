#ifndef DUMMYDOMAINBAEXTENSION_HPP
#define DUMMYDOMAINBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
class Action;
class State;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief Bayes-adaptive extensions for the dummy domain
 **/
class DummyDomainBAExtension : public BADomainExtension
{

public:
    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(std::string index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
};

}} // namespace bayes_adaptive::domain_extensions

#endif // DUMMYDOMAINBAEXTENSION_HPP
