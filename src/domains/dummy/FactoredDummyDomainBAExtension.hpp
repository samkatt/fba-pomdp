#ifndef FACTOREDDUMMYDOMAINBAEXTENSION_HPP
#define FACTOREDDUMMYDOMAINBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
class Action;
class State;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief Bayes-adaptive extensions for the factored dummy domain
 **/
class FactoredDummyDomainBAExtension : public BADomainExtension
{

public:
    explicit FactoredDummyDomainBAExtension(int size);

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(int index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    size_t _size;
    size_t _num_states;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // FACTOREDDUMMYDOMAINBAEXTENSION_HPP
