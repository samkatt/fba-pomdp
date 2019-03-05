#ifndef SYSADMINBAEXTENSION_HPP
#define SYSADMINBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include <vector>

#include "domains/sysadmin/SysAdmin.hpp"

#include "bayes-adaptive/models/Domain_Size.hpp"

#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"

class Action;
class State;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief Bayes-adaptive extensions for the SysAdmin domain
 **/
class SysAdminBAExtension : public BADomainExtension
{

public:
    explicit SysAdminBAExtension(int size);

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(int index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    size_t const _size;
    Domain_Size const _domain_size{0x1 << _size, static_cast<int>(2 * _size), 2};

    std::vector<domains::SysAdminState> _states;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // SYSADMINBAEXTENSION_HPP
