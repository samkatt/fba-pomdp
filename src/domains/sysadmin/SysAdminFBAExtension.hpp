#ifndef SYSADMINFBAEXTENSION_HPP
#define SYSADMINFBAEXTENSION_HPP

#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "utils/distributions.hpp"

namespace bayes_adaptive { namespace domain_extensions {

/**
 * @brief Extends SysAdmin to be used in FBA-POMDP
 **/
class SysAdminFBAExtension : public FBADomainExtension
{
public:
    explicit SysAdminFBAExtension(int nr_comp);

    /**** FBADomainExtension interface ****/
    Domain_Feature_Size domainFeatureSize() const final;
    utils::categoricalDistr const* statePrior() const final;

private:
    int _size; // number of computers
    utils::categoricalDistr _state_prior;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // SYSADMINFBAEXTENSION_HPP
