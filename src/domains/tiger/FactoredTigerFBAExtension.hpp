#ifndef FACTOREDTIGERFBAEXTENSION_HPP
#define FACTOREDTIGERFBAEXTENSION_HPP

#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "utils/distributions.hpp"

namespace bayes_adaptive { namespace domain_extensions {

/**
 * @brief Extends FactoredTiger to be used in FBA-POMDP
 **/
class FactoredTigerFBAExtension : public FBADomainExtension
{
public:
    explicit FactoredTigerFBAExtension(size_t nr_irrelevant_features);

    /**** FBADomainExtension interface ****/
    Domain_Feature_Size domainFeatureSize() const final;
    utils::categoricalDistr const* statePrior() const final;

private:
    size_t _size;
    utils::categoricalDistr _state_prior;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // FACTOREDTIGERFBAEXTENSION_HPP
