#ifndef FACTOREDDUMMYDOMAINFBAEXTENSION_HPP
#define FACTOREDDUMMYDOMAINFBAEXTENSION_HPP

#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "utils/distributions.hpp"

namespace bayes_adaptive { namespace domain_extensions {

/**
 * @brief Extends FactoredDummyDomain to be used in FBA-POMDP
 **/
class FactoredDummyDomainFBAExtension : public FBADomainExtension
{
public:
    explicit FactoredDummyDomainFBAExtension(size_t nr_features);

    /**** FBADomainExtension interface ****/
    Domain_Feature_Size domainFeatureSize() const final;
    utils::categoricalDistr const* statePrior() const final;

private:
    size_t _size; // nr of features
    utils::categoricalDistr _state_prior;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // FACTOREDDUMMYDOMAINFBAEXTENSION_HPP
