//
// Created by rolf on 25-01-21.
//

#ifndef GridWorldCoffeeBigFBAEXTENSION_HPP
#define GridWorldCoffeeBigFBAEXTENSION_HPP


#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "utils/distributions.hpp"

namespace bayes_adaptive { namespace domain_extensions {

/**
 * @brief Extends GridWorld to be used in FBA-POMDP
 **/
class GridWorldCoffeeBigFBAExtension : public FBADomainExtension
{
public:
    explicit GridWorldCoffeeBigFBAExtension(size_t extra_features);

    /**** FBADomainExtension interface ****/
    Domain_Feature_Size domainFeatureSize() const final; // final TODO need final?
    utils::categoricalDistr const* statePrior() const final; // final

private:
    size_t _size, _extra_features;
    utils::categoricalDistr _state_prior;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // GridWorldCoffeeBigFBAEXTENSION_HPP



