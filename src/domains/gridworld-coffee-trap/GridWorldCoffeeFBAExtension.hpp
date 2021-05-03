//
// Created by rolf on 25-01-21.
//

#ifndef GRIDWORLDCOFFEEFBAEXTENSION_HPP
#define GRIDWORLDCOFFEEFBAEXTENSION_HPP


#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "utils/distributions.hpp"

namespace bayes_adaptive { namespace domain_extensions {

/**
 * @brief Extends GridWorld to be used in FBA-POMDP
 **/
class GridWorldCoffeeFBAExtension : public FBADomainExtension
{
public:
    explicit GridWorldCoffeeFBAExtension();

    /**** FBADomainExtension interface ****/
    Domain_Feature_Size domainFeatureSize() const final; // final TODO need final?
    utils::categoricalDistr const* statePrior() const final; // final

private:
    size_t _size, _carpet_configurations;
    utils::categoricalDistr _state_prior;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // GRIDWORLDCOFFEEFBAEXTENSION_HPP



