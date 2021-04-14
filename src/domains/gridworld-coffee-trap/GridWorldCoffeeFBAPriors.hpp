#ifndef GRIDWORLDCOFFEEFBAEXTENSION_HPP
#define GRIDWORLDCOFFEEFBAEXTENSION_HPP

#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "utils/distributions.hpp"

namespace bayes_adaptive { namespace domain_extensions {

/**
 * @brief Extends GridWorldCoffee to be used in FBA-POMDP
 **/
class GridWorldCoffeeFBAExtension : public FBADomainExtension
{
public:
    explicit GridWorldCoffeeFBAExtension();

    /**** FBADomainExtension interface ****/
    Domain_Feature_Size domainFeatureSize() const final;
    utils::categoricalDistr const* statePrior() const final;

private:
    size_t _size;
    utils::categoricalDistr _state_prior;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // GRIDWORLDCOFFEEFBAEXTENSION_HPP
