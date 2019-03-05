#ifndef GRIDWORLDFBAEXTENSION_HPP
#define GRIDWORLDFBAEXTENSION_HPP

#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "utils/distributions.hpp"

namespace bayes_adaptive { namespace domain_extensions {

/**
 * @brief Extends GridWorld to be used in FBA-POMDP
 **/
class GridWorldFBAExtension : public FBADomainExtension
{
public:
    explicit GridWorldFBAExtension(size_t size);

    /**** FBADomainExtension interface ****/
    Domain_Feature_Size domainFeatureSize() const final;
    utils::categoricalDistr const* statePrior() const final;

private:
    size_t _size, _goal_amount;
    utils::categoricalDistr _state_prior;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // GRIDWORLDFBAEXTENSION_HPP
