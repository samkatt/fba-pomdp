#include "FactoredTigerFBAExtension.hpp"

namespace bayes_adaptive { namespace domain_extensions {

FactoredTigerFBAExtension::FactoredTigerFBAExtension(size_t nr_irrelevant_features) :
        _size(nr_irrelevant_features),
        _state_prior(2 << _size, static_cast<float>(1) / static_cast<float>(2 << _size))
{
}

Domain_Feature_Size FactoredTigerFBAExtension::domainFeatureSize() const
{
    return {std::vector<int>(_size + 1, 2), {2}};
}

utils::categoricalDistr const* FactoredTigerFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions
