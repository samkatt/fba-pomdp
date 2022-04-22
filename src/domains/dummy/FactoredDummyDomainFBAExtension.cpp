#include "FactoredDummyDomainFBAExtension.hpp"

namespace bayes_adaptive { namespace domain_extensions {

FactoredDummyDomainFBAExtension::FactoredDummyDomainFBAExtension(size_t nr_features) :
        _size(nr_features), _state_prior(_size * _size)
{

    _state_prior.setRawValue(0, 1); // prior concentrates on the first state
}

Domain_Feature_Size FactoredDummyDomainFBAExtension::domainFeatureSize() const
{
    return {{(int)_size, (int)_size}, {1}};
}

utils::categoricalDistr const* FactoredDummyDomainFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions
