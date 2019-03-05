#include "SysAdminFBAExtension.hpp"

#include <vector>

namespace bayes_adaptive { namespace domain_extensions {

SysAdminFBAExtension::SysAdminFBAExtension(int nr_comp) : _size(nr_comp), _state_prior(0x1 << _size)
{

    // always start in the 'last state' where all computers are 'on'
    _state_prior.setRawValue((0x1 << _size) - 1, 1);
}

Domain_Feature_Size SysAdminFBAExtension::domainFeatureSize() const
{
    return {std::vector<int>(_size, 2), {2}};
}

utils::categoricalDistr const* SysAdminFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions
