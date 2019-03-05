#include "POMDP.hpp"

#include "configurations/DomainConf.hpp"

namespace factory {

std::unique_ptr<POMDP> makePOMDP(configurations::DomainConf const& c)
{
    return std::unique_ptr<POMDP>(dynamic_cast<POMDP*>(factory::makeEnvironment(c).release()));
}

} // namespace factory
