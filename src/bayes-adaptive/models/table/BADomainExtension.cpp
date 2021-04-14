#include "BADomainExtension.hpp"

#include "configurations/BAConf.hpp"
#include "domains/collision-avoidance/CollisionAvoidanceBAExtension.hpp"
#include "domains/dummy/DummyDomainBAExtension.hpp"
#include "domains/dummy/FactoredDummyDomainBAExtension.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffeeBAExtension.hpp"
#include "domains/gridworld/GridWorldBAExtension.hpp"
#include "domains/sysadmin/SysAdminBAExtension.hpp"
#include "domains/tiger/FactoredTigerBAExtension.hpp"
#include "domains/tiger/TigerBAExtension.hpp"

namespace factory {

std::unique_ptr<BADomainExtension> makeBADomainExtension(configurations::BAConf const& c)
{

    auto const d = c.domain_conf.domain;

    namespace ext = bayes_adaptive::domain_extensions;

    if (d == "dummy")
        return std::unique_ptr<BADomainExtension>(new ext::DummyDomainBAExtension());
    if (d == "factored-dummy")
        return std::unique_ptr<BADomainExtension>(
            new ext::FactoredDummyDomainBAExtension(c.domain_conf.size));

    if (d == "episodic-tiger" || d == "continuous-tiger")
    {
        if (d == "episodic-tiger")
            return std::unique_ptr<BADomainExtension>(
                new ext::TigerBAExtension(domains::Tiger::TigerType::EPISODIC));
        else
            return std::unique_ptr<BADomainExtension>(
                new ext::TigerBAExtension(domains::Tiger::TigerType::CONTINUOUS));
    }

    if (d == "independent-sysadmin" || d == "linear-sysadmin")
        return std::unique_ptr<BADomainExtension>(new ext::SysAdminBAExtension(c.domain_conf.size));

    if (d == "episodic-factored-tiger" || d == "continuous-factored-tiger")
    {
        if (d == "episodic-factored-tiger")
            return std::unique_ptr<BADomainExtension>(new ext::FactoredTigerBAExtension(
                domains::FactoredTiger::FactoredTigerDomainType::EPISODIC, c.domain_conf.size));
        else
            return std::unique_ptr<BADomainExtension>(new ext::FactoredTigerBAExtension(
                domains::FactoredTiger::FactoredTigerDomainType::CONTINUOUS, c.domain_conf.size));
    }

    if (d == "gridworld")
        return std::unique_ptr<BADomainExtension>(
            new ext::GridWorldBAExtension(c.domain_conf.size));

    if (d == "gridworldcoffee")
        return std::unique_ptr<BADomainExtension>(
            new ext::GridWorldCoffeeBAExtension());

    if (d == "random-collision-avoidance" || d == "centered-collision-avoidance")
        return std::unique_ptr<BADomainExtension>(new ext::CollisionAvoidanceBAExtension(
            c.domain_conf.width, c.domain_conf.height, c.domain_conf.size));

    throw "incorrect domain provided, " + d + " is not supported as tabular BA-POMDP";
}

} // namespace factory
