#include "Environment.hpp"

#include "configurations/DomainConf.hpp"
#include "domains/agr/AGR.hpp"
#include "domains/coffee/CoffeeProblem.hpp"
#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "domains/collision-avoidance-big/CollisionAvoidanceBig.hpp"
#include "domains/dummy/DummyDomain.hpp"
#include "domains/dummy/FactoredDummyDomain.hpp"
#include "domains/dummy/LinearDummyDomain.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffee.hpp"
#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBig.hpp"
#include "domains/gridworld/GridWorld.hpp"
#include "domains/sysadmin/SysAdmin.hpp"
#include "domains/tiger/FactoredTiger.hpp"
#include "domains/tiger/Tiger.hpp"

namespace factory {

std::unique_ptr<Environment> makeEnvironment(configurations::DomainConf const& c)
{

    if (c.domain == "dummy")
        return std::unique_ptr<Environment>(new domains::DummyDomain());
    if (c.domain == "linear_dummy")
        return std::unique_ptr<Environment>(new domains::LinearDummyDomain());
    if (c.domain == "factored-dummy")
        return std::unique_ptr<Environment>(new domains::FactoredDummyDomain(c.size));
    if (c.domain == "episodic-tiger")
        return std::unique_ptr<Environment>(
            new domains::Tiger(domains::Tiger::TigerType::EPISODIC));
    if (c.domain == "continuous-tiger")
        return std::unique_ptr<Environment>(
            new domains::Tiger(domains::Tiger::TigerType::CONTINUOUS));
    if (c.domain == "agr")
        return std::unique_ptr<Environment>(new domains::AGR(10));
    if (c.domain == "coffee")
        return std::unique_ptr<Environment>(new domains::CoffeeProblem(""));
    if (c.domain == "boutilier-coffee")
        return std::unique_ptr<Environment>(new domains::CoffeeProblem("boutilier"));
    if (c.domain == "independent-sysadmin")
        return std::unique_ptr<Environment>(
            new domains::SysAdmin(static_cast<int>(c.size), "independent"));
    if (c.domain == "linear-sysadmin")
        return std::unique_ptr<Environment>(
            new domains::SysAdmin(static_cast<int>(c.size), "linear"));
    if (c.domain == "episodic-factored-tiger")
        return std::unique_ptr<Environment>(
            new domains::FactoredTiger(domains::FactoredTiger::EPISODIC, c.size));
    if (c.domain == "continuous-factored-tiger")
        return std::unique_ptr<Environment>(
            new domains::FactoredTiger(domains::FactoredTiger::CONTINUOUS, c.size));
    if (c.domain == "gridworld")
        return std::unique_ptr<Environment>(new domains::GridWorld(c.size));
    if (c.domain == "gridworldcoffee")
        return std::unique_ptr<Environment>(new domains::GridWorldCoffee());
    if (c.domain == "gridworldcoffeebig")
        return std::unique_ptr<Environment>(new domains::GridWorldCoffeeBig(c.size));
    if (c.domain == "random-collision-avoidance")
    {

        return std::unique_ptr<Environment>(new domains::CollisionAvoidance(
            static_cast<int>(c.width),
            static_cast<int>(c.height),
            static_cast<int>(c.size),
            domains::CollisionAvoidance::VERSION::INIT_RANDOM_POSITION));
    }
    if (c.domain == "centered-collision-avoidance")
    {
        return std::unique_ptr<Environment>(new domains::CollisionAvoidance(
            static_cast<int>(c.width),
            static_cast<int>(c.height),
            static_cast<int>(c.size),
            domains::CollisionAvoidance::VERSION::INITIALIZE_CENTRE));
    }
    if (c.domain == "random-collision-avoidance-big")
    {

        return std::unique_ptr<Environment>(new domains::CollisionAvoidanceBig(
                static_cast<int>(c.width),
                static_cast<int>(c.height),
                static_cast<int>(c.size),
                domains::CollisionAvoidanceBig::VERSION::INIT_RANDOM_POSITION));
    }
    if (c.domain == "centered-collision-avoidance-big")
    {
        return std::unique_ptr<Environment>(new domains::CollisionAvoidanceBig(
                static_cast<int>(c.width),
                static_cast<int>(c.height),
                static_cast<int>(c.size),
                domains::CollisionAvoidanceBig::VERSION::INITIALIZE_CENTRE));
    }

    throw "incorrect domain provided";
}

} // namespace factory
