#include "BAPOMDPPrior.hpp"

#include <utility>

#include "domains/POMDP.hpp"

#include "bayes-adaptive/models/table/BADomainExtension.hpp"
#include "bayes-adaptive/states/BAState.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/BAConf.hpp"
#include "domains/POMDP.hpp"
#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "domains/collision-avoidance/CollisionAvoidancePriors.hpp"
#include "domains/dummy/DummyDomainPrior.hpp"
#include "domains/dummy/FactoredDummyDomainPriors.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffee.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffeeBAPriors.hpp"
#include "domains/gridworld/GridWorld.hpp"
#include "domains/gridworld/GridWorldBAPriors.hpp"
#include "domains/sysadmin/SysAdmin.hpp"
#include "domains/sysadmin/SysAdminFlatPrior.hpp"
#include "domains/tiger/FactoredTigerPriors.hpp"
#include "domains/tiger/TigerPriors.hpp"
#include "utils/random.hpp"

BAState* BAPOMDPPrior::sample(State const* domain_state) const
{
    assert(domain_state != nullptr);
    return sampleBAPOMDPState(domain_state);
}

namespace factory {

std::unique_ptr<BAPOMDPPrior>
    makeTBAPOMDPPrior(POMDP const& domain, configurations::BAConf const& c)
{

    auto const d = c.domain_conf.domain;

    if (d == "dummy")
        return std::unique_ptr<BAPOMDPPrior>(new priors::DummyBAPrior());
    if (d == "factored-dummy")
        return std::unique_ptr<BAPOMDPPrior>(new priors::FactoredDummyPrior(c.domain_conf.size));
    if (d == "episodic-tiger" || d == "continuous-tiger")
        return std::unique_ptr<BAPOMDPPrior>(new priors::TigerBAPrior(c));
    if (d == "independent-sysadmin" || d == "linear-sysadmin")
        return std::unique_ptr<BAPOMDPPrior>(
            new priors::SysAdminFlatPrior(static_cast<domains::SysAdmin const&>(domain), c));
    if (d == "episodic-factored-tiger" || d == "continuous-factored-tiger")
        return std::unique_ptr<BAPOMDPPrior>(new priors::FactoredTigerFlatPrior(c));
    if (d == "gridworld")
        return std::unique_ptr<BAPOMDPPrior>(
            new priors::GridWorldFlatBAPrior(static_cast<domains::GridWorld const&>(domain), c));
    if (d == "gridworldcoffee")
        return std::unique_ptr<BAPOMDPPrior>(
            new priors::GridWorldCoffeeFlatBAPrior(static_cast<domains::GridWorldCoffee const&>(domain), c));
    if (d == "random-collision-avoidance" || d == "centered-collision-avoidance")
        return std::unique_ptr<BAPOMDPPrior>(new priors::CollisionAvoidanceTablePrior(
            static_cast<domains::CollisionAvoidance const&>(domain), c));

    throw "incorrect domain provided, " + d + " is not supported as tabular BA-POMDP";
}

} // namespace factory
