#include "FBAPOMDPPrior.hpp"

#include "easylogging++.h"

#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"
#include "bayes-adaptive/models/factored/FBAPOMDP.hpp"
#include "bayes-adaptive/models/table/BADomainExtension.hpp"
#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "configurations/FBAConf.hpp"
#include "domains/POMDP.hpp"
#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "domains/collision-avoidance/CollisionAvoidancePriors.hpp"
#include "domains/dummy/FactoredDummyDomainPriors.hpp"
#include "domains/gridworld/GridWorld.hpp"
#include "domains/gridworld/GridWorldBAPriors.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffee.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffeeBAPriors.hpp"
#include "domains/sysadmin/SysAdmin.hpp"
#include "domains/sysadmin/SysAdminFactoredPrior.hpp"
#include "domains/tiger/FactoredTigerPriors.hpp"
#include "domains/tiger/TigerPriors.hpp"
#include "utils/random.hpp"

FBAPOMDPPrior::FBAPOMDPPrior(configurations::FBAConf const& conf) :
        _sample_fully_connected_graphs(conf.structure_prior == "fully-connected")
{
}

BAState* FBAPOMDPPrior::sample(State const* s) const
{
    assert(s != nullptr);

    if (_sample_fully_connected_graphs)
    {
        return sampleFullyConnectedState(s);
    }

    return sampleFBAPOMDPState(s);
}

bayes_adaptive::factored::BABNModel FBAPOMDPPrior::computePriorModel(
    bayes_adaptive::factored::BABNModel::Structure const& /*unused*/) const
{
    LOG(ERROR) << "computePriorModel not implemented by this domain";
    throw "computePriorModel not implemented by this domain";
}

namespace factory {

std::unique_ptr<FBAPOMDPPrior>
    makeFBAPOMDPPrior(POMDP const& domain, configurations::FBAConf const& c)
{

    auto const d = c.domain_conf.domain;

    if (d == "factored-dummy")
        return std::unique_ptr<FBAPOMDPPrior>(
            new priors::FactoredDummyPrior(c, c.domain_conf.size));
    if (d == "independent-sysadmin" || d == "linear-sysadmin")
        return std::unique_ptr<FBAPOMDPPrior>(
            new priors::SysAdminFactoredPrior(static_cast<domains::SysAdmin const&>(domain), c));
    if (d == "episodic-factored-tiger" || d == "continuous-factored-tiger")
        return std::unique_ptr<FBAPOMDPPrior>(new priors::FactoredTigerFactoredPrior(c));
    if (d == "gridworld")
        return std::unique_ptr<FBAPOMDPPrior>(
            new priors::GridWorldFactBAPrior(static_cast<domains::GridWorld const&>(domain), c));
    if (d == "gridworldcoffee")
        return std::unique_ptr<FBAPOMDPPrior>(
                new priors::GridWorldCoffeeFactBAPrior(static_cast<domains::GridWorldCoffee const&>(domain), c));
    if (d == "random-collision-avoidance" || d == "centered-collision-avoidance")
        return std::unique_ptr<FBAPOMDPPrior>(new priors::CollisionAvoidanceFactoredPrior(c));

    throw "incorrect domain provided, " + d + " is not supported as factored BA-POMDP";
}

} // namespace factory
