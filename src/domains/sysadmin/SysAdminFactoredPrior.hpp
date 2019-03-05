#ifndef SYSADMINFACTOREDPRIOR_HPP
#define SYSADMINFACTOREDPRIOR_HPP

#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "bayes-adaptive/states/factored/BABNModel.hpp"
#include "bayes-adaptive/states/factored/DBNNode.hpp"
#include "domains/sysadmin/SysAdmin.hpp"
#include "domains/sysadmin/SysAdminParameters.hpp"
class Action;
class FBAPOMDPState;
class State;

namespace configurations {
struct FBAConf;
}

namespace priors {

/**
 * @brief The Factored bayes-adaptive prior for the sysadmin problem
 **/
class SysAdminFactoredPrior : public FBAPOMDPPrior
{
public:
    SysAdminFactoredPrior(domains::SysAdmin const& d, configurations::FBAConf const& c);

    /*** interface implementation ***/
    FBAPOMDPState* sampleFullyConnectedState(State const* domain_state) const final;
    FBAPOMDPState* sampleCorrectGraphState(State const* domain_state) const final;
    bayes_adaptive::factored::BABNModel computePriorModel(
        bayes_adaptive::factored::BABNModel::Structure const& structure) const final;
    bayes_adaptive::factored::BABNModel::Structure
        mutate(bayes_adaptive::factored::BABNModel::Structure structure) const final;

private:
    float const _noise, _noisy_total_counts, _known_total_counts = 10000;

    domains::SysAdmin_Parameters const _params;
    domains::SysAdmin::NETWORK_TOPOLOGY const _network_topology;

    Domain_Size const _domain_size;
    Domain_Feature_Size const _domain_feature_size;
    bayes_adaptive::factored::BABNModel::BABNModel::Indexing_Steps const _fbapomdp_step_size;

    mutable std::uniform_int_distribution<int> _action_distr;
    mutable std::uniform_int_distribution<int> _comp_distr;

    std::vector<DBNNode> _fully_connected_transition_nodes = {};
    std::vector<DBNNode> _prior_transition_nodes           = {};
    std::vector<DBNNode> _correct_prior_transition_nodes   = {};
    std::vector<DBNNode> _observation_nodes                = {};
    std::vector<DBNNode> _empty_transition_nodes           = {};

    /**
     * @brief populates _prototype nodes, the prototype models for FBAPOMDPStates
     **/
    void precomputeFactoredPrior(domains::SysAdmin const& d);

    /**
     * @brief sets the transition prior to reflect a disconnect network
     **/
    std::vector<DBNNode> disconnectedTransitions();

    /**
     * @brief sets the transition priors to reflect a linear network
     **/
    std::vector<DBNNode> linearTransitions();

    /**
     * @brief sets the transition function for a fully model
     **/
    std::vector<DBNNode> fullyConnectedT(domains::SysAdmin const& d);

    float computeFailureProbability(
        Action const* a,
        int computer,
        std::vector<int> const* parents,
        std::vector<int> const* parent_values) const;

    /*** interface implementation ***/
    FBAPOMDPState* sampleFBAPOMDPState(State const* domain_state) const final;
};

} // namespace priors

#endif // SYSADMINFACTOREDPRIOR_HPP
