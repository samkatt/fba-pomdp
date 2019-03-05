#ifndef FBAPOMDPPRIOR_HPP
#define FBAPOMDPPRIOR_HPP

#include "bayes-adaptive/priors/BAPrior.hpp"

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "bayes-adaptive/states/BAState.hpp"
#include "bayes-adaptive/states/factored/BABNModel.hpp"
class BADomainExtension;
class FBADomainExtension;
class FBAPOMDPState;
class POMDP;
class State;
namespace configurations {
struct FBAConf;
}

/**
 * @brief A prior distribution over FBAPOMDP state dynamics (DBNs)
 **/
class FBAPOMDPPrior : public BAPrior
{

public:
    explicit FBAPOMDPPrior(configurations::FBAConf const& conf);

    /**
     * @brief returns a model prior w.r..t the given structure
     **/
    virtual bayes_adaptive::factored::BABNModel
        computePriorModel(bayes_adaptive::factored::BABNModel::Structure const& structure) const;

    /**
     * @brief samples a FBAPOMDP state with domain state with fully connected model
     **/
    virtual FBAPOMDPState* sampleFullyConnectedState(State const* domain_state) const = 0;

    /**
     * @brief samples a FBAPOMDPState which correct structure (regardless of noise parameter)
     **/
    virtual FBAPOMDPState const* sampleCorrectGraphState(State const* domain_state) const = 0;

    /**
     * @brief mutates the topology of a set of DBNs describing the dynamics
     **/
    virtual bayes_adaptive::factored::BABNModel::Structure
        mutate(bayes_adaptive::factored::BABNModel::Structure structure) const = 0;

    /*** BAPrior interface ***/
    BAState* sample(State const* s) const final;

private:
    /**
     * @brief samples a FBAPOMDP state with domain state
     **/
    virtual FBAPOMDPState* sampleFBAPOMDPState(State const* domain_state) const = 0;

    // whether or not the prior should sample
    bool const _sample_fully_connected_graphs;
};

namespace factory {

/**
 * @brief Returns a prior for the FBA-POMDP
 *
 * @param domain the domain (for domain knowledge) to set a prior for
 * @param c configurations of the prior
 *
 * @return a FBA-POMDP prior
 */
std::unique_ptr<FBAPOMDPPrior>
    makeFBAPOMDPPrior(POMDP const& domain, configurations::FBAConf const& c);

} // namespace factory

#endif // FBAPOMDPPRIOR_HPP
