#ifndef FACTOREDTIGERPRIORS_HPP
#define FACTOREDTIGERPRIORS_HPP

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "bayes-adaptive/states/factored/BABNModel.hpp"
#include "bayes-adaptive/states/table/BAFlatModel.hpp"
class BAPOMDPState;
class FBAPOMDPState;
class State;

namespace configurations {
struct BAConf;
struct FBAConf;
} // namespace configurations

namespace priors {

/**
 * @brief The flat (bapomdp) prior for the factored tiger problem
 **/
class FactoredTigerFlatPrior : public BAPOMDPPrior
{
public:
    explicit FactoredTigerFlatPrior(configurations::BAConf const& c);

private:
    float const _known_counts = 5000;

    Domain_Size _domain_size;
    bayes_adaptive::table::BAFlatModel _prior, _uniform_count_prior;

    /**** BAPOMDPPrior interface ****/
    BAPOMDPState* sampleBAPOMDPState(State const* s) const final;
};

/**
 * @brief The flat (bapomdp) prior for the factored tiger problem
 **/
class FactoredTigerFactoredPrior : public FBAPOMDPPrior
{
public:
    explicit FactoredTigerFactoredPrior(configurations::FBAConf const& c);

    /*** FBAPOMDPPrior interface ***/
    bayes_adaptive::factored::BABNModel computePriorModel(
        bayes_adaptive::factored::BABNModel::Structure const& structure) const final;
    FBAPOMDPState* sampleFullyConnectedState(State const* domain_state) const final;
    FBAPOMDPState* sampleCorrectGraphState(State const* domain_state) const final;

private:
    Domain_Size const _domain_size;
    Domain_Feature_Size const _domain_feature_size;
    bayes_adaptive::factored::BABNModel::Indexing_Steps const _fbapomdp_step_size;

    int const _tiger_loc_feature = 0;

    float const _known_counts = 5000, _acc_O_count, _inacc_O_count, _uniform_O_count;

    std::string _struct_noise;

    std::vector<DBNNode> _transition_nodes = {}, _unstructured_observation_nodes = {},
                         _fully_connected_observation_nodes      = {},
                         _correctly_structured_observation_nodes = {};

    /**
     * @brief samplse a random set of parents for the (listening) observation model
     **/
    std::vector<int> sampleNoisyObservationParents() const;

    /**
     * @brief sets the observation function for listening according to parents
     *
     * The observation node may _or may not_ depend on any of the features (including the _tiger
     * location_)
     */
    void setObservationModel(
        bayes_adaptive::factored::BABNModel* model,
        std::vector<int> const& parents) const;

    /**** FBAPOMDPPrior interface ****/
    FBAPOMDPState* sampleFBAPOMDPState(State const* domain_state) const final;
    // flips an observation edge
    bayes_adaptive::factored::BABNModel::Structure
        mutate(bayes_adaptive::factored::BABNModel::Structure structure) const final;
};

} // namespace priors

#endif // FACTOREDTIGERPRIORS_HPP
