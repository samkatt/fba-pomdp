#ifndef GRIDWORLDCOFFEEBAPRIORS_HPP
#define GRIDWORLDCOFFEEBAPRIORS_HPP

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "bayes-adaptive/states/factored/BABNModel.hpp"
#include "bayes-adaptive/states/table/BAFlatModel.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffee.hpp"
#include "utils/index.hpp"
class BAPOMDPState;
//class FBAPOMDPState;
class AbstractFBAPOMDPState;
class State;
namespace configurations {
struct BAConf;
struct FBAConf;
} // namespace configurations

namespace priors {

/**
 * @brief The prior for the GridWorld (flat) bayes-adaptive problem
 **/
class GridWorldCoffeeFlatBAPrior : public BAPOMDPPrior
{
public:
    GridWorldCoffeeFlatBAPrior(domains::GridWorldCoffee const& domain, configurations::BAConf const& c);

private:
    size_t const _size;
    size_t const _carpet_configurations;
    std::vector<unsigned int> const rain_values = {0,1};

    float const _unknown_counts_total;
    float const _static_total_count = 100000;

    Domain_Size _domain_size;

    bayes_adaptive::table::BAFlatModel _prior_model;

    void setPriorTransitionProbabilities(
        domains::GridWorldCoffee::GridWorldCoffeeState const* s,
        domains::GridWorldCoffee::GridWorldCoffeeAction const* a,
        domains::GridWorldCoffee const& domain);

    void setPriorObservationProbabilities(
        domains::GridWorldCoffee::GridWorldCoffeeAction const* a,
        domains::GridWorldCoffee::GridWorldCoffeeState const* new_s,
        domains::GridWorldCoffee const& domain);

    /*** BAPrior interface **/
    BAPOMDPState* sampleBAPOMDPState(State const* s) const final;
};

/**
 * FACTORED STUFF
 */
/**
 * @brief The prior for the GridWorld (factored) bayes-adaptive problem
 **/
class GridWorldCoffeeFactBAPrior : public FBAPOMDPPrior
{
public:
    GridWorldCoffeeFactBAPrior(domains::GridWorldCoffee const& domain, configurations::FBAConf const& c);

    /*** BAPrior interface **/
    bayes_adaptive::factored::BABNModel computePriorModel(
        bayes_adaptive::factored::BABNModel::Structure const& structure) const override;
    FBAPOMDPState* sampleFullyConnectedState(State const* domain_state) const final;
    FBAPOMDPState const* sampleCorrectGraphState(State const* domain_state) const final;
    bayes_adaptive::factored::BABNModel::Structure
    mutate(bayes_adaptive::factored::BABNModel::Structure structure) const final;

    // Extra
    Domain_Feature_Size* getDomainFeatureSize();

private:
    size_t const _size;
    size_t const _carpet_configurations;

//    float const _noise;
    float const _unknown_counts_total;
    float const _static_total_count = 100000;

//    bool const _only_know_loc_matters;

    // x, y, rain, velocity, carpet
    int const _agent_x_feature = 0;
    int const _agent_y_feature = 1;
    int const _rain_feature    = 2;
//    int const _velocity_feature    = 3;
    int const _carpet_feature    = 3; // carpet = beforehand we know on which tiles there is carpet
    int const _num_features = 4;

    Domain_Size _domain_size;
    Domain_Feature_Size _domain_feature_size;

    bayes_adaptive::factored::BABNModel::Indexing_Steps _indexing_steps;
    bayes_adaptive::factored::BABNModel _correct_struct_prior;

    domains::GridWorldCoffee const& _domain;

    void preComputePrior();

    void setNoisyTransitionNode(
    bayes_adaptive::factored::BABNModel* model,
    Action const& action,
    int feature,
    std::vector<int> const& parents) const;

    /*** FBAPrior interface **/
    FBAPOMDPState* sampleFBAPOMDPState(State const* domain_state) const final;
    };

} // namespace priors

#endif // GRIDWORLDCOFFEEBAPRIORS_HPP
