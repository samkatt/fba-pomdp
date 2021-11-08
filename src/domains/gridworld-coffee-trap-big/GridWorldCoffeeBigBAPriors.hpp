#ifndef GridWorldCoffeeBigBAPRIORS_HPP
#define GridWorldCoffeeBigBAPRIORS_HPP

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "bayes-adaptive/states/factored/BABNModel.hpp"
#include "bayes-adaptive/states/table/BAFlatModel.hpp"
#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBig.hpp"
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
class GridWorldCoffeeBigFlatBAPrior : public BAPOMDPPrior
{
public:
    GridWorldCoffeeBigFlatBAPrior(domains::GridWorldCoffeeBig const& domain, configurations::BAConf const& c);

private:
    size_t const _size;
    bool const _abstraction;
    size_t const _extra_features;
    std::vector<int> const rain_values = {0,1};

    float const _unknown_counts_total;
    float const _static_total_count = 100000;

    int const _x_feature = 0;
    int const _y_feature = 0;
    int const _rain_feature = 0;

    Domain_Size _domain_size;

    bayes_adaptive::table::BAFlatModel _prior_model;

    void setPriorTransitionProbabilities(
        domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const* s,
        domains::GridWorldCoffeeBig::GridWorldCoffeeBigAction const* a,
        domains::GridWorldCoffeeBig const& domain);

    void setPriorObservationProbabilities(
        domains::GridWorldCoffeeBig::GridWorldCoffeeBigAction const* a,
        domains::GridWorldCoffeeBig::GridWorldCoffeeBigState const* new_s,
        domains::GridWorldCoffeeBig const& domain);

    /*** BAPrior interface **/
    BAPOMDPState* sampleBAPOMDPState(State const* s) const final;
};

/**
 * FACTORED STUFF
 */
/**
 * @brief The prior for the GridWorld (factored) bayes-adaptive problem
 **/
class GridWorldCoffeeBigFactBAPrior : public FBAPOMDPPrior
{
public:
    GridWorldCoffeeBigFactBAPrior(domains::GridWorldCoffeeBig const& domain, configurations::FBAConf const& c);

    /*** BAPrior interface **/
    bayes_adaptive::factored::BABNModel computePriorModel(
        bayes_adaptive::factored::BABNModel::Structure const& structure) const override;
    FBAPOMDPState* sampleFullyConnectedState(State const* domain_state) const final;
    FBAPOMDPState const* sampleCorrectGraphState(State const* domain_state) const final;
    bayes_adaptive::factored::BABNModel::Structure
    mutate(bayes_adaptive::factored::BABNModel::Structure structure) const final;

    // Extra
    Domain_Feature_Size* getDomainFeatureSize();
//    int _num_abstractions;
//    std::vector<int> _minimum_abstraction;

private:
    size_t const _size;
    bool const _abstraction;
    size_t const _carpet_tiles;

    float const _unknown_counts_total;
    bool const _correct_prior;
    float const _static_total_count = 100000;

//    bool const _only_know_loc_matters;

    // x, y, rain, velocity, carpet
    int const _agent_x_feature = 0;
    int const _agent_y_feature = 1;
    int const _rain_feature    = 2;
    int const _num_features = 3 + _carpet_tiles;

    Domain_Size _domain_size;
    Domain_Feature_Size _domain_feature_size;

    bayes_adaptive::factored::BABNModel::Indexing_Steps _indexing_steps;
    bayes_adaptive::factored::BABNModel _correct_struct_prior;

    domains::GridWorldCoffeeBig const& _domain;

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

#endif // GridWorldCoffeeBigBAPRIORS_HPP
