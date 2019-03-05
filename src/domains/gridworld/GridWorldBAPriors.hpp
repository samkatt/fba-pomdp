#ifndef GRIDWORLDBAPRIORS_HPP
#define GRIDWORLDBAPRIORS_HPP

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "bayes-adaptive/states/factored/BABNModel.hpp"
#include "bayes-adaptive/states/table/BAFlatModel.hpp"
#include "domains/gridworld/GridWorld.hpp"
#include "utils/index.hpp"
class BAPOMDPState;
class FBAPOMDPState;
class State;
namespace configurations {
struct BAConf;
struct FBAConf;
} // namespace configurations

namespace priors {

/**
 * @brief The prior for the GridWorld (flat) bayes-adaptive problem
 **/
class GridWorldFlatBAPrior : public BAPOMDPPrior
{
public:
    GridWorldFlatBAPrior(domains::GridWorld const& domain, configurations::BAConf const& c);

private:
    size_t const _size;

    float const _noise;
    float const _unknown_counts_total;
    float const _static_total_count = 100000;

    Domain_Size _domain_size;
    std::vector<domains::GridWorld::GridWorldState::pos> const _goal_locations;

    bayes_adaptive::table::BAFlatModel _prior_model;

    void setPriorTransitionProbabilities(
        domains::GridWorld::GridWorldState const* s,
        domains::GridWorld::GridWorldAction const* a,
        domains::GridWorld const& domain);

    void setPriorObservationProbabilities(
        domains::GridWorld::GridWorldAction const* a,
        domains::GridWorld::GridWorldState const* new_s,
        domains::GridWorld const& domain);

    /*** BAPrior interface **/
    BAPOMDPState* sampleBAPOMDPState(State const* s) const final;
};

/**
 * @brief The prior for the GridWorld (factored) bayes-adaptive problem
 **/
class GridWorldFactBAPrior : public FBAPOMDPPrior
{
public:
    GridWorldFactBAPrior(domains::GridWorld const& domain, configurations::FBAConf const& c);

    /*** BAPrior interface **/
    bayes_adaptive::factored::BABNModel computePriorModel(
        bayes_adaptive::factored::BABNModel::Structure const& structure) const override;
    FBAPOMDPState* sampleFullyConnectedState(State const* domain_state) const final;
    FBAPOMDPState const* sampleCorrectGraphState(State const* domain_state) const final;
    bayes_adaptive::factored::BABNModel::Structure
        mutate(bayes_adaptive::factored::BABNModel::Structure structure) const final;

private:
    size_t const _size;

    float const _noise;
    float const _unknown_counts_total;
    float const _static_total_count = 100000;

    bool const _only_know_loc_matters;

    int const _agent_x_feature = 0;
    int const _agent_y_feature = 1;
    int const _goal_feature    = 2;

    Domain_Size _domain_size;
    Domain_Feature_Size _domain_feature_size;

    bayes_adaptive::factored::BABNModel::Indexing_Steps _indexing_steps;
    bayes_adaptive::factored::BABNModel _correct_struct_prior;

    domains::GridWorld const& _domain;

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

#endif // GRIDWORLDBAPRIORS_HPP
