#ifndef CollisionAvoidanceBigPRIORS_HPP
#define CollisionAvoidanceBigPRIORS_HPP

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"

#include <string>
#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "bayes-adaptive/states/factored/BABNModel.hpp"
#include "bayes-adaptive/states/factored/DBNNode.hpp"
#include "bayes-adaptive/states/table/BAFlatModel.hpp"
#include "domains/collision-avoidance-big/CollisionAvoidanceBig.hpp"

namespace configurations {
struct BAConf;
struct FBAConf;
} // namespace configurations

class Action;
class BAPOMDPState;
class State;

namespace priors {

/**
 * @brief The prior over the CollisionAvoidanceBig domains
 *
 * The observation probabilities are considered known,
 * the behaviour of the obstacle, however, may be noisy.
 *
 * The prior over whether the obstacle moves or not is
 * controlled by a single variable: noise. For no noise
 * (noise = 0), the true transitions, the obstacle is thought
 * to move with 25% probaility to either up or down.
 *
 * For noise =.1, this probability is decreased by .5,
 * increasing the probability of the obstacle staying with 10%.
 **/
class CollisionAvoidanceBigTablePrior : public BAPOMDPPrior
{
public:
    CollisionAvoidanceBigTablePrior(
        domains::CollisionAvoidanceBig const& d,
        configurations::BAConf const& c);

private:
    static int const _NUM_ACTIONS = 3;

    int const _height;
    int const _width;
    int const _num_obstacles;

    int const _num_speeds = 2;
    int const _num_traffics = 3;
    int const _num_timeofdays = 2;
    int const _num_obstacletypes = 3;

    float const _noise;
    float const _total_counts;

    std::vector<int> _obstacle_pos_ranges = std::vector<int>(_num_obstacles, _height);

    Domain_Size const _domain_size;
    bayes_adaptive::table::BAFlatModel _prior;

    BAPOMDPState* sampleBAPOMDPState(State const* domain_state) const final;

    /**
     * @brief sets the prior counts from <x,y,y_block,a>
     **/
    void setTransitionCounts(
        int x,
        int y,
        int a,
        std::vector<int> const& obstacles,
        std::vector<int> const& new_obstacles,
        domains::CollisionAvoidanceBig const& d);

    double obstacleTransProb(int y, int new_y) const;
};

class CollisionAvoidanceBigFactoredPrior : public FBAPOMDPPrior
{
public:
    explicit CollisionAvoidanceBigFactoredPrior(configurations::FBAConf const& conf);

    /*** FBAPOMDPPrior interface ****/
    FBAPOMDPState* sampleFullyConnectedState(State const* domain_state) const final;
    FBAPOMDPState* sampleCorrectGraphState(State const* domain_state) const final;
    bayes_adaptive::factored::BABNModel computePriorModel(
        bayes_adaptive::factored::BABNModel::Structure const& structure) const final;
    bayes_adaptive::factored::BABNModel::Structure
        mutate(bayes_adaptive::factored::BABNModel::Structure structure) const final;

private:
    // TODO only using at most 1 obstacle now
    static int const _NUM_ACTIONS = 3;

    constexpr static double const BLOCK_MOVE_PROB = .5;
    constexpr static double const MOVE_PROB_FAST = 0.85;
    constexpr static double const MOVE_PROB_SLOW = 0.85;

    static int const _AGENT_X_FEATURE = 0;
    static int const _AGENT_Y_FEATURE = 1;
    static int const _SPEED_FEATURE = 2;
    static int const _TRAFFIC_FEATURE = 3;
    static int const _TIMEOFDAY_FEATURE = 4;
    static int const _OBSTACLETYPE_FEATURE = 5;
//    static int const _OBSTACLE_FEATURE = 5; // there are potentially multiple obstacle features I think?

    bool const _abstraction;
    int const _num_obstacles;
    int const _width;
    int const _height;
    int const _num_speeds = 2;
    int const _num_traffics = 3;
    int const _num_timeofdays = 2;
    int const _num_obstacletypes = 3;

    int const _first_obstacle     = 6; // first feature number for obstacle
    int const _num_state_features = _first_obstacle + _num_obstacles;

    float const _noise, _counts_total;
    std::string const _edge_noise;

    Domain_Size const _domain_size;
    Domain_Feature_Size const _domain_feature_size;
    bayes_adaptive::factored::BABNModel::Indexing_Steps const _fbapomdp_step_size;

    // samples obstacles
    mutable std::uniform_int_distribution<int> _action_distr{
        rnd::integerDistribution(0, _NUM_ACTIONS)};
    mutable std::uniform_int_distribution<int> _obst_distr{
        rnd::integerDistribution(0, _num_obstacles)};

    std::vector<DBNNode> _observation_model = {}, _correctly_connected_transition_model = {},
                         _fully_connected_transition_model        = {},
                         _transition_model_without_block_features = {};

    FBAPOMDPState* sampleFBAPOMDPState(State const* domain_state) const final;

    /**
     * @brief sets the new y position for the agent
     *
     * action is the action taken
     * y is the original/current position
     * node should describe the next value of Y given this action and y
     **/
    void setAgentYTransition(Action const& a, int y, DBNNode& node) const; // Todo check if needs to be done

    /**
     * @brief sets the new y position of the obstacle
     *
     * y is the original/current position
     * noise & total_counts describe the accuracy & certainty of the prior
     * node should describe the next value of Y given this action and y
     **/
    std::vector<float> obstacleTransition(int y, int speed, int obstacletype) const;

    /**
     * @brief samples a structure for the obstacle behaviour
     *
     * assumes rest of the counts are already set
     **/
    void sampleBlockTModel(
        bayes_adaptive::factored::BABNModel* model,
        int obstacle_feature,
        std::vector<std::vector<int>> structure) const;

        std::vector<float> speedTransition(int speed, int traffic) const;

        int keepInBounds(int value) const;

        std::vector<float> trafficTransition(int traffic, int timeofday) const;

        std::vector<float> timeofdayTransition(int timeofday) const;

//        std::vector<float> obstacleTransition(int y, int speed);

        std::vector<float> obstacletypeTransition(int obstacletype) const;

        std::vector<float> xTransition(int x, int speed) const;

        std::vector<float> obstacleTransition(int y) const;
    };

} // namespace priors

#endif // CollisionAvoidanceBigPRIORS_HPP
