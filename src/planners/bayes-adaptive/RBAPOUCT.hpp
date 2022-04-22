#ifndef RBAPOUCT_HPP
#define RBAPOUCT_HPP

#include "planners/bayes-adaptive/BAPlanner.hpp"

#include <vector>

#include "environment/Action.hpp"
#include "environment/Discount.hpp"
#include "environment/Return.hpp"
#include "planners/mcts/MCTSTreeNodes.hpp"
class BAPOMDP;
class History;
namespace beliefs {
class BABelief;
}
namespace configurations {
struct Conf;
}

namespace planners {

/**
 * @brief Plans with respect to b(s) and a sampled model ~ p(D)
 *
 * @todo: Most functions are a direct copy of `POUCT`. Design wise, this is a
 * shame, and could be improved upon.
 **/
class RBAPOUCT : public BAPlanner
{

public:
    explicit RBAPOUCT(configurations::Conf const& c);

    Action const* selectAction(
        BAPOMDP const& simulator,
        beliefs::BABelief const& belief,
        History const& history) const final;

private:
    enum UCBExploration { ON, OFF };

    // params
    int const _n; // number of simulations
    int const _max_depth; // max depth of the search tree
    int const _h; // horizon of the problem
    double const _u; // exploration constant
    Discount const _discount; // discount used during simulations

    std::vector<double> _ucb_table; // quick ucb lookup table

    /*
     * @brief memory efficient way of storing action nodes
     *
     * @see `POUCT:_action_nodes`
     */
    mutable std::vector<ActionNode*> _action_nodes = {};

    /*
     * @brief temporary container for actions
     *
     * @see `POUCT:_actions`
     */
    mutable std::vector<Action const*> _actions = {};

    /*
     * @brief container for temporary `ChanceNodes`
     *
     * @see `POUCT:_best_chance_nodes`
     */
    mutable std::vector<ChanceNode*> _best_chance_nodes = {};

    mutable struct treeStatistics
    {
        int max_tree_depth   = 0;
        int tree_depth       = 0;
        int num_action_nodes = 0;
    } _stats{};

    /**
     * @brief returns the next chance node based on current statistics in action
     *
     * @see `POUCT::selectChanceNodeUCB`
     **/
    ChanceNode& selectChanceNodeUCB(ActionNode* n, UCBExploration exploration_option) const;

    /**
     * @brief Traverses recursively the tree from node `n`
     *
     * @see `POUCT::traverseActionNode`
     **/
    Return
        traverseActionNode(ActionNode* n, State const* s, BAPOMDP const& simulator, int depth_to_go)
            const;

    /**
     * @brief traverses into the tree
     *
     * @see `POUCT::tranverseChanceNode`
     **/
    Return
        traverseChanceNode(ChanceNode& n, State const* s, BAPOMDP const& simulator, int depth_to_go)
            const;

    /**
     * @brief looks up / calculates the UCB value
     **/
    double UCB(int m, int n) const;

    /**
     * @brief computes histogram of action selection in current tree
     * starting from node depth
     *
     * Aggregates action densities belonging to the same tree depth
     **/
    void fillHistograms(
        std::vector<std::vector<int>>& histograms,
        ActionNode* n,
        int node_depth = 0) const;

    /**
     * @brief perform a rollout starting from a specific state and depth to go
     **/
    Return rollout(State const* s, BAPOMDP const& simulator, int depth_to_go) const;

    /**
     * @brief creates an action node and returns a pointer to it
     *
     * @see `POUCT::createActionNode`
     **/
    ActionNode* createActionNode(std::vector<Action const*> const& actions) const;

    /**
     * @brief deallocates memory of tree
     *
     * - frees action nodes in _action_nodes
     * - frees actions in chance nodes (in action nodes)
     **/
    void freeTree(BAPOMDP const& simulator) const;

    /**
     * @brief sets up the quick ucb lookup table
     **/
    void initiateUCBTable();
};

} // namespace planners

#endif // RBAPOUCT_HPP
