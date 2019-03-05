#ifndef POUCT_HPP
#define POUCT_HPP

#include "planners/Planner.hpp"

#include <vector>

#include "environment/Discount.hpp"
#include "environment/Return.hpp"
#include "planners/mcts/MCTSTreeNodes.hpp"
class Action;
class Belief;
class History;
class POMDP;
class State;
namespace configurations {
struct Conf;
}

namespace planners {

/**
 * @brief Monte-Carlo tree search method
 **/
class POUCT : public Planner
{
public:
    explicit POUCT(configurations::Conf const& c);

    /**** Planner interface ****/
    Action const* selectAction(POMDP const& simulator, Belief const& belief, History const& history)
        const override;

private:
    enum UCBExploration { ON, OFF };

    // params
    int const _n; // number of simulations
    int const _max_depth; // max depth of the search tree
    int const _h; // horizon of the problem
    double const _u; // exploration constant
    Discount const _discount; // discount used during simulations

    std::vector<double> _ucb_table; // quick ucb lookup table

    mutable std::vector<ActionNode*> _action_nodes =
        {}; // memory efficient way of storing action nodes
    mutable std::vector<Action const*> _actions = {}; // temporary container for actions
    mutable std::vector<ChanceNode*> _action_node_holder =
        {}; // temporary container for ChanceNodes

    mutable struct treeStatistics
    {
        int max_tree_depth   = 0;
        int tree_depth       = 0;
        int num_action_nodes = 0;
    } _stats{};

    /**
     * @brief returns the next chance node based on current statistics in action
     *
     * Applies UCB if UCBExploration::ON is used as input
     **/
    ChanceNode& selectChanceNodeUCB(ActionNode* n, UCBExploration exploration_option) const;

    /**
     * @brief traverses into the tree
     *
     * Here we are at the top of a MCTSTree, meaning
     * that we will need to take an action and continue
     * traversing the tree from that action node
     **/
    Return
        traverseActionNode(ActionNode* n, State const* s, POMDP const& simulator, int depth_to_go)
            const;

    /**
     * @brief traverses into the tree
     *
     * Here we have picked an action at some MCTSTree,
     * meaning that we will simulate a step and continue
     * traversing from the respective child MCTSTree
     **/
    Return
        traverseChanceNode(ChanceNode& n, State const* s, POMDP const& simulator, int depth_to_go)
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
    Return rollout(State const* s, POMDP const& simulator, int depth_to_go) const;

    /**
     * @brief creates an action node and returns a pointer to it
     **/
    ActionNode* createActionNode(std::vector<Action const*> const& actions) const;

    /**
     * @brief deallocates memory of tree
     *
     * - frees action nodes in _action_nodes
     * - frees actions in chance nodes (in action nodes)
     **/
    void freeTree(POMDP const& simulator) const;

    /**
     * @brief sets up the quick ucb lookup table
     **/
    void initiateUCBTable();
};

} // namespace planners

#endif // POUCT_HPP
