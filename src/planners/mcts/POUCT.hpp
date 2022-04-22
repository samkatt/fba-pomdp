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

    /*
     * @brief memory efficient way of storing action nodes
     *
     * All action nodes that make up the tree are pushed directly on this
     * vector. This makes for efficient memory management (e.g.\ looping over
     * vector to delete). Also, since it is a private member, this variable
     * will persist over multiple planning calls, reducing the number of times
     * it is re-allocating memory upon growing.
     *
     * `ChanceNode` do not need this, since those are stored as part of the
     * `ActionNode`
     *
     * This field is mutable, since it is simply an internal data structure
     * *that will chance* during constant operations (planning).
     *
     * @see `createActionNode`
     *
     * @todo we are storing _pointers_, so it is not clear at all this is
     * efficient, we still have the actual nodes randomly on the heap
     */
    mutable std::vector<ActionNode*> _action_nodes =
        {}; /*memory efficient way of storing action nodes */

    /*
     * @brief memory efficient holder for actions
     *
     * During planning there is often a need to store actions in an array.
     * Instead of building up this array every time, we keep this private
     * member around instead.
     *
     * Note that memory management of `Action` is done by `POMDP`, so we should
     * not try to keep a vector of (non-pointer) `Action`.
     *
     * @see `POMDP::addLegalActions`
     */
    mutable std::vector<Action const*> _actions = {};

    /*
     * @brief memory-efficient way of storing nodes
     *
     * Used in `selectChanceNodeUCB` to store and pick `ChanceNode`. The
     * function will incrementally build this vector over and over, so it makes
     * sense to instead keep a global vector to reduce memory (re-)allocation.
     */
    mutable std::vector<ChanceNode*> _best_chance_nodes = {};

    mutable struct treeStatistics
    {
        int max_tree_depth   = 0;
        int tree_depth       = 0;
        int num_action_nodes = 0;
    } _stats{};

    /** @brief returns the next chance node based on current statistics in
     * action
     *
     * This represents the tree-policy: given the current (action) node `n`, we
     * aim to pick the 'best' action. This best action results in the next
     * `ChanceNode`, also called 'after states' in RL. The best action is
     * picked with UCB (if `exploration_option == ON`).
     *
     * Should be called in conjuction with actually traversing the tree,
     * `traverseActionNode` and `traverseChanceNode`
     *
     * @param[in] n: current node we are at to pick next action/`ChanceNode`
     * @param[in] exploration_option: whether to apply exploration bonus
     *
     * @return the 'best' `ChanceNode` child from `n`
     *
     **/
    ChanceNode& selectChanceNodeUCB(ActionNode* n, UCBExploration exploration_option) const;

    /**
     * @brief Traverses recursively the tree from node `n`
     *
     * We are (recursively) traversing the current tree and at `ActionNode`
     * `n`. From here on, we pick an `Action` using `UCB`
     * (`selectChanceNodeUCB`).
     *
     * Note that the input state `s` is const, which seems odd knowing that it
     * gets updated during simulations. This is this way because *all* memory
     * management of states are left to the `simulator`, and thus the rest of the
     * program (including this part) may not modify it.
     *
     * @param[in] n: current node to continue traversing from
     * @param[in] simulator: used to simulate with
     * @param[in] s: current state (const to avoid modifications outside of `simulator`)
     * @param[in] depth_to_go: max depth *from this point on-wards*
     *
     * @return the accumulated (discounted) reward from this point on-wards
     **/
    Return
        traverseActionNode(ActionNode* n, State const* s, POMDP const& simulator, int depth_to_go)
            const;

    /**
     * @brief Traverses recursively the tree from node `n`
     *
     * We are (recursively) traversing the current tree and at `ChanceNode`
     * `n`. From here on, we simulate a step using `simulator` on state `s`
     * using `Action` stored in `n`. This will determine the next `ActionNode`
     * we will traverse (through `traverseActionNode`).
     *
     * Note that here it is possible that either the step is terminal, or that
     * the child `ActionNode` (associated with `Observation` generated in
     * simulation step) does not exist. So it is possible to exit the recursion
     * here and possibly continue with expand and evaluating (`rollout`) the
     * leaf.
     *
     * Note that the input state `s` is const, which seems odd knowing that it
     * gets updated during simulations. This is this way because *all* memory
     * management of states are left to the `simulator`, and thus the rest of the
     * program (including this part) may not modify it.
     *
     * @param[in] n: current node to continue traversing from
     * @param[in] simulator: used to simulate with
     * @param[in] s: current state (const to avoid modifications outside of `simulator`)
     * @param[in] depth_to_go: max depth *from this point on-wards*
     *
     * @return the accumulated (discounted) reward from this point on-wards
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
     *
     * This will create an `ActionNode` (and all appropriate `ChanceNode`
     * children *in* it), store it in `_action_node`, and return its pointer.
     *
     * This is an attempt at memory management of the tree. All action nodes
     * are stored in a vector (e.g. deallocation), so this function must be
     * called whenever a new `ActionNode` is created.
     *
     * @see `_action_nodes`
     *
     * @param[in] actions: all legal actions in current node, for which each a `ChanceNode` is
     *initated
     *
     * @return pointer to created `ActionNode`
     **/
    ActionNode* createActionNode(std::vector<Action const*> const& actions) const;

    /**
     * @brief Deallocates memory of tree
     *
     * - frees action nodes in _action_nodes
     * - frees actions in chance nodes (in action nodes)
     *
     *   @param[in] simulator: responsible (necessary) for memory management of actions
     **/
    void freeTree(POMDP const& simulator) const;

    /**
     * @brief sets up the quick ucb lookup table
     **/
    void initiateUCBTable();
};

} // namespace planners

#endif // POUCT_HPP
