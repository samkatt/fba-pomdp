#ifndef MCTSTREENODES_HPP
#define MCTSTREENODES_HPP

#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

#include "environment/Action.hpp"

/**
 * @brief Nodes used to create MCTS trees
 *
 * A tree consists of action nodes and chance nodes.
 * Action nodes is where the children correspond to
 * actions, while the children of chance nodes correspond
 * to the possible observations.
 *
 * When MCTS is doing an iteration and travels through the tree,
 * it should pick an action whenever it is in an action node, and
 * it should simulate a step whenever in a chance node.
 **/

// forward declaration
class ActionNode;

/**
 * @brief A MCTS node that the is reached after selecting an action, contains a child per each
 *possible observation
 *
 * This node is visited whenever the corresponding action
 * in the parent ActionNode has been picked. It contains
 * statistics for how often it has been visited and what the
 * average return (q value) is
 *
 * The child nodes of this node correspond to the observations
 * that can be generated.
 **/
class ChanceNode
{
private:
    // how often this node has been visited
    int _visit_count = 0;

    // expected return when taking this action (q value)
    double _q = 0;

    // maps observation index (chance) to child node
    std::unordered_map<int, ActionNode*> _children{};

public:
    explicit ChanceNode(Action const* a);

    // disallow shallow copies
    ChanceNode(ChanceNode const&)            = default;
    ChanceNode(ChanceNode&&)                 = default;
    ChanceNode& operator=(ChanceNode const&) = default;
    ChanceNode& operator=(ChanceNode&&)      = default;

    // registers visiting the node with return r
    void addVisit(double r);

    /*** iterators ***/
    auto begin() -> decltype(_children)::iterator { return _children.begin(); }
    auto end() -> decltype(_children)::iterator { return _children.end(); }
    auto cbegin() const -> decltype(_children)::const_iterator const { return _children.cbegin(); }
    auto cend() const -> decltype(_children)::const_iterator const { return _children.cend(); }

    // manipulate tree functions
    ActionNode* child(int i);
    bool hasChild(int i) const;
    void addChild(int i, ActionNode* n);

    int visited() const;
    double qValue() const;

    std::string toString() const;

    // action associated with this node
    Action const* const _action;
};

class ActionNode
{
private:
    // how often this node has been visited
    int _visit_count = 0;

    // maps action to child node
    std::vector<ChanceNode> _children{};

public:
    /**
     * @brief constructs a tree with an action node for each legal action
     **/
    explicit ActionNode(std::vector<Action const*> const& legal_actions);

    void addVisit();

    /*** iterators ***/
    auto begin() -> decltype(_children)::iterator { return _children.begin(); }
    auto end() -> decltype(_children)::iterator { return _children.end(); }
    auto cbegin() const -> decltype(_children)::const_iterator const { return _children.cbegin(); }
    auto cend() const -> decltype(_children)::const_iterator const { return _children.cend(); }

    int visited() const;
    std::string toString() const;
};

#endif // MCTSTREENODES_HPP
