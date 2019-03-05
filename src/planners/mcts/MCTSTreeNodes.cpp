#include "MCTSTreeNodes.hpp"

ChanceNode::ChanceNode(Action const* a) : _action(a)
{
    assert(a != nullptr);
}

void ChanceNode::addVisit(double r)
{
    _visit_count++;
    _q += ((r - _q) / _visit_count);
}

int ChanceNode::visited() const
{
    return _visit_count;
}

double ChanceNode::qValue() const
{
    return _q;
}

std::string ChanceNode::toString() const
{
    return "(a=" + _action->toString() + ", q=" + std::to_string(_q)
           + ", n=" + std::to_string(_visit_count) + ")";
}

ActionNode* ChanceNode::child(int i)
{
    return _children.at(i);
}

bool ChanceNode::hasChild(int i) const
{
    return _children.count(i) != 0u;
}

void ChanceNode::addChild(int i, ActionNode* n)
{
    assert(n != nullptr);
    _children.insert({i, n});
}

int ActionNode::visited() const
{
    return _visit_count;
}

ActionNode::ActionNode(std::vector<Action const*> const& legal_actions)
{
    assert(!legal_actions.empty());

    _children.reserve(legal_actions.size());
    for (auto const& a : legal_actions) { _children.emplace_back(ChanceNode(a)); }
}

void ActionNode::addVisit()
{
    _visit_count++;
}

std::string ActionNode::toString() const
{
    return "(n=" + std::to_string(_visit_count) + ", with " + std::to_string(_children.size())
           + " actions)";
}
