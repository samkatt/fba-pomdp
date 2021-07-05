#include "RBAPOUCT_abstraction.hpp"

#include <iomanip>
#include <string>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <bayes-adaptive/states/factored/AbstractFBAPOMDPState.hpp>

#include "easylogging++.h"
#include "utils/random.hpp"

#include "configurations/Conf.hpp"

#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "beliefs/bayes-adaptive/BABelief.hpp"
#include "environment/History.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"

#include "utils/Entropy.hpp"

namespace planners {

RBAPOUCT_abstraction::RBAPOUCT_abstraction(configurations::Conf const& c) :
        _n(c.planner_conf.mcts_simulation_amount),
        _max_depth(c.planner_conf.mcts_max_depth),
        _h(c.horizon),
        _u(c.planner_conf.mcts_exploration_const),
        _discount(c.discount),
        _ucb_table(_n * _n)
{

    if (_n < 1)
    {
        throw "cannot initiate RBAPOUCT with " + std::to_string(_n)
            + " simulations, must be greater than 0";
    }

    if (_max_depth < 0)
    {
        throw "cannot initiate RBAPOUCT with " + std::to_string(_max_depth)
            + " max depth, must be greater or equal to 0";
    }

    if (_h <= 0)
    {
        throw "cannot initiate RBAPOUCT with " + std::to_string(_h)
            + " horizon, must be greater than 0";
    }

    // make sure we are working with actual discount
    const_cast<Discount*>(&_discount)->increment();

    initiateUCBTable();

    VLOG(1) << "initiated RBAPOUCT planner with " << _n << " simulations, " << _max_depth
            << " max depth, " << _u << " exploration constant, " << _discount.toDouble()
            << " discount and " << _h << " horizon";
}

Action const* RBAPOUCT_abstraction::selectAction(
    BAPOMDP const& simulator,
    beliefs::BABelief const& belief,
    History const& history) const
{
    _stats = treeStatistics();

    simulator.addLegalActions(belief.sample(), &_actions);
    auto const _nactions = _actions.size();
    auto const root      = createActionNode(_actions);

    // does not leak memory: actions stored in root node
    _actions.clear();

    // do not look further than the horizon
    _stats.max_tree_depth = std::min(_h - (int)history.length(), _max_depth);

    // make sure counts are not changed over time (changed back after simulations)
    auto const old_mode = simulator.mode();
    simulator.mode(BAPOMDP::StepType::KeepCounts);
    int sims_per_particle = 200;
    // perform simulations
    for (auto i = 0; i < _n/sims_per_particle; ++i)
    {
        // hier particle aanpassen?
        // do not copy!!
        auto particle = (AbstractFBAPOMDPState *) static_cast<BAState const *>(belief.sample());

        // Adding abstraction, if it's not there yet.
        if (*static_cast<AbstractFBAPOMDPState*>(particle)->getAbstraction() != 0) {
            static_cast<AbstractFBAPOMDPState*>(particle)->setAbstraction(0);
        }

        // but safe old state, so that we can reset the particle (counts are not modified)
        auto const old_domain_state = simulator.copyDomainState(particle->_domain_state); // domain_state = pomdp state
        particle->_domain_state = simulator.copyDomainState(old_domain_state);
        for (auto j = 0; j < sims_per_particle; ++j){
            VLOG(4) << "RBAPOUCT sim " << i + 1 << "/" << _n << ": s_0=" << particle->toString();

            auto r = traverseActionNode(root, particle, simulator, _stats.max_tree_depth);

            VLOG(4) << "RBAPOUCT sim " << i + 1 << "/" << _n << "returned :" << r.toDouble();

            // return particle in correct state
            simulator.releaseDomainState(particle->_domain_state);
            particle->_domain_state = old_domain_state;
        }
    }
    simulator.mode(old_mode);

    // pick best action
    auto const& best_chance_node = selectChanceNodeUCB(root, UCBExploration::OFF);
    auto const best_action       = simulator.copyAction(best_chance_node._action);

    assert(_stats.num_action_nodes == (int)_action_nodes.size());

    if (VLOG_IS_ON(3))
    {
        VLOG(3) << "po-uct picked node " << best_chance_node.toString()
                << " at tree of depth=" << _stats.tree_depth << " and " << _stats.num_action_nodes
                << " action nodes";

        VLOG(3) << "Action stats:";
        for (auto const& n : *root) { VLOG(3) << "\t" << n.toString(); }
    }

    if (VLOG_IS_ON(4))
    {
        auto nlayers = _stats.tree_depth + 2;
        std::vector<double> entropies(nlayers);

        std::vector<std::vector<int>> histograms(nlayers, std::vector<int>(_nactions, 0));

        fillHistograms(histograms, root);

        for (unsigned int i = 0; i < histograms.size(); i++)
        {
            // Computing percentual entropy
            entropies[i] = ent::H(histograms[i]) / log2(_nactions);
        }

        std::stringstream ss;
        ss << "Entropies:";
        for (auto h : entropies)
        { ss << " " << std::fixed << std::setprecision(1) << 100 * h << "%"; }
        VLOG(4) << ss.str();
    }

    freeTree(simulator);
    return best_action;
}

double RBAPOUCT_abstraction::UCB(int m, int n) const
{
    assert(m >= 0 && n >= 0);

    return _ucb_table[m * _n + n];
}

ChanceNode& RBAPOUCT_abstraction::selectChanceNodeUCB(ActionNode* n, UCBExploration exploration_option) const
{
    assert(n != nullptr);

    _action_node_holder.clear();

    double best_q = -std::numeric_limits<double>::max();

    auto const m = n->visited();
    // loop over all action nodes and extract best action
    for (auto& chance_node : *n)
    {
        auto q = chance_node.qValue();

        if (exploration_option == UCBExploration::ON)
        {
            q += UCB(m, chance_node.visited());
        }

        if (q >= best_q)
        {
            if (q > best_q)
            {
                _action_node_holder.clear();
            }

            best_q = q;
            _action_node_holder.emplace_back(&chance_node);
        }
    }

    assert(!_action_node_holder.empty());

    // return random action
    return *_action_node_holder[rnd::slowRandomInt(0, (int)_action_node_holder.size())];
}

Return RBAPOUCT_abstraction::traverseActionNode(
    ActionNode* n,
    State const* s,
    BAPOMDP const& simulator,
    int depth_to_go) const
{
    assert(n != nullptr && s != nullptr);
    assert(depth_to_go >= 0);

    VLOG(5) << "at depth " << _stats.max_tree_depth - depth_to_go << " in action node "
            << n->toString();

    _stats.tree_depth = std::max(_stats.tree_depth, _stats.max_tree_depth - depth_to_go);

    if (depth_to_go == 0)
    {
        return Return(0);
    }

    auto& chance_node = selectChanceNodeUCB(n, UCBExploration::ON);
    auto const ret    = traverseChanceNode(chance_node, s, simulator, depth_to_go);

    n->addVisit();
    return ret;
}

Return RBAPOUCT_abstraction::traverseChanceNode(
    ChanceNode& n,
    State const* s,
    BAPOMDP const& simulator,
    int depth_to_go) const
{
    assert(s != nullptr);
    assert(depth_to_go > 0);

    VLOG(5) << "at depth " << _stats.max_tree_depth - depth_to_go << " in action node "
            << n.toString();

    Observation const* o(nullptr);
    Reward immediate_reward(0);
    Return delayed_return;

    auto terminal = simulator.step(&s, n._action, &o, &immediate_reward, BAPOMDP::SampleType::Abstract);

    // continue traverse if not terminated
    if (!terminal.terminated())
    {
        // continue in tree if node exists
        if (n.hasChild(o->index()))
        {
            delayed_return = traverseActionNode(n.child(o->index()), s, simulator, depth_to_go - 1);
        } else // else create leaf and end with rollout
        {
            simulator.addLegalActions(s, &_actions);
            n.addChild(o->index(), createActionNode(_actions));

            // does not leak memory, actions stored in nodes
            _actions.clear();

            delayed_return = rollout(s, simulator, depth_to_go - 1);
        }
    }

    // collect results
    auto const ret = immediate_reward.toDouble() + _discount.toDouble() * delayed_return.toDouble();
    n.addVisit(ret);

    // return
    simulator.releaseObservation(o);
    return Return(ret);
}

void RBAPOUCT_abstraction::fillHistograms(
    std::vector<std::vector<int>>& histograms,
    ActionNode* n,
    int node_depth) const
{
    for (auto& chance_node : (*n))
    {
        histograms[node_depth][chance_node._action->index()] += chance_node.visited();

        for (auto& action_node : chance_node)
        { fillHistograms(histograms, action_node.second, node_depth + 1); }
    }
}

Return RBAPOUCT_abstraction::rollout(State const* s, BAPOMDP const& simulator, int depth_to_go) const
{
    assert(s != nullptr && depth_to_go >= 0);

    auto immediate_reward = Reward(0);
    auto ret              = Return();
    auto t                = Terminal(false);
    auto discount         = Discount(_discount.toDouble());

    Observation const* o;

    // rollout until termination
    while (depth_to_go > 0 && !t.terminated())
    {
        auto const a = simulator.generateRandomAction(s);
        t            = simulator.step(&s, a, &o, &immediate_reward, BAPOMDP::SampleType::Abstract);

        ret.add(immediate_reward, discount);
        discount.increment();

        simulator.releaseAction(a);
        simulator.releaseObservation(o);

        depth_to_go--;
    }

    VLOG(5) << "RBAPOUCT finished rollout to depth " << _stats.max_tree_depth - depth_to_go;
    return ret;
}

ActionNode* RBAPOUCT_abstraction::createActionNode(std::vector<Action const*> const& actions) const
{
    _stats.num_action_nodes++;
    _action_nodes.emplace_back(new ActionNode(actions));
    return _action_nodes.back();
}

void RBAPOUCT_abstraction::freeTree(BAPOMDP const& simulator) const
{
    for (auto& n : _action_nodes)
    {

        for (auto& chance_node : *n) { simulator.releaseAction(chance_node._action); }

        delete (n);
    }

    _action_nodes.clear();
}

void RBAPOUCT_abstraction::initiateUCBTable()
{
    for (auto m = 0; m < _n; ++m)
    {
        _ucb_table[m * _n] = std::numeric_limits<double>::max();

        for (auto n = 1; n < _n; ++n) { _ucb_table[m * _n + n] = _u * sqrt(log1p(m) / n); }
    }
}

} // namespace planners
