#include "RBAPOUCT.hpp"

#include <iomanip>
#include <string>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <boost/timer/timer.hpp>

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

RBAPOUCT::RBAPOUCT(configurations::Conf const& c) :
        _n(c.planner_conf.mcts_simulation_amount),
        _milliseconds_thinking(c.planner_conf.milliseconds_thinking),
        _max_depth(c.planner_conf.mcts_max_depth),
        _h(c.horizon),
        _u(c.planner_conf.mcts_exploration_const),
        _discount(c.discount) //,
//        _ucb_table(_n * _n)
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

//    initiateUCBTable();

    VLOG(1) << "initiated RBAPOUCT planner with " << _n << " simulations, " << _max_depth
            << " max depth, " << _u << " exploration constant, " << _discount.toDouble()
            << " discount and " << _h << " horizon";
}

Action const* RBAPOUCT::selectAction(
    BAPOMDP const& simulator,
    beliefs::BABelief const& belief,
    History const& history,
    int& total_simulations) const
{
    _stats = treeStatistics();

    simulator.addLegalActions(belief.sample(), &_actions);
    auto const _nactions = _actions.size();
    auto const root      = createActionNode(_actions);

    boost::timer::nanosecond_type const one_millisecond(1000000LL);

    // does not leak memory: actions stored in root node
    _actions.clear();

    // do not look further than the horizon
    _stats.max_tree_depth = std::min(_h - (int)history.length(), _max_depth);

    // make sure counts are not changed over time (changed back after simulations)
    auto const old_mode = simulator.mode();
    simulator.mode(BAPOMDP::StepType::KeepCounts);
//    int extra = 0;
//    if (_n % _sims_per_sample != 0) {
//        extra = 1;
//    }

    // perform simulations
    boost::timer::cpu_timer timer;
    int sims_done = 0;
//    float test = (float) _sims_per_sample/100;
    if (_milliseconds_thinking <= 0.01) {
        while (sims_done < _n) {
            sims_done++;
//        for (auto i = 0; i < (extra + _n / _sims_per_sample); ++i) {
            // do not copy!!
            auto particle = static_cast<BAState const *>(belief.sample());

            // but safe old state, so that we can reset the particle (counts are not modified)
            auto const old_domain_state = simulator.copyDomainState(particle->_domain_state);
            const_cast<BAState *>(particle)->_domain_state = simulator.copyDomainState(old_domain_state);
//            int sims_to_do = std::min(_sims_per_sample, _n - _sims_per_sample * i);
//            for (auto j = 0; j < sims_to_do; ++j) {

//                VLOG(4) << "RBAPOUCT sim " << i + 1 << "/" << _n << ": s_0=" << particle->toString();

//                auto r =
            traverseActionNode(root, particle, simulator, _stats.max_tree_depth);

//                VLOG(4) << "RBAPOUCT sim " << i + 1 << "/" << _n << "returned :" << r.toDouble();

            // return particle in correct state
            simulator.releaseDomainState(particle->_domain_state);
            const_cast<BAState *>(particle)->_domain_state = old_domain_state;
//            }
//        }
        }
    } else{
        boost::timer::cpu_times elapsed_time = timer.elapsed();
        do {
            for (auto i = 0; i < 100; ++i) {
                sims_done++;
                //        for (auto i = 0; i < (extra + _n / _sims_per_sample); ++i) {
                // do not copy!!
                auto particle = static_cast<BAState const *>(belief.sample());

                // but safe old state, so that we can reset the particle (counts are not modified)
                auto const old_domain_state = simulator.copyDomainState(particle->_domain_state);
                const_cast<BAState *>(particle)->_domain_state = simulator.copyDomainState(old_domain_state);
                //            int sims_to_do = std::min(_sims_per_sample, _n - _sims_per_sample * i);
                //            for (auto j = 0; j < sims_to_do; ++j) {

                //                VLOG(4) << "RBAPOUCT sim " << i + 1 << "/" << _n << ": s_0=" << particle->toString();

                //                auto r =
                traverseActionNode(root, particle, simulator, _stats.max_tree_depth);

                //                VLOG(4) << "RBAPOUCT sim " << i + 1 << "/" << _n << "returned :" << r.toDouble();

                // return particle in correct state
                simulator.releaseDomainState(particle->_domain_state);
                const_cast<BAState *>(particle)->_domain_state = old_domain_state;
            }
//            }
            elapsed_time = timer.elapsed();
////            VLOG(1) << "Elapsed time " << (elapsed_time.user + elapsed_time.system);
//
//        } while ((elapsed_time.user + elapsed_time.system) < (_milliseconds_thinking*one_millisecond));
        } while ((elapsed_time.wall) < (_milliseconds_thinking*one_millisecond));
    }

//    VLOG(1) << "Simulations done: " << sims_done;
    simulator.mode(old_mode);
    total_simulations += sims_done;

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

double RBAPOUCT::UCB(int m, int n) const
{
    assert(m >= 0 && n >= 0);
    if (n == 0) {
        return std::numeric_limits<double>::max();
    }
    return _u * sqrt(log1p(m) / n);
//    return _ucb_table[m * _n + n];
}

ChanceNode& RBAPOUCT::selectChanceNodeUCB(ActionNode* n, UCBExploration exploration_option) const
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

Return RBAPOUCT::traverseActionNode(
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

Return RBAPOUCT::traverseChanceNode(
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

    auto terminal = simulator.step(&s, n._action, &o, &immediate_reward);

    // continue traverse if not terminated
    if (!terminal.terminated())
    {
        // continue in tree if node exists
        if (n.hasChild(std::stoi(o->index())))
        {
            delayed_return = traverseActionNode(n.child(std::stoi(o->index())), s, simulator, depth_to_go - 1);
        } else // else create leaf and end with rollout
        {
            simulator.addLegalActions(s, &_actions);
            n.addChild(std::stoi(o->index()), createActionNode(_actions));

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

void RBAPOUCT::fillHistograms(
    std::vector<std::vector<int>>& histograms,
    ActionNode* n,
    int node_depth) const
{
    for (auto& chance_node : (*n))
    {
        histograms[node_depth][std::stoi(chance_node._action->index())] += chance_node.visited();

        for (auto& action_node : chance_node)
        { fillHistograms(histograms, action_node.second, node_depth + 1); }
    }
}

Return RBAPOUCT::rollout(State const* s, BAPOMDP const& simulator, int depth_to_go) const
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
        t            = simulator.step(&s, a, &o, &immediate_reward);

        ret.add(immediate_reward, discount);
        discount.increment();

        simulator.releaseAction(a);
        simulator.releaseObservation(o);

        depth_to_go--;
    }

    VLOG(5) << "RBAPOUCT finished rollout to depth " << _stats.max_tree_depth - depth_to_go;
    return ret;
}

ActionNode* RBAPOUCT::createActionNode(std::vector<Action const*> const& actions) const
{
    _stats.num_action_nodes++;
    _action_nodes.emplace_back(new ActionNode(actions));
    return _action_nodes.back();
}

void RBAPOUCT::freeTree(BAPOMDP const& simulator) const
{
    for (auto& n : _action_nodes)
    {

        for (auto& chance_node : *n) { simulator.releaseAction(chance_node._action); }

        delete (n);
    }

    _action_nodes.clear();
}

//void RBAPOUCT::initiateUCBTable()
//{
//    for (auto m = 0; m < _n; ++m)
//    {
//        _ucb_table[m * _n] = std::numeric_limits<double>::max();
//
//        for (auto n = 1; n < _n; ++n) { _ucb_table[m * _n + n] = _u * sqrt(log1p(m) / n); }
//    }
//}

} // namespace planners
