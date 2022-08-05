#ifndef AGR_HPP
#define AGR_HPP

#include "domains/POMDP.hpp"

#include <random>
#include <sstream>

#include "utils/random.hpp"
#include <cassert>

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"

namespace domains {

class AGRState : public State
{
public:
    AGRState(int n, int i);
    AGRState(int n_in, int target_pos_in, int target_goal_in);

    static int indexOf(int n_in, int target_pos_in, int target_goal_in);
    void index(int i) override;
    int index() const override;

    std::string toString() const override;

    int target_pos;
    int target_goal;

private:
    int _n;
};

class AGRAction : public Action
{
public:
    AGRAction(int n, int i);
    AGRAction(int n_in, std::string type_in, int help_pos_in = -1);

    static int indexOf(int n, std::string const& type, int help_pos = -1);
    void index(int i) override;
    int index() const override;

    std::string toString() const override;

    std::string type;
    int help_pos;

private:
    int _n;
};

class AGRObservation : public Observation
{
public:
    AGRObservation(int n, int i);
    AGRObservation(int n_in, std::string type_in, int target_pos_in = -1);

    static int indexOf(int n, std::string const& type, int target_pos = -1);
    void index(int i) override;
    int index() const override;

    std::string toString() const override;

    std::string type;
    int target_pos;

private:
    int _n;
};

/**
 * @brief Simple 1D Active Goal Recognition (AGR) environment.
 *
 * A human (target) inhabits a 1D environment, and needs to travel from its
 * initial position to a goal position which he knows.  Once he is there, he
 * will need the agent's help as soon as possible.
 *
 * The agent (observer) is generally busy performing its own task (action
 * "work"), but should also actively monitor the state of the human (action
 * "observe") so that it knows when and where to help (action "help").
 *
 * To complicate things further, the agent starts without knowing the target
 * goal, and so needs to balance its own work duties and its observation
 * actions in order to obtain an accurate estimate of the goal state and when
 * too help without having the human wait too long.
 *
 * The domain is parametrized by `n` which indicates how many rooms there are
 * both to the left and to the right of the target's initial position  (i.e.
 * there are 2n + 1 cells in total.  Each of them could be the possible target.
 *
 * @see `POMDP` and `Environment` for memory management of `State`, `Action`, and `Observation`
 **/
class AGR : public POMDP
{
public:
    explicit AGR(int n);
    ~AGR() override;

    // allow shallow copies
    AGR(AGR const&)            = default;
    AGR& operator=(AGR const&) = default;

    State const* sampleStartState() const override;
    Action const* generateRandomAction(State const* s) const override;
    Terminal
        step(State const** s, Action const* a, Observation const** o, Reward* r) const override;

    double computeObservationProbability(Observation const* o, Action const* a, State const* s)
        const final;

    void addLegalActions(State const* s, std::vector<Action const*>* actions) const override;

    void releaseAction(Action const* a) const override;
    void releaseObservation(Observation const* o) const override;
    void releaseState(State const* s) const override;

    Action const* copyAction(Action const* a) const override;
    Observation const* copyObservation(Observation const* o) const override;
    State const* copyState(State const* s) const override;

    int _n;

    int _nstates;
    AGRState const** _states;

    int _nstart_states;
    AGRState const** _start_states;
    // cannot generate when const...
    mutable std::uniform_int_distribution<int> _start_state_distr;

    int _nactions;
    AGRAction const** _actions;
    // cannot generate when const...
    mutable std::uniform_int_distribution<int> _action_distr;

    int _nobservations;
    AGRObservation const** _observations;

private:
    void legalActionCheck(Action const* a) const;
    void legalObservationCheck(Observation const* o) const;
    void legalStateCheck(State const* s) const;
};

} // namespace domains

#endif // AGR_HPP
