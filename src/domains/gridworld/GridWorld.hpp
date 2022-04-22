#ifndef GRIDWORLD_HPP
#define GRIDWORLD_HPP

#include "domains/POMDP.hpp"

#include <memory>
#include <string>
#include <vector>

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
class Reward;

namespace domains {

/**
 * @brief An 2-dimensional grid with reward-dispensing cells
 *
 * A 2-d grid world where the agent needs to go to a goal location (part of
 * the state space). The agent has 4 actions, a step in each direction,
 * that is carried out succesfully 95% of the time (and is a no-op
 * otherwise), except for some 'bad' cells, where the successrate drops to
 * 10%. The observation function is noisy, with some sort of gaussian
 * probability around the agent's real location (that accumulates around
 * the edges).
 *
 * @see `POMDP` and `Environment` for memory management of `State`, `Action`, and `Observation`
 **/
class GridWorld : public POMDP
{
public:
    // parameters
    constexpr static double const goal_reward    = 1;
    constexpr static double const step_reward    = 0;
    constexpr static double const move_prob      = .95;
    constexpr static double const slow_move_prob = .15;
    constexpr static double const wrong_obs_prob = .2;

    /**
     * @brief A state in the grid world problem
     **/
    class GridWorldState : public State
    {

    public:
        /**
         * @brief A position in the gridworld
         **/
        struct pos
        {
            unsigned int const x, y;

            bool operator==(pos const& other) const { return x == other.x && y == other.y; }
            bool operator!=(pos const& other) const { return !(*this == other); }

            std::string toString() const
            {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
            }
        };

        GridWorldState(pos agent_pos, pos goal_pos, int i) :
                _agent_position(agent_pos), _goal_position(goal_pos), _index(i)
        {
        }

        /***** state implementation *****/
        void index(int) final { throw "do not change gridworld states"; };
        int index() const final { return _index; }
        std::string toString() const final
        {
            return "agent" + _agent_position.toString() + ", goal" + _goal_position.toString();
        }

        pos const _agent_position, _goal_position;

    private:
        int const _index;
    };

    /**
     * @brief An obsdervation in the grid world problem (agent position)
     **/
    class GridWorldObservation : public Observation
    {
    public:
        GridWorldObservation(
            ::domains::GridWorld::GridWorldState::pos agent_pos,
            ::domains::GridWorld::GridWorldState::pos goal_pos,
            int i) :
                _agent_pos(agent_pos), _goal_pos(goal_pos), _index(i)
        {
        }

        /**** observation interface ***/
        void index(int /*i*/) final { throw "GridWorldObservation::index(i) not allowed"; }
        int index() const final { return _index; };
        std::string toString() const final { return _agent_pos.toString(); }

        ::domains::GridWorld::GridWorldState::pos const _agent_pos, _goal_pos;

    private:
        int const _index;
    };

    /**
     * @brief An action in the grid world problem
     **/
    class GridWorldAction : public Action
    {
    public:
        explicit GridWorldAction(int i) : _index(i) {}

        enum ACTION { UP, RIGHT, DOWN, LEFT };
        static std::vector<std::string> const action_descriptions; // initialized in cpp

        /*** action interface ***/
        void index(int /*i*/) final { throw "GridWorldAction should not edit index"; }
        int index() const final { return _index; }
        std::string toString() const final { return action_descriptions[_index]; }

    private:
        int const _index;
    };

    explicit GridWorld(size_t size);

    static std::vector<GridWorldState::pos> const start_locations;

    /***** getters of parameters and settings of the domain ****/
    size_t size() const;
    double goalReward() const;
    double correctObservationProb() const;
    static std::vector<GridWorldState::pos> goalLocations(int size);
    GridWorldState::pos const* goalLocation(unsigned int goal_index) const;
    std::vector<GridWorldState::pos> const* slowLocations() const;
    bool agentOnSlowLocation(GridWorldState::pos const& agent_pos) const;
    State const* sampleRandomState() const;

    /**
     * @brief returns probability of observing a location given the real one (1 dimensional)
     **/
    float obsDisplProb(unsigned int loc, unsigned int observed_loc) const;

    bool foundGoal(GridWorldState const* s) const;

    GridWorldState const*
        getState(GridWorldState::pos const& agent_pos, GridWorldState::pos const& goal_pos) const;
    GridWorldObservation const* getObservation(
        GridWorldState::pos const& agent_pos,
        GridWorldState::pos const& goal_pos) const;

    /**
     * @brief applies a move a on the old_pos to get a new pos
     **/
    GridWorldState::pos applyMove(GridWorldState::pos const& old_pos, Action const* a) const;

    /**** POMDP interface ****/
    Action const* generateRandomAction(State const* s) const final;
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;
    double computeObservationProbability(Observation const* o, Action const* a, State const* new_s)
        const final;
    void releaseAction(Action const* a) const final;
    Action const* copyAction(Action const* a) const final;

    /***** environment interface *****/
    State const* sampleStartState() const final;
    Terminal step(State const** s, Action const* a, Observation const** o, Reward* r) const final;
    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;
    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;

private:
    // problem settings
    size_t const _size;

    // initiated in constructor
    std::vector<GridWorldState::pos> _slow_locations;
    std::vector<GridWorldState::pos> _goal_locations;

    size_t _goal_amount;
    int _A_size = 4;
    int _S_size;

    // describes the probability of displacement in our observation in 1 dimension
    std::vector<float> _obs_displacement_probs = {};

    std::vector<GridWorldState> _S       = {};
    std::vector<GridWorldObservation> _O = {};

    // constants in the problem
    int const _goal_feature = 2;

    /**
     * @brief returns an observation from positions agent_pos and goal_pos
     **/
    Observation const* generateObservation(
        GridWorldState::pos const& agent_pos,
        GridWorldState::pos const& goal_pos) const;

    int positionsToIndex(GridWorldState::pos const& agent_pos, GridWorldState::pos const& goal_pos)
        const;

    /**
     * @brief populates _slow_locations
     */
    void generateSlowLocations();

    /** some functions to check input from system **/
    void assertLegal(Action const* a) const;
    void assertLegal(Observation const* o) const;
    void assertLegal(State const* s) const;
    void assertLegal(GridWorldState::pos const& position) const;
    void assertLegalGoal(GridWorldState::pos const& position) const;
};

} // namespace domains

#endif // GRIDWORLD_HPP
