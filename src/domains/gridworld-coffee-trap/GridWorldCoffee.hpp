//
// Created by rolf on 28-09-20.
//

#ifndef GRIDWORLDCOFFEE_HPP
#define GRIDWORLDCOFFEE_HPP

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
 * @brief A 2-dimensional grid with trap states
 *
 * A 2-d 5x5 grid world where the agent needs to go to a goal location. The agent has 4 actions, a step in each direction.
 * There is a state variable movement speed, that tells how fast you move more or less.
 * In most states this is 0.9 tiles per second, giving a 90% probability to successfully move in the desired direction.
 * There are some trap positions, where the probability to successfully move is only 15% (0.15 tiles per second speed).
 * There is a carpet between the start and goal location, the agent can stay of the carpet if it walks around the edge.
 * Initially it thinks the carpet will slow itself down.
 * Sometimes it also rains.
 * It also thinks that when it rains the carpet will slow himself down even more.
 *
 * The agent can observe its position, the rain, and whether or not he is on the carpet.
 * He can not observe the movement speed that he has at his location.
 * The goal and starting location are fixed, and at the start it's always dry.
 *
 **/
class GridWorldCoffee : public POMDP
{
public:
    // parameters
    constexpr static double const goal_reward    = 1;
    constexpr static double const step_reward    = 0;
    constexpr static double const same_weather_prob = .7;
    constexpr static double const move_prob      = .95;
    constexpr static double const slow_move_prob = .15;

    /**
     * @brief A state in the grid world problem
     **/
    class GridWorldCoffeeState : public State
    {

    public:
        /**
     * @brief A position in the gridworld
     * x and y are the agent's location, v the "velocity", r the rain, c the carpet
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

        static unsigned int carpet_func(pos agent_pos) {
            // carpet states, rectangular area
            if (agent_pos.x < 3 && agent_pos.y > 0 && agent_pos.y < 4)
            {
                return 1; // Carpet
            }
            return 0; // No carpet
        }

        GridWorldCoffeeState(pos agent_pos, unsigned int rain, unsigned int carpet_config, int i) :
            _agent_position(agent_pos),
            _rain(rain),
            _carpet_config(carpet_config),
            _index(i)
        {
        }

        /***** state implementation *****/
        void index(int) final { throw "do not change gridworldcoffee states"; };
        int index() const final { return _index; }
        std::string toString() const final
        {
            return "agent" + _agent_position.toString() + " rain " + std::to_string(_rain) + " carpet configuration " + std::to_string(_carpet_config);
        }

        pos const _agent_position;
        unsigned int const _rain;
        unsigned int const _carpet_config;

    private:
        int const _index;
    };

    /**
     * @brief An observation in the grid world problem (agent x position)
     **/
    class GridWorldCoffeeObservation : public Observation
    {
    public:
        GridWorldCoffeeObservation(GridWorldCoffeeState::pos agent_pos, unsigned int rain, int carpet_config, int i) :
            _agent_pos(agent_pos),
            _rain(rain),
            _carpet_config(carpet_config), // initiated below
            _index(i)
        {
        }

        /**** observation interface ***/
        void index(int /*i*/) final { throw "GridWorldCoffeeObservation::index(i) not allowed"; }
        int index() const final { return _index; };
        std::string toString() const final { return _agent_pos.toString(); }

        GridWorldCoffeeState::pos const _agent_pos;
        unsigned int const _rain;
        unsigned int const _carpet_config;

    private:
        int const _index;
    };

    /**
     * @brief An action in the grid world problem
     **/
    class GridWorldCoffeeAction : public Action
    {
    public:
        explicit GridWorldCoffeeAction(int i) : _index(i) {}

        enum ACTION { UP, RIGHT, DOWN, LEFT };
        static std::vector<std::string> const action_descriptions; // initialized in cpp

        /*** action interface ***/
        void index(int /*i*/) final { throw "GridWorldCoffeeAction should not edit index"; }
        int index() const final { return _index; }
        std::string toString() const final { return action_descriptions[_index]; }

    private:
        int const _index;
    };

    explicit GridWorldCoffee();

    static GridWorldCoffeeState::pos const start_location; // = {0,0};
    static GridWorldCoffeeState::pos const goal_location;

    /***** getters of parameters and settings of the domain ****/
    size_t size() const;
    static double goalReward() ;
    State const* sampleRandomState() const;
    bool agentOnSlowLocation(GridWorldCoffeeState::pos const& agent_pos) const;

    bool foundGoal(GridWorldCoffeeState const* s) const;

    GridWorldCoffeeState const*
    getState(GridWorldCoffeeState::pos const& agent_pos, unsigned int const& rain, unsigned int const& carpet_config) const;
    GridWorldCoffeeObservation const* getObservation(
            GridWorldCoffeeState::pos const& agent_pos,
        unsigned int const& rain, unsigned int const& carpet_config) const;

    /**
     * @brief applies a move a on the old_pos to get a new pos
     **/
    GridWorldCoffeeState::pos applyMove(GridWorldCoffeeState::pos const& old_pos, Action const* a) const;

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
    size_t const _size = 5;
    size_t const _carpet_configurations = 2;

    // initiated in constructor
//    std::vector<GridWorldCoffeeState::pos> _goal_locations;

    int _A_size = 4;
    int _S_size;
    int _O_size;

    std::vector<GridWorldCoffeeState> _S       = {};
    std::vector<GridWorldCoffeeObservation> _O = {};

    /**
     * @brief returns an observation from position agent_pos and rain
     **/
    Observation const* generateObservation(
        GridWorldCoffeeState::pos const& agent_pos,
        unsigned int const& rain,
        unsigned int const& carpet_config) const;

    int positionsToIndex(GridWorldCoffeeState::pos const& agent_pos, unsigned int const& rain, unsigned int const& carpet_config)
    const;
    int positionsToObservationIndex(GridWorldCoffeeState::pos const& agent_pos, unsigned int const& rain, unsigned int const& carpet_config)
    const;


    /** some functions to check input from system **/
    void assertLegal(Action const* a) const;
    void assertLegal(Observation const* o) const;
    void assertLegal(State const* s) const;
    void assertLegal(GridWorldCoffeeState::pos const& position) const;
//    void assertLegal(int const& x_position) const;
//    void assertLegalGoal(pos const& position) const;
};

} // namespace domains


#endif // GRIDWORLDCOFFEE_HPP





