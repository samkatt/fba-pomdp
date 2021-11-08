#ifndef GridWorldButtons_HPP
#define GridWorldButtons_HPP

#include "domains/POMDP.hpp"

#include <memory>
#include <string>
#include <vector>

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
#include "utils/index.hpp"
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
class GridWorldButtons : public POMDP
{
public:
    // parameters
    constexpr static double const goal_reward_big    = 1000;
    constexpr static double const goal_reward_small    = 11;
    constexpr static double const button_press_reward = 50;
    constexpr static double const step_reward    = 0;
    constexpr static double const move_prob      = .95;
    constexpr static double const door_max_hp = 1;
    constexpr static double const wrong_obs_prob = .1;
    constexpr static double const button_A_open_prob = 0.9;
    constexpr static double const button_B_open_prob = 0.9;
    constexpr static double const button_C_open_prob = 0.9;

    /**
     * @brief A state in the grid world problem
     **/
    class GridWorldButtonsState : public State
    {

    public:
        /**
     * @brief A position in the gridworld
     * x and y are the agent's location, correct_button the correct button, d1, door 1 open, d2, door 2 open.
     * d2hp, hit points of door 2
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

        GridWorldButtonsState(pos agent_pos, unsigned int correct_button, unsigned int d_open, unsigned int d_hp, unsigned int feature_config, int i) :
            _agent_position(agent_pos),
            _correct_button(correct_button),
            _d_open(d_open),
            _d_hp(d_hp),
            _feature_config(feature_config),
            _index(i)
        {
        }

        /***** state implementation *****/
        void index(std::string) final { throw "do not change GridWorldButtons states"; };
        std::string index() const final { return std::to_string(_index); }
        std::string toString() const final
        {
            return "agent" + _agent_position.toString() + " correct button " + std::to_string(_correct_button)
            + " door open " + std::to_string(_d_open)
            + " door hp " + std::to_string(_d_hp);
        }

        pos const _agent_position;
        unsigned int const _correct_button;
        unsigned int const _d_open;
        unsigned int const _d_hp;
        unsigned int const _feature_config; // the features we can abstract away for sure. Could also abstract away _d2_hp and _d2_open

    private:
        int const _index;
    };

    /**
     * @brief An observation in the grid world problem (agent x position)
     **/
    class GridWorldButtonsObservation : public Observation
    {
    public:
        GridWorldButtonsObservation(
            ::domains::GridWorldButtons::GridWorldButtonsState::pos agent_pos, unsigned int d_open, int i) :
            _location_observation(),
            _d_open(d_open),
            _index(i)
        {
            _location_observation = locationToObservationLocation(agent_pos);
        }

        /**** observation interface ***/
        void index(std::string /*i*/) final { throw "GridWorldButtonsObservation::index(i) not allowed"; }
        std::string index() const final { return std::to_string(_index); };
        std::string toString() const final {
                return "observation" + std::to_string(_location_observation) + " door1open " + std::to_string(_d_open);
            }

        unsigned int _location_observation;
        unsigned int const _d_open;

    private:
        int const _index;
    };

    /**
     * @brief An action in the grid world problem
     **/
    class GridWorldButtonsAction : public Action
    {
    public:
        explicit GridWorldButtonsAction(int i) : _index(i) {}

        enum ACTION { UP, RIGHT, DOWN, LEFT, PRESS, HIT };
        static std::vector<std::string> const action_descriptions; // initialized in cpp

        /*** action interface ***/
        void index(std::string /*i*/) final { throw "GridWorldButtonsAction should not edit index"; }
        std::string index() const final { return std::to_string(_index); }
        std::string toString() const final { return action_descriptions[_index]; }

    private:
        int const _index;
    };

    explicit GridWorldButtons(size_t extra_features);

    static GridWorldButtonsState::pos const start_location; // = {0,0};
    static std::vector<GridWorldButtons::GridWorldButtonsState::pos> const goal_locations;
    static std::vector<GridWorldButtons::GridWorldButtonsState::pos> const wall_tiles;

    /***** getters of parameters and settings of the domain ****/
    size_t size_width() const;
    size_t size_height() const;

    bool foundGoal(GridWorldButtonsState const* s) const;

    GridWorldButtonsState const*
        getState(GridWorldButtonsState::pos const& agent_pos, unsigned int const& correct_button,
                 unsigned int const& d_open, unsigned int const& d_hp, unsigned int const& feature_config) const;
    GridWorldButtonsObservation const* getObservation(GridWorldButtonsState::pos const& agent_pos,
                                                      unsigned int const& d_open) const;
    static unsigned int locationToObservationLocation(GridWorldButtons::GridWorldButtonsState::pos agent_pos);

    /**
     * @brief applies a move a on the old_pos to get a new pos
     **/
    GridWorldButtonsState::pos applyMove(GridWorldButtonsState::pos const& old_pos, Action const* a,
                                         unsigned int const& d_open) const;

    /**** POMDP interface ****/
    Action const* generateRandomAction(State const* s) const final;
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;
    double computeObservationProbability(Observation const* o, Action const* a, State const* new_s) const final;
    void releaseAction(Action const* a) const final;
    Action const* copyAction(Action const* a) const final;

    /***** environment interface *****/
    State const* sampleStartState() const final;
    Terminal step(State const** s, Action const* a, Observation const** o, Reward* r) const final;
    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;
    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;
    void clearCache() const final;

private:
    // problem settings
    size_t const _size_width = 3;
    size_t const _size_height = 6;
    size_t const _extra_features;
    std::vector<int> _stepSizes = indexing::stepSize(std::vector<int>(_extra_features, 2));

    // initiated in constructor
    int _A_size = 6;
    int _S_size;
    int _O_size;

    std::vector<GridWorldButtonsState> _S       = {};
    std::vector<GridWorldButtonsObservation> _O = {};

    int positionsToIndex(GridWorldButtonsState::pos const& agent_pos, unsigned int const& correct_button,
                         unsigned int const& d_open, unsigned int const& d_hp, unsigned int const& feature_config)
        const;
    int positionsToObservationIndex(GridWorldButtonsState::pos const& agent_pos, unsigned int const& d_open) const;

    /** some functions to check input from system **/
    void assertLegal(Action const* a) const;
    void assertLegal(Observation const* o) const;
    void assertLegal(State const* s) const;
    void assertLegal(GridWorldButtonsState::pos const& position) const;
};

} // namespace domains


#endif // GridWorldButtons_HPP
