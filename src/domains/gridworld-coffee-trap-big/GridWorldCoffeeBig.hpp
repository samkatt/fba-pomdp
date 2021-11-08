#ifndef GridWorldCoffeeBig_HPP
#define GridWorldCoffeeBig_HPP

#include "domains/POMDP.hpp"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

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
class GridWorldCoffeeBig : public POMDP
{
public:
    // parameters
    constexpr static double const goal_reward    = 1;
    constexpr static double const step_reward    = 0;
    constexpr static double const same_weather_prob = .7;
    constexpr static double const move_prob      = .95;
    constexpr static double const rain_move_prob = .3; // the probability that the agent believes
    constexpr static double const move_prob_reduction = .9; // per feature that is "on" how much does the agent believe the movement probability is lowered
    constexpr static double const slow_move_prob = .1;
    constexpr static double const wrong_obs_prob = .1;

    /**
     * @brief A state in the grid world problem
     **/
    class GridWorldCoffeeBigState : public State
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

//        GridWorldCoffeeBigState(pos agent_pos, unsigned int rain, std::vector<int> feature_config, int i) :
//            _state_vector(),
//            _index(i)
//        {
//            _state_vector = std::vector<int>(3 + feature_config.size());
//            _state_vector[0] = agent_pos.x;
//            _state_vector[1] = agent_pos.y;
//            _state_vector[2] = rain;
//            for (auto k=0; k < (int) feature_config.size(); ++k) {
//                _state_vector[k+3] = feature_config[k];
//            }
//        }

        GridWorldCoffeeBigState(std::vector<int> state_vector, std::string i) :
                _state_vector(std::move(state_vector)),
                _index(std::move(i))
        {
        }

        /***** state implementation *****/
        void index(std::string) final { throw "do not change GridWorldCoffeeBig states"; };
        std::string index() const final { return _index; }
        std::string toString() const final
        {
            return "agent" + pos({static_cast<unsigned int>(_state_vector[0]),
                                  static_cast<unsigned int>(_state_vector[1])}).toString()
                                  + " rain " + std::to_string(_state_vector[2]) + " carpet configuration ";
        }

        std::vector<int> getFeatureValues() const final;
        std::vector<int> const _state_vector;
//        pos const _agent_position;
//        unsigned int const _rain;
//        unsigned int const _feature_config;

    private:
        std::string const _index;
    };

    /**
     * @brief An observation in the grid world problem (agent x position)
     **/
    class GridWorldCoffeeBigObservation : public Observation
    {
    public:
        GridWorldCoffeeBigObservation(
            ::domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos agent_pos,
            int i) :
            _observation_vector({static_cast<int>(agent_pos.x), static_cast<int>(agent_pos.y)}),
            _index(i)
        {
        }

        /**** observation interface ***/
        void index(std::string /*i*/) final { throw "GridWorldCoffeeBigObservation::index(i) not allowed"; }
        std::string index() const final { return std::to_string(_index); };
        std::string toString() const final { return
                                GridWorldCoffeeBigState::pos({static_cast<unsigned int>(_observation_vector[0]),
                                              static_cast<unsigned int>(_observation_vector[1])}).toString(); }

        std::vector<int> getFeatureValues() const final;
        std::vector<int> const _observation_vector;

//        ::domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos const _agent_pos;

    private:
        int const _index;
    };

    /**
     * @brief An action in the grid world problem
     **/
    class GridWorldCoffeeBigAction : public Action
    {
    public:
        explicit GridWorldCoffeeBigAction(int i) : _index(i) {}

        enum ACTION { UP, RIGHT, DOWN, LEFT };
        static std::vector<std::string> const action_descriptions; // initialized in cpp

        /*** action interface ***/
        void index(std::string /*i*/) final { throw "GridWorldCoffeeBigAction should not edit index"; }
        std::string index() const final { return std::to_string(_index); }
        std::string toString() const final { return action_descriptions[_index]; }
        std::vector<int> getFeatureValues() const final { return {_index}; };

    private:
        int const _index;
    };

    explicit GridWorldCoffeeBig(size_t extra_features, bool store_statespace);

    static std::vector<GridWorldCoffeeBigState::pos> const slow_locations;

    static GridWorldCoffeeBigState::pos const start_location; // = {0,0};
    static GridWorldCoffeeBigState::pos const goal_location;

    /***** getters of parameters and settings of the domain ****/
    size_t size() const;
    double goalReward() const;
    bool agentOnSlowLocation(GridWorldCoffeeBigState::pos const& agent_pos) const;
    bool agentOnCarpet(GridWorldCoffeeBigState::pos const& agent_pos, unsigned int const& carpet_config) const;
    static float believedTransitionProb(bool const& rain, int const& features_on);
    State const* sampleRandomState() const;

    /**
     * @brief returns probability of observing a location given the real one (1 dimensional)
     **/
    float obsDisplProb(unsigned int loc, unsigned int observed_loc) const;

    bool foundGoal(GridWorldCoffeeBigState const* s) const;

    GridWorldCoffeeBigState const*
        getState(std::vector<int> const& state_vector) const;
    GridWorldCoffeeBigState const*
    getState(std::string index) const;
    GridWorldCoffeeBigObservation const* getObservation(GridWorldCoffeeBigState::pos const& agent_pos) const;

    /**
     * @brief applies a move a on the old_pos to get a new pos
     **/
    GridWorldCoffeeBigState::pos applyMove(GridWorldCoffeeBigState::pos const& old_pos, Action const* a) const;

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
    size_t const _size = 5;
    size_t const _extra_features;
    std::vector<int> _stepSizes = indexing::stepSize(std::vector<int>(_extra_features, 2));

    int _x_feature = 0;
    int _y_feature = 1;
    int _rain_feature = 2;

    bool _store_statespace;
    std::vector<int> _extrafeatures_space;
    std::vector<int> _feature_space;

    // initiated in constructor
    int _A_size = 4;
    int _S_size;
    int _O_size;

    // describes the probability of displacement in our observation in 1 dimension
    std::vector<float> _obs_displacement_probs = {1-wrong_obs_prob,wrong_obs_prob,0,0,0};


    std::vector<GridWorldCoffeeBigState> _S       = {};
    std::vector<GridWorldCoffeeBigObservation> _O = {};
    mutable std::unordered_map<std::string, GridWorldCoffeeBigState> _S_cache = {};
    mutable std::unordered_map<std::string, GridWorldCoffeeBigObservation> _O_cache = {};

    /**
     * @brief returns an observation from position agent_pos and rain
     **/
    Observation const* generateObservation(GridWorldCoffeeBigState::pos const& agent_pos) const;

//    int positionsToIndex(GridWorldCoffeeBigState::pos const& agent_pos, unsigned int const& rain, std::vector<int> const& feature_config)
//        const;
    std::string positionsToIndex(std::vector<int> state_vector) const;
    int positionsToObservationIndex(GridWorldCoffeeBigState::pos const& agent_pos) const;


    /** some functions to check input from system **/
    void assertLegal(Action const* a) const;
    void assertLegal(Observation const* o) const;
    void assertLegal(State const* s) const;
    void assertLegal(GridWorldCoffeeBigState::pos const& position) const;

    std::vector<int> getStateVectorFromIndex(std::string index) const;
};

} // namespace domains


#endif // GridWorldCoffeeBig_HPP
