#ifndef CollisionAvoidanceBig_HPP
#define CollisionAvoidanceBig_HPP

#include "domains/POMDP.hpp"

#include <memory>
#include <random>
#include <vector>

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
#include "utils/DiscreteSpace.hpp"
#include "utils/distributions.hpp"
#include "utils/random.hpp"

namespace domains {


/**
 * @brief The collision avoidance domain
 *
 * Consist of the <x,y> position of the agent and y position of the obstacles
 **/
struct CollisionAvoidanceBigState : public State
{
    CollisionAvoidanceBigState( std::vector<int> state_vector,
        std::string index) :
            _state_vector(std::move(state_vector)),
            _index(index)
    {
    }

    void index(std::string /*i*/) final
    {
        throw "CollisionAvoidanceBigState::index(i) should not be called...?";
    }

    std::string index() const final { return _index; }

    std::string toString() const final
    {
        std::string obsts = "{";

        for (size_t i = 6; i < (_state_vector.size() - 1); ++i)
        { obsts += std::to_string(_state_vector[i]) + ","; }
        obsts += std::to_string(_state_vector[_state_vector.size()-1]);

        return "(index:" + _index + " (" + std::to_string(_state_vector[0]) + ","
               + std::to_string(_state_vector[1]) + "), " + std::to_string(_state_vector[2]) + ", " + std::to_string(_state_vector[3]) + ", "
              + std::to_string(_state_vector[4]) + ", " + std::to_string(_state_vector[5]) + ", "+ obsts
               + "})";
    }

    std::vector<int> getFeatureValues() const final;
    std::vector<int> const _state_vector;

    std::string _index;

//    std::vector<int> obstacles_pos;
};

/** \brief The collision avoidance domain
  >>>>>>> domain_ba_separation
 *
 * The agent starts moving from a random position in the right-most column
 * of a x by y 30 grid map. An obstacle randomly moves in the left-most
 * column of this map: moves up with probability 0.25, down with 0.25, and
 * stay put with 0.50. The probabilities become 0, 0.25, and 0.75
 * respectively when the obstacle is at the topmost row, and become 0.25,
 * 0, and 0.75 respectively when it is at the bottom-most row.
 * TODO temporary decicion, we are changing the problem as follows:
 * The speed influences the probabilities that the obstacle moves.
 * If we come closer at a higher speed, the obstacle moves less (because it has less time to move essentially)
 * And it is more likely to move when we move slower.
 * Let's say: medium speed: 0.25 (4.5/18), 0.25 (4.5/18), 0.5 (9/18) (up, down, stay)
 * high speed: 3/18, 3/18, 12/18 (0.1666,0.16666,0.6666)
 * Low speed: 6/18, 6/18, 6/18 (0.333,0.333,0.333)
 *
 * Each time,
 * the agent can choose to move upper-left, lower-left or left, with a cost
 * of -1,-1 and 0 respectively. If the agent collides with the obstacle, it
 * receives a penalty of -1000. The task finishes when the agent reaches
 * the left-most column. The agent knows its own position exactly, but
 * observes the obstacle’s position with a Gaussian noise N(0; 1).
 *
 **/
class CollisionAvoidanceBig : public POMDP
{

public:
    enum VERSION { INIT_RANDOM_POSITION, INITIALIZE_CENTRE };
    enum move { MOVE_DOWN, STAY, MOVE_UP };

    constexpr static int const NUM_ACTIONS        = 3;
    constexpr static double const MOVE_PENALTY    = 1;
    constexpr static double const COLLIDE_PENALTY = 1000;
    constexpr static double const BLOCK_MOVE_PROB = .5;
    constexpr static double const MOVE_PROB_FAST = 0.85;
    constexpr static double const MOVE_PROB_SLOW = 0.85;
//    constexpr static double const CORRECT_TYPE = 0.7;

      /**
      * @brief An observation in the collision avoidance problem
      **/
//    class CollisionAvoidanceBigObervation : public Observation
//    {
//    public:
//        CollisionAvoidanceBigObervation(
//
//                int i) :
//                _observation_vector({static_cast<int>(agent_pos.x), static_cast<int>(agent_pos.y)}),
//                _index(i)
//        {
//        }
//
//        /**** observation interface ***/
//        void index(std::string /*i*/) final { throw "GridWorldCoffeeBigObservation::index(i) not allowed"; }
//        std::string index() const final { return std::to_string(_index); };
//        std::string toString() const final { return
//                    GridWorldCoffeeBigState::pos({static_cast<unsigned int>(_observation_vector[0]),
//                                                  static_cast<unsigned int>(_observation_vector[1])}).toString(); }
//
//        std::vector<int> getFeatureValues() const final;
//        std::vector<int> const _observation_vector;
//
////        ::domains::GridWorldCoffeeBig::GridWorldCoffeeBigState::pos const _agent_pos;
//
//    private:
//        int const _index;
//    };

    CollisionAvoidanceBig(
        int grid_width,
        int grid_height,
        int num_obstacles = 1,
        VERSION version   = INIT_RANDOM_POSITION);
    ~CollisionAvoidanceBig() final;

    /***** getters *****/
    /**
     * @brief returns the type of the domain
     *
     * @return VERSION::INIT_RANDOM_POSITION or VERSION::INITIALIZE_CENTRE
     */
    VERSION type() const;
    int xAgent(State const* s) const;
    int yAgent(State const* s) const;
//    std::vector<int> const& yObstacles(State const* s) const;
    int speedRelative(State const* s) const;
    int trafficStatus(State const* s) const;
    int timeofdayStatus(State const* s) const;

    int x_agent_f = 0;
    int y_agent_f = 1;
    int speed_f = 2; // speed takes 0,1,2
    int traffic_f = 3; // traffic takes 0,1,2
    int timeofday_f = 4; // timeofday takes 0, 1
    int obstacle_type_f = 5;
    int obstacle_start = 6;

    State const* getState(int x, int y, int speed_in, int traffic_in,
                          int timeofday_in, int obstacle_type_in, std::vector<int> const& obstacles) const;

    Action const* getAction(move m) const;
    Observation const* getObservation(int y) const;

    /***** domain interface ****/
    Action const* generateRandomAction(State const* s) const final;
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;
    double computeObservationProbability(Observation const* o, Action const* a, State const* new_s)
        const final;

    Action const* copyAction(Action const* a) const final;
    void releaseAction(Action const* a) const final;

    /***** Environment interface *****/
    State const* sampleStartState() const final;
    Terminal step(State const** s, Action const* a, Observation const** o, Reward* r) const final;

    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;
    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;
    void clearCache() const final;

/**
 * @brief returns the state associated with index
 *
 * @param index the index of the state
 *
 * @return the CollisionAvoidanceBigState of index
 */
State const* getState(int index) const;

    private:
    int const _grid_width;
    int const _grid_height;
    int const _num_speeds = 2;
    int const _num_traffics = 3;
    int const _num_timeofdays = 2;
    int const _num_obstacletypes = 3;
    int const _num_obstacles;
    std::vector<double> const _obstacletype_probs = {0.1, 0.3, 0.6, 0.3, 0.3, 0.4}; // first 3 entries for time of day 0, last 3 time of day 1
    VERSION const _version;

    std::vector<int> const _obstacles_space = std::vector<int>(_num_obstacles, _grid_height);

    utils::DiscreteSpace<IndexAction> _actions{NUM_ACTIONS};
    std::vector<Observation*> _observations{
        static_cast<size_t>(static_cast<int>(_grid_width * _num_timeofdays * _num_obstacletypes * std::pow(_grid_height, _num_obstacles)))};

    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<State const*>>>>>>> _states{
            static_cast<size_t>(_grid_width),
            std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<State const*>>>>>>(
                    _grid_height,
            std::vector<std::vector<std::vector<std::vector<std::vector<State const*>>>>>(
                    _num_speeds, //speed
                    std::vector<std::vector<std::vector<std::vector<State const*>>>>(
                            _num_traffics, //traffic
                                std::vector<std::vector<std::vector<State const*>>>(
                                        _num_timeofdays, //timeofday
                                std::vector<std::vector<State const*>>(
                                        _num_obstacletypes, //obstacletype
                                        std::vector<State const*>(std::pow(_grid_height, _num_obstacles)))))))};

    // probability of observation distance
    // element i contains the probability of observing
    // the obstacle i cells away from its actual position
    std::vector<double> _observation_error_probability{};

    // uniform distriution over the actions
    mutable std::uniform_int_distribution<int> _action_distr{
        rnd::integerDistribution(0, NUM_ACTIONS)};

    // describes the noise of the observation
    mutable std::normal_distribution<float> _observation_distr =
        std::normal_distribution<float>(0, 0.5);

    mutable std::uniform_int_distribution<int> _y_sampler{
        rnd::integerDistribution(0, _grid_height)};

    utils::categoricalDistr _state_prior{static_cast<size_t>(
        _grid_width * _grid_height * _num_speeds * _num_traffics * _num_timeofdays * _num_obstacletypes * static_cast<int>(std::pow(_grid_height, _num_obstacles)))};

        /**
         * @brief computes reward associated with <*,a,new_s>
         *
         * @param a
         * @param new_s
         *
         * @return the reward of ending up in new_s after taking action a
         */
    Reward reward(Action const* a, State const* new_s) const;

    /**
     * @brief given some y value, it will make sure to place it within the grid
     *
     * If y is higher than the grid height, then it will be set to
     * the top cell. Similarly any negative value will be set to 0
     **/
    int keepInGrid(int y) const;

    /**
     * @brief returns a new position of the obstacle
     *
     * obstacle moves up & down with 25% and stays put with 50%
     */
    int moveObstacle(int current_position, int speed, int obstacletype) const;
    int moveAgent(int current_position, int speed) const;

    void assertLegal(State const* s) const;
    void assertLegal(Action const* a) const;
    void assertLegal(Observation const* o) const;

    int changeSpeed(int speed, int traffic) const;

    int changeTraffic(int traffic, int timeofday) const;

    int keepInBounds(int value, int num_options) const;

    int getObservationIndex(int x, int timeofday, int obstacletype, int obstacle) const;
    };

} // namespace domains

#endif // CollisionAvoidanceBig_HPP
