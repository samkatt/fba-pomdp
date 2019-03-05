#ifndef COFFEEPROBLEM_HPP
#define COFFEEPROBLEM_HPP

#include "domains/POMDP.hpp"

#include <string>
#include <vector>

#include "domains/coffee/CoffeeProblemAction.hpp"
#include "domains/coffee/CoffeeProblemObservation.hpp"
#include "domains/coffee/CoffeeProblemState.hpp"
#include "environment/Terminal.hpp"
#include "utils/DiscreteSpace.hpp"
#include "utils/random.hpp"
class Action;
class Observation;
class Reward;
class State;

namespace domains {

/**
 * @brief The coffee benchmark problem
 *
 * There are two versions, one used by poupart in his master thesis (default)
 * and the other is used by Boutilier and Poole in their 1996 paper (boutilier)
 *
 * Features a robot that must purchase coffee at a coffee shop and deliver it
 * to a user. The state space is defined by 5 Boolean variables indicating
 * whether the user has coffee HC, the user wants coffee WC,
 * it is raining R, the robot is wet W, the robot carries an umbrella
 * U.  The robot has two actions: get coffee, getC and check whether the user
 * wants coffee checkWC. All variables remain unchanged except those for
 * which a conditional probability table is given. When the rob ot gets coffee,
 * it must go across the street to the local coffee shop to buy coffee.
 * The robot may get wet if there is rain and it doesn’t carry any umbrella.
 * The robot doesn’t make any observation when it gets coffee,
 * however it observes (with some noise) whether the user wants coffee when
 * executing the checkWC action. The robot earns rewards when the user wants
 * coffee and has coffee, and it is penalized otherwise. The robot is further
 * penalized if it gets wet. A small cost of 1.0 is incurred each time it gets
 * coffee, and 0.5 e ach time it checks whether the user wants coffee.
 **/
class CoffeeProblem : public POMDP
{
public:
    explicit CoffeeProblem(std::string const& version);

    /*** POMDP interface ***/
    Action const* generateRandomAction(State const* s) const final;
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;

    double computeObservationProbability(Observation const* o, Action const* a, State const* s)
        const final;

    /*** environment interface ***/
    State const* sampleStartState() const final;
    Terminal step(State const** s, Action const* a, Observation const** o, Reward* r) const final;

    void releaseAction(Action const* a) const final;
    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;

    Action const* copyAction(Action const* a) const final;
    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;

private:
    short unsigned int _S = 32, _O = 2, _A = 2;

    double _fetch_coffee_success_rate = .9, _fetch_coffee_lose_desire_chance = .9,
           _coffee_is_drinked = .3, _acquire_coffee_desire = .3, _correctly_observes_desire = .8,
           _correctly_observes_lack_of_desire = .9;

    utils::DiscreteSpace<CoffeeProblemState> _states;
    utils::DiscreteSpace<CoffeeProblemAction> _actions;
    utils::DiscreteSpace<CoffeeProblemObservation> _observations;

    mutable std::uniform_int_distribution<int> _state_distr =
        rnd::integerDistribution(0, _S); // cannot generate when const...

    /***  performs some assertions  to check validity ***/
    void assertLegal(State const* s) const;
    void assertLegal(Action const* a) const;
    void assertLegal(Observation const* o) const;
};

} // namespace domains

#endif // COFFEEPROBLEM_HPP
