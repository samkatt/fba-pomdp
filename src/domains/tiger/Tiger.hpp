#ifndef TIGER_HPP
#define TIGER_HPP

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

namespace domains {

/**
 * @brief The abstract tiger class, implemented by continuous & episodic subclass
 **/
class Tiger : public POMDP
{
public:
    enum TigerType { EPISODIC, CONTINUOUS };
    enum Literal { LEFT, RIGHT, OBSERVE };

    explicit Tiger(TigerType type);

    /*** environment interface ***/
    State const* sampleStartState() const final;
    Terminal step(State const** s, Action const* a, Observation const** o, Reward* r) const final;

    void releaseAction(Action const* a) const final;
    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;
    void clearCache() const final;

    Action const* copyAction(Action const* a) const final;
    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;

    /*** domain interface ***/
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;
    Action const* generateRandomAction(State const* s) const final;

    double computeObservationProbability(Observation const* o, Action const* a, State const* s)
        const final;

private:
    TigerType const _type;

    IndexState const _tiger_left{std::to_string(Literal::LEFT)}, _tiger_right{std::to_string(Literal::RIGHT)};
    IndexAction const _open_left{std::to_string(Literal::LEFT)}, _open_right{std::to_string(Literal::RIGHT)},
        _listen{std::to_string(Literal::OBSERVE)};
    IndexObservation const _hear_left{std::to_string(Literal::LEFT)}, _hear_right{std::to_string(Literal::RIGHT)};

    utils::DiscreteSpace<IndexAction> _actions{3};
    utils::DiscreteSpace<IndexState> _states{2};
    utils::DiscreteSpace<IndexObservation> _observations{2};

    mutable std::uniform_int_distribution<int> _action_distr{0, 2}; // cannot generate when const...

    /*** assertion functions **/
    void legalActionCheck(Action const* a) const;
    void legalObservationCheck(Observation const* o) const;
    void legalStateCheck(State const* s) const;
};

} // namespace domains

#endif // TIGER_HPP
