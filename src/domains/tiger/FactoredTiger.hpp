#ifndef FACTOREDTIGER_HPP
#define FACTOREDTIGER_HPP

#include "domains/POMDP.hpp"

#include <memory>
#include <string>
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
 * @brief The tiger problem where the state contains additional (useless) features
 *
 * The extra information bit in the tiger state is non-informative,
 * but allows the problem to be factored into two state features.
 *
 * This doubles the number of states into 4, making it one of the smallest
 * possible factored POMDPs.
 **/
class FactoredTiger : public POMDP
{
public:
    enum FactoredTigerDomainType { EPISODIC, CONTINUOUS };
    enum TigerAction { OPEN_LEFT, OPEN_RIGHT, OBSERVE };
    enum TigerLocation { LEFT, RIGHT };

    FactoredTiger(FactoredTigerDomainType type, size_t num_irrelevant_features);

    /**
     * @brief returns location of tiger in s
     **/
    TigerLocation tigerLocation(State const* s) const;

    /**** POMDP interface ****/
    Action const* generateRandomAction(State const* s) const final;
    void addLegalActions(State const* s, std::vector<Action const*>* actions) const final;
    void releaseAction(Action const* a) const final;
    Action const* copyAction(Action const* a) const final;
    double computeObservationProbability(Observation const* o, Action const* a, State const* new_s)
        const final;

    /**** Environment interface ****/
    State const* sampleStartState() const final;
    Terminal step(State const** s, Action const* a, Observation const** o, Reward* r) const final;

    void releaseObservation(Observation const* o) const final;
    void releaseState(State const* s) const final;

    Observation const* copyObservation(Observation const* o) const final;
    State const* copyState(State const* s) const final;
    void clearCache() const final;

private:
    FactoredTigerDomainType const _type;
    int const _S_size, _A_size = 3, _O_size = 2;

    utils::DiscreteSpace<IndexState> _states{_S_size};
    utils::DiscreteSpace<IndexAction> _actions{_A_size};

    IndexObservation const _hear_left{std::to_string(TigerLocation::LEFT)}, _hear_right{std::to_string(TigerLocation::RIGHT)};
    std::vector<IndexObservation> _observations{_hear_left, _hear_right};

    // used to sample states uniformly
    mutable std::uniform_int_distribution<int> _state_distr{0, _S_size - 1};
    mutable std::uniform_int_distribution<int> _action_distr{0, _A_size - 1};

    /**** checks for legal input *****/
    void assertLegal(State const* s) const;
    void assertLegal(Observation const* o) const;
    void assertLegal(Action const* a) const;
};

} // namespace domains

#endif // FACTOREDTIGER_HPP
