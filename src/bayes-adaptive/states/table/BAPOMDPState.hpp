#ifndef BAPOMDPSTATE_HPP
#define BAPOMDPSTATE_HPP

#include "bayes-adaptive/states/BAState.hpp"

#include "bayes-adaptive/states/table/BAFlatModel.hpp"

#include "utils/random.hpp"

class Action;
class State;
class Observation;

/**
 * @brief A state in the Bayes-Adapative POMDP
 **/
class BAPOMDPState : public BAState
{

public:
    /**
     * @brief initiates a BAPOMDP state with T and O counts = phi & psi
     **/
    BAPOMDPState(State const* s, bayes_adaptive::table::BAFlatModel chi);

    /*** BAState interface ***/
    BAState* copy(State const*) const final;
    void logCounts() const final;

    int sampleStateIndex(State const* s, Action const* a, rnd::sample::Dir::sampleMethod m)
        const final;

    int sampleObservationIndex(
        Action const* a,
        State const* new_s,
        rnd::sample::Dir::sampleMethod m) const final;

    double computeObservationProbability(
        Observation const* o,
        Action const* a,
        State const* new_s,
        rnd::sample::Dir::sampleMultinominal m) const final;

    void incrementCountsOf(
        State const* s,
        Action const* a,
        Observation const* o,
        State const* new_s,
        float amount = 1) final;

    bayes_adaptive::table::BAFlatModel* model() { return &_model; }
    bayes_adaptive::table::BAFlatModel const* model() const { return &_model; }

    /*** State interface ***/
    std::string toString() const override;

private:
    bayes_adaptive::table::BAFlatModel _model;
};

#endif // BAPOMDPSTATE_HPP
