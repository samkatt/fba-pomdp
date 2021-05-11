#ifndef ABSTRACTFBAPOMDPSTATE_HPP
#define ABSTRACTFBAPOMDPSTATE_HPP

#include "bayes-adaptive/states/BAState.hpp"

#include "bayes-adaptive/states/factored/BABNModel.hpp"

#include "utils/random.hpp"
#include "FBAPOMDPState.hpp"

#include <utility>

class Action;
class Observation;
class State;

/**
 * @brief The factored bayes-adaptive state
 **/
class AbstractFBAPOMDPState : public FBAPOMDPState
{

public:
    /**
     * @brief initiates AbstractFBAPOMDPState with 0 CPTs
     **/
    AbstractFBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model);

//    bayes_adaptive::factored::BABNModel* model() { return &_model; };
//    bayes_adaptive::factored::BABNModel const* model() const { return &_model; };

    /*** BAState interface ***/
    BAState* copy(State const* domain_state) const final;
    int sampleStateIndex(State const* s, Action const* a, rnd::sample::Dir::sampleMethod m)
        const final;
    int sampleObservationIndex(
        Action const* a,
        State const* new_s,
        rnd::sample::Dir::sampleMethod m) const final;
    double computeObservationProbability(
        Observation const* o,
        Action const* a,
        State const* s,
        rnd::sample::Dir::sampleMultinominal sampleMultinominal) const final;

    void incrementCountsOf(
        State const* s,
        Action const* a,
        Observation const* o,
        State const* new_s,
        float amount = 1) final;

    /**
     * @brief logs the whole model to a file
     **/
    void logCounts() const final;

    std::vector<unsigned int>* getAbstraction();
    void setAbstraction(std::vector<unsigned int>);

    /*** State interface ***/
    std::string toString() const override;

private:
//    bayes_adaptive::factored::BABNModel _model;
    std::vector<unsigned int> _abstraction;
};

#endif // ABSTRACTFBAPOMDPSTATE_HPP
