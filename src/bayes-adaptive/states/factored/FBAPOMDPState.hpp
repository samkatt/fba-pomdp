#ifndef FBAPOMDPSTATE_HPP
#define FBAPOMDPSTATE_HPP

#include "bayes-adaptive/states/BAState.hpp"

#include "bayes-adaptive/states/factored/BABNModel.hpp"

#include "utils/random.hpp"

#include <utility>

class Action;
class Observation;
class State;

/**
 * @brief The factored bayes-adaptive state
 **/
class FBAPOMDPState : public BAState
{

public:
    /**
     * @brief initiates FBAPOMDPState with 0 CPTs
     **/
    FBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model);

    bayes_adaptive::factored::BABNModel* model() { return &_model; };
    bayes_adaptive::factored::BABNModel model_real() const { return _model; };
    bayes_adaptive::factored::BABNModel const* model() const { return &_model; };

    /*** BAState interface ***/
    BAState* copy(State const* domain_state) const override;
    int sampleStateIndex(State const* s, Action const* a, rnd::sample::Dir::sampleMethod m)
        const override;
    int sampleObservationIndex(
        Action const* a,
        State const* new_s,
        rnd::sample::Dir::sampleMethod m) const override;
    double computeObservationProbability(
        Observation const* o,
        Action const* a,
        State const* s,
        rnd::sample::Dir::sampleMultinominal sampleMultinominal) const override;

    void incrementCountsOf(
        State const* s,
        Action const* a,
        Observation const* o,
        State const* new_s,
        float amount = 1) override;

    /**
     * @brief logs the whole model to a file
     **/
    void logCounts() const override;

    /*** State interface ***/
    std::string toString() const override;

//    std::vector<unsigned int>* getAbstraction();
//    void setAbstraction(std::vector<unsigned int>);
//    std::vector<unsigned int> abstraction = {};

private:
    bayes_adaptive::factored::BABNModel _model;
//    std::vector<unsigned int> _abstraction;
};

#endif // FBAPOMDPSTATE_HPP
