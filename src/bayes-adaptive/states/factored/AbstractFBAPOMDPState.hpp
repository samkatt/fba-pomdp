#ifndef ABSTRACTFBAPOMDPSTATE_HPP
#define ABSTRACTFBAPOMDPSTATE_HPP

#include "bayes-adaptive/states/BAState.hpp"

#include "bayes-adaptive/states/factored/BABNModel.hpp"

#include "utils/random.hpp"
#include "FBAPOMDPState.hpp"

#include <utility>
#include <bayes-adaptive/models/Domain_Size.hpp>
#include <bayes-adaptive/models/factored/Domain_Feature_Size.hpp>

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
    AbstractFBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model,
                          bayes_adaptive::factored::BABNModel abstract_model, std::vector<int> used_features);


    /*** BAState interface ***/
    BAState* copy(State const* domain_state) const final;
    int sampleStateIndex(State const* s, Action const* a, rnd::sample::Dir::sampleMethod m)
        const final;
    int sampleStateIndexAbstract(const State *s, const Action *a, rnd::sample::Dir::sampleMethod m) const;
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

    void incrementCountsOfAbstract(
            State const* s,
            Action const* a,
            Observation const* o,
            State const* new_s,
            float amount = 1);

    /**
     * @brief logs the whole model to a file
     **/
    void logCounts() const final;

    int* getAbstraction();
//    void setAbstraction(int);
    void setAbstraction(bayes_adaptive::factored::BABNModel abstr_model);
//    void setAbstractionNormalized(int);

    /*** State interface ***/
    std::string toString() const override;

    std::vector<int> feature_set = {};

private:
//    bayes_adaptive::factored::BABNModel _model;
    int _abstraction; // Vector with the features that are included in the abstract model
    bayes_adaptive::factored::BABNModel _abstract_model;  // Abstract model that only uses the feature in _abstraction
//    bayes_adaptive::factored::BABNModel construct_abstract_model(bayes_adaptive::factored::BABNModel model, bool normalize) const;

//    Domain_Size _abstract_domain_size;
//    Domain_Feature_Size _abstract_domain_feature_size;
//    bayes_adaptive::factored::BABNModel::Indexing_Steps _step_size;
};

#endif // ABSTRACTFBAPOMDPSTATE_HPP
