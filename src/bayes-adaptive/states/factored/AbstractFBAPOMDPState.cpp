#include "easylogging++.h"

#include <cstddef>
#include <utility>

#include "AbstractFBAPOMDPState.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"


#include "utils/index.hpp"
AbstractFBAPOMDPState::AbstractFBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model) :
        FBAPOMDPState(domain_state, std::move(model)),
        _abstraction(-1), // Empty initialization.
        _abstract_model() // Initialized later, when abstraction is added
//        _abstract_domain_size(25,4,25),
//        _abstract_domain_feature_size({5,5}, {5,5}),
//        _step_size({5,1},{5,1})
{
    assert(model.domainFeatureSize());
}

AbstractFBAPOMDPState::AbstractFBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model,
                                             bayes_adaptive::factored::BABNModel abstract_model) :
        FBAPOMDPState(domain_state, std::move(model)),
        _abstraction(0), // Empty initialization.
        _abstract_model(std::move(abstract_model)) // Initialized later, when abstraction is added
//        _abstract_domain_size(25,4,25),
//        _abstract_domain_feature_size({5,5}, {5,5}),
//        _step_size({5,1},{5,1})
{
    assert(model.domainFeatureSize());
}

BAState* AbstractFBAPOMDPState::copy(State const* domain_state) const
{
//    auto toReturn = new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real());
//    toReturn->_abstraction = _abstraction;
    if (_abstraction == 0) {
        return new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real(), _abstract_model);
//        toReturn->_abstract_model = bayes_adaptive::factored::BABNModel( _abstract_model.domainSize(),
//                _abstract_model.domainFeatureSize(), _abstract_model.stepSizes(), _abstract_model._transition_nodes, _abstract_model._observation_nodes);
                // &toReturn->_abstract_domain_size, &toReturn->_abstract_domain_feature_size, &toReturn->_step_size, _abstract_model._transition_nodes, _abstract_model._observation_nodes);
    } else {
        return new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real());
    }
//    return toReturn; // new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real());
}

// this samples a new state
int AbstractFBAPOMDPState::sampleStateIndex(
    State const* s,
    Action const* a,
    rnd::sample::Dir::sampleMethod m) const
{
    return model()->sampleStateIndex(s,a,m);
}

// this samples a new state, using the abstract model
int AbstractFBAPOMDPState::sampleStateIndexAbstract(
        State const* s,
        Action const* a,
        rnd::sample::Dir::sampleMethod m) const
{
    if (_abstraction == 0) {
        auto parent_values = model()->stateFeatureValues(s);

        auto next_values = std::vector<int>(parent_values.size());
        for (auto n = 0; n < 2; ++n)
        { next_values[n] = _abstract_model.transitionNode(a, n).sample(parent_values, m); }
        for (auto n = 2; n < (int)_abstract_model.domainFeatureSize()->_S.size(); ++n)
        { next_values[_abstract_model.transitionNode(a,n).parents()->at(0)] = _abstract_model.transitionNode(a, n).sample(parent_values, m); }

//        std::vector<int> next_values = {0, 0};
//        std::vector<int> actual_parent_values = {parent_values[0], parent_values[1]};
//        next_values[0] = _abstract_model.transitionNode(a, 0).sample(parent_values, m);
//        next_values[1] = _abstract_model.transitionNode(a, 1).sample(parent_values, m);
//        auto next_values = std::vector<int>(_abstract_model.domainFeatureSize()->_S.size());
        //fill the vector by sampling next stage feature 1 by 1
//        for (auto n = 0; n < (int)_abstract_model.domainFeatureSize()->_S.size(); ++n)
//        { next_values[n] = _abstract_model.transitionNode(a, n).sample(parent_values, m); }
        // TODO this needs to be changed?
        return model()->sampleStateIndexThroughAbstraction(s,a, next_values);


    }
    return model()->sampleStateIndex(s,a,m);
//    return _abstract_model.sampleStateIndex(s,a,m);
}

int AbstractFBAPOMDPState::sampleObservationIndex(
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMethod m) const
{
    return model()->sampleObservationIndex(a, new_s, m);
}

// only called at initialization?
double AbstractFBAPOMDPState::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* s,
    rnd::sample::Dir::sampleMultinominal sampleMultinominal) const
{
    return FBAPOMDPState::model()->computeObservationProbability(o, a, s, sampleMultinominal);
}

void AbstractFBAPOMDPState::incrementCountsOfAbstract(
    State const* s,
    Action const* a,
    Observation const* o,
    State const* new_s,
    float amount)
{
    if(_abstraction == 0) {
        auto parent_values = model()->stateFeatureValues(s);

        auto state_feature_values = model()->stateFeatureValues(new_s);
        // need to do something like before, to find the new values
        std::vector<int> actual_parent_values; // = std::vector<int>(_abstract_model.domainFeatureSize()->_S.size());
//        auto next_values = std::vector<int>(_abstract_model.domainFeatureSize()->_S.size());

        // for each node
        // get the parent values
        // get the next value
        // increment counts
        for (auto n = 0; n < 2; ++n) {
            actual_parent_values = std::vector<int>(_abstract_model.transitionNode(a,n).parents()->size());
            for (auto i = 0; i < (int) _abstract_model.transitionNode(a,n).parents()->size(); i++) {
                actual_parent_values[i] = parent_values[_abstract_model.transitionNode(a,n).parents()->at(i)];
            }
            _abstract_model.transitionNode(a, n).increment(actual_parent_values, state_feature_values[n], amount);
        }
        for (auto n = 2; n < (int)_abstract_model.domainFeatureSize()->_S.size(); ++n) {
            auto input_node = {parent_values[_abstract_model.transitionNode(a,n).parents()->at(0)]};
            auto next_togo = state_feature_values[_abstract_model.transitionNode(a,n).parents()->at(0)];
            _abstract_model.transitionNode(a, n).increment(
                    input_node,
                    next_togo, amount);
        }

//        for (auto n = 0; n < 2; ++n) {
//            next_values[n] = state_feature_values[n];
//            actual_parent_values[n] = parent_values[n];
//        }
//        for (auto n = 0; n < (int)_abstract_model.domainFeatureSize()->_S.size(); ++n) {
//            next_values[n] = state_feature_values[_abstract_model.transitionNode(a,n).parents()->at(0)];
//            actual_parent_values[n] = parent_values[_abstract_model.transitionNode(a,n).parents()->at(0)];
//        }


//        std::vector<int> next_values = {state_feature_values[0], state_feature_values[1]};
//        std::vector<int> actual_parent_values = {parent_values[0], parent_values[1]};
        // update transition DBN
//        for (auto n = 0; n < 2; ++n)
//        { _abstract_model.transitionNode(a, n).increment(actual_parent_values, next_values[n], amount); }
//        for (auto n = 2; n < (int) _abstract_model.domainFeatureSize()->_S.size(); ++n)
//        { _abstract_model.transitionNode(a, n).increment(actual_parent_values, next_values[n], amount); }

        auto observation_feature_values = model()->observationFeatureValues(o);
        // update observation DBN
        for (auto n = 0; n < (int)_abstract_model.domainFeatureSize()->_O.size(); ++n)
        { _abstract_model.observationNode(a, n).increment(parent_values, observation_feature_values[n], amount); }
    }
    FBAPOMDPState::model()->incrementCountsOf(s, a, o, new_s, amount);
}

void AbstractFBAPOMDPState::incrementCountsOf(
        State const* s,
        Action const* a,
        Observation const* o,
        State const* new_s,
        float amount)
{
    FBAPOMDPState::model()->incrementCountsOf(s, a, o, new_s, amount);
}

int* AbstractFBAPOMDPState::getAbstraction(){
    return &_abstraction;
}

//void AbstractFBAPOMDPState::setAbstraction(int k){
//    _abstraction = k;
//    _abstract_model = construct_abstract_model(FBAPOMDPState::model_real(), false);
//}

void AbstractFBAPOMDPState::setAbstraction(bayes_adaptive::factored::BABNModel abstr_model){
    _abstraction = 0;
    _abstract_model = std::move(abstr_model);
}


//void AbstractFBAPOMDPState::setAbstractionNormalized(int k){
//    _abstraction = k;
//    _abstract_model = construct_abstract_model(FBAPOMDPState::model_real(), true);
//}

// Construct abstract model from the model given the features to keep in the abstraction
//bayes_adaptive::factored::BABNModel AbstractFBAPOMDPState::construct_abstract_model(bayes_adaptive::factored::BABNModel model, bool normalize) const {
////    if (_abstraction == 0) {
//    return model.abstract(_abstraction, model.structure(), _abstract_model.domainSize(), _abstract_model.domainFeatureSize(),
//                          _abstract_model.stepSizes(), normalize);
////    }
//}

void AbstractFBAPOMDPState::logCounts() const {
    FBAPOMDPState::model()->log();
}

std::string AbstractFBAPOMDPState::toString() const
{
    return "AbstractFBAPOMDPState with s=" + _domain_state->toString();
}
