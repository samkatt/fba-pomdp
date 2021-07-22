#include "easylogging++.h"

#include <cstddef>
#include <utility>

#include "AbstractFBAPOMDPState.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"


#include "utils/index.hpp"
// TODO after every "real" timestep, will these be constructed again? I.e. can I assume _abstraction will be empty?
AbstractFBAPOMDPState::AbstractFBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model) :
        FBAPOMDPState(domain_state, std::move(model)),
        _abstraction(-1), // Empty initialization.
        _abstract_model(), // Initialized later, when abstraction is added
        _abstract_domain_size(25,4,25),
        _abstract_domain_feature_size({5,5}, {5,5}),
        _step_size({5,1},{5,1})
{
    assert(model.domainFeatureSize());
}

BAState* AbstractFBAPOMDPState::copy(State const* domain_state) const
{
    auto toReturn = new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real());
    toReturn->_abstraction = _abstraction;
    if (_abstraction == 0) {
        toReturn->_abstract_model = bayes_adaptive::factored::BABNModel(
                &toReturn->_abstract_domain_size, &toReturn->_abstract_domain_feature_size, &toReturn->_step_size, _abstract_model._transition_nodes, _abstract_model._observation_nodes);
    }
    return toReturn; // new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real());
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
        std::vector<int> next_values = {0, 0};
        std::vector<int> actual_parent_values = {parent_values[0], parent_values[1]};
        next_values[0] = _abstract_model.transitionNode(a, 0).sample(actual_parent_values, m);
        next_values[1] = _abstract_model.transitionNode(a, 1).sample(actual_parent_values, m);
        // TODO this needs to be changed?
        return model()->sampleStateIndexThroughAbstraction(s,a, next_values);
    }
    return _abstract_model.sampleStateIndex(s,a,m);
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

bool updateAbstractModel = true;

void AbstractFBAPOMDPState::incrementCountsOf(
    State const* s,
    Action const* a,
    Observation const* o,
    State const* new_s,
    float amount)
{
    if (updateAbstractModel && _abstraction == 0) {
        auto parent_values = model()->stateFeatureValues(s);

        auto state_feature_values = model()->stateFeatureValues(new_s);

        std::vector<int> next_values = {state_feature_values[0], state_feature_values[1]};
        std::vector<int> actual_parent_values = {parent_values[0], parent_values[1]};
        // update transition DBN
        for (auto n = 0; n < (int)_abstract_domain_feature_size._S.size(); ++n)
        { _abstract_model.transitionNode(a, n).increment(actual_parent_values, next_values[n], amount); }

        auto observation_feature_values = model()->observationFeatureValues(o);
        // update observation DBN
        for (auto n = 0; n < (int)_abstract_domain_feature_size._O.size(); ++n)
        { _abstract_model.observationNode(a, n).increment(parent_values, observation_feature_values[n], amount); }
    }
    FBAPOMDPState::model()->incrementCountsOf(s, a, o, new_s, amount);
}

int* AbstractFBAPOMDPState::getAbstraction(){
    return &_abstraction;
}

void AbstractFBAPOMDPState::setAbstraction(int k){
    _abstraction = k;
    _abstract_model = construct_abstract_model(FBAPOMDPState::model_real());
}

// Construct abstract model from the model given the features to keep in the abstraction
bayes_adaptive::factored::BABNModel AbstractFBAPOMDPState::construct_abstract_model(bayes_adaptive::factored::BABNModel model) const {
//    if (_abstraction == 0) {
    return model.abstract(_abstraction, model.structure(), &_abstract_domain_size, &_abstract_domain_feature_size, &_step_size);
//    }
}

void AbstractFBAPOMDPState::logCounts() const {
    FBAPOMDPState::model()->log();
}

std::string AbstractFBAPOMDPState::toString() const
{
    return "AbstractFBAPOMDPState with s=" + _domain_state->toString();
}
