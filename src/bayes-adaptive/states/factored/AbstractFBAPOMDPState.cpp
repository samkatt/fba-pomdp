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
{
    assert(model.domainFeatureSize());
}

AbstractFBAPOMDPState::AbstractFBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model,
                                             bayes_adaptive::factored::BABNModel abstract_model) :
        FBAPOMDPState(domain_state, std::move(model)),
        _abstraction(0), // Empty initialization.
        _abstract_model(std::move(abstract_model)) // Initialized later, when abstraction is added
{
    assert(model.domainFeatureSize());
}

BAState* AbstractFBAPOMDPState::copy(State const* domain_state) const
{
    if (_abstraction == 0) {
        return new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real(), _abstract_model);
    } else {
        return new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real());
    }
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

        return model()->sampleStateIndexThroughAbstraction(s,a, next_values);
    }
    return model()->sampleStateIndex(s,a,m);
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


void AbstractFBAPOMDPState::setAbstraction(bayes_adaptive::factored::BABNModel abstr_model){
    _abstraction = 0;
    _abstract_model = std::move(abstr_model);
}

void AbstractFBAPOMDPState::logCounts() const {
    FBAPOMDPState::model()->log();
}

std::string AbstractFBAPOMDPState::toString() const
{
    return "AbstractFBAPOMDPState with s=" + _domain_state->toString();
}
