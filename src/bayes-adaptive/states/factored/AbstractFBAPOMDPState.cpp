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
                                             bayes_adaptive::factored::BABNModel abstract_model, std::vector<int> used_features) :
        FBAPOMDPState(domain_state, std::move(model)),
        feature_set(std::move(used_features)),
        _abstraction(0), // Empty initialization.
        _abstract_model(std::move(abstract_model))
        // Initialized later, when abstraction is added
{
    assert(model.domainFeatureSize());
}

BAState* AbstractFBAPOMDPState::copy(State const* domain_state) const
{
    if (_abstraction == 0) {
        return new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real(), _abstract_model, feature_set);
    } else {
        return new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real());
    }
}

// this samples a new state
std::string AbstractFBAPOMDPState::sampleStateIndex(
    State const* s,
    Action const* a,
    rnd::sample::Dir::sampleMethod m) const
{
    return model()->sampleStateIndex(s,a,m);
}

// From https://www.delftstack.com/howto/cpp/how-to-determine-if-a-string-is-number-cpp/
// TODO add parameter to BABNModel that tells if an index is a string or number?
bool isNumber(const std::string& str)
{
    return !std::any_of(&str[0], &str[str.size()-1], [] (char c) { return !std::isdigit(c); });
}

// this samples a new state, using the abstract model
std::string AbstractFBAPOMDPState::sampleStateIndexAbstract(
        State const* s,
        Action const* a,
        rnd::sample::Dir::sampleMethod m) const
{
    if (_abstraction == 0) {
        auto parent_values = s->getFeatureValues(); // model()->stateFeatureValues(s);

        if (!isNumber(s->index())) { // TODO should be better way
            std::string index = s->index();
            // TODO this only works if all the variables are at most 1 character long
            for (auto n = 0; n < (int) feature_set.size(); ++n) {
                index.replace(feature_set[n]*2, 1, std::to_string(_abstract_model.transitionNode(a, n).sample(parent_values, m)));
            }

            return index;
        }

        auto next_values = parent_values; // std::vector<int>(parent_values.size(), 0);
        for (auto n = 0; n < (int) feature_set.size(); ++n) {
            next_values[feature_set[n]] = _abstract_model.transitionNode(a, n).sample(parent_values, m);
        }

        return model()->sampleStateIndexThroughAbstraction(&next_values);
    }
    return model()->sampleStateIndex(s,a,m);
}

//auto parent_values = stateFeatureValues(s);
//
////create a vector for the next-stage variables - XXX: this is a memory allocation... expensive!?
//auto feature_values = std::vector<int>(_domain_feature_size->_S.size());
////fill the vector by sampling next stage feature 1 by 1
//for (auto n = 0; n < (int) _domain_feature_size->_S.size(); ++n)
//{ feature_values[n] = transitionNode(a, n).sample(parent_values, m); }
//
//return indexing::project(feature_values, _domain_feature_size->_S);

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
        _abstract_model.incrementCountsOfAbstract(a, o, amount, s->getFeatureValues(), // model()->stateFeatureValues(s),
                                                  new_s->getFeatureValues(),feature_set); // model()->stateFeatureValues(new_s), feature_set);
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
