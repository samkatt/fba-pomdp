#include "easylogging++.h"

#include <cstddef>
#include <utility>

#include "AbstractFBAPOMDPState.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"

//#include "bayes-adaptive/states/factored/BABNModel.hpp"

#include "utils/index.hpp"
// TODO after every "real" timestep, will these be constructed again? I.e. can I assume _abstraction will be empty?
// cppcheck-suppress passedByValue
AbstractFBAPOMDPState::AbstractFBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model) :
        FBAPOMDPState(domain_state, model),
        _abstraction({}), // Empty initialization.
        _abstract_model(model) // Initialized later, when abstraction is added
{
    assert(model.domainFeatureSize());
}

BAState* AbstractFBAPOMDPState::copy(State const* domain_state) const
{
    return new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real()); //_model); // TODO better way?
}

// TODO change for abstraction
// this samples a new state...?
int AbstractFBAPOMDPState::sampleStateIndex(
    State const* s,
    Action const* a,
    rnd::sample::Dir::sampleMethod m) const
{
//    if (_abstraction.size() > 0) {
    return _abstract_model.sampleStateIndex(s,a,m);
//    }
//    return model()->sampleStateIndex(s,a,m);
}

// TODO do I need this?
// this samples a new state
int AbstractFBAPOMDPState::sampleStateIndexAbstract(
        State const* s,
        Action const* a,
        rnd::sample::Dir::sampleMethod m) const
{
        return _abstract_model.sampleStateIndex(s,a,m);
}

// TODO change for abstraction
int AbstractFBAPOMDPState::sampleObservationIndex(
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMethod m) const
{
//    if (_abstraction.size() > 0) {
//        return _abstract_model.sampleObservationIndex(a, new_s, m);
//    }
    // TODO do I need to do something with abstraction?
    return model()->sampleObservationIndex(a, new_s, m); // _model
}

// only called at initialization?
double AbstractFBAPOMDPState::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* s,
    rnd::sample::Dir::sampleMultinominal sampleMultinominal) const
{
    return FBAPOMDPState::model()->computeObservationProbability(o, a, s, sampleMultinominal); // _model
}

// TODO change for abstraction
void AbstractFBAPOMDPState::incrementCountsOf(
    State const* s,
    Action const* a,
    Observation const* o,
    State const* new_s,
    float amount)
{
    FBAPOMDPState::model()->incrementCountsOf(s, a, o, new_s, amount); // _model
}

std::vector<int>* AbstractFBAPOMDPState::getAbstraction(){
    return &_abstraction;
}

void AbstractFBAPOMDPState::setAbstraction(std::vector<int> new_abstraction){
    _abstraction = new_abstraction;
    _abstract_model = construct_abstract_model(FBAPOMDPState::model_real());
}

// Construct abstract model from the model given the features to keep in the abstraction
bayes_adaptive::factored::BABNModel AbstractFBAPOMDPState::construct_abstract_model(bayes_adaptive::factored::BABNModel model) {
    return model.abstract(_abstraction, model.structure());
}

void AbstractFBAPOMDPState::logCounts() const
{
    FBAPOMDPState::model()->log(); // _model
}

std::string AbstractFBAPOMDPState::toString() const
{
    return "AbstractFBAPOMDPState with s=" + _domain_state->toString();
}
