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
        _abstract_model() // Initialized later, when abstraction is added
{
    assert(model.domainFeatureSize());
}

BAState* AbstractFBAPOMDPState::copy(State const* domain_state) const
{
    return new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real());
}

// this samples a new state
int AbstractFBAPOMDPState::sampleStateIndex(
    State const* s,
    Action const* a,
    rnd::sample::Dir::sampleMethod m) const
{
    return model()->sampleStateIndex(s,a,m);
}

// this samples a new state, using the abstract model (TODO currently it still samples from all the variables)
int AbstractFBAPOMDPState::sampleStateIndexAbstract(
        State const* s,
        Action const* a,
        rnd::sample::Dir::sampleMethod m) const
{
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

void AbstractFBAPOMDPState::setAbstraction(int k){
    _abstraction = k;
    _abstract_model = construct_abstract_model(FBAPOMDPState::model_real());
}

// Construct abstract model from the model given the features to keep in the abstraction
bayes_adaptive::factored::BABNModel AbstractFBAPOMDPState::construct_abstract_model(bayes_adaptive::factored::BABNModel model) {
    return model.abstract(_abstraction, model.structure());
}

void AbstractFBAPOMDPState::logCounts() const {
    FBAPOMDPState::model()->log();
}

std::string AbstractFBAPOMDPState::toString() const
{
    return "AbstractFBAPOMDPState with s=" + _domain_state->toString();
}
