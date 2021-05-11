#include "easylogging++.h"

#include <cstddef>
#include <utility>

#include "AbstractFBAPOMDPState.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "utils/index.hpp"

// cppcheck-suppress passedByValue
AbstractFBAPOMDPState::AbstractFBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model) :
        FBAPOMDPState(domain_state, model),
        _abstraction({})
{
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
    return FBAPOMDPState::model()->sampleStateIndex(s, a, m); // _model
}

// TODO change for abstraction
int AbstractFBAPOMDPState::sampleObservationIndex(
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMethod m) const
{
    return FBAPOMDPState::model()->sampleObservationIndex(a, new_s, m); // _model
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

std::vector<unsigned int>* AbstractFBAPOMDPState::getAbstraction(){
    return &_abstraction;
}

void AbstractFBAPOMDPState::setAbstraction(std::vector<unsigned int> new_abstraction){
    _abstraction = new_abstraction;
}

void AbstractFBAPOMDPState::logCounts() const
{
    FBAPOMDPState::model()->log(); // _model
}

std::string AbstractFBAPOMDPState::toString() const
{
    return "AbstractFBAPOMDPState with s=" + _domain_state->toString();
}
