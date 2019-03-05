#include "FBAPOMDPState.hpp"

#include "easylogging++.h"

#include <cstddef>
#include <utility>

#include "FBAPOMDPState.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "utils/index.hpp"

// cppcheck-suppress passedByValue
FBAPOMDPState::FBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model) :
        BAState(domain_state),
        _model(std::move(model))
{
}

BAState* FBAPOMDPState::copy(State const* domain_state) const
{
    return new FBAPOMDPState(domain_state, _model);
}

int FBAPOMDPState::sampleStateIndex(
    State const* s,
    Action const* a,
    rnd::sample::Dir::sampleMethod m) const
{
    return _model.sampleStateIndex(s, a, m);
}

int FBAPOMDPState::sampleObservationIndex(
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMethod m) const
{
    return _model.sampleObservationIndex(a, new_s, m);
}

double FBAPOMDPState::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* s,
    rnd::sample::Dir::sampleMultinominal sampleMultinominal) const
{
    return _model.computeObservationProbability(o, a, s, sampleMultinominal);
}

void FBAPOMDPState::incrementCountsOf(
    State const* s,
    Action const* a,
    Observation const* o,
    State const* new_s,
    float amount)
{
    _model.incrementCountsOf(s, a, o, new_s, amount);
}

void FBAPOMDPState::logCounts() const
{
    _model.log();
}

std::string FBAPOMDPState::toString() const
{
    return "FBAPOMDPState with s=" + _domain_state->toString();
}
