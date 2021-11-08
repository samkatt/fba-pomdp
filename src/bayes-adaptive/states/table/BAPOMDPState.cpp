#include "BAPOMDPState.hpp"

using bayes_adaptive::table::BAFlatModel;

BAPOMDPState::BAPOMDPState(State const* s, BAFlatModel chi) : BAState(s), _model(std::move(chi)) {}

BAState* BAPOMDPState::copy(State const* s) const
{
    return new BAPOMDPState(s, _model);
}

std::string BAPOMDPState::sampleStateIndex(
    State const* s,
    Action const* a,
    rnd::sample::Dir::sampleMethod m) const
{
    return _model.sampleStateIndex(s, a, m);
}

int BAPOMDPState::sampleObservationIndex(
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMethod m) const
{
    return _model.sampleObservationIndex(a, new_s, m);
}

double BAPOMDPState::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMultinominal m) const
{
    return _model.computeObservationProbability(o, a, new_s, m);
}

void BAPOMDPState::incrementCountsOf(
    State const* s,
    Action const* a,
    Observation const* o,
    State const* new_s,
    float amount)
{
    _model.incrementCountsOf(s, a, o, new_s, amount);
}

std::string BAPOMDPState::toString() const
{
    return "BAPOMDPState with s=" + _domain_state->toString();
}

void BAPOMDPState::logCounts() const
{
    _model.logCounts();
}
