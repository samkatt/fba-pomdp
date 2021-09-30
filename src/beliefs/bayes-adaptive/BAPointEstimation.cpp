#include "BAPointEstimation.hpp"

#include <cassert>

#include "easylogging++.h"

#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "domains/POMDP.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"

namespace beliefs {

BAPointEstimation::BAPointEstimation()
{
    VLOG(1) << "Initiated point estimation belief";
}

BAPointEstimation::BAPointEstimation(State const* s)
{
    assert(s != nullptr);

    _point_estimate = s;

    VLOG(1) << "Initiated point estimation belief with state " + s->toString();
}

State const* BAPointEstimation::sample() const
{
    assert(_point_estimate != nullptr);
    return _point_estimate;
}

void BAPointEstimation::initiate(POMDP const& simulator)
{
    free(simulator);
    _point_estimate = simulator.sampleStartState();
}

void BAPointEstimation::free(POMDP const& simulator)
{
    if (_point_estimate != nullptr)
    {
        simulator.releaseState(_point_estimate);
    }

    _point_estimate = nullptr;
}

void BAPointEstimation::updateEstimation(
    Action const* a,
    Observation const* o,
    POMDP const& simulator)
{
    assert(a != nullptr && o != nullptr);
    Observation const* temp_observation = nullptr;
    Reward temp_reward(0);

    auto temp_state = simulator.copyState(_point_estimate);
    simulator.step(&temp_state, a, &temp_observation, &temp_reward);

    // rejection sampling- keep state if observation matches
    while (o->index() != temp_observation->index())
    {
        simulator.releaseState(temp_state);
        simulator.releaseObservation(temp_observation);

        temp_state = simulator.copyState(_point_estimate);
        simulator.step(&temp_state, a, &temp_observation, &temp_reward);
    }

    simulator.releaseState(_point_estimate);
    simulator.releaseObservation(temp_observation);

    // new estimate is (a copy of) the generated state
    _point_estimate = simulator.copyState(temp_state);
    simulator.releaseState(temp_state);

    VLOG(3) << "Belief estimates: s=" << _point_estimate->index();
}

void BAPointEstimation::resetDomainStateDistribution(BAPOMDP const& bapomdp)
{
    assert(_point_estimate != nullptr);
    bapomdp.resetDomainState(dynamic_cast<BAState const*>(_point_estimate));
}

    void BAPointEstimation::resetDomainStateDistributionAndAddAbstraction(const BAPOMDP &bapomdp,
                                                                          Abstraction &abstraction, int i) {
        assert(_point_estimate != nullptr);
        bapomdp.resetDomainState(dynamic_cast<BAState const*>(_point_estimate));
        VLOG( 3) << abstraction.printSomething();
        VLOG( 3) << i;
    }


} // namespace beliefs
