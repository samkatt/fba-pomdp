#include "BAPOMDP.hpp"

#include "easylogging++.h"

#include <utility> // std::move

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "configurations/BAConf.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/Reward.hpp"

BAPOMDP::BAPOMDP(
    std::unique_ptr<POMDP> domain,
    std::unique_ptr<BADomainExtension> ba_domain_ext,
    std::unique_ptr<BAPrior> prior,
    rnd::sample::Dir::sampleMethod sample_method,
    rnd::sample::Dir::sampleMultinominal compute_mult_method) :
        _domain(std::move(domain)),
        _ba_domain_ext(std::move(ba_domain_ext)),
        _ba_prior(std::move(prior)),
        _observations(_ba_domain_ext->domainSize()._O),
        _domain_size(_ba_domain_ext->domainSize()),
        _sample_method(sample_method),
        _compute_mult_method(compute_mult_method)
{
    assert(_domain != nullptr);
    assert(_domain_size._A > 0 && _domain_size._O > 0 && _domain_size._S > 0);

    VLOG(1) << "Initiated BAPOMDP with (S:" << _domain_size._S << ", A:" << _domain_size._A
            << ", O:" << _domain_size._O << ")";
}

BAPOMDP::StepType BAPOMDP::mode() const
{
    return _mode;
}

void BAPOMDP::mode(StepType new_mode) const
{
    _mode = new_mode;
}

Domain_Size const* BAPOMDP::domainSize() const
{
    return &_domain_size;
}

State const* BAPOMDP::sampleDomainState() const
{
    return _domain->sampleStartState();
}

State const* BAPOMDP::domainState(int i) const
{
    return _ba_domain_ext->getState(i);
}

void BAPOMDP::releaseDomainState(State const* s) const
{
    _domain->releaseState(s);
}

State const* BAPOMDP::copyDomainState(State const* domain_state) const
{
    return _domain->copyState(domain_state);
}

void BAPOMDP::resetDomainState(BAState const* s) const
{
    assert(s != nullptr);

    // underlying domain owns the domain state
    _domain->releaseState(s->_domain_state);

    const_cast<BAState*>(s)->_domain_state = _domain->sampleStartState();
}

Action const* BAPOMDP::generateRandomAction(State const* s) const
{
    assert(s != nullptr);

    return _domain->generateRandomAction(static_cast<BAState const*>(s)->_domain_state);
}

void BAPOMDP::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assert(s != nullptr);

    _domain->addLegalActions(static_cast<BAState const*>(s)->_domain_state, actions);
}

double BAPOMDP::computeObservationProbability(Observation const* o, Action const* a, State const* s)
    const
{

    return static_cast<BAState const*>(s)->computeObservationProbability(
        o, a, s, _compute_mult_method);
}

State const* BAPOMDP::sampleStartState() const
{
    return _ba_prior->sample(_domain->sampleStartState());
}

Terminal BAPOMDP::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    return step(s, a, o, r, _mode);
}

Terminal BAPOMDP::step(
    State const** s,
    Action const* a,
    Observation const** o,
    Reward* r,
    StepType step_type) const
{
    assert(s != nullptr && *s != nullptr && (*s)->index() >= 0);
    assert(a != nullptr && a->index() >= 0);
    assert(o != nullptr);
    assert(r != nullptr);

    auto ba_s               = const_cast<BAState*>(static_cast<BAState const*>(*s));
    auto const domain_state = ba_s->_domain_state;

    // sample state
    auto const new_s =
        _ba_domain_ext->getState(ba_s->sampleStateIndex(domain_state, a, _sample_method));
    *o = _observations.get(ba_s->sampleObservationIndex(a, new_s, _sample_method));

    auto const t = _ba_domain_ext->terminal(domain_state, a, new_s);
    *r           = _ba_domain_ext->reward(domain_state, a, new_s);

    if (step_type == StepType::UpdateCounts)
    {
        ba_s->incrementCountsOf(domain_state, a, *o, new_s);
    }

    _domain->releaseState(ba_s->_domain_state);
    ba_s->_domain_state = new_s;

    return t;
}

void BAPOMDP::releaseAction(Action const* a) const
{
    _domain->releaseAction(a);
}

void BAPOMDP::releaseObservation(Observation const* o) const
{
    assert(o != nullptr && o->index() >= 0 && o->index() < _domain_size._O);
    // all stored in _observations
}

void BAPOMDP::releaseState(State const* s) const
{
    assert(s != nullptr);

    _domain->releaseState(static_cast<BAState const*>(s)->_domain_state);

    delete s;
}

Action const* BAPOMDP::copyAction(Action const* a) const
{
    return _domain->copyAction(a);
}

Observation const* BAPOMDP::copyObservation(Observation const* o) const
{
    assert(o != nullptr && o->index() >= 0 && o->index() < _domain_size._O);

    // all stored in _observations
    return o;
}

State const* BAPOMDP::copyState(State const* s) const
{
    assert(s != nullptr && s->index() >= 0 && s->index() < _domain_size._S);

    auto ba_s = static_cast<BAState const*>(s);

    return ba_s->copy(_domain->copyState(ba_s->_domain_state));
}

namespace factory {

std::unique_ptr<BAPOMDP> makeTBAPOMDP(configurations::BAConf const& c)
{

    auto domain = dynamic_cast<POMDP*>(factory::makeEnvironment(c.domain_conf).release());

    auto ba_domain_ext = factory::makeBADomainExtension(c);
    auto prior         = makeTBAPOMDPPrior(*domain, c);

    auto const sample_method       = (c.bayes_sample_method == 0)
                                         ? rnd::sample::Dir::sampleFromSampledMult
                                         : rnd::sample::Dir::sampleFromExpectedMult;
    auto const compute_mult_method = (c.bayes_sample_method == 0) ? rnd::sample::Dir::sampleMult
                                                                  : rnd::sample::Dir::expectedMult;

    return std::unique_ptr<BAPOMDP>(new BAPOMDP(
        std::unique_ptr<POMDP>(domain),
        std::move(ba_domain_ext),
        std::move(prior),
        sample_method,
        compute_mult_method));
}

} // namespace factory
