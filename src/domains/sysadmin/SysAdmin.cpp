#include "SysAdmin.hpp"

#include <algorithm>

#include "easylogging++.h"

#include "environment/Reward.hpp"
#include "utils/index.hpp"

namespace domains {
SysAdmin_Parameters const SysAdmin::param = SysAdmin_Parameters(.025f, .95f, .95f, 1.0f, .075f);

SysAdmin::SysAdmin(int n, std::string const& version) :
        _version(UNINITIALIZED), _size(n), _states({}) // initiated below
{

    if (_size < 1)
    {
        throw("Cannot initiate Sysadmin with n " + std::to_string(_size));
    }

    _states.reserve(_S_size);
    for (auto i = 0; i < _S_size; ++i) { _states.emplace_back(SysAdminState(i, n)); }

    if (version == "independent")
    {
        _version = INDEPENDENT;
    } else if (version == "linear")
    {
        _version = LINEAR;
    }

    if (_version == UNINITIALIZED)
    {
        throw(" Sysadmin only accepts 'independent' and 'linear' as versions, given: " + version);
    }

    VLOG(1) << "initiated " << _version << " sysadmin of " << _size << " computers";
}

int SysAdmin::size() const
{
    return _size;
}

State const* SysAdmin::breakComputer(State const* s, int computer) const
{
    assertLegal(s);
    assert(computer >= 0 && computer < _size);

    return &_states[s->index() & ~(0x1 << computer)];
}

State const* SysAdmin::fixComputer(State const* s, int computer) const
{
    assertLegal(s);
    assert(computer >= 0 && computer < _size);

    return &_states[s->index() | (0x1 << computer)];
}

Action const* SysAdmin::observeAction(int comp) const
{
    return _actions.get(comp);
}

Action const* SysAdmin::rebootAction(int comp) const
{
    return _actions.get(comp + _size);
}

SysAdminState const* SysAdmin::getState(std::vector<int> config) const
{

    assert(config.size() == static_cast<size_t>(_size));
    for (auto c : config) { assert(c == 0 || c == 1); }

    std::reverse(config.begin(), config.end());
    auto const i = indexing::project(config, _S_space);

    return &_states[i];
}

float SysAdmin::failProbability(State const* s, Action const* a, int c) const
{

    double fail_probability =
        static_cast<SysAdminState const*>(s)->isOperational(c)
            ? 1
                  - (1 - param._fail_prob)
                        * pow(1 - param._fail_neighbour_factor, numFailingNeighbours(c, s))
            : 1;

    if (isRebootingAction(a) && rebootAction(c)->index() == a->index())
    {
        fail_probability *= (1 - params()->_reboot_success_rate);
    }

    return fail_probability;
}

State const* SysAdmin::sampleStartState() const
{
    return &_states[_S_size - 1];
}

Terminal SysAdmin::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    assertLegal(*s);
    assertLegal(a);

    auto const rebooting         = isRebootingAction(a);
    auto const operated_computer = (rebooting) ? a->index() - _size : a->index();
    auto sys_state               = static_cast<SysAdminState const*>(*s);

    // step
    auto index = sys_state->index();
    for (auto c = 0; c < _size; ++c)
    {

        auto const keeps_running_prob =
            (1 - param._fail_prob)
            * pow(1 - param._fail_neighbour_factor, numFailingNeighbours(c, sys_state));

        if (rnd::uniform_rand01() > keeps_running_prob)
        {
            index = index & ~(0x1 << c); // fail this computer
        }
    }

    if (rebooting && rnd::uniform_rand01() < param._reboot_success_rate)
    {
        index = index | (0x1 << operated_computer); // make this computer working
    }

    *s        = &_states[index];
    sys_state = static_cast<SysAdminState const*>(*s);

    // return the correct observation with _observe_prob,
    // and the opposite otherwise
    *o = ((rnd::uniform_rand01() < param._observe_prob)
          == (sys_state->isOperational(operated_computer)))
             ? &_operational
             : &_failing;

    // reward is just really the number of working computers (-1 if rebooting)
    r->set(
        ((float)sys_state->numOperationalComputers())
        - param._reboot_cost * static_cast<int>(rebooting));

    return Terminal(false);
}

double SysAdmin::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* new_s) const
{

    auto const operated_computer = (isRebootingAction(a)) ? a->index() - _size : a->index();
    auto const is_operational =
        static_cast<SysAdminState const*>(new_s)->isOperational(operated_computer);

    return (o->index() == static_cast<int>(is_operational)) ? param._observe_prob
                                                            : 1 - param._observe_prob;
}

Action const* SysAdmin::generateRandomAction(State const* /*s*/) const
{
    return _actions.get(_action_distr(rnd::rng()));
}

void SysAdmin::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assertLegal(s);
    assert(actions->empty());

    for (auto i = 0; i < _A_size; ++i) { actions->emplace_back(_actions.get(i)); }
}

void SysAdmin::releaseAction(Action const* a) const
{
    assertLegal(a);
    // actions are stored locally & never modified so memory need to be free'd
}

void SysAdmin::releaseObservation(Observation const* o) const
{
    assertLegal(o);
    // observations are stored locally & never modified so memory need to be free'd
}

void SysAdmin::releaseState(State const* s) const
{
    assertLegal(s);
    // states are stored locally & never modified so memory need to be free'd
}

Action const* SysAdmin::copyAction(Action const* a) const
{
    assertLegal(a);
    // we never modify a so a itself is a legit copy
    return a;
}

Observation const* SysAdmin::copyObservation(Observation const* o) const
{
    assertLegal(o);
    // we never modify o so o itself is a legit copy
    return o;
}

State const* SysAdmin::copyState(State const* s) const
{
    assertLegal(s);
    // we never modify s so s itself is a legit copy
    return s;
}

unsigned int SysAdmin::numFailingNeighbours(int c, State const* state) const
{

    if (_version == INDEPENDENT)
    {
        return 0;
    }

    // linear (hard coded) topology
    auto const s = static_cast<SysAdminState const*>(state);
    auto n       = 0;

    if (c > 0 && !s->isOperational(c - 1))
    {
        n++;
    }

    if (c < (_size - 1) && !s->isOperational(c + 1))
    {
        n++;
    }

    return n;
}

bool SysAdmin::isRebootingAction(Action const* a) const
{
    return a->index() >= _size;
}

void SysAdmin::assertLegal(State const* s) const
{
    assert(s != nullptr && s->index() >= 0 && s->index() < _S_size);
}

void SysAdmin::assertLegal(Action const* a) const
{
    assert(a != nullptr && a->index() >= 0 && a->index() < _A_size);
}

void SysAdmin::assertLegal(Observation const* o) const
{
    assert(o != nullptr && o->index() >= 0 && o->index() < _O_size);
}

} // namespace domains
