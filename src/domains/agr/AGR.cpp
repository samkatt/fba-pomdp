#include "AGR.hpp"

// State indices
// =============
//
// i = 0                   : target_goal = -n, target_pos = -n
// i = 1                   : target_goal = -n, target_pos = -n + 1
// ...
// i = n                   : target_goal = -n, target_pos = 0
// ...
// i = 2n                  : target_goal = -n, target_pos = n
// i = 2n + 1              : target_goal = -n + 1, target_pos = -n
// i = 2n + 2              : target_goal = -n + 1, target_pos = -n + 1
// ...
// i = (2n + 1) * (2n + 1) : target_goal = n, target_pos = n

namespace domains {

AGRState::AGRState(int n, int i) : target_pos(0), target_goal(0), _n(n)
{
    index(i); // stupid way to get initialization working
}

AGRState::AGRState(int n_in, int target_pos_in, int target_goal_in) :
        target_pos(target_pos_in), target_goal(target_goal_in), _n(n_in)
{
}

int AGRState::indexOf(int n_in, int target_pos_in, int target_goal_in)
{
    return (2 * n_in + 1) * (target_goal_in + n_in) + target_pos_in + n_in;
}

int AGRState::index() const
{
    return AGRState::indexOf(_n, target_pos, target_goal);
}

void AGRState::index(int i)
{
    target_goal = i / (2 * _n + 1) - _n;
    target_pos  = i % (2 * _n + 1) - _n;
}

std::string AGRState::toString() const
{
    std::stringstream ss;
    ss << "S(target_pos=" << target_pos << ", target_goal=" << target_goal << ")";
    return ss.str();
}

// Action indices
// ==============
//
// i = 0      : action = help pos -n
// i = 1      : action = help pos -n + 1
// ...
// i = n      : action = help pos 0
// ...
// i = 2n     : action = help pos n
// i = 2n + 1 : action = work
// i = 2n + 2 : action = observe

AGRAction::AGRAction(int n, int i) : type(""), help_pos(), _n(n)
{
    index(i); // stupid way to get initialization working
}

// cppcheck-suppress passedByValue
AGRAction::AGRAction(int n_in, std::string type_in, int help_pos_in) :
        type(std::move(type_in)), help_pos(help_pos_in), _n(n_in)
{
}

int AGRAction::indexOf(int n, std::string const& type, int help_pos)
{
    assert(type == "help" or type == "work" or type == "observe");

    if (type == "help")
    {
        return help_pos + n;
    } else if (type == "work")
    {
        return 2 * n + 1;
    } else // type == "observe"
    {
        return 2 * n + 2;
    }
}

int AGRAction::index() const
{
    return AGRAction::indexOf(_n, type, help_pos);
}

void AGRAction::index(int i)
{
    assert(i <= 2 * _n + 2);

    if (i <= 2 * _n)
    {
        type     = "help";
        help_pos = i - _n;
    } else if (i == 2 * _n + 1)
    {
        type     = "work";
        help_pos = -1;
    } else // i == 2 * _n + 2
    {
        type     = "observe";
        help_pos = -1;
    }
}

std::string AGRAction::toString() const
{
    std::stringstream ss;

    ss << "A(type=" << type;
    if (type == "help")
    {
        ss << ", help_pos=" << help_pos;
    }
    ss << ")";
    return ss.str();
}

// Observation indices
// ===================
//
// i = 0      : observation = target_pos = -n
// i = 1      : observation = target_pos = -n + 1
// ...
// i = n      : observation = target_pos = 0
// ...
// i = 2n     : observation = target_pos = n
// i = 2n + 1 : observation = none

AGRObservation::AGRObservation(int n, int i) : type(""), target_pos(0), _n(n)
{
    index(i); // stupid way to get initialization working
}

// cppcheck-suppress passedByValue
AGRObservation::AGRObservation(int n_in, std::string type_in, int target_pos_in) :
        type(std::move(type_in)), target_pos(target_pos_in), _n(n_in)
{
}

int AGRObservation::indexOf(int n, std::string const& type, int target_pos)
{
    assert(type == "target_pos" or type == "none");

    if (type == "target_pos")
    {
        return target_pos + n;
    } else // type == "none"
    {
        return 2 * n + 1;
    }
}

int AGRObservation::index() const
{
    return AGRObservation::indexOf(_n, type, target_pos);
}

void AGRObservation::index(int i)
{
    if (i <= 2 * _n)
    {
        type       = "target_pos";
        target_pos = i - _n;
    } else if (i == 2 * _n + 1)
    {
        type       = "none";
        target_pos = -1;
    } else
    {
        throw "AGRObservation::index i not defined.";
    }
}

std::string AGRObservation::toString() const
{
    std::stringstream ss;

    ss << "O(type=" << type;
    if (type == "target_pos")
    {
        ss << ", target_pos=" << target_pos;
    }
    ss << ")";
    return ss.str();
}

AGR::AGR(int n) :
        _n(n),
        _nstates((2 * _n + 1) * (2 * _n + 1)),
        _states(new const AGRState*[_nstates]),
        _nstart_states(2 * _n + 1),
        _start_states(new const AGRState*[_nstart_states]),
        _start_state_distr(rnd::integerDistribution(0, _nstart_states)),
        _nactions(2 * _n + 3),
        _actions(new const AGRAction*[_nactions]),
        _action_distr(rnd::integerDistribution(0, _nactions)),
        _nobservations(2 * _n + 2),
        _observations(new const AGRObservation*[_nobservations])
{

    for (int i = 0, j = 0; i < _nstates; i++)
    {
        _states[i] = new AGRState(_n, i);
        if (_states[i]->target_pos == 0)
            _start_states[j++] = _states[i];
    }

    for (auto i = 0; i < _nactions; i++) _actions[i] = new AGRAction(_n, i);

    for (auto i = 0; i < _nobservations; i++) _observations[i] = new AGRObservation(_n, i);
}

AGR::~AGR()
{
    for (auto i = 0; i < _nstates; i++) { delete _states[i]; }
    delete[] _states;
    delete[] _start_states;

    for (auto i = 0; i < _nactions; i++) delete _actions[i];
    delete[] _actions;

    for (auto i = 0; i < _nobservations; i++) delete _observations[i];
    delete[] _observations;
}

State const* AGR::sampleStartState() const
{
    return _start_states[_start_state_distr(rnd::rng())];
}

Action const* AGR::generateRandomAction(State const* s) const
{
    legalStateCheck(s);
    return _actions[_action_distr(rnd::rng())];
}

Terminal AGR::step(State const** s, Action const* a, Observation const** o, Reward* r) const
{
    legalStateCheck(*s);
    legalActionCheck(a);
    assert(o != 0);
    assert(r != 0);

    const AGRState* agr_s  = dynamic_cast<const AGRState*>(*s);
    const AGRAction* agr_a = dynamic_cast<const AGRAction*>(a);

    // computing reward
    bool helped = false;
    if (agr_a->type == "help")
    {
        if (agr_a->help_pos == agr_s->target_goal && agr_s->target_pos == agr_s->target_goal)
        {
            r->set(100.);
            helped = true;
        } else
        {
            r->set(-100.);
        }
    } else if (agr_a->type == "work")
    {
        r->set(-5.);
    } else if (agr_a->type == "observe")
    {
        r->set(-10.);
    }

    // computing next state:
    //  * target goal remains the same,
    //  * target steps towards goal
    int target_diff = agr_s->target_goal - agr_s->target_pos;
    // int target_step = std::clamp(target_diff, -1, 1);
    int target_step = target_diff > 1 ? 1 : target_diff;
    target_step     = target_step < -1 ? -1 : target_step;
    AGRState s_next(_n, agr_s->target_pos + target_step, agr_s->target_goal);
    *s = _states[s_next.index()];

    // computing observation
    std::string o_next_type = "none";
    int o_next_pos          = -1;
    if (agr_a->type == "observe")
    {
        o_next_type = "target_pos";
        o_next_pos  = s_next.target_pos;
    }
    AGRObservation o_next(_n, o_next_type, o_next_pos);
    *o = _observations[o_next.index()];

    legalObservationCheck(*o);
    legalStateCheck(*s);

    return Terminal(helped);
}

double AGR::computeObservationProbability(Observation const*, Action const*, State const*) const
{
    throw "AGR::computeObservationProbability nyi";
}

void AGR::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assert(actions->empty());
    legalStateCheck(s);

    for (auto i = 0; i < _nactions; i++) { actions->emplace_back(_actions[i]); }
}

void AGR::releaseAction(Action const* a) const
{
    legalActionCheck(a);
    // action is a member of the object
}

void AGR::releaseObservation(Observation const* o) const
{
    legalObservationCheck(o);
    // observation is a member of the object
}

void AGR::releaseState(State const* s) const
{
    legalStateCheck(s);
    // state is a member of the object
}

Action const* AGR::copyAction(Action const* a) const
{
    legalActionCheck(a);
    return a;
}

Observation const* AGR::copyObservation(Observation const* o) const
{
    legalObservationCheck(o);
    return o;
}

State const* AGR::copyState(State const* s) const
{
    legalStateCheck(s);
    return s;
}

void AGR::legalActionCheck(Action const* a) const
{
    assert(a != 0 && a->index() >= 0 && a->index() < _nactions);
}

void AGR::legalObservationCheck(Observation const* o) const
{
    assert(o != 0 && o->index() >= 0 && o->index() < _nobservations);
}

void AGR::legalStateCheck(State const* s) const
{
    assert(s != 0 && s->index() >= 0 && s->index() < _nstates);
}

} // namespace domains
