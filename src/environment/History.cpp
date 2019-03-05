#include "History.hpp"

#include "domains/POMDP.hpp"
#include "environment/Environment.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"

History::Interaction::Interaction(Action const* a, Observation const* o) : action(a), observation(o)
{
    assert(action != nullptr);
    assert(observation != nullptr);
}

void History::add(Action const* a, Observation const* o)
{
    _history.emplace_back(Interaction(a, o));
}

void History::clear(POMDP const& action_owner, Environment const& observation_owner)
{
    for (auto i : _history)
    {
        action_owner.releaseAction(i.action);
        observation_owner.releaseObservation(i.observation);
    }
}

std::vector<History::Interaction>::iterator History::begin()
{
    return _history.begin();
}

std::vector<History::Interaction>::iterator History::end()
{
    return _history.end();
}

std::vector<History::Interaction>::reference History::back()
{
    return _history.back();
}

std::vector<History::Interaction>::const_iterator History::begin() const
{
    return _history.begin();
}

std::vector<History::Interaction>::const_iterator History::end() const
{
    return _history.end();
}

std::vector<History::Interaction>::const_reference History::back() const
{
    return _history.back();
}

size_t History::length() const
{
    return _history.size();
}

History::Interaction History::operator[](int i) const
{
    return _history[i];
}
