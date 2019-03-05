#include "LinearDummyDomain.hpp"

#include "easylogging++.h"

#include "environment/Action.hpp"
#include "environment/State.hpp"

#include "utils/random.hpp"

namespace domains {

LinearDummyDomain::LinearDummyDomain()
{
    VLOG(1) << "initiated linear dummy domain";
}

Action const* LinearDummyDomain::generateRandomAction(State const* s) const
{
    assert(s != nullptr);

    if (s->index() > 0)
    {
        // returns action 0 or 1
        return (rnd::boolean()) ? new IndexAction(0) : new IndexAction(1);
    }

    return new IndexAction(Actions::FORWARD);
}

void LinearDummyDomain::addLegalActions(State const* s, std::vector<Action const*>* actions) const
{
    assert(actions->empty());
    assert(s != nullptr);

    actions->emplace_back(new IndexAction(Actions::FORWARD));
    if (s->index() != 0)
    {
        actions->emplace_back(new IndexAction(Actions::BACKWARD));
    }
}

Terminal LinearDummyDomain::step(State const** s, Action const* a, Observation const** o, Reward* r)
    const
{
    assert(a->index() < 2);
    auto state = const_cast<State*>(*s);

    // do traditional dummy step
    auto const t = DummyDomain::step(s, a, o, r);

    // increment or decrement depending on action
    if (a->index() == Actions::BACKWARD)
    {
        state->index(state->index() - 1);
    } else
    {
        state->index(state->index() + 1);
    }

    assert(state->index() >= 0);
    return t;
}

} // namespace domains
