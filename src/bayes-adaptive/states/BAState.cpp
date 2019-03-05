#include "BAState.hpp"

BAState::BAState(State const* domain_state) : _domain_state(domain_state)
{
    assert(_domain_state);
}

int BAState::index() const
{
    return _domain_state->index();
}

void BAState::index(int /*i*/)
{
    throw "BAState::index() should not be called. Inheriters of BAState should implement this function";
}
