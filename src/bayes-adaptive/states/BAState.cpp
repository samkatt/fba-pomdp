#include "BAState.hpp"

BAState::BAState(State const* domain_state) : _domain_state(domain_state)
{
    assert(_domain_state);
}

std::string BAState::index() const
{
    return _domain_state->index();
}

void BAState::index(std::string /*i*/)
{
    throw "BAState::index() should not be called. Inheriters of BAState should implement this function";
}

std::vector<int> BAState::getFeatureValues() const {
    return _domain_state->getFeatureValues();
}