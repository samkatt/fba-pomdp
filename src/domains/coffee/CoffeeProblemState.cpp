#include "CoffeeProblemState.hpp"

#include "domains/coffee/CoffeeProblemIndices.hpp"

namespace domains {

CoffeeProblemState::CoffeeProblemState(int i) : _s(i)
{
    assert(i < 32);
}

bool CoffeeProblemState::wet() const
{
    return (index() & WET) != 0;
}

bool CoffeeProblemState::umbrella() const
{
    return (index() & UMBRELLA) != 0;
}

bool CoffeeProblemState::hasCoffee() const
{
    return (index() & HAS_COFEE) != 0;
}

bool CoffeeProblemState::wantsCoffee() const
{
    return (index() & WANTS_COFFEE) != 0;
}

bool CoffeeProblemState::rains() const
{
    return (index() & RAINS) != 0;
}

void CoffeeProblemState::index(int i)
{
    _s.index(i);
}

int CoffeeProblemState::index() const
{
    return _s.index();
}

std::string CoffeeProblemState::toString() const
{
    std::string result = "(";

    if (wet())
    {
        result += " w ";
    }

    if (umbrella())
    {
        result += " u ";
    }

    if (rains())
    {
        result += " r ";
    }

    if (hasCoffee())
    {
        result += " hc ";
    }

    if (wantsCoffee())
    {
        result += " wc ";
    }

    return result + ")";
}

} // namespace domains
