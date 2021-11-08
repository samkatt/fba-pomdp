#include "CoffeeProblemState.hpp"

#include "domains/coffee/CoffeeProblemIndices.hpp"

namespace domains {

CoffeeProblemState::CoffeeProblemState(std::basic_string<char> i) : _s(i)
{
    assert(std::stoi(i) < 32);
}

bool CoffeeProblemState::wet() const
{
    return (std::stoi(index()) & WET) != 0;
}

bool CoffeeProblemState::umbrella() const
{
    return (std::stoi(index()) & UMBRELLA) != 0;
}

bool CoffeeProblemState::hasCoffee() const
{
    return (std::stoi(index()) & HAS_COFEE) != 0;
}

bool CoffeeProblemState::wantsCoffee() const
{
    return (std::stoi(index()) & WANTS_COFFEE) != 0;
}

bool CoffeeProblemState::rains() const
{
    return (std::stoi(index()) & RAINS) != 0;
}

void CoffeeProblemState::index(std::string i)
{
    _s.index(i);
}

std::string CoffeeProblemState::index() const
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

std::vector<int> CoffeeProblemState::getFeatureValues() const {
    // TODO implement
    return std::vector<int>();
}

} // namespace domains
