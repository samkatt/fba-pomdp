#include "CoffeeProblemAction.hpp"

#include "domains/coffee/CoffeeProblemIndices.hpp"

namespace domains {

CoffeeProblemAction::CoffeeProblemAction(int i) : _a(i) {}

void CoffeeProblemAction::index(int i)
{
    _a.index(i);
}

int CoffeeProblemAction::index() const
{
    return _a.index();
}

std::string CoffeeProblemAction::toString() const
{
    return (index() == GetCoffee) ? "get coffee" : "check";
}

} // namespace domains
