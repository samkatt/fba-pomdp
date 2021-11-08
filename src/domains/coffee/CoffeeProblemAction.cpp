#include "CoffeeProblemAction.hpp"

#include "domains/coffee/CoffeeProblemIndices.hpp"

namespace domains {

CoffeeProblemAction::CoffeeProblemAction(std::basic_string<char> i) : _a(i) {}

void CoffeeProblemAction::index(std::string i)
{
    _a = i;
    //_a.index(i);
}

std::string CoffeeProblemAction::index() const
{
    return _a; //.index();
}

std::string CoffeeProblemAction::toString() const
{
    // TODO fix
    return "get cofee";
//    return (index() == GetCoffee) ? "get coffee" : "check";
}

std::vector<int> CoffeeProblemAction::getFeatureValues() const {
    // TODO Fix
    return std::vector<int>();
}

} // namespace domains
