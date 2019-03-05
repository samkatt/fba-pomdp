#include "CoffeeProblemObservation.hpp"

#include "domains/coffee/CoffeeProblemIndices.hpp"

namespace domains {

CoffeeProblemObservation::CoffeeProblemObservation(int i) : _o(i) {}

void CoffeeProblemObservation::index(int i)
{
    _o.index(i);
}

int CoffeeProblemObservation::index() const
{
    return _o.index();
}

std::string CoffeeProblemObservation::toString() const
{
    return (index() == Want_Coffee) ? "Want coffee" : "Want no coffee";
}

} // namespace domains
