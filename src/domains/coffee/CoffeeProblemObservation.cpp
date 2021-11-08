#include "CoffeeProblemObservation.hpp"

#include "domains/coffee/CoffeeProblemIndices.hpp"

namespace domains {

CoffeeProblemObservation::CoffeeProblemObservation(std::basic_string<char> i) : _o(i) {}

void CoffeeProblemObservation::index(std::string i)
{
    _o.index(i);
}

std::string CoffeeProblemObservation::index() const
{
    return _o.index();
}

std::string CoffeeProblemObservation::toString() const
{
    return (std::stoi(index()) == Want_Coffee) ? "Want coffee" : "Want no coffee";
}

std::vector<int> CoffeeProblemObservation::getFeatureValues() const {
    // TODO implement
    return std::vector<int>();
}

} // namespace domains
