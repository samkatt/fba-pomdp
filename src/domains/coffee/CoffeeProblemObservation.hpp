#ifndef COFFEEPROBLEMOBSERVATION_HPP
#define COFFEEPROBLEMOBSERVATION_HPP

#include "environment/Observation.hpp"

#include <string>

namespace domains {

/**
 * @brief An observation in the coffee problem class
 **/
class CoffeeProblemObservation : public Observation

{
public:
    explicit CoffeeProblemObservation(std::basic_string<char> i);

    /**** Observation interface ***/
    void index(std::string i) final;
    std::string index() const final;
    std::string toString() const final;
    std::vector<int> getFeatureValues() const final;

private:
    IndexObservation _o;
};

} // namespace domains

#endif // COFFEEPROBLEMOBSERVATION_HPP
