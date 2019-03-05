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
    explicit CoffeeProblemObservation(int i);

    /**** Observation interface ***/
    void index(int i) final;
    int index() const final;
    std::string toString() const final;

private:
    IndexObservation _o;
};

} // namespace domains

#endif // COFFEEPROBLEMOBSERVATION_HPP
