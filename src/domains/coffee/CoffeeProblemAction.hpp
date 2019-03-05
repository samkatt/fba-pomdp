#ifndef COFFEEPROBLEMACTION_HPP
#define COFFEEPROBLEMACTION_HPP

#include "environment/Action.hpp"

#include <string>

namespace domains {

/**
 * @brief An action in the coffee problem domain
 *
 * Implements toString
 **/
class CoffeeProblemAction : public Action

{
public:
    explicit CoffeeProblemAction(int i);

    /*** Action interface ***/
    void index(int i) final;
    int index() const final;
    std::string toString() const final;

private:
    IndexAction _a;
};

} // namespace domains

#endif // COFFEEPROBLEMACTION_HPP
