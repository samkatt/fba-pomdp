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
    explicit CoffeeProblemAction(std::basic_string<char> i);

    /*** Action interface ***/
    void index(std::string i) final;
    std::string index() const final;
    std::string toString() const final;
    std::vector<int> getFeatureValues() const final; // { return {std::stoi(index())}; };

private:
//    IndexAction _a;
        // TODO fix
        std::basic_string<char> _a;
    };

} // namespace domains

#endif // COFFEEPROBLEMACTION_HPP
