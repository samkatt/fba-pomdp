#ifndef COFFEEPROBLEMSTATE_HPP
#define COFFEEPROBLEMSTATE_HPP

#include "environment/State.hpp"

#include <string>

namespace domains {

/**
 * @brief This class represents a state in the coffee domain
 *
 * Only overrides the toString method and adds some getters
 **/
class CoffeeProblemState : public State
{
public:
    explicit CoffeeProblemState(std::basic_string<char> i);

    /*** getters ***/
    bool wet() const;
    bool umbrella() const;
    bool hasCoffee() const;
    bool wantsCoffee() const;
    bool rains() const;

    /**** State interface ***/
    void index(std::string i) final;
    std::string index() const final;
    std::string toString() const final;
    std::vector<int> getFeatureValues() const final;

private:
    IndexState _s;
};

} // namespace domains

#endif // COFFEEPROBLEMSTATE_HPP
