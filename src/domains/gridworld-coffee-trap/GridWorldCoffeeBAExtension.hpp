//
// Created by rolf on 21-01-21.
//

#ifndef GRIDWORLDCOFFEEBAEXTENSION_HPP
#define GRIDWORLDCOFFEEBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "domains/gridworld-coffee-trap/GridWorldCoffee.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
class Action;
class State;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief Bayes-adaptive extensions for the GridWorldCoffee domain
 **/
class GridWorldCoffeeBAExtension : public BADomainExtension
{

public:
    explicit GridWorldCoffeeBAExtension();

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(int index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    size_t const _size;
    std::vector<domains::GridWorldCoffee::GridWorldCoffeeState> _states; // initialized in the constructor
    Domain_Size _domain_size;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // GRIDWORLDCOFFEEBAEXTENSION_HPP
