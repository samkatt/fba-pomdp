#ifndef GRIDWORLDBAEXTENSION_HPP
#define GRIDWORLDBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "domains/gridworld/GridWorld.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
class Action;
class State;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief Bayes-adaptive extensions for the GridWorld domain
 **/
class GridWorldBAExtension : public BADomainExtension
{

public:
    explicit GridWorldBAExtension(size_t size);

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(std::string index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    size_t const _size;
    std::vector<domains::GridWorld::GridWorldState> _states; // initialized in the constructor
    Domain_Size _domain_size;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // GRIDWORLDBAEXTENSION_HPP
