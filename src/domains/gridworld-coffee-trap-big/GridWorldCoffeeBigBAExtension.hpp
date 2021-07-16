#ifndef GridWorldCoffeeBigBAEXTENSION_HPP
#define GridWorldCoffeeBigBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBig.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
class Action;
class State;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief Bayes-adaptive extensions for the GridWorldCoffeeBig domain
 **/
class GridWorldCoffeeBigBAExtension : public BADomainExtension
{

public:
    explicit GridWorldCoffeeBigBAExtension();

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(int index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    size_t const _size;
    size_t const _carpet_tiles;
    std::vector<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState> _states; // initialized in the constructor
    Domain_Size _domain_size;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // GridWorldCoffeeBigBAEXTENSION_HPP
