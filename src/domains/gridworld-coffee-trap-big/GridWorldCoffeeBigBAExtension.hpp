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
    explicit GridWorldCoffeeBigBAExtension(size_t extra_features, domains::GridWorldCoffeeBig const& problem_domain);

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(std::string index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    size_t const _size;
    size_t const _extra_features;
    int const _x_feature = 0;
    int const _y_feature = 1;
    int const _rain_feature = 2;
//    std::vector<domains::GridWorldCoffeeBig::GridWorldCoffeeBigState> _states; // initialized in the constructor
    Domain_Size _domain_size;
    domains::GridWorldCoffeeBig const& gridworldcoffeebig;
//    bool _store_statespace;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // GridWorldCoffeeBigBAEXTENSION_HPP
