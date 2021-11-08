#ifndef COLLISIONAVOIDANCEBAEXTENSION_HPP
#define COLLISIONAVOIDANCEBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
class Action;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief The extensions required to run CollisionAvoidance as BA-POMDPs
 **/
class CollisionAvoidanceBAExtension : public BADomainExtension
{

public:
    CollisionAvoidanceBAExtension(int grid_width, int grid_height, int num_obstacles);
    ~CollisionAvoidanceBAExtension();

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(std::string index) const final;
//    Observation const* getObservation(std::string index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    int _grid_width, _grid_height, _num_obstacles;

    Domain_Size _domain_size = {_grid_width * _grid_height
                                    * static_cast<int>(std::pow(_grid_height, _num_obstacles)),
                                domains::CollisionAvoidance::NUM_ACTIONS,
                                static_cast<int>(std::pow(_grid_height, _num_obstacles))};

    std::vector<std::vector<std::vector<State const*>>> _states{
        static_cast<size_t>(_grid_width),
        std::vector<std::vector<State const*>>(
            _grid_height,
            std::vector<State const*>(std::pow(_grid_height, _num_obstacles)))};
};

}} // namespace bayes_adaptive::domain_extensions

#endif // COLLISIONAVOIDANCEBAEXTENSION_HPP
