#ifndef CollisionAvoidanceBigBAEXTENSION_HPP
#define CollisionAvoidanceBigBAEXTENSION_HPP

#include "bayes-adaptive/models/table/BADomainExtension.hpp"

#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "domains/collision-avoidance-big/CollisionAvoidanceBig.hpp"
#include "environment/Reward.hpp"
#include "environment/State.hpp"
#include "environment/Terminal.hpp"
class Action;

namespace bayes_adaptive { namespace domain_extensions {

/**
 * \brief The extensions required to run CollisionAvoidanceBig as BA-POMDPs
 **/
class CollisionAvoidanceBigBAExtension : public BADomainExtension
{

public:
    CollisionAvoidanceBigBAExtension(int grid_width, int grid_height, int num_obstacles);
    ~CollisionAvoidanceBigBAExtension();

    /*** BADomainExtension interface implementation ****/
    Domain_Size domainSize() const final;
    State const* getState(std::string index) const final;
    Terminal terminal(State const* s, Action const* a, State const* new_s) const final;
    Reward reward(State const* s, Action const* a, State const* new_s) const final;

private:
    int _grid_width, _grid_height, _num_obstacles;
    int const _num_speeds = 2;
    int const _num_traffics = 3;
    int const _num_timeofdays = 2;
    int const _num_obstacletypes = 3;
    int x_agent_f = 0;
    int y_agent_f = 1;
    int speed_f = 2;
    int traffic_f = 3;
    int timeofday_f = 4;
    int obstacle_type_f = 5;
    int obstacle_start = 6;

    Domain_Size _domain_size = {_grid_width * _grid_height * _num_speeds * _num_traffics * _num_timeofdays * _num_obstacletypes
                                    * static_cast<int>(std::pow(_grid_height, _num_obstacles)),
                                domains::CollisionAvoidanceBig::NUM_ACTIONS,
                                static_cast<int>(_grid_width * _num_timeofdays * _num_obstacletypes * std::pow(_grid_height, _num_obstacles))};

    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<State const*>>>>>>> _states{
            static_cast<size_t>(_grid_width),
            std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<State const*>>>>>>(
                    _grid_height,
                    std::vector<std::vector<std::vector<std::vector<std::vector<State const*>>>>>(
                            _num_speeds, //speed
                            std::vector<std::vector<std::vector<std::vector<State const*>>>>(
                                    _num_traffics, //traffic
                                    std::vector<std::vector<std::vector<State const*>>>(
                                            _num_timeofdays, //timeofday
                                            std::vector<std::vector<State const*>>(
                                                    _num_obstacletypes, //obstacletype
                                                    std::vector<State const*>(std::pow(_grid_height, _num_obstacles)))))))};
};

}} // namespace bayes_adaptive::domain_extensions

#endif // CollisionAvoidanceBigBAEXTENSION_HPP
