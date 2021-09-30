#ifndef TSPLANNER_HPP
#define TSPLANNER_HPP

#include "planners/Planner.hpp"

#include "environment/Action.hpp"
#include "planners/mcts/POUCT.hpp"
class Belief;
class History;
class POMDP;
namespace configurations {
struct Conf;
}

namespace planners {

/**
 * @brief Samples and assumes a single state from the estimation
 *
 * This class uses another planner to actually plan from the (assumed) state
 **/
class TSPlanner : public Planner
{
public:
    explicit TSPlanner(configurations::Conf const& c);

    Action const*
        selectAction(POMDP const& simulator, Belief const& belief, History const& h, int& total_simulations) const final;

private:
    POUCT _planner;
};

} // namespace planners

#endif // TSPLANNER_HPP
