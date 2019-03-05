#ifndef RANDOMPLANNER_HPP
#define RANDOMPLANNER_HPP

#include "planners/Planner.hpp"

class Action;
class Belief;
class History;
class POMDP;

namespace planners {

/**
 * @brief This planner returns a random action
 **/
class RandomPlanner : public Planner
{
public:
    RandomPlanner();
    Action const*
        selectAction(POMDP const& simulator, Belief const& belief, History const& h) const final;
};

} // namespace planners

#endif // RANDOMPLANNER_HPP
