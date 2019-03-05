#ifndef BAPLANNER_HPP
#define BAPLANNER_HPP

#include "planners/Planner.hpp"

#include <memory>

#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "beliefs/Belief.hpp"
#include "beliefs/bayes-adaptive/BABelief.hpp"
#include "environment/Action.hpp"
class History;
class POMDP;

namespace configurations {
struct Conf;
}

namespace planners {

/**
 * @brief interface for planners in BA-POMDPs
 **/
class BAPlanner : public Planner
{
public:
    ~BAPlanner() override = default;

    /***** interface *****/
    /**
     * @brief selects an action in the BA-POMDP environment
     **/
    virtual Action const* selectAction(
        BAPOMDP const& bapomdp,
        beliefs::BABelief const& belief,
        History const& h) const = 0;

    /**** implementation planner ****/
    /**
     * @brief delegates to implementation, casting the input
     **/
    Action const*
        selectAction(POMDP const& simulator, Belief const& belief, History const& h) const override
    {
        return selectAction(
            dynamic_cast<BAPOMDP const&>(simulator),
            dynamic_cast<beliefs::BABelief const&>(belief),
            h);
    }
};
} // namespace planners

namespace factory {

std::unique_ptr<Planner> makeBAPlanner(configurations::Conf const& c);

} // namespace factory

#endif // BAPLANNER_HPP
