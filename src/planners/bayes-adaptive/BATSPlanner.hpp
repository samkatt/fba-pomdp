#ifndef BATSPLANNER_HPP
#define BATSPLANNER_HPP

#include "planners/bayes-adaptive/BAPlanner.hpp"
#include "planners/bayes-adaptive/RBAPOUCT.hpp"

#include "environment/Action.hpp"
class BAPOMDP;
class History;
namespace beliefs {
class BABelief;
}
namespace configurations {
struct Conf;
}

namespace planners {

/**
 * @brief Plans with respect to s ~ b(s)
 **/
class BATSPlanner : public BAPlanner
{
public:
    explicit BATSPlanner(configurations::Conf const& c);

    Action const* selectAction(
        BAPOMDP const& bapomdp,
        beliefs::BABelief const& belief,
        History const& h) const final;

private:
    RBAPOUCT _planner;
};

} // namespace planners

#endif // BATSPLANNER_HPP
