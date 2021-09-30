#ifndef PLANNER_HPP
#define PLANNER_HPP

#include <memory>

class Action;
class Belief;
class History;
class POMDP;
namespace configurations {
struct Conf;
}

/**
 * @brief The interface of a planner: it requires to select an action based on a belief and/or
 *history
 **/
class Planner
{
public:
    virtual ~Planner() = default;

    virtual Action const*
        selectAction(POMDP const& simulator, Belief const& belief, History const& h, int& total_simulations) const = 0;
};

namespace factory {

std::unique_ptr<Planner> makePlanner(configurations::Conf const& c);

} // namespace factory

#endif // PLANNER_HPP
