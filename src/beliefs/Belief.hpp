#ifndef BELIEF_HPP
#define BELIEF_HPP

#include <memory>
class Action;
class Observation;
class POMDP;
class State;

namespace configurations {
struct Conf;
}

/**
 * @brief A belief or estimation of the state manager
 **/
class Belief
{
public:
    virtual ~Belief() = default;

    /**
     * @brief creates initial belief
     **/
    virtual void initiate(POMDP const& domain) = 0;

    /**
     * @brief restores belief to null
     **/
    virtual void free(POMDP const& domain) = 0;

    /**
     * @brief samples a state w.r.t. to the belief
     **/
    virtual State const* sample() const = 0;

    /**
     * @brief performs the belief update w.r.t. a new action-observation step
     **/
    virtual void updateEstimation(Action const* a, Observation const* o, POMDP const& domain) = 0;
};

namespace factory {

std::unique_ptr<Belief> makeBelief(configurations::Conf const& c);

} // namespace factory

#endif // BELIEF_HPP
