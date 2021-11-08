#ifndef BADOMAINEXTENSION_HPP
#define BADOMAINEXTENSION_HPP

#include <memory>
#include <domains/POMDP.hpp>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "environment/Reward.hpp"
#include "environment/Terminal.hpp"
class Action;
class State;

namespace configurations {
struct BAConf;
}

/**
 * @brief Implementation for domain extensions to allow usage with BA-POMDPs
 *
 * The BAPOMDP needs some domain knowledge, and this interface
 * determines exactly which functions provide those
 **/
class BADomainExtension
{

public:
    virtual ~BADomainExtension() = default;

    /**
     * @brief provides required domain knowledge
     **/
    virtual Domain_Size domainSize() const = 0;

    /**
     * @brief returns state i
     **/
    virtual State const* getState(std::string index) const = 0;

    /**
    * @brief returns observation i
    **/
//    virtual Observation const* getObservation(std::string index) const = 0;

    /**
     * @brief returns terminality of <s,a,s'> transition
     **/
    virtual Terminal terminal(State const* s, Action const* a, State const* new_s) const = 0;

    /**
     * @brief returns reward for <s,a,s'> transition
     **/
    virtual Reward reward(State const* s, Action const* a, State const* new_s) const = 0;
};

namespace factory {

/**
 * @brief Returns the BADomainExtension according to the configurations
 *
 * @param c the configurations (e.g. what domain, etc)
 *
 * @return the bayes-adaptive extended functionality to make a domain work in BA-POMDP
 */
std::unique_ptr<BADomainExtension> makeBADomainExtension(configurations::BAConf const& c, POMDP const& domain);

} // namespace factory

#endif // BADOMAINEXTENSION_HPP
