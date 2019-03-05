#ifndef BAPOMDPPRIOR_HPP
#define BAPOMDPPRIOR_HPP

#include "bayes-adaptive/priors/BAPrior.hpp"

#include <memory>

#include "bayes-adaptive/models/Domain_Size.hpp"

#include "bayes-adaptive/states/BAState.hpp"

class POMDP;
class BADomainExtension;
class BAPOMDPState;
class State;

namespace configurations {
struct BAConf;
}

/**
 * @brief The tbayes-adaptive prior
 *
 * Provides a function to sample a belief over flat/table BAPOMDP model
 **/
class BAPOMDPPrior : public BAPrior
{
public:
    /*** BAPrior interface **/
    BAState* sample(State const* domain_state) const final;

private:
    /**
     * @brief sample a BAPOMDPState based on domain state s
     **/
    virtual BAPOMDPState* sampleBAPOMDPState(State const* s) const = 0;
};

namespace factory {

/**
 * @brief Returns a prior for the BA-POMDP
 *
 * @param domain the domain (for domain knowledge) to set a prior for
 * @param c configurations of the prior
 *
 * @return a BA-POMDP prior
 */
std::unique_ptr<BAPOMDPPrior>
    makeTBAPOMDPPrior(POMDP const& domain, configurations::BAConf const& c);

} // namespace factory

#endif // BAPOMDPPRIOR_HPP
