#ifndef BAPRIOR_HPP
#define BAPRIOR_HPP

#include "utils/random.hpp"
#include <memory>

class BAState;
class State;

namespace configurations {
struct BAConf;
}

/**
 * @brief A prior distribution over the counts in a Bayes-Adaptive POMDP
 *
 * Is able to sample from the prior distribution
 * to initiate counts in a Bayes-Adaptive state
 **/
class BAPrior
{
public:
    virtual ~BAPrior() = default;

    /**
     * @brief samples a bayes-adaptive state, given a domain state and the sizes
     **/
    virtual BAState* sample(State const* s) const = 0;
};

#endif // BAPRIOR_HPP
