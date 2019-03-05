#ifndef TIGERPRIORS_HPP
#define TIGERPRIORS_HPP

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/states/table/BAFlatModel.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"

namespace configurations {
struct BAConf;
}
class State;

namespace priors {

/**
 * @brief The bayes-adaptive count prior for tiger problem
 **/
class TigerBAPrior : public BAPOMDPPrior
{
public:
    /**
     * @brief sets the noise on the observation model belief
     *
     * noise is the probability that will be substracted
     * from the real observation reliability. So a noise of n
     * will lead to a belief that the agent hears correctly
     * with 8.5 - n probability (in terms of counts).
     *
     * The total number of counts is 10. So if n = 2,
     * then the counts are {6.5, 4.5}
     **/
    explicit TigerBAPrior(configurations::BAConf const& c);

private:
    Domain_Size _domain_sizes = {2, 3, 2};
    BAPOMDPState* sampleBAPOMDPState(State const* domain_state) const final
    {
        return new BAPOMDPState(domain_state, _prior);
    }

    bayes_adaptive::table::BAFlatModel _prior;
};

} // namespace priors

#endif // TIGERPRIORS_HPP
