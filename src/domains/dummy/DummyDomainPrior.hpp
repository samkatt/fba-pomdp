#ifndef DUMMYDOMAINPRIOR_HPP
#define DUMMYDOMAINPRIOR_HPP

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"

#include <memory>

#include "bayes-adaptive/states/table/BAFlatModel.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"

#include "bayes-adaptive/models/Domain_Size.hpp"

namespace priors {

/**
 * @brief An accurate and confident prior for the dummy domain
 **/
class DummyBAPrior : public BAPOMDPPrior
{

private:
    Domain_Size _domain_size = {1, 1, 1};

    std::shared_ptr<std::vector<float> const> _base_counts =
        std::make_shared<std::vector<float>>(1, 100);

    BAPOMDPState* sampleBAPOMDPState(State const* domain_state) const final
    {

        return new BAPOMDPState(
            domain_state,
            bayes_adaptive::table::BAFlatModel(_base_counts, _base_counts, &_domain_size));
    }
};

} // namespace priors

#endif // DUMMYDOMAINPRIOR_HPP
