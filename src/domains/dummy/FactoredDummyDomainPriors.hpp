#ifndef FACTOREDDUMMYDOMAINPRIORS_HPP
#define FACTOREDDUMMYDOMAINPRIORS_HPP

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"
#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "bayes-adaptive/states/factored/BABNModel.hpp"
#include "bayes-adaptive/states/table/BAFlatModel.hpp"

namespace configurations {
struct FBAConf;
}

class State;
class BAPOMDPState;
class FBAPOMDPState;

namespace priors {
/**
 * @brief Bayes-Adaptive priors for the FactoredDummy domain
 **/
class FactoredDummyPrior : public BAPOMDPPrior, public FBAPOMDPPrior
{
public:
    /**
     * @brief constructor for the tabular representation
     **/
    explicit FactoredDummyPrior(size_t size);

    /**
     * @brief constructor for the factored representation
     **/
    FactoredDummyPrior(configurations::FBAConf const& c, size_t size);

    /*** FBAPOMDPPrior interface ***/
    FBAPOMDPState* sampleFullyConnectedState(State const* domain_state) const final;

    FBAPOMDPState* sampleCorrectGraphState(State const* domain_state) const final;
    bayes_adaptive::factored::BABNModel::Structure
        mutate(bayes_adaptive::factored::BABNModel::Structure structure) const final;

private:
    size_t _size;
    Domain_Size const _domain_size;
    Domain_Feature_Size const _domain_feature_size;
    bayes_adaptive::factored::BABNModel::Indexing_Steps const _fbapomdp_step_size;

    bayes_adaptive::table::BAFlatModel _flat_prior;
    bayes_adaptive::factored::BABNModel _factored_prior;
    bayes_adaptive::factored::BABNModel _fully_connected_prior;

    /**
     * @brief precomputes the counts for the BAPOMDP (flat) state
     **/
    void precomputePrior();

    /*** BAPOMDPPrior interface ***/
    BAPOMDPState* sampleBAPOMDPState(State const* domain_state) const final;

    /*** FBAPOMDPPrior interface ***/
    FBAPOMDPState* sampleFBAPOMDPState(State const* domain_state) const final;
};

} // namespace priors

#endif // FACTOREDDUMMYDOMAINPRIORS_HPP
