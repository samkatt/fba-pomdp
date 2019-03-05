#ifndef FBAPOMDP_HPP
#define FBAPOMDP_HPP

#include "bayes-adaptive/models/table/BAPOMDP.hpp"

#include <memory>

#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"
#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"
#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "domains/POMDP.hpp"
#include "utils/random.hpp"
namespace configurations {
struct FBAConf;
}
namespace distributions { namespace categorical {
struct categoricalDistr;
}} // namespace distributions::categorical

namespace bayes_adaptive { namespace factored {

/**
 * @brief The factored BAPOMDP class
 **/
class FBAPOMDP : public BAPOMDP
{
public:
    FBAPOMDP(
        std::unique_ptr<POMDP> domain,
        std::unique_ptr<BADomainExtension> ba_domain_ext,
        std::unique_ptr<FBADomainExtension> fba_domain_ext,
        std::unique_ptr<FBAPOMDPPrior> prior,
        rnd::sample::Dir::sampleMethod sample_method,
        rnd::sample::Dir::sampleMultinominal compute_mult_method);

    Domain_Feature_Size const* domainFeatureSize() const;

    FBAPOMDPPrior const* prior() const;

    /**
     * @brief returns the prior distribution over states 0..S
     **/
    utils::categoricalDistr const* domainStatePrior() const;

    /**
     * @brief mutates the topology of a set of DBNs describing the dynamics
     **/
    bayes_adaptive::factored::BABNModel::Structure
        mutate(bayes_adaptive::factored::BABNModel::Structure structure) const;

    /**
     * @brief samples a FBAPOMDPState where all (unknown) edges are connected
     **/
    FBAPOMDPState const* sampleFullyConnectedState() const;

    /**
     * @brief samples a FBAPOMDPState which correct structure (regardless of noise parameter)
     **/
    FBAPOMDPState const* sampleCorrectGraphState() const;

private:
    // domain dependent extended functionality for the FBA-POMDP
    std::unique_ptr<FBADomainExtension const> _fba_domain_ext;

    Domain_Feature_Size const _domain_feature_size;

    BABNModel::Indexing_Steps _step_sizes;
};

}} // namespace bayes_adaptive::factored

namespace factory {

/**
 * @brief Constructor for a factored BA-POMDP instance according to the configurations
 *
 * @param c the configurations
 *
 * @return a factored, structured BAPOMDP unique pointer
 */
std::unique_ptr<BAPOMDP> makeFBAPOMDP(configurations::FBAConf const& c);

} // namespace factory

#endif // FBAPOMDP_HPP
