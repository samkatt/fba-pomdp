#include "FBAPOMDP.hpp"

#include "easylogging++.h"

#include "bayes-adaptive/priors/BAPrior.hpp"
#include "bayes-adaptive/priors/FBAPOMDPPrior.hpp"
#include "configurations/FBAConf.hpp"

#include "utils/distributions.hpp"
#include "utils/index.hpp"
#include "utils/stl_utils.hpp"

namespace bayes_adaptive { namespace factored {

FBAPOMDP::FBAPOMDP(
    std::unique_ptr<POMDP> domain,
    std::unique_ptr<BADomainExtension> ba_domain_ext,
    std::unique_ptr<FBADomainExtension> fba_domain_ext,
    std::unique_ptr<FBAPOMDPPrior> prior,
    rnd::sample::Dir::sampleMethod sample_method,
    rnd::sample::Dir::sampleMultinominal compute_mult_method) :
        BAPOMDP(
            std::move(domain),
            std::move(ba_domain_ext),
            std::unique_ptr<BAPrior>(prior.release()),
            sample_method,
            compute_mult_method),
        _fba_domain_ext(std::move(fba_domain_ext)),
        _domain_feature_size(_fba_domain_ext->domainFeatureSize()),
        _step_sizes(
            indexing::stepSize(_domain_feature_size._S),
            indexing::stepSize(_domain_feature_size._O))
{

    VLOG(1) << "Initiated FBAPOMDP with (S:" << utils::stl::toString(_domain_feature_size._S)
            << ", O:" << utils::stl::toString(_domain_feature_size._O);
}

Domain_Feature_Size const* FBAPOMDP::domainFeatureSize() const
{
    return &_domain_feature_size;
}

utils::categoricalDistr const* FBAPOMDP::domainStatePrior() const
{
    return _fba_domain_ext->statePrior();
}

FBAPOMDPPrior const* FBAPOMDP::prior() const
{
    return static_cast<FBAPOMDPPrior const*>(_ba_prior.get());
}

/**
 * @brief mutates the topology of a set of DBNs describing the dynamics
 **/
bayes_adaptive::factored::BABNModel::Structure
    FBAPOMDP::mutate(bayes_adaptive::factored::BABNModel::Structure structure) const
{
    return static_cast<FBAPOMDPPrior const*>(_ba_prior.get())->mutate(std::move(structure));
}

FBAPOMDPState const* FBAPOMDP::sampleFullyConnectedState() const
{
    return static_cast<FBAPOMDPPrior const*>(_ba_prior.get())
        ->sampleFullyConnectedState(_domain->sampleStartState());
}

FBAPOMDPState const* FBAPOMDP::sampleCorrectGraphState() const
{
    return static_cast<FBAPOMDPPrior const*>(_ba_prior.get())
        ->sampleCorrectGraphState(_domain->sampleStartState());
}

}} // namespace bayes_adaptive::factored

namespace factory {

std::unique_ptr<BAPOMDP> makeFBAPOMDP(configurations::FBAConf const& c)
{

    auto domain = dynamic_cast<POMDP*>(factory::makeEnvironment(c.domain_conf).release());

    auto ba_domain_ext  = makeBADomainExtension(c);
    auto fba_domain_ext = makeFBADomainExtension(c);
    auto prior          = makeFBAPOMDPPrior(*domain, c);

    auto sample_method = (c.bayes_sample_method == 0) ? rnd::sample::Dir::sampleFromSampledMult
                                                      : rnd::sample::Dir::sampleFromExpectedMult;

    auto compute_mult_method = (c.bayes_sample_method == 0) ? rnd::sample::Dir::sampleMult
                                                            : rnd::sample::Dir::expectedMult;

    return std::unique_ptr<BAPOMDP>(new bayes_adaptive::factored::FBAPOMDP(
        std::unique_ptr<POMDP>(domain),
        std::move(ba_domain_ext),
        std::move(fba_domain_ext),
        std::move(prior),
        sample_method,
        compute_mult_method));
}

} // namespace factory
