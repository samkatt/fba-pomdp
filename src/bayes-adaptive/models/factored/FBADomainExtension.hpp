#ifndef FBADOMAINEXTENSION_HPP
#define FBADOMAINEXTENSION_HPP

#include <memory>

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "utils/distributions.hpp"

namespace configurations {
struct FBAConf;
}

/**
 * @brief The interface for extending domains to be used in the FBA-POMDP
 */
class FBADomainExtension
{

public:
    virtual ~FBADomainExtension() = default;

    /**
     * @brief Returns the factored domain size of the domain
     *
     * @return Returns the factored domain size of the domain
     */
    virtual Domain_Feature_Size domainFeatureSize() const = 0;

    /**
     * @brief Returns the (domain) state prior
     *
     * @return Returns the (domain) state prior
     *
     * Required by some particle reinvigoration strategies
     */
    virtual utils::categoricalDistr const* statePrior() const = 0;
};

namespace factory {

/**
 * @brief Returns the FBADomainExtension according to the configurations
 *
 * @param c the configurations (e.g. what domain, etc)
 *
 * @return the factored bayes-adaptive extended functionality to make a domain work in BA-POMDP
 */
std::unique_ptr<FBADomainExtension> makeFBADomainExtension(configurations::FBAConf const& c);

} // namespace factory

#endif // FBADOMAINEXTENSION_HPP
