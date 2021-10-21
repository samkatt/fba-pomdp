#ifndef CollisionAvoidanceBigFBAEXTENSION_HPP
#define CollisionAvoidanceBigFBAEXTENSION_HPP

#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "domains/collision-avoidance-big/CollisionAvoidanceBig.hpp"
#include "utils/distributions.hpp"

namespace bayes_adaptive { namespace domain_extensions {

/**
 * @brief Extends CollisionAvoidanceBig to be used in FBA-POMDP
 **/
class CollisionAvoidanceBigFBAExtension : public FBADomainExtension
{
public:
    CollisionAvoidanceBigFBAExtension(
        int grid_width,
        int grid_height,
        int num_obstacles,
        domains::CollisionAvoidanceBig::VERSION version);

    /*** FBADomainExtension interface ***/
    Domain_Feature_Size domainFeatureSize() const final;
    utils::categoricalDistr const* statePrior() const final;

private:
    Domain_Feature_Size _domain_feature_size;
    utils::categoricalDistr _state_prior;
    int const _num_speeds = 3;
    int const _num_traffics = 3;
    int const _num_timeofdays = 2;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // CollisionAvoidanceBigFBAEXTENSION_HPP
