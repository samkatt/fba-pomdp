#ifndef COLLISIONAVOIDANCEFBAEXTENSION_HPP
#define COLLISIONAVOIDANCEFBAEXTENSION_HPP

#include "bayes-adaptive/models/factored/FBADomainExtension.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"
#include "domains/collision-avoidance/CollisionAvoidance.hpp"
#include "utils/distributions.hpp"

namespace bayes_adaptive { namespace domain_extensions {

/**
 * @brief Extends CollisionAvoidance to be used in FBA-POMDP
 **/
class CollisionAvoidanceFBAExtension : public FBADomainExtension
{
public:
    CollisionAvoidanceFBAExtension(
        int grid_width,
        int grid_height,
        int num_obstacles,
        domains::CollisionAvoidance::VERSION version);

    /*** FBADomainExtension interface ***/
    Domain_Feature_Size domainFeatureSize() const final;
    utils::categoricalDistr const* statePrior() const final;

private:
    Domain_Feature_Size _domain_feature_size;
    utils::categoricalDistr _state_prior;
};

}} // namespace bayes_adaptive::domain_extensions

#endif // COLLISIONAVOIDANCEFBAEXTENSION_HPP
