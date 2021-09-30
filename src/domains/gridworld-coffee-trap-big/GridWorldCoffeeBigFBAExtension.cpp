//
// Created by rolf on 25-01-21.
//

#include "GridWorldCoffeeBigFBAExtension.hpp"

#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBig.hpp"

namespace bayes_adaptive { namespace domain_extensions {

GridWorldCoffeeBigFBAExtension::GridWorldCoffeeBigFBAExtension(size_t extra_features) :
    _size(5),
    _extra_features(extra_features),
    _state_prior(0) // initialized below
//    _num_abstractions(2), // Minimum: x and y, then "1", variables that effect x and y, can be any of the others. Then we go to full model
//    _minimum_abstraction({0, 1}) // x and y
{

    auto const domain = domains::GridWorldCoffeeBig(_extra_features);
//    auto const goals  = domains::GridWorldCoffeeBig::goalLocations(_size);

//    _goal_amount = 1;

    // generate _state_prior
    // TODO correct?
//    _state_prior = utils::categoricalDistr(static_cast<int>((_size * _size * 2) << _extra_features));

//    float const start_prob =
//            (static_cast<float>(1) / (1 << _extra_features));

    for (unsigned int feature_config = 0; feature_config < (unsigned int) (1 << _extra_features); ++feature_config)
    {
        auto const* s = domain.getState(domains::GridWorldCoffeeBig::start_location, 0, feature_config);

//        _state_prior.setRawValue(s->index(), start_prob);

        domain.releaseState(s);
    }
}

Domain_Feature_Size GridWorldCoffeeBigFBAExtension::domainFeatureSize() const
{
    // For states, x, y, rain, velocity, carpet
    // For observation, x, y, rain, carpet
    auto feature_sizes = std::vector<int>(3 + _extra_features, 2);
    feature_sizes[0] = _size;
    feature_sizes[1] = _size;
    return {feature_sizes,
            {static_cast<int>(_size), static_cast<int>(_size)}};
}

utils::categoricalDistr const* GridWorldCoffeeBigFBAExtension::statePrior() const
{
    return &_state_prior;
}

}} // namespace bayes_adaptive::domain_extensions