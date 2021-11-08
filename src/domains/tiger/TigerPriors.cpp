#include "TigerPriors.hpp"

#include <memory>
#include <vector>

#include "configurations/BAConf.hpp"

#include "domains/tiger/Tiger.hpp"

#include "environment/State.hpp"

namespace priors {

TigerBAPrior::TigerBAPrior(configurations::BAConf const& c) :
        _prior(
            std::make_shared<std::vector<float>>(3 * 2 * 2, 5000),
            std::make_shared<std::vector<float>>(3 * 2 * 2, 5000),
            &_domain_sizes)
{
    if (c.noise <= -.15 || c.noise > .3)
    {
        throw "noise has to be between -.15 and .3";
    }

    auto s_right = IndexState(std::to_string(domains::Tiger::Literal::RIGHT));
    auto s_left  = IndexState(std::to_string(domains::Tiger::Literal::LEFT));

    auto a_listen = IndexAction(std::to_string(domains::Tiger::Literal::OBSERVE));

    auto o_left  = IndexObservation(std::to_string(domains::Tiger::Literal::LEFT));
    auto o_right = IndexObservation(std::to_string(domains::Tiger::Literal::RIGHT));

    _prior.count(&s_right, &a_listen, &s_left) = 0;
    _prior.count(&s_left, &a_listen, &s_right) = 0;

    float correct_observation_prior   = (.85f - c.noise) * c.counts_total,
          incorrect_observation_prior = (.15f + c.noise) * c.counts_total;

    _prior.count(&a_listen, &s_right, &o_right) = correct_observation_prior;
    _prior.count(&a_listen, &s_right, &o_left)  = incorrect_observation_prior;
    _prior.count(&a_listen, &s_left, &o_right)  = incorrect_observation_prior;
    _prior.count(&a_listen, &s_left, &o_left)   = correct_observation_prior;
}

} // namespace priors
