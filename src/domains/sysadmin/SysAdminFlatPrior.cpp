#include "SysAdminFlatPrior.hpp"

#include <cmath>
#include <string>

#include "easylogging++.h"

#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/BAConf.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"
#include "utils/random.hpp"

namespace priors {

SysAdminFlatPrior::SysAdminFlatPrior(
    domains::SysAdmin const& domain,
    configurations::BAConf const& c) :
        _noise(c.noise),
        _noisy_total_counts(c.counts_total),
        _domain_size(0x1 << c.domain_conf.size, 2 * c.domain_conf.size, 2),
        _prior(&_domain_size)
{

    precomputeFlatPrior(domain);
}

BAPOMDPState* SysAdminFlatPrior::sampleBAPOMDPState(State const* domain_state) const
{
    return new BAPOMDPState(domain_state, _prior);
}

void SysAdminFlatPrior::precomputeFlatPrior(domains::SysAdmin const& d)
{

    // set table T counts
    for (auto s = 0; s < _domain_size._S; ++s)
    {
        setTrueTCountsRecur(
            domains::SysAdminState(s, d.size()),
            domains::SysAdminState(_domain_size._S - 1, d.size()),
            d.size() - 1,
            1,
            d);
    }

    auto const num_computers = d.size();

    auto const high_prob_counts = _known_total_counts * d.params()->_observe_prob,
               low_prob_counts  = _known_total_counts * (1 - d.params()->_observe_prob);

    // set O counts
    for (auto new_s = 0; new_s < _domain_size._S; ++new_s)
    {
        auto new_state = domains::SysAdminState(new_s, num_computers);

        for (auto c = 0; c < num_computers; ++c)
        {
            auto high_prob_observation =
                new_state.isOperational(c) ? IndexObservation("1") : IndexObservation("0");
            auto low_prob_observation = IndexObservation(std::to_string(1 - std::stoi(high_prob_observation.index())));

            auto observe = IndexAction(std::to_string(c)), reboot = IndexAction(std::to_string(c + num_computers));

            _prior.count(&observe, &new_state, &high_prob_observation) = high_prob_counts;
            _prior.count(&observe, &new_state, &low_prob_observation)  = low_prob_counts;

            _prior.count(&reboot, &new_state, &high_prob_observation) = high_prob_counts;
            _prior.count(&reboot, &new_state, &low_prob_observation)  = low_prob_counts;
        }
    }
}

void SysAdminFlatPrior::setTrueTCountsRecur(
    domains::SysAdminState const& s,
    domains::SysAdminState const& new_s,
    int computer,
    double accumulated_probability,
    domains::SysAdmin const& d)
{
    assert(accumulated_probability > 0 && accumulated_probability <= 1);
    assert(computer < d.size());

    // base case:
    // all computers are set and we now want to set the count
    // with the appropriate accumulated probability
    if (computer == -1)
    {
        setTrueTCounts(s, new_s, accumulated_probability, d);
    } else
    {

        auto s_fail = domains::SysAdminState(std::stoi(new_s.index()) & ~(0x1 << computer), d.size());

        // recursive call
        if (!s.isOperational(computer))
        {
            // action is not rebooting this computer
            setTrueTCountsRecur(s, s_fail, computer - 1, accumulated_probability, d);

            // reboot failing computer:
            // the scenario that rebooting fails
            setTrueTCountsRecur(
                s,
                s_fail,
                computer - 1,
                accumulated_probability * (1 - d.params()->_reboot_success_rate),
                computer,
                d);

            // the scenario that rebooting works
            setTrueTCountsRecur(
                s,
                new_s,
                computer - 1,
                accumulated_probability * d.params()->_reboot_success_rate,
                computer,
                d);
        } else // computer is operational
        {

            auto const fail_prob = 1
                                   - (1 - d.params()->_fail_prob)
                                         * pow(1 - d.params()->_fail_neighbour_factor,
                                               d.numFailingNeighbours(computer, &s));

            // scenario keeps working
            setTrueTCountsRecur(
                s, new_s, computer - 1, accumulated_probability * (1 - fail_prob), d);

            // scenario starts failing
            setTrueTCountsRecur(s, s_fail, computer - 1, accumulated_probability * fail_prob, d);
        }
    }
}

void SysAdminFlatPrior::setTrueTCounts(
    domains::SysAdminState const& s,
    domains::SysAdminState const& new_s,
    double prob,
    domains::SysAdmin const& d)
{
    assert(prob > 0 && prob <= 1);

    // probabilities of next state when listening
    for (auto a = 0; a < d.size(); ++a)
    {
        auto action                       = IndexAction(std::to_string(a));
        _prior.count(&s, &action, &new_s) = (float)prob * _known_total_counts;
    }

    // probabilities of next state when rebooting
    for (auto a = d.size(); a < 2 * d.size(); ++a)
    {
        auto action = IndexAction(std::to_string(a));

        if (new_s.isOperational(a - d.size()))
        {
            auto const fail_prob =
                1
                - (1 - d.params()->_fail_prob)
                      * pow(1 - d.params()->_fail_neighbour_factor, d.numFailingNeighbours(a, &s));

            // chance of getting to this state is having the computer stay alive PLUS
            // having the specific computer fail and rebooted
            _prior.count(&s, &action, &new_s) =
                _known_total_counts
                * (float)(prob + (prob * fail_prob / (1 - fail_prob) * d.params()->_reboot_success_rate));

        } else // rebooted but not working: we reached this state and had bad luck with rebooting
        {
            _prior.count(&s, &action, &new_s) =
                _known_total_counts * (float)(prob * (1 - d.params()->_reboot_success_rate));
        }
    }
}

void SysAdminFlatPrior::setTrueTCountsRecur(
    domains::SysAdminState const& s,
    domains::SysAdminState const& new_s,
    int computer,
    double accumulated_probability,
    int const& rebooting_computer,
    domains::SysAdmin const& d)
{
    assert(accumulated_probability > 0 && accumulated_probability <= 1);
    assert(rebooting_computer < d.size());
    assert(rebooting_computer >= 0 && rebooting_computer < d.size());

    // base case:
    // all computers are set and we now want to set the count
    // with the appropriate accumulated probability
    if (computer == -1)
    {
        setTrueTCountsForRebootingComputer(
            s, new_s, accumulated_probability, rebooting_computer, d);
    } else // recursive call
    {
        auto s_fail = domains::SysAdminState(std::stoi(new_s.index()) & ~(0x1 << computer), d.size());

        if (!s.isOperational(computer))
        {
            setTrueTCountsRecur(
                s, s_fail, computer - 1, accumulated_probability, rebooting_computer, d);
        } else
        {
            auto const fail_prob = 1
                                   - (1 - d.params()->_fail_prob)
                                         * pow(1 - d.params()->_fail_neighbour_factor,
                                               d.numFailingNeighbours(computer, &s));

            setTrueTCountsRecur(
                s,
                new_s,
                computer - 1,
                accumulated_probability * (1 - fail_prob),
                rebooting_computer,
                d);
            setTrueTCountsRecur(
                s,
                s_fail,
                computer - 1,
                accumulated_probability * fail_prob,
                rebooting_computer,
                d);
        }
    }
}

void SysAdminFlatPrior::setTrueTCountsForRebootingComputer(
    domains::SysAdminState const& s,
    domains::SysAdminState const& new_s,
    double prob,
    int const& rebooting_computer,
    domains::SysAdmin const& d)
{
    assert(prob > 0 && prob <= 1);
    assert(rebooting_computer >= 0 && rebooting_computer < d.size());

    auto const a                 = IndexAction(std::to_string(d.size() + rebooting_computer));
    _prior.count(&s, &a, &new_s) = (float)prob * _known_total_counts;
}

} // namespace priors
