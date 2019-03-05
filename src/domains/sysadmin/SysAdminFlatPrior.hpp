#ifndef SYSADMINFLATPRIOR_HPP
#define SYSADMINFLATPRIOR_HPP

#include "bayes-adaptive/priors/BAPOMDPPrior.hpp"

#include <cstddef>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/states/table/BAFlatModel.hpp"
#include "domains/sysadmin/SysAdmin.hpp"
#include "domains/sysadmin/SysAdminParameters.hpp"
#include "domains/sysadmin/SysAdminState.hpp"
class Action;
class BAPOMDPState;
class State;

namespace configurations {
struct BAConf;
}

namespace priors {

/**
 * @brief The bayes-adaptive priors for the sysadmin problem
 **/
class SysAdminFlatPrior : public BAPOMDPPrior
{

public:
    SysAdminFlatPrior(domains::SysAdmin const& domain, configurations::BAConf const& c);

private:
    float const _noise, _noisy_total_counts, _known_total_counts = 10000;

    Domain_Size const _domain_size;
    bayes_adaptive::table::BAFlatModel _prior;

    void precomputeFlatPrior(domains::SysAdmin const& d);

    /**
     * @brief recursive function that eventually sets the counts for (s,*,*)
     *
     * Will visit all possible computer configurations (new_s) in a tree
     * like fashion while maintaining the probability of that configuration.
     *
     * When all computers have been set (computer == -1), then it calls
     * the non-recursive setTrueTCounts to actually set (s,*,new_s)
     * with the appropriate (accumulated) counts
     */
    void setTrueTCountsRecur(
        domains::SysAdminState const& s,
        domains::SysAdminState const& new_s,
        int computer,
        double accumulated_probability,
        domains::SysAdmin const& d);

    /**
     * @brief recursive function that eventually sets the counts for (s,*,*) while rebooting not
     * working computer x
     *
     * Will visit all possible computer configurations (new_s) in a tree
     * like fashion while maintaining the probability of that configuration.
     *
     * When all computers have been set (computer == -1), then it calls
     * the non-recursive setTrueTCounts to actually set (s,*,new_s)
     * with the appropriate (accumulated) counts
     *
     * This function is called when earlier a non working computer (in s)
     * was found and it was decided to populate the counts for rebooting that computer
     */
    void setTrueTCountsRecur(
        domains::SysAdminState const& s,
        domains::SysAdminState const& new_s,
        int computer,
        double accumulated_probability,
        int const& rebooting_computer,
        domains::SysAdmin const& d);

    /**
     * @brief populate s_bapomdp with counts <s,*,s'>
     *
     * Sets the counts in s_bapomdp for the <s,*,s'> transition, assuming
     * the probability of reacing this transition is prob.
     **/
    void setTrueTCounts(
        domains::SysAdminState const& s,
        domains::SysAdminState const& new_s,
        double prob,
        domains::SysAdmin const& d);

    /**
     * @brief populate s_bapomdp with counts <s,reboot(),s'>
     *
     * Sets the counts in s_bapomdp for the <s,reboot(),s'> transition, assuming
     * the probability of reacing this transition is prob.
     *
     * This function is called when, at some point in figuring out the probability
     * of the transition, it has been decided that a non-functioning computer
     * must be rebooted. Thus the action has already been determined:
     * reboot(rebooting_computer).
     **/
    void setTrueTCountsForRebootingComputer(
        domains::SysAdminState const& s,
        domains::SysAdminState const& new_s,
        double prob,
        int const& rebooting_computer,
        domains::SysAdmin const& d);

    /*** interface implementation ***/
    BAPOMDPState* sampleBAPOMDPState(State const* domain_state) const final;
};

} // namespace priors

#endif // SYSADMINFLATPRIOR_HPP
