#ifndef SYSADMINPARAMETERS
#define SYSADMINPARAMETERS

namespace domains {

/**
 * @brief Parameters of the sysadmin problem
 **/
struct SysAdmin_Parameters
{

    SysAdmin_Parameters(
        float fail_prob,
        float observe_prob,
        float reboot_succ_rate,
        float reboot_cost,
        float fail_neighbour_factor) :
            _fail_prob(fail_prob),
            _observe_prob(observe_prob),
            _reboot_success_rate(reboot_succ_rate),
            _reboot_cost(reboot_cost),
            _fail_neighbour_factor(fail_neighbour_factor)
    {
    }

    float const _fail_prob, // probability a computer starts failing
        _observe_prob, // probability admin observes correctly
        _reboot_success_rate, // probability that rebooting succeeds
        _reboot_cost, // cost of rebooting
        _fail_neighbour_factor; // fail probability induced by each failing neighbour
};

} // namespace domains

#endif // SYSADMINPARAMETERS
