#include "BeliefConf.hpp"

namespace configurations {

void BeliefConf::addOptions(boost::program_options::options_description* descr)
{

    namespace po = boost::program_options;

    // clang-format off
    descr->add_options()
        (
        "particle-amount",
        po::value(&particle_amount)->default_value(particle_amount),
        "The number of particles used by the particle filter (note you will need to set 'belief,E' "
        "to 'rejection_sampling')")
        (
        "resample-amount",
        po::value(&resample_amount)->default_value(resample_amount),
        "The number of particles that are resampled or reinvigorated (used by belief updates "
        "reinvigoration, cheating-reinvigoration and incubator)")
        (
        "threshold",
        po::value(&threshold)->default_value(threshold),
        "The threshold, with different functions for different beliefs:\n\tincubator: probabilty a "
        "particle needs before it is moved from the incubator to the real "
        "belief\n\tcheating-reinvigoration: log likelihood of the belief threshold before "
        "cheating\n\tmh:threshold: log likelihood of the belief threshold before cheating")
        (
        "belief-option",
        po::value(&option)->default_value(option),
        "An additional option to give to the belief. For mh-within-gibbs 'rs' for rejection "
        "sampling");
    // clang-format on
}

void BeliefConf::validate(std::string const& belief) const
{
    using boost::program_options::error;

    if ((resample_amount == 0)
        ^ (belief != "reinvigoration" && belief != "cheating-reinvigoration"
           && belief != "incubator"))
    {
        throw error(
            "You have set the resample amount (" + std::to_string(resample_amount)
            + "), but are not using one of the beliefs (" + belief
            + ") that use it: reinvigoration, cheating-reinvigoration and incubator");
    }

    if (!option.empty() && belief != "mh-within-gibbs")
    {
        throw error(
            "You have set the illegal belief_option '" + option + "' with belief " + belief + ".");
    }
}

} // namespace configurations
