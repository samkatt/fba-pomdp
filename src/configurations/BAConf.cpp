#include "BAConf.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace rnd { namespace sample { namespace Dir {

/**
 * @brief provide streaming functionality to the SAMPLETYPE
 *
 * @param in: the insteam (contains input from user)
 * @param sample_type (out): what sample_type is given in
 *
 * required for program_options to allow SAMPLETYPE as type
 *
 * @return the input instream (side: sets sample_type)
 */
std::istream& operator>>(std::istream& in, SAMPLETYPE& sample_type)
{
    std::string token;
    in >> token;

    if (token == "regular")
    {
        sample_type = rnd::sample::Dir::SAMPLETYPE::Regular;
    } else if (token == "expected")
    {
        sample_type = rnd::sample::Dir::SAMPLETYPE::Expected;
    } else
    {
        in.setstate(std::ios_base::failbit);
    }

    return in;
}

}}} // namespace rnd::sample::Dir

namespace configurations {

void BAConf::addOptions(boost::program_options::options_description* descr)
{
    namespace po = boost::program_options;

    // clang-format off
    descr->add_options()
        (
        "episodes",
        po::value(&num_episodes)->default_value(num_episodes), 
        "Number of episodes")
        (
        "dirichlet_sampling_method",
        po::value<rnd::sample::Dir::SAMPLETYPE>(&bayes_sample_method) ->default_value(bayes_sample_method),
        "Whether to use regular or expected dirichlet sampling ('regular' or (default) " "'expected')")
        (
         "noise",
         po::value(&noise)->default_value(noise),
         "The noise parameter")
        (
        "counts-total,C",
        po::value(&counts_total)->default_value(counts_total),
        "Total number of initial counts in a dirichlet distibution that needs to be learned");
    // clang-format on

    // add all options inherited from conf
    Conf::addOptions(descr);
}

void BAConf::validate() const
{
    std::vector<std::string> const valid_bapomdps = {"dummy",
                                                     "factored-dummy",
                                                     "episodic-tiger",
                                                     "continuous-tiger",
                                                     "episodic-factored-tiger",
                                                     "continuous-factored-tiger",
                                                     "independent-sysadmin",
                                                     "linear-sysadmin",
                                                     "random-collision-avoidance",
                                                     "centered-collision-avoidance",
                                                     "gridworld"};

    Conf::validate();

    using boost::program_options::error;
    if (num_episodes < 1)
    {
        throw error("please enter a positive number for episode");
    }

    if (std::find(valid_bapomdps.begin(), valid_bapomdps.end(), domain_conf.domain)
        == valid_bapomdps.end())
    {
        throw error(
            "please enter a Bayes-Adapative domain (dummy, episodic-tiger, continuous-tiger, "
            "episodic-factored-tiger, continuous-factored-tiger, independent-sysadmin, or "
            "linear-sysadmin)");
    }

    if (counts_total < 1)
    {
        throw error("please enter a positive total amount of counts");
    }

    if (bayes_sample_method != rnd::sample::Dir::Regular
        && bayes_sample_method != rnd::sample::Dir::Expected)
    {
        throw error(
            "please enter either 0 or 1 for bayes_sample_method, given: "
            + std::to_string(bayes_sample_method));
    }
}

} // namespace configurations
