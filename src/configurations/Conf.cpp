#include "Conf.hpp"

#include <vector>

namespace configurations {

void Conf::addOptions(boost::program_options::options_description* descr)
{
    namespace po = boost::program_options;

    // clang-format off
    descr->add_options()
        (
        "help,h",
        po::bool_switch(&help)->default_value(help),
        "Print help options")
        (
        "verbose,v",
        po::value(&verbose)->default_value(verbose),
        "Verbose, 1: track episodes, 2: environment interactions, 3: algorithm summaries, 4 and "
        "up: algorithm details")
        (
        "output-file,f",
        po::value(&output_file)->default_value(output_file),
        "Verbose")
        (
        "runs",
        po::value(&num_runs)->default_value(num_runs),
        "Number of runs")
        (
        "horizon,H",
        po::value(&horizon)->default_value(horizon),
        "Horizon, number of steps per episode")
        (
        "discount,d",
        po::value(&discount)->default_value(discount),
        "Discount for future rewards")
        (
        "planner,P",
        po::value(&planner)->default_value(planner),
        "The planner to use (random, ts or po-uct)")
        (
        "belief,B",
        po::value(&belief)->default_value(belief),
        "The state estimator to use (point_estimate, rejection_sampling, importance_sampling, "
        "reinvigoration, mh-nips, cheating-reinvigoration, incubator or nested)")
        (
        "seed",
        po::value(&seed)->default_value(seed),
        "Global seed for all random samples")
        (
        "id",
        po::value(&id)->default_value(id),
        "The id to give this process (used in logging)");
    // clang-format on

    domain_conf.addOptions(descr);
    planner_conf.addOptions(descr);
    belief_conf.addOptions(descr);
}

void Conf::validate() const
{
    using boost::program_options::error;

    if (num_runs < 1)
    {
        throw error("please enter a positive number for runs");
    }

    if (horizon < 1)
    {
        throw error("please enter a positive number for horizon");
    }

    if (discount < 0 || discount > 1)
    {
        throw error("please enter a number between 0 and 1 for the discount");
    }

    if (planner != "random" && planner != "ts" && planner != "po-uct" && planner != "po-uct-abstraction")
    {
        throw error("Please enter a legit planner: random, ts, po-uct or po-uct-abstraction");
    }

    std::vector<std::string> valid_belief = {"point_estimate",
                                             "rejection_sampling",
                                             "importance_sampling",
                                             "reinvigoration",
                                             "mh-nips",
                                             "mh-within-gibbs",
                                             "cheating-reinvigoration",
                                             "incubator",
                                             "nested"};

    if (std::find(valid_belief.begin(), valid_belief.end(), belief) == valid_belief.end())
    {
        throw error(
            "please enter a legit state stimator: point, rejection_sampling, importance_sampling, "
            "mh-nips, mh-within-gibbs, (cheating-)reinvigoration or incubator, provided: "
            + belief);
    }

    planner_conf.validate();
    domain_conf.validate();
    belief_conf.validate(belief);
}

} // namespace configurations
