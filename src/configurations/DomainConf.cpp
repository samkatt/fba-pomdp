#include "DomainConf.hpp"

namespace configurations {

void DomainConf::addOptions(boost::program_options::options_description* descr)
{

    namespace po = boost::program_options;

    // clang-format off
    descr->add_options()
        (
        "domain,D",
        po::value(&domain),
        "The domain to experiment in (dummy, linear_dummy, factored-dummy, episodic-tiger, "
        "continuous-tiger, agr, (boutilier-)coffee, episodic-factored-tiger, "
        "continuous-factored-tiger, gridworld, gridworldcoffee or (random/centered)-collision-avoidance)")
        (
        "size",
        po::value(&size)->default_value(size),
        "The size of the domain: factored-dummy: size of 1 dimension")
        (
        "height",
        po::value(&height)->default_value(height),
        "The height of the domain: collision avoidance grid size")
        (
        "width",
        po::value(&width)->default_value(width),
        "The width of the domain: collision avoidance grid size");
    // clang-format on
}

void DomainConf::validate() const
{
    using boost::program_options::error;

    std::vector<std::string> valid_domains = {"dummy",
                                              "linear_dummy",
                                              "factored-dummy",
                                              "episodic-tiger",
                                              "continuous-tiger",
                                              "agr",
                                              "coffee",
                                              "boutilier-coffee",
                                              "independent-sysadmin",
                                              "linear-sysadmin",
                                              "episodic-factored-tiger",
                                              "continuous-factored-tiger",
                                              "gridworld",
                                              "gridworldcoffee",
                                              "random-collision-avoidance",
                                              "centered-collision-avoidance"};

    if (std::find(valid_domains.begin(), valid_domains.end(), domain) == valid_domains.end())
    {
        throw error(
            "please enter a legit domain: dummy, linear_dummy, factored-dummy, episodic-tiger, "
            "continuous-tiger, agr, (boutilier-)coffee, episodic-factored-tiger, "
            "continuous-factored-tiger, independent-sysadmin, linear-sysadmin, gridworld, gridworldcoffee, or "
            "(random/centered)-collision-avoidance. Given: "
            + domain);
    }

    // only accept if both size & factored domain is set, or neither
    if ((size != 0)
        ^ (domain == "factored-dummy" || domain == "independent-sysadmin"
           || domain == "linear-sysadmin" || domain == "episodic-factored-tiger"
           || domain == "continuous-factored-tiger" || domain == "gridworld"
           || domain == "random-collision-avoidance" || domain == "centered-collision-avoidance"))
    {
        throw error(
            "--size and -D should be used together, either one is useless without the other (size "
            "&& factored-dummy, *-collision-avoidance, sysadmin or a factored_tiger domain), "
            "provided domain "
            + domain + " with size " + std::to_string(size));
    }

    if ((height != 0 || width != 0) && domain != "random-collision-avoidance"
        && domain != "centered-collision-avoidance")
    {
        throw error(
            "--height and --width can only be used with domain collision-avoidance (not " + domain
            + ")");
    }
}

} // namespace configurations
