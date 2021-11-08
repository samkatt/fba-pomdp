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
        "continuous-factored-tiger, gridworld, gridworldcoffee(big) or (random/centered)-collision-avoidance(-big))")
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
        "The width of the domain: collision avoidance grid size")
        (
        "abstraction",
        po::value(&abstraction)->default_value(abstraction),
        "Whether or not we use abstraction: gridworldcoffee, gridworldbuttons, and gridworldcoffeebig")
        (
        "store_statespace",
        po::value(&store_statespace)->default_value(store_statespace),
        "Whether or not the domain stores the state space in memory or uses a cache: gridworldcoffeebig");
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
                                              "gridworldcoffeebig",
                                              "gridworldbuttons",
                                              "random-collision-avoidance",
                                              "centered-collision-avoidance",
                                              "random-collision-avoidance-big",
                                              "centered-collision-avoidance-big"};

    if (std::find(valid_domains.begin(), valid_domains.end(), domain) == valid_domains.end())
    {
        throw error(
            "please enter a legit domain: dummy, linear_dummy, factored-dummy, episodic-tiger, "
            "continuous-tiger, agr, (boutilier-)coffee, episodic-factored-tiger, "
            "continuous-factored-tiger, independent-sysadmin, linear-sysadmin, gridworld, gridworldcoffee(big), or "
            "(random/centered)-collision-avoidance(-big). Given: "
            + domain);
    }

    // only accept if both size & factored domain is set, or neither
    if ((size != 0)
        ^ (domain == "factored-dummy" || domain == "independent-sysadmin"
           || domain == "linear-sysadmin" || domain == "episodic-factored-tiger"
           || domain == "continuous-factored-tiger" || domain == "gridworld"
           || domain == "gridworldcoffeebig" || domain == "gridworldbuttons"
           || domain == "random-collision-avoidance" || domain == "centered-collision-avoidance"
           || domain == "random-collision-avoidance-big" || domain == "centered-collision-avoidance-big"))
    {
        throw error(
            "--size and -D should be used together, either one is useless without the other (size "
            "&& factored-dummy, *-collision-avoidance(-big), sysadmin, gridworldbuttons, "
            "gridworldcoffeebig or a factored_tiger domain), "
            "provided domain "
            + domain + " with size " + std::to_string(size));
    }

    if ((height != 0 || width != 0) && domain != "random-collision-avoidance"
        && domain != "centered-collision-avoidance" && domain != "random-collision-avoidance-big"
           && domain != "centered-collision-avoidance-big")
    {
        throw error(
            "--height and --width can only be used with domain collision-avoidance(-big) (not " + domain
            + ")");
    }
}

} // namespace configurations
