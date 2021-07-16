#include "FBAConf.hpp"

namespace configurations {

void FBAConf::addOptions(boost::program_options::options_description* descr)
{

    namespace po = boost::program_options;

    // clang-format off
    descr->add_options()
        (
         "structure-prior",
         po::value(&structure_prior)->default_value(structure_prior),
         "The prior on the structure of the (uncertain parts of the) dynamics. If not given, "
         "assumes prior knowledge on the true model, otherwise accepts 'uniform', 'match-counts', "
         "match-uniform' and 'fully-connected'. These respectively apply a uniform distribution, a "
         "distribution that simply allows to represent the prior counts and the one that represents "
         "the prior counts and uniform over the other edges. 'fully-connected' assumed the true "
         "model is fully connected."
        );
    // clang-format on

    BAConf::addOptions(descr);
}

void FBAConf::validate() const
{
    using boost::program_options::error;

    if (structure_prior == "fully-connected")
    {
        if (!(domain_conf.domain == "episodic-factored-tiger"
              || domain_conf.domain == "continuous-factored-tiger"
              || domain_conf.domain == "random-collision-avoidance"
              || domain_conf.domain == "centered-collision-avoidance"))
        {
            throw error(
                "the only legit domains for 'fully-connected' noise option is one of the factored "
                "tigers domain, you provided "
                + domain_conf.domain
                + " with instead, and do not forget to flag factored structure noise on");
        }
    }

    std::vector<std::string> const fpomdps = {
        "factored-dummy",
        "episodic-factored-tiger",
        "continuous-factored-tiger",
        "independent-sysadmin",
        "linear-sysadmin",
        "random-collision-avoidance",
        "centered-collision-avoidance",
        "gridworld",
        "gridworldcoffee",
        "gridworldcoffeebig"
    };

    if (std::find(fpomdps.begin(), fpomdps.end(), domain_conf.domain) == fpomdps.end())
    {
        throw error(
            "please enter a factored domain if you wish to use factored representations: "
            "factored-dummy, a (factored) tiger domain, independent/linear-sysadmin, "
            " gridworldcoffee(big) or random/centered-collision-avoidance, you provided: "
            + domain_conf.domain);
    }

    BAConf::validate();
}

} // namespace configurations
