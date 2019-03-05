#ifndef CONF_HPP
#define CONF_HPP

#include <boost/program_options.hpp>
#include <ctime>
#include <string>

#include "configurations/BeliefConf.hpp"
#include "configurations/DomainConf.hpp"
#include "configurations/PlannerConf.hpp"

namespace configurations {

struct Conf
{
    virtual ~Conf() = default;

    std::string seed = "";
    std::string id   = std::to_string(time(nullptr));

    bool help               = false;
    unsigned short verbose  = 0;
    std::string output_file = "results.txt";

    int num_runs    = 1;
    int horizon     = 10;
    double discount = .95;

    std::string planner = "po-uct";
    std::string belief  = "rejection_sampling";

    PlannerConf planner_conf = PlannerConf();
    DomainConf domain_conf   = DomainConf();
    BeliefConf belief_conf   = BeliefConf();

    /**
     * /brief adds options in this structure to descr
     **/
    virtual void addOptions(boost::program_options::options_description* descr);

    /**
     * @brief validates its own parameters
     **/
    virtual void validate() const;
};

} // namespace configurations

#endif // CONF_HPP
