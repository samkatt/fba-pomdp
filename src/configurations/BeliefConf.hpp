#ifndef BELIEFCONF_HPP
#define BELIEFCONF_HPP

#include <boost/program_options.hpp>
#include <cstddef>
#include <string>

namespace configurations {

/**
 * @brief Configurations for the beliefs
 **/
struct BeliefConf
{

    size_t particle_amount = 100;
    size_t resample_amount = 0;

    double threshold = 0;

    std::string option = "";

    /**
     * /brief adds options in this structure to descr
     **/
    void addOptions(boost::program_options::options_description* descr);

    /**
     * @brief validates its own parameters based on the belief chosen
     **/
    void validate(std::string const& belief) const;
};

} // namespace configurations

#endif // BELIEFCONF_HPP
