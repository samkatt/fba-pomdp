#ifndef FBACONF_HPP
#define FBACONF_HPP

#include "configurations/BAConf.hpp"

#include <boost/program_options.hpp>
#include <string>

namespace configurations {

/**
 * @brief Configurations for FBA-POMDP experiments
 **/
struct FBAConf : public BAConf
{

    ~FBAConf() override = default;

    std::string structure_prior = "";

    void addOptions(boost::program_options::options_description* descr) override;
    void validate() const override;
};

} // namespace configurations
#endif // FBACONF_HPP
