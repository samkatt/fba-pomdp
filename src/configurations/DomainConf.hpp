#ifndef DOMAINCONF_HPP
#define DOMAINCONF_HPP

#include <boost/program_options.hpp>
#include <cstddef>
#include <string>

namespace configurations {
/**
 * @brief Configurations for domains
 **/
struct DomainConf
{
    std::string domain = "";

    size_t size = 0; // domain specific

    size_t height = 0; // domain specific
    size_t width  = 0; // domain specific

    /**
     * /brief adds options in this structure to descr
     **/
    void addOptions(boost::program_options::options_description* descr);

    /**
     * @brief validates its own parameters
     **/
    void validate() const;
};

} // namespace configurations

#endif // DOMAINCONF_HPP
