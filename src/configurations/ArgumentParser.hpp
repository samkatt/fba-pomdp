#ifndef ARGUMENTPARSER_HPP
#define ARGUMENTPARSER_HPP

#include <boost/program_options.hpp>

#include "easylogging++.h"

namespace configurations {
struct Conf;
}

namespace argument_parser {

namespace po = boost::program_options;

/**
 * @brief processes arguments and configurations, modifies descr & conf
 **/
void parse(int argc, char* argv[], po::options_description* descr, configurations::Conf* conf);

} // namespace argument_parser

#endif // ARGUMENTPARSER_HPP
