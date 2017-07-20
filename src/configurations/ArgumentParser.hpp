#ifndef ARGUMENTPARSER_HPP
#define ARGUMENTPARSER_HPP

#include <boost/program_options.hpp>
#include "easylogging++.h"
#include "configurations/Conf.hpp"

namespace argument_parser
{
    namespace po = boost::program_options;
    configurations::Conf parse(int argc, char* argv[], po::options_description* descr);
}

#endif // ARGUMENTPARSER_HPP
