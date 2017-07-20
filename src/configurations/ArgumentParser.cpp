#include "configurations/ArgumentParser.hpp"


namespace argument_parser
{

    configurations::Conf parse(int argc, char* argv[], po::options_description* descr)
    {
        configurations::Conf conf;

        descr->add_options()
            ("help,h",    po::value<bool>(&conf.help),   "Print help options")
            ("verbose,v", po::value<unsigned short>(&conf.verbose), "Verbose");

        po::variables_map vm; 
        po::store(po::parse_command_line(argc, argv, *descr), vm);
        po::notify(vm); 

        // easylogging
        START_EASYLOGGINGPP(argc, argv);

        el::Configurations el_conf;
        el_conf.setToDefault();
        el_conf.setGlobally(el::ConfigurationType::Format, "%level: %msg");
        el_conf.set(el::Level::Verbose, el::ConfigurationType::Format, "v%vlevel: %msg");
        el::Loggers::setVerboseLevel(conf.verbose);

        el::Loggers::reconfigureAllLoggers(el_conf);

        return conf;
    }

}
