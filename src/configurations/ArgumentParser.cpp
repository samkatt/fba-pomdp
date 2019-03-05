#include "ArgumentParser.hpp"

#include "configurations/Conf.hpp"
#include "utils/random.hpp"

namespace argument_parser {
void parse(int argc, char* argv[], po::options_description* descr, configurations::Conf* conf)
{
    rnd::initiate();
    el::Configurations el_conf;

    /* setup logging */
    START_EASYLOGGINGPP(argc, argv);

    el_conf.setToDefault();
    el_conf.setGlobally(el::ConfigurationType::Format, "%level: \t%msg");
    el_conf.setGlobally(el::ConfigurationType::ToFile, "false");
    el_conf.set(el::Level::Verbose, el::ConfigurationType::Format, "V%vlevel: %fbase\t%msg");
    el_conf.set(
        el::Level::Info, el::ConfigurationType::Format, "%level: %msg (%datetime{%a %H:%m})");

    el::Loggers::reconfigureAllLoggers(el_conf);

    /* process arguments */
    conf->addOptions(descr);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, *descr), vm);
    po::notify(vm);

    if (conf->help)
    {
        LOG(INFO) << *descr;
        return;
    }

    if (conf->planner_conf.mcts_max_depth < 0)
    {
        conf->planner_conf.mcts_max_depth = conf->horizon;
    }

    conf->validate();

    // update verbosity of logging
    el::Loggers::setVerboseLevel(conf->verbose);
    el::Loggers::reconfigureAllLoggers(el_conf);

    if (!conf->seed.empty())
    {
        rnd::seed(conf->seed);
    }
}

} // namespace argument_parser
