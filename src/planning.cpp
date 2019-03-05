#include <boost/program_options.hpp>
#include <string>

#include "easylogging++.h"

#include "configurations/ArgumentParser.hpp"
#include "configurations/Conf.hpp"

#include "experiments/PlanningExperiment.hpp"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    /**** parse arguments ****/
    configurations::Conf conf;
    boost::program_options::options_description descr("Options");

    try
    {
        argument_parser::parse(argc, argv, &descr, &conf);

        if (conf.help)
        {
            return 0;
        }

    } catch (boost::program_options::error& e)
    {
        LOG(ERROR) << e.what();
        return 1;
    }

    /***** run program *****/
    try
    {
        LOG(INFO) << "(" << conf.id << "): Starting planning experiment";

        auto const res = experiment::planning::run(conf);

        std::ofstream f(conf.output_file);
        f << res << std::endl;

        LOG(INFO) << "(" << conf.id << "): Succesfully ran planning experiment";

    } catch (char const* e)
    {
        LOG(ERROR) << e;
        return 1;
    } catch (std::string const& e)
    {
        LOG(ERROR) << e;
        return 1;
    }

    return 0;
}
