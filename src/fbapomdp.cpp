#include <boost/program_options.hpp>
#include <string>

#include "easylogging++.h"

#include "configurations/ArgumentParser.hpp"
#include "configurations/FBAConf.hpp"

#include "bayes-adaptive/models/factored/FBAPOMDP.hpp"
#include "experiments/BAPOMDPExperiment.hpp"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    /**** parse arguments ****/
    configurations::FBAConf conf;
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
        VLOG(1) << "INFO " << "(" << conf.id << "): Starting FBAPOMDP experiment";

        auto const fbapomdp = factory::makeFBAPOMDP(conf);
        auto const res      = experiment::bapomdp::run(fbapomdp.get(), conf);

        std::ofstream f(conf.output_file);
        f << res << std::endl;

        LOG(INFO) << "(" << conf.id << "): Succesfully ran BAPOMDP experiment";

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
