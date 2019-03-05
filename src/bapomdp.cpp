#include <boost/program_options.hpp>
#include <string>

#include "easylogging++.h"

#include "configurations/ArgumentParser.hpp"
#include "configurations/BAConf.hpp"

#include "bayes-adaptive/models/table/BAPOMDP.hpp"
#include "experiments/BAPOMDPExperiment.hpp"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    /**** parse arguments ****/
    configurations::BAConf conf;
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
        LOG(INFO) << "(" << conf.id << "): Starting BAPOMDP experiment";

        auto const bapomdp = factory::makeTBAPOMDP(conf);
        auto const res     = experiment::bapomdp::run(bapomdp.get(), conf);

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
