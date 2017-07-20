#include <boost/program_options.hpp>
#include "easylogging++.h"

#include "configurations/ArgumentParser.hpp"
#include "configurations/Conf.hpp"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[]) 
{
    /**** parse arguments ****/
    configurations::Conf conf;
    boost::program_options::options_description descr("Options");

    try
    {
        conf = argument_parser::parse(argc, argv, &descr);

        if (conf.help)
        {
            LOG(INFO) << descr; 
            return 0;
        }

    } catch(boost::program_options::error& e)
    {
        LOG(ERROR) << e.what() << "\n\n" << descr;
        return 1;
    }

    /***** run program *****/
    LOG(INFO) << "hello world";
    return 0;
}
