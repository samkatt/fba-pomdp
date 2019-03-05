#ifndef BACONF_HPP
#define BACONF_HPP

#include <boost/program_options.hpp>

#include "configurations/Conf.hpp"
#include "utils/random.hpp"

namespace configurations {

/**
 * @brief configurations for the learning experiments
 **/
struct BAConf : public Conf
{

    int num_episodes = 1;

    float noise        = 0;
    float counts_total = 10000;

    rnd::sample::Dir::SAMPLETYPE bayes_sample_method = rnd::sample::Dir::Expected;

    void addOptions(boost::program_options::options_description* descr) override;
    void validate() const override;
};

} // namespace configurations

#endif // BACONF_HPP
