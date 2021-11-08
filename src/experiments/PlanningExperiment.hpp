#ifndef PLANNINGEXPERIMENT_HPP
#define PLANNINGEXPERIMENT_HPP

#include "boost/timer/timer.hpp"
#include <vector>

#include "easylogging++.h"

#include "utils/Statistic.hpp"

namespace configurations {
struct Conf;
}

namespace experiment { namespace planning {

/*
 * @brief The result of an planning experiment
 **/
struct Result : public el::Loggable
{
    utils::Statistic episode_return = utils::Statistic(), episode_duration = utils::Statistic();

    void log(el::base::type::ostream_t& os) const final;
};

/**
 * @brief runs an experiment for planners (no learning)
 **/
Result run(configurations::Conf const& conf);

}} // namespace experiment::planning

#endif // PLANNINGEXPERIMENT_HPP
