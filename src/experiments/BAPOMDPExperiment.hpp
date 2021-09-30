#ifndef BAPOMDPEXPERIMENT_HPP
#define BAPOMDPEXPERIMENT_HPP

#include "boost/timer.hpp"
#include <vector>

#include "easylogging++.h"

#include "utils/Statistic.hpp"
class BAPOMDP;
namespace configurations {
struct BAConf;
}

namespace experiment { namespace bapomdp {

/**
 * @brief Contains results of a learning experiment
 **/
struct Result : public el::Loggable
{
    /**
     * @brief The result of an episode
     **/
    struct episode_result
    {
        utils::Statistic ret = utils::Statistic(), duration = utils::Statistic(), simulations = utils::Statistic();
    };

    std::vector<episode_result> r;

    explicit Result(int size);

    void log(el::base::type::ostream_t& os) const final;
};

/**
 * @brief An experiment that tests planners & learners
 **/
Result run(BAPOMDP const* bapomdp, configurations::BAConf const& conf);

}} // namespace experiment::bapomdp

#endif // BAPOMDPEXPERIMENT_HPP
