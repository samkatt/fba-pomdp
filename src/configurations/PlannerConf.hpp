#ifndef PLANNERCONF_HPP
#define PLANNERCONF_HPP

#include <boost/program_options.hpp>
#include <cstddef>
#include <string>

namespace configurations {

/**
 * @brief Configurations for planners
 **/
struct PlannerConf
{

    int mcts_simulation_amount    = 1000;
    int milliseconds_thinking     = 0;
    int mcts_max_depth            = -1;
    double mcts_exploration_const = 100;
    int remake_abstract_model     = 0;
    int update_abstract_model    = 0;
    int update_abstract_model_normalized = 0;
    int abstraction_k = 0;


    /**
     * /brief adds options in this structure to descr
     **/
    void addOptions(boost::program_options::options_description* descr);

    /**
     * @brief validates its own parameters
     **/
    void validate() const;

};

} // namespace configurations
#endif // PLANNERCONF_HPP
