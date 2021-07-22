#include "PlannerConf.hpp"

namespace configurations {

void PlannerConf::addOptions(boost::program_options::options_description* descr)
{
    namespace po = boost::program_options;

    // clang-format off
    descr->add_options()
        (
        "simulation-amount,s",
        po::value(&mcts_simulation_amount)->default_value(mcts_simulation_amount),
        "The number of simulations used by Tree Search methods (note you will need to set the "
        "planner accordingly)")
        (
        "sims-per-sample,s",
        po::value(&sims_per_sample)->default_value(sims_per_sample),
        "The number of simulations done per sampled particle by Tree Search methods (note you will need to set the "
        "planner accordingly)")
        (
        "remake_abstract_model",
        po::value(&remake_abstract_model)->default_value(remake_abstract_model),
        "Whether or not the abstract model is constructed every episode or just once")
        (
        "mcts-max-depth,max-depth",
        po::value(&mcts_max_depth)->default_value(mcts_max_depth),
        "The maximum depth Tree Search methods will search- will be set to the horizon if "
        "negative")
        (
        "exploration-constant,u",
        po::value(&mcts_exploration_const)->default_value(mcts_exploration_const),
        "The exploration constant used by UCB in PO-UCT");
    // clang-format on
}

void PlannerConf::validate() const
{
    using boost::program_options::error;

    if (mcts_simulation_amount < 0)
    {
        throw error("Please set a positive number of simulations");
    }

    if (sims_per_sample < 0 or sims_per_sample > mcts_simulation_amount)
    {
        throw error("Please set a number between 0 and mcts_simulation_amount");
    }

    if (mcts_max_depth < 0)
    {
        throw error("Tree Search max depth cannot be negative");
    }

    if (mcts_exploration_const < 0)
    {
        throw error("Please set a positive exploration constant");
    }
}

} // namespace configurations
