#ifndef EPISODE_HPP
#define EPISODE_HPP

#include <cassert>
#include <vector>

#include "easylogging++.h"

#include "environment/Discount.hpp"
#include "environment/Return.hpp"
class Belief;
class Environment;
class Horizon;
class POMDP;
class Planner;

namespace episode {

/**
 * @brief The result of an episode
 *
 * Stores the return and number of timesteps (length)
 **/
struct Result
{
    Return ret;
    int length;
    int simulations;

    Result(Return r, int l, int sims) : ret(r), length(l), simulations(sims) {}
};

/**
 * @brief runs an episode and returns an evaluation
 *
 * NOTE: assumes belief has got a legit estimation
 */
Result
    run(Planner const& planner,
        Belief& belief,
        Environment const& env,
        POMDP const& simulator,
        Horizon const& h,
        Discount discount);

} // namespace episode

#endif // EPISODE_HPP
