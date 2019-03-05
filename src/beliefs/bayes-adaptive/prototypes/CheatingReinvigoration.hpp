#ifndef CHEATINGREINVIGORATION
#define CHEATINGREINVIGORATION

#include "beliefs/bayes-adaptive/BABelief.hpp"

#include <cstdlib>

#include "beliefs/particle_filters/FlatFilter.hpp"
#include "beliefs/particle_filters/WeightedFilter.hpp"

class BAPOMDP;
class POMDP;
class State;
class Action;
class Observation;
class FBAPOMDPState;

namespace beliefs { namespace bayes_adaptive { namespace prototypes {

/**
 * @brief A belief that cheats and reinvigorates with correct structures
 *
 * Will resample correct structures into the belief with probability
 * 'cheat_percentage' whenever the complete likelihood drops below
 * some threshold.
 **/
class CheatingReinvigoration : public BABelief
{
public:
    CheatingReinvigoration(size_t size, size_t cheat_amount, double resample_threshold);

    /*** BABelief interface ***/
    void resetDomainStateDistribution(BAPOMDP const& bapomdp) final;

    /*** Belief interface ***/
    void initiate(POMDP const& domain) final;
    void free(POMDP const& domain) final;
    State const* sample() const final;
    void updateEstimation(Action const* a, Observation const* o, POMDP const& domain) final;

private:
    // params
    size_t _size;
    size_t _cheat_amount;
    double _resample_threshold;

    double _likelihood = 1;

    // beliefs
    WeightedFilter<FBAPOMDPState const*> _belief{};
    FlatFilter<FBAPOMDPState const*> _correct_structured_belief{};

    /**
     * @brief replaces random graphs in the belief with correctly structured ones
     **/
    void cheat(POMDP const& pomdp);
};

}}} // namespace beliefs::bayes_adaptive::prototypes

#endif // CHEATINGREINVIGORATION
