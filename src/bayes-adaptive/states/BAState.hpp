#ifndef BASTATE_HPP
#define BASTATE_HPP

#include "environment/State.hpp"

#include "utils/random.hpp"

class Action;
class Observation;

/**
 * @brief A state in bayes-adaptive frameworks
 *
 * Interface for the various possible implementations of states
 * in Bayes-Adaptive frameworks such as BAPOMDP and FBAPOMDP
 **/
class BAState : public State
{
public:
    State const* _domain_state;

    explicit BAState(State const* domain_state);

    // deny shallow copies
    BAState(const BAState&) = delete;
    BAState& operator=(const BAState&) = delete;

    /**
     * @brief virtual copy constructor idiom
     **/
    virtual BAState* copy(State const*) const = 0;

    /**
     * @brief returns the observation probability of o given <a,s'>
     **/
    virtual double computeObservationProbability(
        Observation const* o,
        Action const* a,
        State const* new_s,
        rnd::sample::Dir::sampleMultinominal m) const = 0;

    /**
     * @brief samples a state index for <s,a>
     **/
    virtual std::string sampleStateIndex(State const* s, Action const* a, rnd::sample::Dir::sampleMethod m)
        const = 0;

    /**
     * @brief samples an observation index for <s,a,s'>
     **/
    virtual int sampleObservationIndex(
        Action const* a,
        State const* new_s,
        rnd::sample::Dir::sampleMethod m) const = 0;

    /**
     * @brief updates counts associated with <s,a,s',o>
     **/
    virtual void incrementCountsOf(
        State const* s,
        Action const* a,
        Observation const* o,
        State const* new_s,
        float amount = 1) = 0;

    /**
     * @brief logs the whole model
     **/
    virtual void logCounts() const = 0;

    /*** state interface ***/
    std::string index() const final;
    void index(std::string) final;
    std::vector<int> getFeatureValues() const final;
};

#endif // BASTATE_HPP
