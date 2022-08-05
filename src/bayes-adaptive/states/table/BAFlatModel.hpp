#ifndef BAFLATMODEL_HPP
#define BAFLATMODEL_HPP

#include <map>
#include <memory>
#include <vector>

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "utils/random.hpp"
class State;
class Action;
class Observation;

namespace bayes_adaptive { namespace table {

/**
 * @brief A set of dirichlets that represents a belief over dynamics
 **/
class BAFlatModel
{
public:
    /**
     * @brief Initiates empty model (cannot directly be used)
     */
    BAFlatModel();

    /**
     * @brief initiates model counts with 0's
     **/
    explicit BAFlatModel(Domain_Size const* domain_size);

    /**
     * @brief initiates model with counts ph, psi
     **/
    BAFlatModel(
        std::shared_ptr<std::vector<float> const> phi,
        std::shared_ptr<std::vector<float> const> psi,
        Domain_Size const* domain_size);

    // allow shallow copies
    BAFlatModel(BAFlatModel const&);
    BAFlatModel(BAFlatModel&&) noexcept = default;
    BAFlatModel& operator=(BAFlatModel const&);
    BAFlatModel& operator=(BAFlatModel&&) = default;

    float& count(State const* s, Action const* a, State const* new_s);
    float& count(Action const* a, State const* new_s, Observation const* o);

    /**
     * @brief Returns the expected transition probabilities of state-action s-a
     *
     * @param s a current state
     * @param a the taken action
     *
     * @return a vector of length of state space size describing transition probabilities
     */
    std::vector<float> transitionExpectation(State const* s, Action const* a) const;

    /**
     * @brief Returns the expected observation probabilities of action-new_state a-new_s
     *
     * @param a the taken action
     * @param new_s a next state
     *
     * @return a vector of length of observation space size describing observation probabilities
     */
    std::vector<float> observationExpectation(Action const* a, State const* new_s) const;

    int sampleStateIndex(State const* s, Action const* a, rnd::sample::Dir::sampleMethod m) const;

    int sampleObservationIndex(
        Action const* a,
        State const* new_s,
        rnd::sample::Dir::sampleMethod m) const;

    double computeObservationProbability(
        Observation const* o,
        Action const* a,
        State const* new_s,
        rnd::sample::Dir::sampleMultinominal m) const;

    void incrementCountsOf(
        State const* s,
        Action const* a,
        Observation const* o,
        State const* new_s,
        float amount = 1);

    void logCounts() const;

private:
    Domain_Size const* _domain_size;

    std::shared_ptr<std::vector<float> const> _phi = {}; // base P(T)
    std::shared_ptr<std::vector<float> const> _psi = {}; // base P(O)

    // updated counts, to be updated over time
    std::map<unsigned int, std::vector<float>> _phi_cache = {};
    std::map<unsigned int, std::vector<float>> _psi_cache = {};

    double _cache_ratio_threshold = .1;

    /**
     * @brief returns count in phi
     **/
    float const& phi(int s, int a, int new_s) const;
    float& phi(int s, int a, int new_s);

    /**
     * @brief returns count in psi
     **/
    float const& psi(int a, int new_s, int o) const;
    float& psi(int a, int new_s, int o);

    /**
     * @brief returns index into updated storages
     **/
    unsigned int phi_cache_index(int s, int a) const;
    unsigned int psi_cache_index(int a, int new_s) const;

    void assertLegal(State const* s) const;
    void assertLegal(Observation const* o) const;
    void assertLegal(Action const* a) const;
};

}} // namespace bayes_adaptive::table

#endif // BAFLATMODEL_HPP
