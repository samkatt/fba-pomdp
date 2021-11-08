#ifndef BABNMODEL_HPP
#define BABNMODEL_HPP

#include <vector>

#include "bayes-adaptive/states/factored/DBNNode.hpp"

#include "utils/random.hpp"

struct Domain_Size;
struct Domain_Feature_Size;

class Action;
class State;
class Observation;

namespace bayes_adaptive { namespace factored {

/**
 * @brief Bayesian Network based model of the dynamics in a BAPOMDP
 **/
class BABNModel
{
public:
    std::vector<DBNNode> _transition_nodes  = {};
    std::vector<DBNNode> _observation_nodes = {};
    /**
     * @brief the step sizes used to project/index from features to indices
     **/
    struct Indexing_Steps
    {
        // cppcheck-suppress passedByValue
        Indexing_Steps(std::vector<int> t, std::vector<int> o) : T(std::move(t)), O(std::move(o)) {}

        std::vector<int> T, O;
    };

    struct Structure
    {
        // [a][f]  contains parents of feature f under action a
        std::vector<std::vector<std::vector<int>>> T = {};
        std::vector<std::vector<std::vector<int>>> O = {};

        /**
         * @brief flips 1 edges randomly (adds an edge of up to edge_range)
         *
         * Note this means that, with 50/50, we either
         * remove an edge, or add one (that does not exist yet)
         **/
        static void flip_random_edge(std::vector<int>* edges, int edge_range);
    };

    BABNModel();

    BABNModel(
        Domain_Size const* domain_size,
        Domain_Feature_Size const* domain_feature_size,
        Indexing_Steps const* step_sizes);

    BABNModel(
        Domain_Size const* domain_size,
        Domain_Feature_Size const* domain_feature_size,
        Indexing_Steps const* step_sizes,
        std::vector<DBNNode> transition_nodes,
        std::vector<DBNNode> observation_nodes);

    // allow shallow copies
    BABNModel(BABNModel const&) = default;
    BABNModel(BABNModel&&)      = default;
    BABNModel& operator=(BABNModel const&) = default;
    BABNModel& operator=(BABNModel&&) = default;

    std::string sampleStateIndex(State const* s, Action const* a, rnd::sample::Dir::sampleMethod m) const;
    std::string sampleStateIndexThroughAbstraction(const std::vector<int> *parent_values) const;
    int sampleObservationIndex(
        Action const* a,
        State const* new_s,
        rnd::sample::Dir::sampleMethod m) const;

    const Domain_Size * domainSize() const;
    const Domain_Feature_Size * domainFeatureSize() const;
    const Indexing_Steps * stepSizes() const;

    /**
     * @brief returns a table of probabilities [s][a][new_s]
     **/
    std::vector<std::vector<std::vector<float>>> flattenT() const;

    /**
     * @brief returns a table of probabilities [s][a][new_s]
     **/
    std::vector<std::vector<std::vector<float>>> flattenO() const;

    void incrementCountsOf(
        State const* s,
        Action const* a,
        Observation const* o,
        State const* new_s,
        float amount = 1);

    double computeObservationProbability(
        Observation const* o,
        Action const* a,
        State const* s,
        rnd::sample::Dir::sampleMultinominal sampleMultinominal) const;

    /**
     * @brief calculates the BD score of a graph given its prior and posterior
     **/
    double LogBDScore(BABNModel const& prior) const;

    /**
     * @brief takes a structure with fewer connections and returns a marginalized-out model
     **/
    BABNModel marginalizeOut(Structure new_structure) const;

    /**
     * @brief takes a abstraction with fewer features and marginalizes out the other features
     **/
    BABNModel abstract(int abstraction, Structure structure, const Domain_Size *ds, const Domain_Feature_Size *dfs,
                       const Indexing_Steps *is, bool normalize) const;
    BABNModel abstract(std::vector<int> abstraction_set, Structure structure, const Domain_Size *ds, const Domain_Feature_Size *dfs,
                       const Indexing_Steps *is, bool normalize) const;

    void abstractionNormalizeCounts(BABNModel prior, BABNModel prior_normalized);

    /**
     * @brief returns the structure of its nodes
     *
     * NOTE: <generates> this on the spot and is slightly expensive
     **/
    bayes_adaptive::factored::BABNModel::Structure structure() const;

    auto copyT() const -> decltype(_transition_nodes) const& { return _transition_nodes; }
    auto copyO() const -> decltype(_observation_nodes) const& { return _observation_nodes; }

    /**
     * @brief sets the parents for a transition node
     **/
    void resetTransitionNode(Action const* a, int state_feature, std::vector<int> parents);

    /**
     * @brief sets the parents for a transition node
     **/
    void resetObservationNode(Action const* a, int observation_feature, std::vector<int> parents);

    DBNNode& transitionNode(Action const* a, int feature);
    DBNNode const& transitionNode(Action const* a, int feature) const;

    DBNNode& observationNode(Action const* a, int feature);
    DBNNode const& observationNode(Action const* a, int feature) const;

    void log() const;

/**
 * @brief returns the state feature values associated with provided state
 **/
std::vector<int> stateFeatureValues(State const* s) const;

/**
 * @brief returns the observation feature values associated with provided observation
 **/
std::vector<int> observationFeatureValues(Observation const* o) const;

    void incrementCountsOfAbstract(const Action *a, const Observation *o, float amount,
                                   const std::vector<int> &parent_values,
                                   std::vector<int> state_feature_values, std::vector<int> vector);

private:
    Domain_Size const* _domain_size;
    Domain_Feature_Size const* _domain_feature_size;

    Indexing_Steps const* _step_sizes;

    void assertLegal(State const* s) const;
    void assertLegal(Observation const* o) const;
    void assertLegal(Action const* a) const;
    void assertLegalStateFeature(int f) const;
    void assertLegalObservationFeature(int f) const;
};

}} // namespace bayes_adaptive::factored

#endif // BABNMODEL_HPP
