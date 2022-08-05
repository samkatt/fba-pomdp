#ifndef DBNNODE_HPP
#define DBNNODE_HPP

#include <vector>

#include "utils/random.hpp"

#include <cstddef>

#include <utility>

/**
 * @brief A node in a Dynamic Bayesian Network
 **/
class DBNNode
{

public:
    /**
     * @brief constructs a node from graph sizes with specific parents and output range
     *
     * graph_input_size is the size of the values each potential parent node can take
     * parent_nodes are the actual parents of this node
     * output_size is the size of the output (range) of this node
     **/
    DBNNode(
        std::vector<int> const* graph_input_size,
        std::vector<int> parent_nodes,
        int output_size);

    // allow shallow copies
    DBNNode(DBNNode const&)            = default;
    DBNNode& operator=(DBNNode const&) = default;

    /**
     * @brief returns the range, or the number of outputs, this node can have
     **/
    size_t range() const;

    /**
     * @brief returns the number of parameters (counts, the size of the CPTs) of this node
     **/
    size_t numParams() const;

    /**
     * @brief returns parents of node
     **/
    std::vector<int> const* parents() const;

    /**
     * @brief returns the BD score given the prior
     **/
    double LogBDScore(DBNNode const& prior) const;

    /**
     * @brief returns the expected probabilities conditioned on node_input
     **/
    std::vector<float> expectation(std::vector<int> const& node_input) const;

    /**
     * @brief takes graph input values and samples a value for the node
     **/
    int sample(std::vector<int> const& node_input, rnd::sample::Dir::sampleMethod m) const;

    /**
     * @brief returns a multinominal over the output range associated with graph_input according to
     *sampling method m
     **/
    std::vector<float> sampleMultinominal(
        std::vector<int> const& node_input,
        rnd::sample::Dir::sampleMultinominal sampleMethod) const;

    /**
     * @brief increments the counts associated with the provided transition <parent_values> to
     *<value> of the node
     **/
    void increment(std::vector<int> const& node_input, int node_output, float amount = 1);

    /**
     * @brief returns the count of the <X,a,X'> cpt
     */
    float& count(std::vector<int> const& node_input, int node_output);

    /**
     * @brief sets the counts for a particular set of parent values
     */
    void setDirichletDistribution(std::vector<int> const& node_input, std::vector<float> counts);

    /**
     * @brief instantiates a node from its CPTS according to given parents
     **/
    DBNNode marginalizeOut(std::vector<int> new_parents) const;

    /**
     * @brief writes of its cpts in LOG(INFO)
     **/
    void logCPTs() const;

private:
    /**
     * @brief range of values this node can take on
     **/
    int _output_size;

    /**
     * @brief which features (by id) are its parents
     **/
    std::vector<int> _parent_nodes;

    /**
     * @brief a mapping from parent id to it's size
     **/
    std::vector<int> _parent_sizes;

    /**
     * @brief the actual counts that reprents the dirichlet distributions governing the cpts
     **/
    std::vector<float> _cpts = {};

    /**
     * @brief the range of all nodes in the graph (also those not connected to *this*)
     *
     * Is a const reference because we do not want to actually store it physically.
     * It is owned by whoever created this object
     *
     * NOTE: assumes the owner -- fbapomdpstate, fbapomdp -- outlives us
     **/
    std::vector<int> const* _graph_range;

    /**
     * @brief many internal functions require to accumulate parent values, this is a memory
     *efficient way of doing that
     **/
    static std::vector<int> _parent_value_holder;

    /**
     * @brief returns the index into the cpt given parent values and desired output
     **/
    int cptIndex(std::vector<int> const& node_input, int node_output) const;

    /**
     * @brief takes graph values as input and extract the parent values
     **/
    void parentValues(std::vector<int> const& node_input, std::vector<int>* parent_values) const;
};

#endif // DBNNODE_HPP
