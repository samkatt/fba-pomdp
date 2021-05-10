#include "DBNNode.hpp"

#include <cassert>
#include <string>

#include <algorithm> // for std::transform
#include <functional> // for std::plus

#include "easylogging++.h"

#include "utils/index.hpp"
#include "utils/random.hpp"

std::vector<int> DBNNode::_parent_value_holder;

DBNNode::DBNNode(
    std::vector<int> const* graph_input_size,
    // cppcheck-suppress passedByValue
    std::vector<int> parent_nodes,
    int output_size) :
        _output_size(output_size),
        _parent_nodes(std::move(parent_nodes)),
        _parent_sizes(_parent_nodes.size()),
        _graph_range(graph_input_size)
{
    assert(_parent_nodes.size() <= graph_input_size->size());

    auto max_parent_values = 1;

    // set parent sizes & initiate cpts
    for (size_t i = 0; i < _parent_nodes.size(); ++i)
    {
        _parent_sizes[i] = graph_input_size->at(_parent_nodes[i]);
        max_parent_values *= _parent_sizes[i];
    }

    _cpts = std::vector<float>(max_parent_values * _output_size);
}

DBNNode DBNNode::marginalizeOut(std::vector<int> new_parents) const
{
    assert(new_parents.size() <= _parent_nodes.size());

    // easy corner case: exactly same parents need no marginalizing
    if (new_parents == _parent_nodes)
    {
        return *this;
    }

    auto new_node = DBNNode(_graph_range, std::move(new_parents), _output_size);

    // base case, no parents
    if (_parent_nodes.empty())
    {
        new_node.setDirichletDistribution({}, _cpts);
        return new_node;
    }

    // loop over all our distributions and add them to our new node
    size_t cpt_start   = 0;
    auto parent_values = std::vector<int>(_parent_nodes.size());
    do
    {

        // add our dirichlet counts to the new node
        // the new node transforms our parent values into
        auto new_cpt_start = new_node.cptIndex(parent_values, 0);
        std::transform(
            &new_node._cpts[new_cpt_start],
            &new_node._cpts[new_cpt_start + _output_size],
            &_cpts[cpt_start],
            &new_node._cpts[new_cpt_start],
            std::plus<float>());

        cpt_start += _output_size;
    } while (!indexing::increment(parent_values, _parent_sizes));

    assert(cpt_start == _cpts.size());

    return new_node;
}

double DBNNode::LogBDScore(DBNNode const& prior) const
{
    assert(_output_size == prior._output_size);
    assert(_parent_nodes.size() == prior._parent_nodes.size());

    for (size_t i = 0; i < _parent_nodes.size(); ++i)
    { assert(_parent_nodes[i] == prior._parent_nodes[i]); }

    assert(_cpts.size() == prior._cpts.size());

    double bd_score = 0;

    // loop over all dirichlet distributions
    // takes advantage of the knowledge of the distribution layout
    for (size_t distr_start = 0; distr_start < _cpts.size(); distr_start += _output_size)
    {
        double distr_total       = 0;
        double prior_distr_total = 0;

        for (auto v = 0; v < _output_size; ++v)
        {
            auto i = distr_start + v;

            distr_total += _cpts[i];
            prior_distr_total += prior._cpts[i];

            bd_score += rnd::math::logGamma(_cpts[i]) - rnd::math::logGamma(prior._cpts[i]);
        }

        bd_score += rnd::math::logGamma(prior_distr_total) - rnd::math::logGamma(distr_total);
    }

    return bd_score;
}

std::vector<float> DBNNode::expectation(std::vector<int> const& node_input) const
{
    return rnd::sample::Dir::expectedMult(&_cpts[cptIndex(node_input, 0)], _output_size);
}

void DBNNode::increment(std::vector<int> const& node_input, int node_output, float amount)
{
    _cpts[cptIndex(node_input, node_output)] += amount;
}

void DBNNode::setDirichletDistribution(
    std::vector<int> const& node_input,
    std::vector<float> counts)
{
    assert(counts.size() == (size_t)_output_size);

    std::move(counts.begin(), counts.end(), &_cpts[cptIndex(node_input, 0)]);
}

float& DBNNode::count(std::vector<int> const& node_input, int node_output)
{
    return _cpts[cptIndex(node_input, node_output)];
}

size_t DBNNode::range() const
{
    return _output_size;
}

size_t DBNNode::numParams() const
{
    return _cpts.size();
}

std::vector<int> const* DBNNode::parents() const
{
    return &_parent_nodes;
}

int DBNNode::sample(std::vector<int> const& node_input, rnd::sample::Dir::sampleMethod m) const
{
    // sample from dirichlet starting from joint index for _ouput_size counts
    // XXX FAO let me see if I git this right:
    //     - output_size = the number of values that this node can take
    //     - cptIndex(node_input, 0) = the index (within _cpts) of the probability table for parent values=node_input
    //     - 0 here means the first entry in this table
    //     - and this table is of _output_size 
    // E.g.,  to find the index of  P( X | <y1=2, y2=42> ), we ask where P( X=0 | <y1=2, y2=42> ) lies and then take the next _output_size entries
    return m(&_cpts[cptIndex(node_input, 0)], _output_size);
}

std::vector<float> DBNNode::sampleMultinominal(
    std::vector<int> const& node_input, //e.g. parent values
    rnd::sample::Dir::sampleMultinominal sampleMethod //e.g. sample from expected model - cf random.cpp
    ) const
{
    return sampleMethod(&_cpts[cptIndex(node_input, 0)], _output_size);
}

int DBNNode::cptIndex(std::vector<int> const& node_input, int node_output) const
{
    assert(node_output < _output_size);

    // base case: if we have no parents then our cpt counts consists
    // of a single dirichlet distribution and we can safely return the output's
    // count
    if (_parent_nodes.empty())
    {
        return node_output;
    }

    // Now we are dealing with two possibilities:
    // input is either the parents input, or the graph input
    //
    // we should really make a proper differentiation between them, but for now
    // there is no clear way of implementing this.
    //
    // Throughout the rest of the code for DBBNodes we did not assume whether
    // we are dealing with parent input or graph- it has all been delayed until
    // here.

    // easiest case: input size is our parents, and we can safely assume
    // the input is our parents input (or this node is fully connected
    // and it is the graphs input, which comes down to the same)
    if (node_input.size() == _parent_nodes.size())
    {
        return indexing::project(node_input, _parent_sizes) * _output_size + node_output;
    }

    // option 2: if input != parents, then we should be dealing with the
    // whole graph input (which is at least bigger than our parents at this point)
    parentValues(node_input, &_parent_value_holder);
    return indexing::project(_parent_value_holder, _parent_sizes) * _output_size + node_output;
}

// TODO Sammie, what does this function do?
// It populates the parent values. Node input is the DBN input
void DBNNode::parentValues(std::vector<int> const& node_input, std::vector<int>* parent_values)
    const
{
    assert(node_input.size() > _parent_sizes.size());

    // make sure none of the input exceeds the allowed values
    for (size_t i = 0; i < _parent_nodes.size(); ++i)
    { assert(node_input[_parent_nodes[i]] < _parent_sizes[i]); }

    // populate parent values
    parent_values->clear();
    for (auto const& p : _parent_nodes) { parent_values->emplace_back(node_input[p]); }
}

void DBNNode::logCPTs() const
{
    // corner case: no parents
    if (_parent_nodes.empty())
    {
        std::string descr = std::to_string(_cpts[0]);
        for (auto output = 1; output < _output_size; ++output)
        { descr += "," + std::to_string(_cpts[output]); }

        LOG(INFO) << descr;
    } else // at least one parent
    {

        std::vector<int> input(_parent_nodes.size(), 0);

        // print the CPT for each parent value
        for (size_t i = 0; i < _cpts.size() / _output_size; ++i)
        {
            // parent description
            auto descr = "\t{" + std::to_string(_parent_nodes[0]) + ":" + std::to_string(input[0]);
            for (size_t p = 1; p < input.size(); ++p)
            { descr += "," + std::to_string(_parent_nodes[p]) + ":" + std::to_string(input[p]); }
            descr += "}: {" + std::to_string(_cpts[i * _output_size]);

            // dirichlet description
            for (auto output = 1; output < _output_size; ++output)
            { descr += "," + std::to_string(_cpts[i * _output_size + output]); }
            descr += "}";

            LOG(INFO) << descr;

            indexing::increment(input, _parent_sizes);
        }
    }
}
