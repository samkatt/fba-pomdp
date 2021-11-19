#include <unordered_set>
#include "BABNModel.hpp"

#include "easylogging++.h"

#include "bayes-adaptive/models/Domain_Size.hpp"
#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "utils/index.hpp"

namespace bayes_adaptive { namespace factored {

void BABNModel::Structure::flip_random_edge(std::vector<int>* edges, int edge_range)
{
    assert(edges);
    assert(edge_range >= 0);

    auto edge_to_flip = rnd::slowRandomInt(0, edge_range);
    auto lower_bound  = std::lower_bound(edges->begin(), edges->end(), edge_to_flip);

    if (lower_bound != edges->end() && *lower_bound == edge_to_flip) // found the edge, remove
    {
        edges->erase(lower_bound);
    } else // edge was not there, add!
    {
        edges->insert(lower_bound, edge_to_flip);
    }
}

BABNModel::BABNModel() : _domain_size(0), _domain_feature_size(0), _step_sizes(0) {}

BABNModel::BABNModel(
    Domain_Size const* domain_size,
    Domain_Feature_Size const* domain_feature_size,
    Indexing_Steps const* step_sizes) :
        _domain_size(domain_size),
        _domain_feature_size(domain_feature_size),
        _step_sizes(step_sizes)
{

    assert(_domain_size);
    assert(_domain_feature_size);
    assert(_step_sizes);

    // init our model
    _transition_nodes.reserve(_domain_size->_A * _domain_feature_size->_S.size());
    _observation_nodes.reserve(_domain_size->_A * _domain_feature_size->_O.size());

    for (auto a = 0; a < _domain_size->_A; ++a)
    {
        for (auto const& feature_size : _domain_feature_size->_S)
        { _transition_nodes.emplace_back(DBNNode(&_domain_feature_size->_S, {}, feature_size)); }

        for (auto const& feature_size : _domain_feature_size->_O)
        { _observation_nodes.emplace_back(DBNNode(&_domain_feature_size->_S, {}, feature_size)); }
    }
}

BABNModel::BABNModel(
    Domain_Size const* domain_size,
    Domain_Feature_Size const* domain_feature_size,
    Indexing_Steps const* step_sizes,
    // cppcheck-suppress passedByValue
    std::vector<DBNNode> transition_nodes,
    // cppcheck-suppress passedByValue
    std::vector<DBNNode> observation_nodes) :
        _transition_nodes(std::move(transition_nodes)),
        _observation_nodes(std::move(observation_nodes)),
        _domain_size(domain_size),
        _domain_feature_size(domain_feature_size),
        _step_sizes(step_sizes)
{

    assert(_domain_size);
    assert(_domain_feature_size);
    assert(_step_sizes);

    assert(_transition_nodes.size() == _domain_size->_A * _domain_feature_size->_S.size());
    assert(_observation_nodes.size() == _domain_size->_A * _domain_feature_size->_O.size());
}

std::vector<std::vector<std::vector<float>>> BABNModel::flattenT() const
{

    std::vector<std::vector<std::vector<float>>> T(
        _domain_size->_S,
        std::vector<std::vector<float>>(_domain_size->_A, std::vector<float>(_domain_size->_S, 1)));

    for (auto a = 0; a < _domain_size->_A; ++a)
    {

        IndexAction action(std::to_string(a));

        std::vector<int> features(_domain_feature_size->_S.size());
        auto s = 0;
        do
        {

            // store probabilities for each feature
            std::vector<std::vector<float>> conditional_expectations;
            for (size_t f = 0; f < _domain_feature_size->_S.size(); ++f)
            {
                conditional_expectations.emplace_back(
                    transitionNode(&action, f).expectation(features));
            }

            // loop over new_s
            std::vector<int> new_features(_domain_feature_size->_S.size());
            auto new_s = 0;
            do
            {

                // probability is the product of each conditional
                for (size_t f = 0; f < _domain_feature_size->_S.size(); ++f)
                { T[s][a][new_s] *= conditional_expectations[f][new_features[f]]; }

                new_s++;
            } while (!indexing::increment(new_features, _domain_feature_size->_S));

            s++;
        } while (!indexing::increment(features, _domain_feature_size->_S));
    }

    return T;
}

std::vector<std::vector<std::vector<float>>> BABNModel::flattenO() const
{

    std::vector<std::vector<std::vector<float>>> O(
        _domain_size->_A,
        std::vector<std::vector<float>>(_domain_size->_S, std::vector<float>(_domain_size->_O, 1)));

    // loop over a
    for (auto a = 0; a < _domain_size->_A; ++a)
    {

        IndexAction action(std::to_string(a));

        // loop over new_s
        std::vector<int> state_features(_domain_feature_size->_S.size());
        auto new_s = 0;
        do
        {

            // store probabilities for each feature
            std::vector<std::vector<float>> conditional_expectations;
            for (size_t f = 0; f < _domain_feature_size->_O.size(); ++f)
            {
                conditional_expectations.emplace_back(
                    observationNode(&action, f).expectation(state_features));
            }

            // loop over o
            std::vector<int> observation_features(_domain_feature_size->_O.size());
            auto o = 0;
            do
            {

                // probability is the product of each conditional
                for (size_t f = 0; f < _domain_feature_size->_O.size(); ++f)
                { O[a][new_s][o] *= conditional_expectations[f][observation_features[f]]; }

                o++;
            } while (!indexing::increment(observation_features, _domain_feature_size->_O));
            new_s++;
        } while (!indexing::increment(state_features, _domain_feature_size->_S));
    }

    return O;
}

// TODO Sammie, this is also used as initiation of a node?
// Initializes transition node, for a state feature and its parents given the action.
void BABNModel::resetTransitionNode(Action const* a, int state_feature, std::vector<int> parents)
{
    assertLegalStateFeature(state_feature);
    assert(parents.size() <= _domain_feature_size->_S.size());

    // initiates a DBNNode for the <a,feature>
    transitionNode(a, state_feature) = DBNNode(
        &_domain_feature_size->_S, std::move(parents), _domain_feature_size->_S[state_feature]);
}

// TODO Sammie, this is also used as initiation of a node?
// Initializes observation node, for an observation feature and its parents given the action.
void BABNModel::resetObservationNode(
    Action const* a,
    int observation_feature,
    std::vector<int> parents)
{
    assertLegalObservationFeature(observation_feature);
    assert(parents.size() <= _domain_feature_size->_S.size());

    // initiates a DBNNode for the <a,feature>
    observationNode(a, observation_feature) = DBNNode(
        &_domain_feature_size->_S,
        std::move(parents),
        _domain_feature_size->_O[observation_feature]);
}

BABNModel BABNModel::marginalizeOut(Structure new_structure) const
{
    std::vector<DBNNode> T_marginalized, O_marginalized;

    auto action = IndexAction(std::to_string(0));
    for (auto a = 0; a < static_cast<int>(_domain_size->_A); ++a)
    {
        action.index(std::to_string(a));

        for (auto f = 0; f < static_cast<int>(_domain_feature_size->_S.size()); ++f)
        {
            T_marginalized.emplace_back(
                transitionNode(&action, f).marginalizeOut(std::move(new_structure.T[a][f])));
        }

        for (auto f = 0; f < static_cast<int>(_domain_feature_size->_O.size()); ++f)
        {
            O_marginalized.emplace_back(
                observationNode(&action, f).marginalizeOut(std::move(new_structure.O[a][f])));
        }
    }

    return BABNModel(
        _domain_size, _domain_feature_size, _step_sizes, T_marginalized, O_marginalized);
}


void remove_parents(std::vector<int>& a, std::vector<int>& b){
    std::unordered_multiset<int> st;
    st.insert(b.begin(), b.end());
    auto predicate = [&st](const int& k){ return st.count(k) < 1; };
    a.erase(std::remove_if(a.begin(), a.end(), predicate), a.end());
}

BABNModel
BABNModel::abstract(std::vector<int> abstraction_set, Structure structure, const Domain_Size *ds, const Domain_Feature_Size *dfs,
                    const Indexing_Steps *is, bool normalize) const {
    auto new_structure = std::move(structure);

    for (auto a = 0; a < static_cast<int>(_domain_size->_A); ++a)
    {
        // remove parents that are not part of the abstraction_set, from the features in the abstraction_set
        for (auto const& f: abstraction_set) {
            remove_parents(new_structure.T[a][f], abstraction_set);
        }
    }

    // marginalize and reduce
    std::vector<DBNNode> T_marginalized, O_marginalized;

    auto action = IndexAction(std::to_string(0));
    for (auto a = 0; a < static_cast<int>(_domain_size->_A); ++a)
    {
        action.index(std::to_string(a));

        for (auto f = 0; f < static_cast<int>(_domain_feature_size->_S.size()); ++f)
        {
            if(normalize) {
                T_marginalized.emplace_back(
                        transitionNode(&action, f).marginalizeOutAndNormalize(std::move(new_structure.T[a][f])));
            } else {
                T_marginalized.emplace_back(
                        transitionNode(&action, f).marginalizeOut(std::move(new_structure.T[a][f])));
            }
        }

        for (auto f = 0; f < static_cast<int>(_domain_feature_size->_O.size()); ++f)
        {
            O_marginalized.emplace_back(
                    observationNode(&action, f).marginalizeOut(std::move(new_structure.O[a][f])));
        }
    }

    // remove nodes when they are not in the abstraction set
    T_marginalized.erase(std::remove_if(T_marginalized.begin(), T_marginalized.end(),
                                        [abstraction_set](DBNNode i) {return
                                        find(abstraction_set.begin(), abstraction_set.end(), i.parents()->at(0))
                                        == abstraction_set.end();}), T_marginalized.end()); // remove if not in the abstraction_set

    return BABNModel(
            ds, dfs, is, T_marginalized, O_marginalized);
}

void BABNModel::abstractionNormalizeCounts(BABNModel prior, BABNModel prior_normalized)
{
    auto action = IndexAction(std::to_string(0));
    for (auto a = 0; a < static_cast<int>(_domain_size->_A); ++a)
    {
        action.index(std::to_string(a));

        for (auto f = 0; f < static_cast<int>(_domain_feature_size->_S.size()); ++f)
        {
            transitionNode(&action, f).doAbstractionNormalizeCounts(prior.transitionNode(&action, f), prior_normalized.transitionNode(&action, f));
        }
    }
}

DBNNode& BABNModel::transitionNode(Action const* a, int feature)
{
    assertLegal(a);
    assertLegalStateFeature(feature);
    return _transition_nodes[indexing::twoToOne(
        std::stoi(a->index()), feature, (int)_domain_feature_size->_S.size())];
}

DBNNode const& BABNModel::transitionNode(Action const* a, int feature) const
{
    assertLegal(a);
    assertLegalStateFeature(feature);
    return _transition_nodes[indexing::twoToOne(
        std::stoi(a->index()), feature, (int)_domain_feature_size->_S.size())];
}

DBNNode& BABNModel::observationNode(Action const* a, int feature)
{
    assertLegal(a);
    assertLegalObservationFeature(feature);
    return _observation_nodes[indexing::twoToOne(
        std::stoi(a->index()), feature, (int)_domain_feature_size->_O.size())];
}

DBNNode const& BABNModel::observationNode(Action const* a, int feature) const
{
    assertLegal(a);
    assertLegalObservationFeature(feature);
    return _observation_nodes[indexing::twoToOne(
        std::stoi(a->index()), feature, (int)_domain_feature_size->_O.size())];
}

bayes_adaptive::factored::BABNModel::Structure BABNModel::structure() const
{

    auto res    = bayes_adaptive::factored::BABNModel::Structure();
    auto action = IndexAction(std::to_string(0));

    res.O = std::vector<std::vector<std::vector<int>>>(_domain_size->_A);
    res.T = std::vector<std::vector<std::vector<int>>>(_domain_size->_A);
    for (auto a = 0; a < static_cast<int>(_domain_size->_A); ++a)
    {

        action.index(std::to_string(a));

        res.O[a] = std::vector<std::vector<int>>(_domain_feature_size->_O.size());
        for (auto f = 0; f < static_cast<int>(_domain_feature_size->_O.size()); ++f)
        { res.O[a][f] = *observationNode(&action, f).parents(); }

        res.T[a] = std::vector<std::vector<int>>(_domain_feature_size->_S.size());
        for (auto f = 0; f < static_cast<int>(_domain_feature_size->_S.size()); ++f)
        { res.T[a][f] = *transitionNode(&action, f).parents(); }
    }

    return res;
}

// From https://www.delftstack.com/howto/cpp/how-to-determine-if-a-string-is-number-cpp/
// TODO add parameter to BABNModel that tells if an index is a string or number?
bool isNumber(const std::string& str)
{
    return !std::any_of(&str[0], &str[str.size()-1], [] (char c) { return !std::isdigit(c); });
}

std::string BABNModel::sampleStateIndex(State const* s, Action const* a, rnd::sample::Dir::sampleMethod m)
    const
{
    assertLegal(s);
    assertLegal(a);

    auto parent_values = s->getFeatureValues(); // stateFeatureValues(s);

    if (!isNumber(s->index())) { // TODO should be better way
        std::string index;
        for (int i=0; i < (int) _domain_feature_size->_S.size() - 1; i++){
            index += std::to_string(transitionNode(a, i).sample(parent_values, m));
            index += '+';
        }
        index += std::to_string(transitionNode(a, _domain_feature_size->_S.size() - 1).sample(parent_values, m));
        return index;
    }

    //create a vector for the next-stage variables - XXX: this is a memory allocation... expensive!?
    auto feature_values = std::vector<int>(_domain_feature_size->_S.size());
    //fill the vector by sampling next stage feature 1 by 1
    for (auto n = 0; n < (int) _domain_feature_size->_S.size(); ++n)
    { feature_values[n] = transitionNode(a, n).sample(parent_values, m); }

    return std::to_string(indexing::project(feature_values, _domain_feature_size->_S));
}

std::string BABNModel::sampleStateIndexThroughAbstraction(const std::vector<int> *newfeature_values) const {
    // If we call this function we already should now that the index is a number
    return std::to_string(indexing::project(*newfeature_values, _domain_feature_size->_S));
}

int BABNModel::sampleObservationIndex(
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMethod m) const
{
    assertLegal(a);
    assertLegal(new_s);

    auto parent_values = new_s->getFeatureValues(); // stateFeatureValues(new_s);

    auto feature_values = std::vector<int>(_domain_feature_size->_O.size());
    for (auto n = 0; n < (int)_domain_feature_size->_O.size(); ++n)
    { feature_values[n] = observationNode(a, n).sample(parent_values, m); }

    return indexing::project(feature_values, _domain_feature_size->_O);
}

double BABNModel::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* s,
    rnd::sample::Dir::sampleMultinominal sampleMultinominal) const
{
    assertLegal(a);
    assertLegal(o);
    assertLegal(s);

    auto nodes_input    = s->getFeatureValues(); // stateFeatureValues(s);
    auto feature_values = observationFeatureValues(o); // o->getFeatureValues(); // observationFeatureValues(o);

    double prob = 1;

    // probability of this observation is the multiplication of
    // the probbility of each feature
    for (auto n = 0; n < static_cast<int>(_domain_feature_size->_O.size()); ++n)
    {
        prob *= observationNode(a, n).sampleMultinominal(
            nodes_input, sampleMultinominal)[feature_values[n]];
    }

    return prob;
}

void BABNModel::incrementCountsOfAbstract(const Action *a, const Observation *o, float amount,
                                          const std::vector<int> &parent_values,
                                          std::vector<int> state_feature_values, std::vector<int> feature_set) {
    // update transition DBN
    for (auto n = 0; n < (int)_domain_feature_size->_S.size(); ++n)
    { transitionNode(a, n).increment(parent_values, state_feature_values[feature_set[n]], amount); } // goes wrong here? state_feature_values[n] is not necesarily correct here...

    auto observation_feature_values = observationFeatureValues(o);
    // update observation DBN
    for (auto n = 0; n < (int)_domain_feature_size->_O.size(); ++n)
    { observationNode(a, n).increment(parent_values, observation_feature_values[n], amount); }
}

void BABNModel::incrementCountsOf(
    State const* s,
    Action const* a,
    Observation const* o,
    State const* new_s,
    float amount)
{
    assertLegal(s);
    assertLegal(a);
    assertLegal(o);
    assertLegal(new_s);

    auto parent_values = s->getFeatureValues(); // stateFeatureValues(s);

    auto state_feature_values = new_s->getFeatureValues(); // stateFeatureValues(new_s);

    // update transition DBN
    for (auto n = 0; n < (int)_domain_feature_size->_S.size(); ++n)
    { transitionNode(a, n).increment(parent_values, state_feature_values[n], amount); }

    auto observation_feature_values = observationFeatureValues(o); // o->getFeatureValues(); // observationFeatureValues(o);
    // update observation DBN
    for (auto n = 0; n < (int)_domain_feature_size->_O.size(); ++n)
    { observationNode(a, n).increment(parent_values, observation_feature_values[n], amount); }
}

void BABNModel::log() const
{
    LOG(INFO) << "CPTs for FBAPOMDP state:";
    LOG(INFO) << "Transition:";

    auto action = IndexAction(std::to_string(0));
    for (auto f = 0; f < (int)_domain_feature_size->_S.size(); ++f)
    {
        for (auto a = 0; a < _domain_size->_A; ++a)
        {
            action.index(std::to_string(a));
            LOG(INFO) << "Feature " << f << ", Action " << a;
            transitionNode(&action, f).logCPTs();
        }
    }

    LOG(INFO) << "Observation:";
    for (auto f = 0; f < (int)_domain_feature_size->_O.size(); ++f)
    {
        for (auto a = 0; a < _domain_size->_A; ++a)
        {
            action.index(std::to_string(a));
            LOG(INFO) << "Feature " << f << ", Action " << a;
            observationNode(&action, f).logCPTs();
        }
    }
}

std::vector<int> BABNModel::stateFeatureValues(State const* s) const
{
    assertLegal(s);
    // W
    // TODO unnecessary if we can get it from the state directly?
    return s->getFeatureValues();
//    if (_step_sizes->T[0] <= 0) { // This happens when the state space is too large, the int overflows
//        auto state_vector = std::vector<int>(_step_sizes->T.size());
//        std::string value;
//        int entry = 0;
//        for (char const &c: s->index()) {
//            if (c == '+') {
//                state_vector[entry] = std::stoi(value);
//                value.clear();
//                entry++;
//            } else {
//                value += c;
//            }
//        }
//        state_vector[state_vector.size()-1] = std::stoi(value);
//
//        return state_vector;
//    } else {
//        return indexing::projectUsingStepSize(std::stoi(s->index()), _step_sizes->T);
//    }
//    return s->getFeatureValues();
}

std::vector<int> BABNModel::observationFeatureValues(Observation const* o) const
{
    assertLegal(o);
//    return o->getFeatureValues();
    // TODO needs to be changed?
    return indexing::projectUsingStepSize(std::stoi(o->index()), _step_sizes->O);
}

void BABNModel::assertLegal(State const* s) const
{
    assert(s != nullptr); // &&std::stoi(s->index())< _domain_size->_S);
}

void BABNModel::assertLegal(Action const* a) const
{
    assert(a != nullptr && std::stoi(a->index()) < _domain_size->_A);
}

void BABNModel::assertLegal(Observation const* o) const
{
    assert(o != nullptr && std::stoi(o->index()) < _domain_size->_O);
}

void BABNModel::assertLegalStateFeature(int f) const
{
    assert(f >= 0 && f < static_cast<int>(_domain_feature_size->_S.size()));
}

void BABNModel::assertLegalObservationFeature(int f) const
{
    assert(f >= 0 && f < static_cast<int>(_domain_feature_size->_O.size()));
}

double BABNModel::LogBDScore(BABNModel const& prior) const
{
    assert(_transition_nodes.size() == prior._transition_nodes.size());
    assert(_observation_nodes.size() == prior._observation_nodes.size());

    double bd_score = 0;

    auto action      = IndexAction(std::to_string(0));
    auto state       = IndexState(std::to_string(0));
    auto observation = IndexObservation(std::to_string(0));

    for (auto a = 0; a < _domain_size->_A; ++a)
    {
        action.index(std::to_string(a));

        for (auto f = 0; f < static_cast<int>(_domain_feature_size->_S.size()); ++f)
        { bd_score += transitionNode(&action, f).LogBDScore(prior.transitionNode(&action, f)); }

        for (auto f = 0; f < static_cast<int>(_domain_feature_size->_O.size()); ++f)
        { bd_score += observationNode(&action, f).LogBDScore(prior.observationNode(&action, f)); }
    }

    return bd_score;
}

    const Domain_Feature_Size* BABNModel::domainFeatureSize() const {
        return _domain_feature_size;
    }

    const Domain_Size* BABNModel::domainSize() const {
        return _domain_size;
    }

    const BABNModel::Indexing_Steps* BABNModel::stepSizes() const {
        return _step_sizes;
    }

}} // namespace bayes_adaptive::factored
