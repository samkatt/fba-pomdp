#include "easylogging++.h"

#include <cstddef>
#include <utility>

#include "AbstractFBAPOMDPState.hpp"

#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"

#include "bayes-adaptive/models/factored/Domain_Feature_Size.hpp"

//#include "bayes-adaptive/states/factored/BABNModel.hpp"

#include "utils/index.hpp"
// TODO after every "real" timestep, will these be constructed again? I.e. can I assume _abstraction will be empty?
// cppcheck-suppress passedByValue
AbstractFBAPOMDPState::AbstractFBAPOMDPState(State const* domain_state, bayes_adaptive::factored::BABNModel model) :
        FBAPOMDPState(domain_state, model),
        _abstraction({}), // Empty initialization.
        _abstract_model() // Initialized later, when abstraction is added
{
}

BAState* AbstractFBAPOMDPState::copy(State const* domain_state) const
{
    return new AbstractFBAPOMDPState(domain_state, FBAPOMDPState::model_real()); //_model); // TODO better way?
}

// TODO change for abstraction
// this samples a new state...?
int AbstractFBAPOMDPState::sampleStateIndex(
    State const* s,
    Action const* a,
    rnd::sample::Dir::sampleMethod m) const
{
    if (_abstraction.size() > 0) {
        return _abstract_model.sampleStateIndex(s,a,m); // TODO check if abstraction exists maybe?
    }
    return FBAPOMDPState::model()->sampleStateIndex(s, a, m); // _model
}

// TODO change for abstraction
int AbstractFBAPOMDPState::sampleObservationIndex(
    Action const* a,
    State const* new_s,
    rnd::sample::Dir::sampleMethod m) const
{
//    return FBAPOMDPState::model()->sampleObservationIndex(a, new_s, m); // _model
    return _abstract_model.sampleObservationIndex(a, new_s, m); // TODO check if abstraction exists maybe?
}

// only called at initialization?
double AbstractFBAPOMDPState::computeObservationProbability(
    Observation const* o,
    Action const* a,
    State const* s,
    rnd::sample::Dir::sampleMultinominal sampleMultinominal) const
{
    return FBAPOMDPState::model()->computeObservationProbability(o, a, s, sampleMultinominal); // _model
}

// TODO change for abstraction
void AbstractFBAPOMDPState::incrementCountsOf(
    State const* s,
    Action const* a,
    Observation const* o,
    State const* new_s,
    float amount)
{
    FBAPOMDPState::model()->incrementCountsOf(s, a, o, new_s, amount); // _model
}

std::vector<int>* AbstractFBAPOMDPState::getAbstraction(){
    return &_abstraction;
}

void AbstractFBAPOMDPState::setAbstraction(std::vector<int> new_abstraction){
    _abstraction = new_abstraction;
    _abstract_model = construct_abstract_model(FBAPOMDPState::model_real());
}

// Construct abstract model from the model given the features to keep in the abstraction
bayes_adaptive::factored::BABNModel AbstractFBAPOMDPState::construct_abstract_model(bayes_adaptive::factored::BABNModel model) {
    bayes_adaptive::factored::BABNModel abstract_model = model; // TODO this is a copy? expensive
//    int abstraction_size = _abstraction.size();
//    const Domain_Feature_Size* domainFeatureSize = _abstract_model.domainFeatureSize(); // maybe no need?
//    for (int i : _abstraction) {
////         TODO HELP what is this? "Member access into incomplete type 'const Domain_Feature_Size'"
//        for (auto x = 0; x < domainFeatureSize->_S[i]; ++x) {
//
//        }
//    }
    // Hardcoding since I can't figure out how to use domainFeatureSize
//    auto old_structure = abstract_model.structure();
//    bayes_adaptive::factored::BABNModel::Structure new_structure = old_structure;
//    for (int it = new_structure.T.begin(); it != new_structure.T.end(); it++) {
//
//    }
//    for (DBNNode node : abstract_model._observation_nodes) {
//
//    }
//    for (int i = 0; i < 2; i++){
//        abstract_model.marginalizeOut()
//        abstract_model.resetTransitionNode(&action, i, parents);
//        model.transitionNode(&action, feature).count({x, y, rain, carpet_config}, loc) +=
//                (1 - success_prob) * _unknown_counts_total;
//    }

    abstract_model = abstract_model.abstract(_abstraction, abstract_model.structure());
    return abstract_model;
}

void AbstractFBAPOMDPState::logCounts() const
{
    FBAPOMDPState::model()->log(); // _model
}

std::string AbstractFBAPOMDPState::toString() const
{
    return "AbstractFBAPOMDPState with s=" + _domain_state->toString();
}
