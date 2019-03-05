#include "FactoredDummyDomainPriors.hpp"

#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/FBAConf.hpp"
#include "domains/dummy/FactoredDummyDomain.hpp"
#include "environment/State.hpp"
#include "utils/index.hpp"

namespace priors {
FactoredDummyPrior::FactoredDummyPrior(size_t size) :
        FBAPOMDPPrior(configurations::FBAConf()),
        _size(size),
        _domain_size({static_cast<int>(size * size), 2, 1}),
        _domain_feature_size({{(int)size, (int)size}, {1}}),
        _fbapomdp_step_size(
            indexing::stepSize(_domain_feature_size._S),
            indexing::stepSize(_domain_feature_size._O)),
        _flat_prior(&_domain_size),
        _factored_prior(&_domain_size, &_domain_feature_size, &_fbapomdp_step_size),
        _fully_connected_prior(&_domain_size, &_domain_feature_size, &_fbapomdp_step_size)
{

    precomputePrior();
}

FactoredDummyPrior::FactoredDummyPrior(configurations::FBAConf const& c, size_t size) :
        FBAPOMDPPrior(c),
        _size(size),
        _domain_size({static_cast<int>(size * size), 2, 1}),
        _domain_feature_size({{(int)size, (int)size}, {1}}),
        _fbapomdp_step_size(
            indexing::stepSize(_domain_feature_size._S),
            indexing::stepSize(_domain_feature_size._O)),
        _flat_prior(&_domain_size),
        _factored_prior(&_domain_size, &_domain_feature_size, &_fbapomdp_step_size),
        _fully_connected_prior(&_domain_size, &_domain_feature_size, &_fbapomdp_step_size)
{

    precomputePrior();
}

BAPOMDPState* FactoredDummyPrior::sampleBAPOMDPState(State const* domain_state) const
{
    assert(domain_state != nullptr && domain_state->index() < _domain_size._S);

    // we simply set the counts that we pre computed, since
    // the prior counts are the same for each state
    return new BAPOMDPState(domain_state, _flat_prior);
}

FBAPOMDPState* FactoredDummyPrior::sampleFBAPOMDPState(State const* domain_state) const
{
    assert(domain_state != nullptr && domain_state->index() < _domain_size._S);

    return new FBAPOMDPState(domain_state, _factored_prior);
}

FBAPOMDPState* FactoredDummyPrior::sampleFullyConnectedState(State const* domain_state) const
{
    assert(domain_state != nullptr && domain_state->index() < _domain_size._S);

    return new FBAPOMDPState(domain_state, _fully_connected_prior);
}

FBAPOMDPState* FactoredDummyPrior::sampleCorrectGraphState(State const* /*domain_state*/) const
{
    throw "CollisionAvoidance::sampleCorrectGraphState nyi";
    return nullptr;
}

bayes_adaptive::factored::BABNModel::Structure
    FactoredDummyPrior::mutate(bayes_adaptive::factored::BABNModel::Structure structure) const
{
    // this domain does not have any unknown variables in the structure
    return structure;
}

void FactoredDummyPrior::precomputePrior()
{
    /* here we loop over all states and compute the next state
     * and observation for each action, which is deterministic
     *
     * we then increment the corresponding count to make sure
     * the prior is set correctly
     *
     * we do this for the factored and flat case simultaneously */

    // tmp variables used to increment the counts
    auto temp_state = IndexState(0), temp_next_state = IndexState(0);
    auto up        = IndexAction(domains::FactoredDummyDomain::UP),
         right     = IndexAction(domains::FactoredDummyDomain::RIGHT);
    auto o         = IndexObservation(0);
    int num_states = static_cast<int>(_size * _size);

    auto x_feature = 1, y_feature = 0;

    // we start with a complete zero'd out models
    BAPOMDPState s(&temp_state, bayes_adaptive::table::BAFlatModel(&_domain_size));
    FBAPOMDPState s_factored(&temp_state, _factored_prior);
    FBAPOMDPState s_fact_fully_connected(&temp_state, _fully_connected_prior);

    // add connections
    s_factored.model()->resetTransitionNode(&up, y_feature, {y_feature});
    s_factored.model()->resetTransitionNode(&right, y_feature, {y_feature});
    s_factored.model()->resetTransitionNode(&up, x_feature, {x_feature});
    s_factored.model()->resetTransitionNode(&right, x_feature, {x_feature});

    s_fact_fully_connected.model()->resetTransitionNode(&up, y_feature, {x_feature, y_feature});
    s_fact_fully_connected.model()->resetTransitionNode(&right, y_feature, {x_feature, y_feature});
    s_fact_fully_connected.model()->resetTransitionNode(&up, x_feature, {x_feature, y_feature});
    s_fact_fully_connected.model()->resetTransitionNode(&right, x_feature, {x_feature, y_feature});

    for (auto i = 0; i < num_states; ++i)
    {
        temp_state.index(i);

        // going up gets +1 unless on top edge already
        temp_next_state.index(i + ((((i + 1) % _size) != 0u) ? 1 : 0));
        s.model()->count(&temp_state, &up, &temp_next_state)++;
        s.model()->count(&up, &temp_next_state, &o)++;

        s_factored.incrementCountsOf(&temp_state, &up, &o, &temp_next_state);
        s_fact_fully_connected.incrementCountsOf(&temp_state, &up, &o, &temp_next_state);

        // going right +_size unless on right edge already
        temp_next_state.index(
            i + ((i < static_cast<int>((_size - 1) * _size)) ? static_cast<int>(_size) : 0));
        s.model()->count(&temp_state, &right, &temp_next_state)++;
        s.model()->count(&right, &temp_next_state, &o)++;

        s_factored.incrementCountsOf(&temp_state, &right, &o, &temp_next_state);
        s_fact_fully_connected.incrementCountsOf(&temp_state, &right, &o, &temp_next_state);
    }

    // here we save the counts, so we can directly set the counts
    // whenever we need to sample a BAPOMDPState from this prior
    _flat_prior            = *s.model();
    _factored_prior        = std::move(*s_factored.model());
    _fully_connected_prior = std::move(*s_fact_fully_connected.model());
}

} // namespace priors
