#include "CollisionAvoidancePriors.hpp"

#include <algorithm>
#include <cstdlib>

#include "bayes-adaptive/states/factored/FBAPOMDPState.hpp"
#include "bayes-adaptive/states/table/BAPOMDPState.hpp"
#include "configurations/BAConf.hpp"
#include "configurations/FBAConf.hpp"
#include "environment/Action.hpp"
#include "environment/Observation.hpp"
#include "environment/State.hpp"
#include "utils/index.hpp"
#include "utils/random.hpp"

namespace priors {

/**
 * @brief returns the probability of each observation
 **/
std::vector<float> observationDistr(int height, std::vector<int> const& obstacles_pos)
{
    auto obs_distr = std::vector<float>(height, 1);

    for (auto const& b : obstacles_pos)
    {
        obs_distr[0] *= static_cast<float>(rnd::normal::cdf(-b + .5, 0, 1) * 10000);

        for (auto y = 1; y < height - 1; ++y)
        {
            auto dist = std::abs(y - b);
            obs_distr[y] *= static_cast<float>(
                (rnd::normal::cdf(dist + .5, 0, 1) - rnd::normal::cdf(dist - .5, 0, 1)) * 10000);
        }

        auto dist             = height - 1 - b;
        obs_distr[height - 1] = static_cast<float>(rnd::normal::cdf(-dist + .5, 0, 1) * 10000);
    }

    return obs_distr;
}

/**
 * @brief returns the probability of each observation
 **/
std::vector<float> observationDistr(int height, int obstacle_pos)
{
    auto obs_distr = std::vector<float>(height, 1);

    obs_distr[0] = static_cast<float>(rnd::normal::cdf(-obstacle_pos + .5, 0, 1));

    for (auto y = 1; y < height - 1; ++y)
    {
        auto dist    = std::abs(y - obstacle_pos);
        obs_distr[y] = static_cast<float>(
            (rnd::normal::cdf(dist + .5, 0, 1) - rnd::normal::cdf(dist - .5, 0, 1)));
    }

    auto dist             = height - 1 - obstacle_pos;
    obs_distr[height - 1] = static_cast<float>(rnd::normal::cdf(-dist + .5, 0, 1));

    return obs_distr;
}

CollisionAvoidanceTablePrior::CollisionAvoidanceTablePrior(
    domains::CollisionAvoidance const& d,
    configurations::BAConf const& c) :
        _height(c.domain_conf.height),
        _width(c.domain_conf.width),
        _num_obstacles(c.domain_conf.size),
        _noise(c.noise),
        _total_counts(c.counts_total),
        _domain_size(
            static_cast<int>(_width * std::pow(_height, _num_obstacles + 1)),
            _NUM_ACTIONS,
            static_cast<int>(std::pow(_height, _num_obstacles))),
        _prior(&_domain_size)
{

    assert(c.noise < .5 && c.noise > -.5);

    {

        // prepare observation probabilities (distribution per location for a single obstacle)
        std::vector<std::vector<float>> obs_distr(_height);
        for (auto i = 0; i < _height; ++i) { obs_distr[i] = observationDistr(_height, i); }

        std::vector<int> obstacles(_num_obstacles);
        do
        {

            for (auto x = 0; x < _width; ++x)
            {
                for (auto y = 0; y < _height; ++y)
                {

                    auto const s = d.getState(x, y, obstacles);

                    for (auto a = 0; a < _NUM_ACTIONS; ++a)
                    {

                        IndexAction const action(a);

                        // loop over position of obstacles
                        std::vector<int> new_obstacles(_num_obstacles);
                        auto count = 0;
                        do
                        {
                            IndexObservation const o(count);

                            /***** transition counts ****/
                            setTransitionCounts(x, y, a, obstacles, new_obstacles, d);

                            /*** observation counts ****/
                            double observation_prob = 1;
                            for (auto i = 0; i < _num_obstacles; ++i)
                            { observation_prob *= obs_distr[obstacles[i]][new_obstacles[i]]; }

                            _prior.count(&action, s, &o) = observation_prob * 10000;

                        } while (++count
                                 && !indexing::increment(new_obstacles, _obstacle_pos_ranges));
                    }
                }
            }

        } while (!indexing::increment(obstacles, _obstacle_pos_ranges));
    }
}

BAPOMDPState* CollisionAvoidanceTablePrior::sampleBAPOMDPState(State const* domain_state) const
{
    return new BAPOMDPState(domain_state, _prior);
}

double CollisionAvoidanceTablePrior::obstacleTransProb(int y, int new_y) const
{
    assert(y >= 0 && y < _height);
    assert(new_y >= 0 && new_y < _height);

    auto const dist = std::abs(y - new_y);

    // base case: further than 1 apart: 0 probability
    if (dist > 1)
    {
        return 0;
    }

    // 2 possibilities of y: (1) top or bottom, or (2) neither
    if (y == 0 || y == _height - 1) // bottom
    {
        if (dist == 0)
        {
            return .75 + .5 * _noise;
        } else
        {
            return .25 - .5 * _noise;
        }

    } else // somewhere in middle
    {
        if (dist == 0)
        {
            return .5 + _noise;
        } else
        {
            return .25 - .5 * _noise;
        }
    }
}

void CollisionAvoidanceTablePrior::setTransitionCounts(
    int x,
    int y,
    int a,
    std::vector<int> const& obstacles,
    std::vector<int> const& new_obstacles,
    domains::CollisionAvoidance const& d)
{
    if (x == 0)
        return;

    // compute deterministic parts
    auto new_x = x - 1;
    auto new_y = y + a - 1;

    if ((new_y == -1) || (new_y == _height))
    {
        new_y = y;
    }

    double prob = 1;
    for (auto i = 0; i < _num_obstacles; ++i)
    {
        prob *= obstacleTransProb(obstacles[i], new_obstacles[i]);

        if (prob == 0)
        {
            return;
        }
    }

    auto const s      = d.getState(x, y, obstacles);
    auto const new_s  = d.getState(new_x, new_y, new_obstacles);
    auto const action = IndexAction(a);

    _prior.count(s, &action, new_s) = static_cast<float>(prob * _total_counts);
}

CollisionAvoidanceFactoredPrior::CollisionAvoidanceFactoredPrior(
    configurations::FBAConf const& conf) :
        FBAPOMDPPrior(conf),
        _num_obstacles(conf.domain_conf.size),
        _width(conf.domain_conf.width),
        _height(conf.domain_conf.height),
        _noise(conf.noise),
        _counts_total(conf.counts_total),
        _edge_noise(conf.structure_prior),
        _domain_size(
            _width * _height * static_cast<int>(std::pow(_height, _num_obstacles)),
            _NUM_ACTIONS,
            static_cast<int>(std::pow(_height, _num_obstacles))),
        _domain_feature_size(
            std::vector<int>(2 + _num_obstacles, _height),
            std::vector<int>(_num_obstacles, _height)),
        _fbapomdp_step_size({}, {})
{
    const_cast<Domain_Feature_Size&>(_domain_feature_size)._S[0] = _width;
    const_cast<bayes_adaptive::factored::BABNModel::Indexing_Steps&>(_fbapomdp_step_size) =
        bayes_adaptive::factored::BABNModel::Indexing_Steps(
            indexing::stepSize(_domain_feature_size._S),
            indexing::stepSize(_domain_feature_size._O));

    if (_noise > .5 || _noise < -.5)
    {
        throw "CollisionAvoidanceFactoredPrior must be intiiated with -.5 < noise < .5 (is: "
            + std::to_string(_noise) + ")";
    }

    std::vector<std::string> accepted_priors = {
        "uniform", "match-counts", "match-uniform", "fully-connected"};

    if (!_edge_noise.empty()
        && std::find(accepted_priors.begin(), accepted_priors.end(), _edge_noise)
               == accepted_priors.end())
    {
        throw "CollisionAvoidanceFactoredPrior does not accept " + _edge_noise + " as noise";
    }

    bayes_adaptive::factored::BABNModel model(
        &_domain_size, &_domain_feature_size, &_fbapomdp_step_size);

    /*** set known part of the transition function (how agent transitions) ***/
    for (auto a = 0; a < _NUM_ACTIONS; a++)
    {
        auto action = IndexAction(a);

        // set agent x feature [0] for each action
        model.resetTransitionNode(&action, _AGENT_X_FEATURE, {_AGENT_X_FEATURE});
        for (auto x = 1; x < _width; ++x)
        { model.transitionNode(&action, _AGENT_X_FEATURE).count({x}, x - 1) = 1; }

        // set agent y feature [1] for each action
        model.resetTransitionNode(&action, _AGENT_Y_FEATURE, {_AGENT_Y_FEATURE});
        for (auto y = 0; y < _height; ++y)
        { setAgentYTransition(action, y, model.transitionNode(&action, _AGENT_Y_FEATURE)); }
    }

    /////// store it
    _transition_model_without_block_features = model.copyT();

    /*** compute transition when edges are known ***/
    for (auto a = 0; a < _NUM_ACTIONS; a++)
    {
        // set block y features for each action
        for (auto f = _first_obstacle; f < _num_state_features; ++f)
        {
            auto action = IndexAction(a);
            model.resetTransitionNode(&action, f, {f});
            for (auto y = 0; y < _height; ++y)
            {
                model.transitionNode(&action, f)
                    .setDirichletDistribution({y}, obstacleTransition(y));
            }
        }
    }

    ////// store it
    _correctly_connected_transition_model = model.copyT();

    /**** compute observation dynamics (all are known) ****/
    for (auto a = 0; a < _NUM_ACTIONS; ++a)
    {

        for (auto f = 0; f < _num_obstacles; ++f)
        {
            auto action = IndexAction(a);
            model.resetObservationNode(&action, f, {f + _first_obstacle});

            for (auto y = 0; y < _height; ++y)
            {

                auto obsDistr = observationDistr(_height, y);
                for (auto& p : obsDistr) p *= 10000;

                model.observationNode(&action, f).setDirichletDistribution({y}, obsDistr);
            }
        }
    }

    ////// store it
    _observation_model = model.copyO();

    /**** compute fully connected transition nodes, using the regular as starting point ****/
    // setup fully parents & their values
    auto parents       = std::vector<int>(_num_state_features);
    auto parent_values = std::vector<int>(_num_state_features);
    auto parent_ranges = std::vector<int>(_num_state_features);

    for (auto i = 0; i < _num_state_features; ++i)
    {
        parents[i]       = i;
        parent_ranges[i] = _domain_feature_size._S[i];
    }

    for (auto a = 0; a < _NUM_ACTIONS; ++a)
    {
        auto action = IndexAction(a);

        for (auto f = _first_obstacle; f < _num_state_features; ++f)
        {

            model.resetTransitionNode(&action, f, parents);
            do
            {
                model.transitionNode(&action, f)
                    .setDirichletDistribution(parent_values, obstacleTransition(parent_values[f]));
            } while (!indexing::increment(parent_values, parent_ranges));
        }
    }

    ///// store it
    _fully_connected_transition_model = model.copyT();
}

FBAPOMDPState* CollisionAvoidanceFactoredPrior::sampleFBAPOMDPState(State const* domain_state) const
{

    // if no edge noise, we can simply return a pre-computed one
    if (_edge_noise.empty() || _edge_noise == "match-counts")
    {
        return sampleCorrectGraphState(domain_state);
    }

    // with edge noise: create a (structurally) noisy transition model
    bayes_adaptive::factored::BABNModel model(
        &_domain_size,
        &_domain_feature_size,
        &_fbapomdp_step_size,
        _transition_model_without_block_features,
        _observation_model);

    // generate random structure for each obstacle
    for (auto f = _first_obstacle; f < _num_state_features; ++f)
    {

        auto parents = std::vector<std::vector<int>>(_NUM_ACTIONS);

        for (auto a = 0; a < _NUM_ACTIONS; ++a)
        {
            for (auto f_parent = 0; f_parent < _num_state_features; ++f_parent)
            {

                // add f_parent as parent randomly,
                // or for sure if we want to match and f_parent is itself
                if (rnd::boolean() || (f_parent == f && _edge_noise == "match-uniform"))
                {
                    parents[a].emplace_back(f_parent);
                }
            }
        }

        sampleBlockTModel(&model, f, std::move(parents));
    }

    return new FBAPOMDPState(domain_state, std::move(model));
}

void CollisionAvoidanceFactoredPrior::setAgentYTransition(Action const& a, int y, DBNNode& node)
    const
{
    // move directly (and deterministically) determines
    // the new y -- just making sure it stays on the grid here
    auto new_y = std::max(0, std::min(_height - 1, y + a.index() - 1));

    node.increment({y}, new_y);
}

std::vector<float> CollisionAvoidanceFactoredPrior::obstacleTransition(int y) const
{
    auto move_prob = static_cast<float>(.25 - .5 * _noise);

    // probability of staying is most definitely twice the probability
    // of moving- only not if the obstacle is at the top or bottom:
    // this is takes on 3 times as many probability
    auto stay_prob = (y == 0 || y == _height - 1) ? static_cast<float>(3 * .25 + .5 * _noise)
                                                  : static_cast<float>(2 * .25 + _noise);

    std::vector<float> position_counts(_height);

    if (y != 0)
    {
        position_counts[y - 1] = move_prob * _counts_total;
    }

    if (y != _domain_feature_size._S[_first_obstacle] - 1)
    {
        position_counts[y + 1] = move_prob * _counts_total;
    }

    position_counts[y] = stay_prob * _counts_total;

    return position_counts;
}

FBAPOMDPState*
    CollisionAvoidanceFactoredPrior::sampleFullyConnectedState(State const* domain_state) const
{
    return new FBAPOMDPState(
        domain_state,
        bayes_adaptive::factored::BABNModel(
            &_domain_size,
            &_domain_feature_size,
            &_fbapomdp_step_size,
            _fully_connected_transition_model,
            _observation_model));
}

FBAPOMDPState*
    CollisionAvoidanceFactoredPrior::sampleCorrectGraphState(State const* domain_state) const
{
    return new FBAPOMDPState(
        domain_state,
        bayes_adaptive::factored::BABNModel(
            &_domain_size,
            &_domain_feature_size,
            &_fbapomdp_step_size,
            _correctly_connected_transition_model,
            _observation_model));
}

bayes_adaptive::factored::BABNModel::Structure CollisionAvoidanceFactoredPrior::mutate(
    bayes_adaptive::factored::BABNModel::Structure structure) const
{
    assert(structure.T.size() == _NUM_ACTIONS);
    assert(structure.O.size() == _NUM_ACTIONS);

    for (auto a = 0; a < _NUM_ACTIONS; ++a)
    {
        for (auto i = 0; i < _num_obstacles; ++i)
        {
            // parents of observaiton nodes is always the obstacle itself
            assert(structure.O[a][i] == std::vector<int>({i + 2}));
        }
    }

    for (auto a = 0; a < _NUM_ACTIONS; ++a)
    {
        // we have 2 nodes for each action in T + number of obstacles
        assert(structure.T[a].size() == static_cast<unsigned int>(_num_obstacles + 2));
    }

    // the structure noise in this problem is the transition
    // function of the obstacle for each action. Here we pick which to change
    auto const a  = _action_distr(rnd::rng());
    auto obstacle = 2 + _obst_distr(rnd::rng());

    // then we just flip an edge in that structure
    bayes_adaptive::factored::BABNModel::Structure::flip_random_edge(
        &structure.T[a][obstacle], // '2' == object feature
        _domain_feature_size._S.size() // max features to add or remove
    );

    return structure;
}

bayes_adaptive::factored::BABNModel CollisionAvoidanceFactoredPrior::computePriorModel(
    bayes_adaptive::factored::BABNModel::Structure const& structure) const
{
    assert(structure.T.size() == static_cast<size_t>(_domain_size._A));
    assert(structure.O.size() == static_cast<size_t>(_domain_size._A));

    for (auto a = 0; a < _NUM_ACTIONS; ++a)
    {
        assert(structure.T[a].size() == static_cast<size_t>(_num_state_features));
        assert(structure.O[a].size() == static_cast<size_t>(_num_obstacles));
    }

    // load all known parts of the prior first
    auto model = bayes_adaptive::factored::BABNModel(
        &_domain_size,
        &_domain_feature_size,
        &_fbapomdp_step_size,
        _transition_model_without_block_features,
        _observation_model);

    for (auto f = _first_obstacle; f < _num_state_features; ++f)
    {

        // extract parents of block feature from structure
        std::vector<std::vector<int>> block_feature_structure(_domain_size._A);
        for (auto a = 0; a < _domain_size._A; ++a)
        { block_feature_structure[a] = structure.T[a][f]; }

        // actually add a model for the block feature according
        // to the extracted structure
        sampleBlockTModel(&model, f, block_feature_structure);
    }

    return model;
}

void CollisionAvoidanceFactoredPrior::sampleBlockTModel(
    bayes_adaptive::factored::BABNModel* model,
    int obstacle_feature,
    std::vector<std::vector<int>> structure) const
{
    // populate node according to its parents
    for (auto a = 0; a < _NUM_ACTIONS; ++a)
    {
        auto const action = IndexAction(a);

        // extract some info from parents
        auto parent_values = std::vector<int>();
        auto parent_ranges = std::vector<int>();
        auto correct_edge  = -1;

        for (auto const& p : structure[a])
        {

            if (p == obstacle_feature)
            {
                correct_edge = parent_values.size();
            }

            parent_values.emplace_back(0);
            parent_ranges.emplace_back(_domain_feature_size._S[p]);
        }

        // set counts according to parents
        model->resetTransitionNode(&action, obstacle_feature, structure[a]);

        if (correct_edge != -1)
        {

            do
            {
                model->transitionNode(&action, obstacle_feature)
                    .setDirichletDistribution(
                        parent_values, obstacleTransition(parent_values[correct_edge]));
            } while (!indexing::increment(parent_values, parent_ranges));

        } else
        {

            // not correct edge: uniform prior
            auto counts = std::vector<float>(
                _domain_feature_size._S[obstacle_feature],
                _counts_total / static_cast<float>(_domain_feature_size._S[obstacle_feature]));

            // base case: no parents
            if (parent_values.empty())
            {
                model->transitionNode(&action, obstacle_feature)
                    .setDirichletDistribution({0}, std::move(counts));
            } else
            {
                // at least one parent
                do
                {
                    model->transitionNode(&action, obstacle_feature)
                        .setDirichletDistribution(parent_values, counts);
                } while (!indexing::increment(parent_values, parent_ranges));
            }
        }
    }
}

} // namespace priors
