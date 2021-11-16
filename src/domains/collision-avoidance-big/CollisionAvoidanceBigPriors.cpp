#include "CollisionAvoidanceBigPriors.hpp"

#include <algorithm>
#include <cstdlib>
#include <bayes-adaptive/states/factored/AbstractFBAPOMDPState.hpp>

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
std::vector<float> observationDistrBig(int height, std::vector<int> const& obstacles_pos)
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
std::vector<float> observationDistrBig(int height, int obstacle_pos)
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

CollisionAvoidanceBigTablePrior::CollisionAvoidanceBigTablePrior(
    domains::CollisionAvoidanceBig const& d,
    configurations::BAConf const& c) :
        _height(c.domain_conf.height),
        _width(c.domain_conf.width),
        _num_obstacles(c.domain_conf.size),
        _noise(c.noise),
        _total_counts(c.counts_total),
        _domain_size(
            static_cast<int>(_width * std::pow(_height, _num_obstacles + 1) * _num_speeds * _num_traffics * _num_timeofdays * _num_obstacles),
            _NUM_ACTIONS,
            static_cast<int>(std::pow(_height, _num_obstacles))),
        _prior(&_domain_size)
{
    // TODO fix this function if it's needed for the factored things
    assert(c.noise < .5 && c.noise > -.5);

    {
        // prepare observation probabilities (distribution per location for a single obstacle)
        std::vector<std::vector<float>> obs_distr(_height);
        for (auto i = 0; i < _height; ++i) { obs_distr[i] = observationDistrBig(_height, i); }

        std::vector<int> obstacles(_num_obstacles);
        do {
            for (auto x = 0; x < _width; ++x) {
                for (auto y = 0; y < _height; ++y) {
                    for (auto speed = 0; speed < _num_speeds; ++speed) {
                        for (auto traffic = 0; traffic < _num_traffics; ++traffic) {
                            for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) {
                                for (auto obstacletype = 0; obstacletype < _num_obstacletypes; ++obstacletype) {
                                    auto const s = d.getState(x, y, speed, traffic, timeofday, obstacletype, obstacles);

                                    for (auto a = 0; a < _NUM_ACTIONS; ++a) {
                                        IndexAction const action(std::to_string(a));

                                        // loop over position of obstacles
                                        std::vector<int> new_obstacles(_num_obstacles);
                                        auto count = 0;
                                        do {
                                            IndexObservation const o(std::to_string(count));

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
                        }
                    }
                }
            }

        } while (!indexing::increment(obstacles, _obstacle_pos_ranges));
    }
}

BAPOMDPState* CollisionAvoidanceBigTablePrior::sampleBAPOMDPState(State const* domain_state) const
{
    return new BAPOMDPState(domain_state, _prior);
}

double CollisionAvoidanceBigTablePrior::obstacleTransProb(int y, int new_y) const
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

void CollisionAvoidanceBigTablePrior::setTransitionCounts(
    int x,
    int y,
    int a,
    std::vector<int> const& obstacles,
    std::vector<int> const& new_obstacles,
    domains::CollisionAvoidanceBig const& d)
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

    auto const s      = d.getState(x, y, 0, 0, 0, 0, obstacles); // TODO change if I need this
    auto const new_s  = d.getState(new_x, new_y, 0, 0, 0, 0, new_obstacles); // TODO change if I need this
    auto const action = IndexAction(std::to_string(a));

    _prior.count(s, &action, new_s) = static_cast<float>(prob * _total_counts);
}

CollisionAvoidanceBigFactoredPrior::CollisionAvoidanceBigFactoredPrior(
    configurations::FBAConf const& conf) :
        FBAPOMDPPrior(conf),
        _abstraction(conf.domain_conf.abstraction),
        _num_obstacles(conf.domain_conf.size),
        _width(conf.domain_conf.width),
        _height(conf.domain_conf.height),
        _noise(conf.noise),
        _counts_total(conf.counts_total),
        _edge_noise(conf.structure_prior),
        _domain_size(
            _width * _height * _num_speeds * _num_traffics * _num_timeofdays * _num_obstacletypes * static_cast<int>(std::pow(_height, _num_obstacles)),
            _NUM_ACTIONS,
            _width * _num_timeofdays * _num_obstacletypes * static_cast<int>(std::pow(_height, _num_obstacles))),
        _domain_feature_size(
            std::vector<int>(_first_obstacle + _num_obstacles, _height),
            std::vector<int>(3 + _num_obstacles, _height)),
        _fbapomdp_step_size({}, {})
{
    const_cast<Domain_Feature_Size&>(_domain_feature_size)._S[0] = _width;
    const_cast<Domain_Feature_Size&>(_domain_feature_size)._S[_SPEED_FEATURE] = _num_speeds;
    const_cast<Domain_Feature_Size&>(_domain_feature_size)._S[_TRAFFIC_FEATURE] = _num_traffics;
    const_cast<Domain_Feature_Size&>(_domain_feature_size)._S[_TIMEOFDAY_FEATURE] = _num_timeofdays;
    const_cast<Domain_Feature_Size&>(_domain_feature_size)._S[_OBTACLETYPE_FEATURE] = _num_obstacletypes;
    const_cast<Domain_Feature_Size&>(_domain_feature_size)._O[0] = _width;
    const_cast<Domain_Feature_Size&>(_domain_feature_size)._O[1] = _num_timeofdays;
    const_cast<Domain_Feature_Size&>(_domain_feature_size)._O[2] = _num_obstacletypes;
    const_cast<bayes_adaptive::factored::BABNModel::Indexing_Steps&>(_fbapomdp_step_size) =
        bayes_adaptive::factored::BABNModel::Indexing_Steps(
            indexing::stepSize(_domain_feature_size._S),
            indexing::stepSize(_domain_feature_size._O));

    if (_noise > .5 || _noise < -.5)
    {
        throw "CollisionAvoidanceBigFactoredPrior must be intiiated with -.5 < noise < .5 (is: "
            + std::to_string(_noise) + ")";
    }

    std::vector<std::string> accepted_priors = {
        "uniform", "match-counts", "match-uniform", "fully-connected"};

    if (!_edge_noise.empty()
        && std::find(accepted_priors.begin(), accepted_priors.end(), _edge_noise)
               == accepted_priors.end())
    {
        throw "CollisionAvoidanceBigFactoredPrior does not accept " + _edge_noise + " as noise";
    }

    bayes_adaptive::factored::BABNModel model(
        &_domain_size, &_domain_feature_size, &_fbapomdp_step_size);

    /*** set known part of the transition function (how agent transitions on Y) ***/
    for (auto a = 0; a < _NUM_ACTIONS; a++)
    {
        auto action = IndexAction(std::to_string(a));

        // set agent y feature [1] for each action
        model.resetTransitionNode(&action, _AGENT_Y_FEATURE, {_AGENT_Y_FEATURE});
        for (auto y = 0; y < _height; ++y)
        { setAgentYTransition(action, y, model.transitionNode(&action, _AGENT_Y_FEATURE)); }
    }

    /*** new transitions for speed, traffic and timeofday, when known ***/
    for (auto a = 0; a < _NUM_ACTIONS; a++) {
        auto action = IndexAction(std::to_string(a));

        // set agent x feature [0] for each action
        model.resetTransitionNode(&action, _AGENT_X_FEATURE, {_AGENT_X_FEATURE, _SPEED_FEATURE});
        for (auto x = 1; x < _width; ++x) {
            for (auto speed = 0; speed < _num_speeds; ++speed) {  // loop over node input
                model.transitionNode(&action, _AGENT_X_FEATURE)
                        .setDirichletDistribution({x, speed}, xTransition(x, speed));
            }
        }

        // set speed feature for each action
        model.resetTransitionNode(&action, _SPEED_FEATURE, {_SPEED_FEATURE, _TRAFFIC_FEATURE});
        for (auto speed = 0; speed < _num_speeds; ++speed) {  // loop over node input
            for (auto traffic = 0; traffic < _num_traffics; ++traffic) { // loop over node input
                model.transitionNode(&action, _SPEED_FEATURE)
                        .setDirichletDistribution({speed, traffic}, speedTransition(speed, traffic));
            }
        }

        // set traffic feature for each action
        model.resetTransitionNode(&action, _TRAFFIC_FEATURE, {_TRAFFIC_FEATURE, _TIMEOFDAY_FEATURE});
        for (auto traffic = 0; traffic < _num_traffics; ++traffic) {  // loop over node input
            for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) { // loop over node input
                model.transitionNode(&action, _TRAFFIC_FEATURE)
                        .setDirichletDistribution({traffic, timeofday}, trafficTransition(traffic, timeofday));
            }
        }

        // set time of day feature for each action
        model.resetTransitionNode(&action, _TIMEOFDAY_FEATURE, {_TIMEOFDAY_FEATURE});
        for (auto timeofday = 0; timeofday < _num_timeofdays; ++timeofday) { // loop over node input
            model.transitionNode(&action, _TIMEOFDAY_FEATURE)
                    .setDirichletDistribution({timeofday}, timeofdayTransition(timeofday));
        }

        // set obstacle type feature for each action
        model.resetTransitionNode(&action, _OBTACLETYPE_FEATURE, {_OBTACLETYPE_FEATURE});
        for (auto obstacletype = 0; obstacletype < _num_obstacletypes; ++obstacletype) { // loop over node input
            model.transitionNode(&action, _OBTACLETYPE_FEATURE)
                    .setDirichletDistribution({obstacletype}, obstacletypeTransition(obstacletype));
        }
    }

    /////// store it
    _transition_model_without_block_features = model.copyT();

    /*** compute transition when edges are known ***/
    for (auto a = 0; a < _NUM_ACTIONS; a++) {
        auto action = IndexAction(std::to_string(a));
        // set block y features for each action
        for (auto f = _first_obstacle; f < (_first_obstacle + _num_obstacles); ++f) {
            model.resetTransitionNode(&action, f, {_SPEED_FEATURE, _OBTACLETYPE_FEATURE, f});
            for (auto y = 0; y < _height; ++y) {
                for (auto speed = 0; speed < _num_speeds; ++speed) {
                    for (auto obstacletype = 0; obstacletype < _num_obstacletypes; ++obstacletype) { // loop over node input
                        model.transitionNode(&action, f)
                                .setDirichletDistribution({speed, obstacletype, y}, obstacleTransition(y, speed, obstacletype)); // TODO does the order here matter...?
                    }
                }
            }
        }
    }

    ////// store it
    _correctly_connected_transition_model = model.copyT();

    /**** compute observation dynamics (all are known) ****/
    for (auto a = 0; a < _NUM_ACTIONS; ++a)
    {
        auto action = IndexAction(std::to_string(a));

        // X position
        model.resetObservationNode(&action, 0, {_AGENT_X_FEATURE});
        for (auto x = 0; x < _width; ++x) {
            std::vector<float> x_obs_counts(_width, 0);
            x_obs_counts[x] += 1;

            model.observationNode(&action, 0).setDirichletDistribution({x}, x_obs_counts);
        }

        // Time of day
        model.resetObservationNode(&action, 1, {_TIMEOFDAY_FEATURE});
        for (auto tod = 0; tod < _num_timeofdays; ++tod) {
            std::vector<float> tod_obs_counts(_num_timeofdays, 0);
            tod_obs_counts[tod] += 1;

            model.observationNode(&action, 1).setDirichletDistribution({tod}, tod_obs_counts);
        }

        // Obstacle type
        model.resetObservationNode(&action, 2, {_OBTACLETYPE_FEATURE});
        for (auto obsttype = 0; obsttype < _num_obstacletypes; ++obsttype) {
            std::vector<float> obsttype_obs_counts(_num_obstacletypes, 0);
            obsttype_obs_counts[obsttype] += 1;

            model.observationNode(&action, 2).setDirichletDistribution({obsttype}, obsttype_obs_counts);
        }

        // Obstacles
        for (auto f = 3; f < 3 + _num_obstacles; ++f) {
            model.resetObservationNode(&action, f, {f - 3 + _first_obstacle});

            for (auto y = 0; y < _height; ++y) {
                auto obsDistr = observationDistrBig(_height, y);
                for (auto& p : obsDistr) p *= 10000;

                model.observationNode(&action, f).setDirichletDistribution({y}, obsDistr);
            }
        }
    }

    ////// store it
    _observation_model = model.copyO();

    /**** compute fully connected transition nodes, using the regular as starting point ****/
    // setup fully parents & their values
//    auto parents       = std::vector<int>(_num_state_features);
//    auto parent_values = std::vector<int>(_num_state_features);
//    auto parent_ranges = std::vector<int>(_num_state_features);
//
//    for (auto i = 0; i < _num_state_features; ++i)
//    {
//        parents[i]       = i;
//        parent_ranges[i] = _domain_feature_size._S[i];
//    }
//
//    for (auto a = 0; a < _NUM_ACTIONS; ++a)
//    {
//        auto action = IndexAction(std::to_string(a));
//
//        for (auto f = _first_obstacle; f < (_first_obstacle + _num_obstacles); ++f)
//        {
//
//            model.resetTransitionNode(&action, f, parents);
//            do
//            {
//                model.transitionNode(&action, f)
//                    .setDirichletDistribution(parent_values, obstacleTransition(parent_values[f]));
//            } while (!indexing::increment(parent_values, parent_ranges));
//        }
//    }
//
//    ///// store it
//    _fully_connected_transition_model = model.copyT();
}

FBAPOMDPState* CollisionAvoidanceBigFactoredPrior::sampleFBAPOMDPState(State const* domain_state) const
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
    for (auto f = _first_obstacle; f < (_first_obstacle + _num_obstacles); ++f) {
        auto parents = std::vector<std::vector<int>>(_NUM_ACTIONS);


        for (auto f_parent = 0; f_parent < _num_state_features; ++f_parent) {
            // add f_parent as parent randomly,
            // or for sure if we want to match and f_parent is itself
            // TODO perhaps make this part of the environment, the probability for a feature to be added as parent
            if ((rnd::slowRandomInt(1,100) <= 50) || (f_parent == f && _edge_noise == "match-uniform")) {
                for (auto a = 0; a < _NUM_ACTIONS; ++a) {
                    parents[a].emplace_back(f_parent);
                }
            }
        }
        sampleBlockTModel(&model, f, std::move(parents));
    }

    if (_abstraction) {
        return new AbstractFBAPOMDPState(domain_state, std::move(model));
    }
    return new FBAPOMDPState(domain_state, std::move(model));
}

void CollisionAvoidanceBigFactoredPrior::setAgentYTransition(Action const& a, int y, DBNNode& node)
    const
{
    // move directly (and deterministically) determines
    // the new y -- just making sure it stays on the grid here
    auto new_y = std::max(0, std::min(_height - 1, y + std::stoi(a.index()) - 1));

    node.increment({y}, new_y);
}

std::vector<float> CollisionAvoidanceBigFactoredPrior::obstacleTransition(int y, int speed, int obstacletype) const
{
    // Only used when correct graph is used
    // TODO change this if I want to use this as a "wrong" prior
    // overestimate the move speed when the relative speed is low, underestimate the movement when the speed is high
//    auto move_prob = static_cast<float>(.2 + 0.05*(obstacletype+1)*speed - 0.05*(obstacletype+1)*(speed - 1)-.5 * _noise*speed + -.5 *_noise*(speed -1));
    // probability that it moves up, as well as for moving down (i.e. the probability that the block moves is 2* move_prob)
    auto move_prob = 0.5*(BLOCK_MOVE_PROB + 0.1*(obstacletype+1)*speed + 0.1*(obstacletype+1)*(speed - 1));

    // probability of staying is different if the obstacle is at the top or bottom:
    auto stay_prob = (y == 0 || y == _height - 1) ? static_cast<float>(1 - move_prob)
                                                  : static_cast<float>(1 - 2*move_prob);

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

//std::vector<float> CollisionAvoidanceBigFactoredPrior::obstacleTransition(int y, int speed) {
//    float move_prob = 1.0/3 + 1.0/6 * speed - _noise * std::max(2.f/3, (float) (speed%2));
//
//    // probability of staying is 1 - the probability
//    // of moving- only not if the obstacle is at the top or bottom:
//    // then it is 1 - 0.5 * the probability of moving
//    auto stay_prob = (y == 0 || y == _height - 1) ? (1 - 0.5 * move_prob) : (1 - move_prob);
//
//    std::vector<float> position_counts(_height);
//
//    if (y != 0)
//    {
//        position_counts[y - 1] = move_prob * 0.5 * _counts_total;
//    }
//
//    if (y != _domain_feature_size._S[_first_obstacle] - 1)
//    {
//        position_counts[y + 1] = move_prob * 0.5 * _counts_total;
//    }
//
//    position_counts[y] = stay_prob * _counts_total;
//
//    return position_counts;
//}


int CollisionAvoidanceBigFactoredPrior::keepInBounds(int value) const {
    return std::max(0, std::min(2, value));
}

std::vector<float> CollisionAvoidanceBigFactoredPrior::xTransition(int x, int speed) const {
    // TODO use _noise here?
    std::vector<float> x_counts(_width, 0);

    if (speed == 1) {
        x_counts[x - 1] += MOVE_PROB_FAST * _counts_total;
        x_counts[std::max(x - 2, 0)] += (1 - MOVE_PROB_FAST) * _counts_total;
    } else {
        x_counts[x - 1] += MOVE_PROB_SLOW * _counts_total;
        x_counts[x] += (1 - MOVE_PROB_SLOW) * _counts_total;
    }

    return x_counts;
}

std::vector<float> CollisionAvoidanceBigFactoredPrior::speedTransition(int speed, int traffic) const {
    // TODO use _noise here?
    std::vector<float> speed_counts(_num_speeds, 0);

    if (speed == 1) {
        if (traffic == 2) {
            speed_counts[keepInBounds(speed - 1)] += 0.75 * _counts_total;
            speed_counts[speed] += 0.25 * _counts_total;
        } else if (traffic == 1) {
            speed_counts[keepInBounds(speed - 1)] += 0.5 * _counts_total;
            speed_counts[speed] += 0.5 * _counts_total;
        } else {
            speed_counts[keepInBounds(speed - 1)] += 0.25 * _counts_total;
            speed_counts[speed] += 0.75 * _counts_total;
        }
    } else {
        if (traffic == 2) {
            speed_counts[keepInBounds(speed + 1)] += 0.25 * _counts_total;
            speed_counts[speed] += 0.75 * _counts_total;
        } else if (traffic == 1) {
            speed_counts[keepInBounds(speed + 1)] += 0.5 * _counts_total;
            speed_counts[speed] += 0.5 * _counts_total;
        } else {
            speed_counts[keepInBounds(speed + 1)] += 0.75 * _counts_total;
            speed_counts[speed] += 0.25 * _counts_total;
        }
    }

    return speed_counts;
}

std::vector<float> CollisionAvoidanceBigFactoredPrior::trafficTransition(int traffic, int timeofday) const {
    // TODO use _noise here?
    std::vector<float> traffic_counts(_num_traffics, 0);

    if (traffic == 2) {
        if (timeofday == 1) {
            traffic_counts[traffic] += 0.8 * _counts_total;
            traffic_counts[keepInBounds(traffic - 1)] += 0.2 * _counts_total;
        } else {
            traffic_counts[traffic] += 0.2 * _counts_total;
            traffic_counts[keepInBounds(traffic - 1)] += 0.8 * _counts_total;
        }
    } else if (traffic == 1) {
        if (timeofday == 1) {
            traffic_counts[traffic] += 0.6 * _counts_total;
            traffic_counts[keepInBounds(traffic + 1)] += 0.3 * _counts_total;
            traffic_counts[keepInBounds(traffic - 1)] += 0.1 * _counts_total;
        } else {
            traffic_counts[traffic] += 0.6 * _counts_total;
            traffic_counts[keepInBounds(traffic + 1)] += 0.1 * _counts_total;
            traffic_counts[keepInBounds(traffic - 1)] += 0.3 * _counts_total;
        }
    } else {
        if (timeofday == 1) {
            traffic_counts[traffic] += 0.2 * _counts_total;
            traffic_counts[keepInBounds(traffic + 1)] += 0.8 * _counts_total;
        } else {
            traffic_counts[traffic] += 0.8 * _counts_total;
            traffic_counts[keepInBounds(traffic + 1)] += 0.2 * _counts_total;
        }
    }

    return traffic_counts;
}

std::vector<float> CollisionAvoidanceBigFactoredPrior::timeofdayTransition(int timeofday) const {
    std::vector<float> timeofday_counts(_num_timeofdays, 0);

    timeofday_counts[timeofday] += 1;

    return timeofday_counts;
}

std::vector<float> CollisionAvoidanceBigFactoredPrior::obstacletypeTransition(int obstacletype) const {
    std::vector<float> obstacletype_counts(_num_obstacletypes, 0);

    obstacletype_counts[obstacletype] += 1;

    return obstacletype_counts;
}

FBAPOMDPState*
    CollisionAvoidanceBigFactoredPrior::sampleFullyConnectedState(State const* domain_state) const
{
    if (_abstraction) {
        return new AbstractFBAPOMDPState(
                domain_state,
                bayes_adaptive::factored::BABNModel(
                        &_domain_size,
                        &_domain_feature_size,
                        &_fbapomdp_step_size,
                        _fully_connected_transition_model,
                        _observation_model));
    }
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
    CollisionAvoidanceBigFactoredPrior::sampleCorrectGraphState(State const* domain_state) const
{
    if (_abstraction) {
        return new AbstractFBAPOMDPState(
                domain_state,
                bayes_adaptive::factored::BABNModel(
                        &_domain_size,
                        &_domain_feature_size,
                        &_fbapomdp_step_size,
                        _correctly_connected_transition_model,
                        _observation_model));
    }
    return new FBAPOMDPState(
        domain_state,
        bayes_adaptive::factored::BABNModel(
            &_domain_size,
            &_domain_feature_size,
            &_fbapomdp_step_size,
            _correctly_connected_transition_model,
            _observation_model));
}

bayes_adaptive::factored::BABNModel::Structure CollisionAvoidanceBigFactoredPrior::mutate(
    bayes_adaptive::factored::BABNModel::Structure structure) const
{
    // TODO change if I want to use reinvigoration
    assert(structure.T.size() == _NUM_ACTIONS);
    assert(structure.O.size() == _NUM_ACTIONS);

    for (auto a = 0; a < _NUM_ACTIONS; ++a)
    {
        for (auto i = 0; i < _num_obstacles; ++i)
        {
            // parents of observation nodes is always the obstacle itself
            assert(structure.O[a][i] == std::vector<int>({i + _first_obstacle}));
        }
    }

    for (auto a = 0; a < _NUM_ACTIONS; ++a)
    {
        // we have 2 nodes for each action in T + number of obstacles (TODO now not 2 but 5? should be equal to _first _obstacle)
        assert(structure.T[a].size() == static_cast<unsigned int>(_num_obstacles + _first_obstacle));
    }

    // the structure noise in this problem is the transition
    // function of the obstacle for each action. Here we pick which to change
    auto const a  = _action_distr(rnd::rng());
    auto obstacle = _first_obstacle + _obst_distr(rnd::rng());

    // then we just flip an edge in that structure
    bayes_adaptive::factored::BABNModel::Structure::flip_random_edge(
        &structure.T[a][obstacle], // '2' == object feature
        _domain_feature_size._S.size() // max features to add or remove
    );

    return structure;
}

bayes_adaptive::factored::BABNModel CollisionAvoidanceBigFactoredPrior::computePriorModel(
    bayes_adaptive::factored::BABNModel::Structure const& structure) const
{
    assert(structure.T.size() == static_cast<size_t>(_domain_size._A));
    assert(structure.O.size() == static_cast<size_t>(_domain_size._A));

    for (auto a = 0; a < _NUM_ACTIONS; ++a) {
        assert(structure.T[a].size() == static_cast<size_t>(_num_state_features));
        assert(structure.O[a].size() == static_cast<size_t>(_num_obstacles + 3));
    }

    // load all known parts of the prior first
    auto model = bayes_adaptive::factored::BABNModel(
        &_domain_size,
        &_domain_feature_size,
        &_fbapomdp_step_size,
        _transition_model_without_block_features,
        _observation_model);

    for (auto f = _first_obstacle; f < (_first_obstacle + _num_obstacles); ++f) {
        // extract parents of block feature from structure
        std::vector<std::vector<int>> block_feature_structure(_domain_size._A);
        for (auto a = 0; a < _domain_size._A; ++a) {
            block_feature_structure[a] = structure.T[a][f];
        }

        // actually add a model for the block feature according
        // to the extracted structure
        sampleBlockTModel(&model, f, block_feature_structure);
    }

    return model;
}

void CollisionAvoidanceBigFactoredPrior::sampleBlockTModel(
    bayes_adaptive::factored::BABNModel* model,
    int obstacle_feature,
    std::vector<std::vector<int>> structure) const
{
    // populate node according to its parents
    for (auto a = 0; a < _NUM_ACTIONS; ++a) {
        auto const action = IndexAction(std::to_string(a));

        // extract some info from parents
        auto parent_values = std::vector<int>();
        auto parent_ranges = std::vector<int>();
//        auto correct_edge  = -1;

        for (auto const& p : structure[a]) {

//            if (p == obstacle_feature)
//            {
//                correct_edge = parent_values.size();
//            }

            parent_values.emplace_back(0);
            parent_ranges.emplace_back(_domain_feature_size._S[p]);
        }

        // set counts according to parents
        model->resetTransitionNode(&action, obstacle_feature, structure[a]);

//        if (correct_edge != -1) { // if the feature itself is part of the parents...?
//            do {
//                model->transitionNode(&action, obstacle_feature)
//                    .setDirichletDistribution(
//                        parent_values, obstacleTransition(parent_values[correct_edge]));
//            } while (!indexing::increment(parent_values, parent_ranges));
//
//        } else { // if the feature itself is not part of the parents...?
            // not correct edge: uniform prior
        auto counts = std::vector<float>(
            _domain_feature_size._S[obstacle_feature],
            _counts_total / static_cast<float>(_domain_feature_size._S[obstacle_feature]));

        // base case: no parents
        if (parent_values.empty()) {
            model->transitionNode(&action, obstacle_feature)
                .setDirichletDistribution({0}, std::move(counts));
        } else {
            // at least one parent
            do {
                model->transitionNode(&action, obstacle_feature)
                    .setDirichletDistribution(parent_values, counts);
            } while (!indexing::increment(parent_values, parent_ranges));
        }
//        }
    }
}

} // namespace priors
