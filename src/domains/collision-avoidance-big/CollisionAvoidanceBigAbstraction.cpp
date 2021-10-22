#include <bayes-adaptive/models/factored/Domain_Feature_Size.hpp>
#include <bayes-adaptive/models/Domain_Size.hpp>
#include <bayes-adaptive/models/factored/FBAPOMDP.hpp>
#include "CollisionAvoidanceBigAbstraction.hpp"

#include "utils/index.hpp"

abstractions::CollisionAvoidanceBigAbstraction::CollisionAvoidanceBigAbstraction(
        configurations::BAConf c) :
        _num_abstractions(),
        _minimum_abstraction(),
        _abstraction_updating(false),
        _abstraction_normalization(c.planner_conf.update_abstract_model_normalized),
        _qvalues(),
        _counts(),
        _abstract_domain_sizes(),
        _abstract_domain_feature_sizes(),
        _abstract_step_sizes()
{
    // TODO now assuming that size = 1, and that num_speeds = num_traffics
    // abstraction, 0 = {x, y, obst1}
    // abstraction, 1 = {x, y, obst1, speed} (minimum)
    // abstraction, 2 = {x, y, ,obst1, speed, traffic} (minimum)
    // abstraction, 3 = {x, y, ,obst1, speed, traffic, timeofday} (maximum full model)
    // abstraction, 4 = full model
    // minimum = also possible to have extra features if connected to obst1
    _num_abstractions    = 5;
    // x, y, speed, traffic, timeofday, obst (0, 1, 2, 3, 4, 5)
    _minimum_abstraction = {0,1,2};
    _qvalues = std::vector<double> (_num_abstractions,0);
    _counts = std::vector<int> (_num_abstractions, 0);

    // Possible combinations (everything contains x, y, obst at least)
    // { }, {speed/traffic}, {timeofday}, {speed/traffic, speed/traffic}, {speed/traffic, timeofday},
    // {speed, traffic, timeofday}
    // speed: 3, traffic: 4, timeofday: 5
    _abstract_domain_sizes = std::vector<Domain_Size> (6,
           Domain_Size(c.domain_conf.width * c.domain_conf.height * static_cast<int>(std::pow(c.domain_conf.height, c.domain_conf.size)),
           3, static_cast<int>(std::pow(c.domain_conf.height, c.domain_conf.size))));
    _abstract_domain_feature_sizes = std::vector<Domain_Feature_Size> (6,
                       Domain_Feature_Size(std::vector<int>(2 + c.domain_conf.size, c.domain_conf.height),
                                           std::vector<int>(c.domain_conf.size, c.domain_conf.height)));
    _abstract_step_sizes = std::vector<bayes_adaptive::factored::BABNModel::Indexing_Steps> (6,
                                                                                             bayes_adaptive::factored::BABNModel::Indexing_Steps(
                                                                                                     indexing::stepSize(_abstract_domain_feature_sizes[0]._S),
                                                                                                     indexing::stepSize(_abstract_domain_feature_sizes[0]._O)));

    std::vector<int> num_features =  {4, 4, 5, 5, 6}; // first one is skipped below
    std::vector<std::vector<int>> feature_sizes_sets = {{_num_speeds}, {_num_timeofdays}, {_num_speeds, _num_traffics},
                                   {_num_speeds, _num_timeofdays}, {_num_speeds, _num_traffics, _num_timeofdays}};
    for (unsigned int i = 1; i < 6; ++i) {
        int dom_size = c.domain_conf.width * c.domain_conf.height * static_cast<int>(std::pow(c.domain_conf.height, c.domain_conf.size));

        auto feature_sizes = std::vector<int>(num_features[i-1], c.domain_conf.height);
        feature_sizes[0] = c.domain_conf.width;

        for (int j = 0; j < num_features[i-1]-3; j++) {
            dom_size *= feature_sizes_sets[i-1][j];
            feature_sizes[j+3] = feature_sizes_sets[i-1][j];
        }
        _abstract_domain_sizes[i] = Domain_Size(dom_size, 3, static_cast<int>(std::pow(c.domain_conf.height, c.domain_conf.size)));

        _abstract_domain_feature_sizes[i] = Domain_Feature_Size(feature_sizes,
                                    std::vector<int>(c.domain_conf.size, c.domain_conf.height));
        _abstract_step_sizes[i] = bayes_adaptive::factored::BABNModel::Indexing_Steps(
                indexing::stepSize(_abstract_domain_feature_sizes[i]._S),
                indexing::stepSize(_abstract_domain_feature_sizes[i]._O));
    }
}

bayes_adaptive::factored::BABNModel
abstractions::CollisionAvoidanceBigAbstraction::constructAbstractModel(bayes_adaptive::factored::BABNModel model, int k, POMDP const& domain) {
    std::vector<int> abstraction_set; ; // = {};
    unsigned int index_to_use;
    if (k == 0) {
        abstraction_set = _minimum_abstraction;
    } else if (k >= 1 && k < 4) {
        // We assume the same model for each action here
        // TODO Could make one abstraction_set per action?
        auto action = IndexAction(0);
        std::set_union(_minimum_abstraction.begin(),
                       _minimum_abstraction.end(),
                       model.transitionNode(&action, 2).parents()->begin(),
                       model.transitionNode(&action, 2).parents()->end(),
                       std::back_inserter(abstraction_set));
        if (k > 1) {
            for (int i = 1; i < k; ++i) {
                std::vector<int> to_loop_over;
                std::vector<int> new_set = abstraction_set;
                std::set_difference(_minimum_abstraction.begin(),
                               _minimum_abstraction.end(),
                                    abstraction_set.begin(),
                                    abstraction_set.end(),
                               std::back_inserter(to_loop_over));
                for (auto const& value: to_loop_over) {
                    std::set_union(abstraction_set.begin(),
                                   abstraction_set.end(),
                                   model.transitionNode(&action, value).parents()->begin(),
                                   model.transitionNode(&action, value).parents()->end(),
                                   std::back_inserter(new_set));
                }
                abstraction_set = new_set;
            }
        }
    } else {
        return model;
    }

    if (abstraction_set.size() == 3) { // {x, y, obst}
        index_to_use = 0;
    } else if (abstraction_set.size() == 6) { // {x, y, obst, speed, traffic, timeofday}
        index_to_use = 5;
    } else if (abstraction_set.size() == 4) {
        if (abstraction_set[3] == _tod_feature) {
            index_to_use = 2;
        } else {
            index_to_use = 1;
        }
    } else { // size 5
        if (abstraction_set[4] == _tod_feature) {
            index_to_use = 4;
        } else {
            index_to_use = 3;
        }
    }

    if (_abstraction_normalization) {
        auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(domain);
        auto priorModel = fbapomdp.prior()->computePriorModel(model.structure());
        auto abstractPriorModel_normalized = priorModel.abstract(abstraction_set, model.structure(), &_abstract_domain_sizes[index_to_use],
                                                                 &_abstract_domain_feature_sizes[index_to_use],
                                                                 &_abstract_step_sizes[index_to_use], true);
        auto abstractPriorModel = priorModel.abstract(abstraction_set, model.structure(), &_abstract_domain_sizes[index_to_use],
                                                      &_abstract_domain_feature_sizes[index_to_use],
                                                      &_abstract_step_sizes[index_to_use], false);
        auto newModel = model.abstract(abstraction_set, model.structure(), &_abstract_domain_sizes[index_to_use],
                                       &_abstract_domain_feature_sizes[index_to_use],
                                       &_abstract_step_sizes[index_to_use], false);
        newModel.abstractionNormalizeCounts(abstractPriorModel, abstractPriorModel_normalized);
        return newModel;
    } else {
        return model.abstract(abstraction_set, model.structure(), &_abstract_domain_sizes[index_to_use],
                              &_abstract_domain_feature_sizes[index_to_use],
                              &_abstract_step_sizes[index_to_use], false);
    }
}

int abstractions::CollisionAvoidanceBigAbstraction::selectAbstraction() {
    // Check if we haven't chosen any
    int totalCounts = 0;

    for (int i = 0; i < _num_abstractions; i++) {
        if (_counts[i] == 0) {
            _counts[i]++;
            return i;
        }
        totalCounts += _counts[i];
    }

    int toSelect = 0;
    double maxvalue = _qvalues[0]/_counts[0] + _UCB_exploration * sqrt(log1p(totalCounts) / _counts[0]);

    for (int i = 1; i < _num_abstractions; i++) {
        if (_qvalues[i]/_counts[i] + _UCB_exploration * sqrt(log1p(totalCounts) / _counts[i]) > maxvalue) {
            maxvalue = _qvalues[i]/_counts[i] + _UCB_exploration * sqrt(log1p(totalCounts) / _counts[i]);
            toSelect = i;
        }
    }
    _counts[toSelect]++;
    return toSelect;
}

void abstractions::CollisionAvoidanceBigAbstraction::addReturn(int abstraction, double reward) {
    _qvalues[abstraction] += reward;
}

int abstractions::CollisionAvoidanceBigAbstraction::printSomething() const {
    return 0;
}

bool abstractions::CollisionAvoidanceBigAbstraction::isFullModel(int abstraction) const {
    return abstraction == _num_abstractions-1;
}