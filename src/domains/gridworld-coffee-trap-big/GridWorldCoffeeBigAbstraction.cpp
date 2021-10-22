#include <bayes-adaptive/models/factored/Domain_Feature_Size.hpp>
#include <bayes-adaptive/models/Domain_Size.hpp>
#include <bayes-adaptive/models/factored/FBAPOMDP.hpp>
#include "GridWorldCoffeeBigAbstraction.hpp"

abstractions::GridWorldCoffeeBigAbstraction::GridWorldCoffeeBigAbstraction(
        configurations::BAConf c) :
        _num_abstractions(),
        _minimum_abstraction(),
        _abstraction_updating(false),
        _abstraction_normalization(c.planner_conf.update_abstract_model_normalized),
        _qvalues(),
        _counts(),
//        _used_abstraction_sets(),
        _abstract_domain_sizes(),
        _abstract_domain_feature_sizes(),
        _abstract_step_sizes()
{
    // abstraction, 0 = {x, y}
    // abstraction, 1 = {x, y, rain and/or other extra features if connected to x or y}
    // abstraction, 2 = no abstraction, i.e. full model

    _num_abstractions    = 3;
    _minimum_abstraction = {0,1};
    _qvalues = std::vector<double> (_num_abstractions,0);
    _counts = std::vector<int> (_num_abstractions, 0);
//    _used_abstraction_sets.emplace_back(_minimum_abstraction);

    _abstract_domain_sizes = std::vector<Domain_Size> (1 + c.domain_conf.size + 1, Domain_Size(25,4,25));
    _abstract_domain_feature_sizes = std::vector<Domain_Feature_Size> (1 + c.domain_conf.size + 1, Domain_Feature_Size({5,5}, {5,5}));
    _abstract_step_sizes = std::vector<bayes_adaptive::factored::BABNModel::Indexing_Steps> (1 + c.domain_conf.size + 1,
                                                                                             bayes_adaptive::factored::BABNModel::Indexing_Steps(
                                                                                                     indexing::stepSize(_abstract_domain_feature_sizes[0]._S),
                                                                                                     indexing::stepSize(_abstract_domain_feature_sizes[0]._O)));
    // to size + 1 (rain)
    for (unsigned int i = 1; i <= c.domain_conf.size + 1; ++i) {
        _abstract_domain_sizes[i] = Domain_Size((5*5) << i, 4, 25);
        auto feature_sizes = std::vector<int>(i+2, 2);
        feature_sizes[0] = 5;
        feature_sizes[1] = 5;
        _abstract_domain_feature_sizes[i] = Domain_Feature_Size(feature_sizes, {5,5});
        _abstract_step_sizes[i] = bayes_adaptive::factored::BABNModel::Indexing_Steps(
                indexing::stepSize(_abstract_domain_feature_sizes[i]._S),
                indexing::stepSize(_abstract_domain_feature_sizes[i]._O));
    }


}

bayes_adaptive::factored::BABNModel
abstractions::GridWorldCoffeeBigAbstraction::constructAbstractModel(bayes_adaptive::factored::BABNModel model, int k, POMDP const& domain) {
    std::vector<int> abstraction_set; // = {};
    if (k == 0) {
        abstraction_set = {0,1};

        if (_abstraction_normalization) {
            auto const& fbapomdp = dynamic_cast<::bayes_adaptive::factored::FBAPOMDP const&>(domain);

            auto priorModel = fbapomdp.prior()->computePriorModel(model.structure());
            auto abstractPriorModel_normalized = priorModel.abstract(abstraction_set, model.structure(), &_abstract_domain_sizes[0], &_abstract_domain_feature_sizes[0],
                                                                     &_abstract_step_sizes[0], true);
            auto abstractPriorModel = priorModel.abstract(abstraction_set, model.structure(), &_abstract_domain_sizes[0], &_abstract_domain_feature_sizes[0],
                                                          &_abstract_step_sizes[0], false);
            auto newModel = model.abstract(abstraction_set, model.structure(), &_abstract_domain_sizes[0], &_abstract_domain_feature_sizes[0],
                                           &_abstract_step_sizes[0], false);
            newModel.abstractionNormalizeCounts(abstractPriorModel, abstractPriorModel_normalized);
            return newModel;
        } else {
            return model.abstract(abstraction_set, model.structure(), &_abstract_domain_sizes[0], &_abstract_domain_feature_sizes[0],
                                  &_abstract_step_sizes[0], false);
        }

    } else if (k == 1) {
        // We assume the same model for each action here
        // TODO Could make one abstraction_set per action?
        auto action = IndexAction(0);
        std::set_union(model.transitionNode(&action, 0).parents()->begin(),
                              model.transitionNode(&action, 0).parents()->end(),
                              model.transitionNode(&action, 1).parents()->begin(),
                              model.transitionNode(&action, 1).parents()->end(),
                              std::back_inserter(abstraction_set));
        unsigned int index_to_use = abstraction_set.size() - 2;

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

    } else {
        return model;

    }
}

int abstractions::GridWorldCoffeeBigAbstraction::selectAbstraction() {
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

void abstractions::GridWorldCoffeeBigAbstraction::addReturn(int abstraction, double reward) {
    _qvalues[abstraction] += reward;
}

int abstractions::GridWorldCoffeeBigAbstraction::printSomething() const {
    return 0;
}

bool abstractions::GridWorldCoffeeBigAbstraction::isFullModel(int abstraction) const {
    return abstraction == _num_abstractions-1;
}



