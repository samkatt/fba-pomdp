#ifndef GridWorldCoffeeBigAbstraction_HPP
#define GridWorldCoffeeBigAbstraction_HPP

#include <bayes-adaptive/abstractions/Abstraction.hpp>
#include "GridWorldCoffeeBig.hpp"

namespace abstractions {

class GridWorldCoffeeBigAbstraction : public Abstraction
{
public:
    GridWorldCoffeeBigAbstraction(configurations::BAConf c);

    /*** Abstraction interface **/
    bayes_adaptive::factored::BABNModel constructAbstractModel(bayes_adaptive::factored::BABNModel model, int k, POMDP const& domain) final;
    int  selectAbstraction() final;
    int printSomething() const final;
    bool isFullModel(int abstraction) const final;



private:
    int _num_abstractions;
    std::vector<int> _minimum_abstraction;
    bool _abstraction_updating;
    bool _abstraction_normalization;
    std::vector<double> _qvalues;
    int _UCB_exploration = 1;
    std::vector<int> _counts;
//    std::vector<std::vector<int>> _used_abstraction_sets;
    std::vector<Domain_Size> _abstract_domain_sizes;
    std::vector<Domain_Feature_Size> _abstract_domain_feature_sizes;
    std::vector<bayes_adaptive::factored::BABNModel::Indexing_Steps> _abstract_step_sizes;

//    bayes_adaptive::factored::BABNModel
//    subTractCounts(bayes_adaptive::factored::BABNModel new_model, bayes_adaptive::factored::BABNModel prior_model_normalized,
//                   bayes_adaptive::factored::BABNModel prior_model) const;
        void addReturn(int abstraction, double reward);

};

} // namespace abstractions















#endif //GridWorldCoffeeBigAbstraction_HPP
