#ifndef CollisionAvoidanceBigAbstraction_HPP
#define CollisionAvoidanceBigAbstraction_HPP

#include <bayes-adaptive/abstractions/Abstraction.hpp>
#include "CollisionAvoidanceBig.hpp"

namespace abstractions {

    class CollisionAvoidanceBigAbstraction : public Abstraction
    {
    public:
        CollisionAvoidanceBigAbstraction(configurations::BAConf c);

        /*** Abstraction interface **/
        bayes_adaptive::factored::BABNModel
        constructAbstractModel(bayes_adaptive::factored::BABNModel model, int k, const POMDP &domain,
                               std::vector<int> *feature_set) final;
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

        static int const _speed_feature = 2;
        static int const _traffic_feature = 3;
        static int const _tod_feature = 4;
        static int const _obstacletype_feature = 5;
        static int const _first_obstacle = 6;

        const int _num_speeds = 2;
        const int _num_traffics = 2;
        const int _num_timeofdays = 2;
        const int _num_obstacletypes = 3;

        std::vector<Domain_Size> _abstract_domain_sizes;
        std::vector<Domain_Feature_Size> _abstract_domain_feature_sizes;
        std::vector<bayes_adaptive::factored::BABNModel::Indexing_Steps> _abstract_step_sizes;

        void addReturn(int abstraction, double reward);
    };

} // namespace abstractions

#endif //GridWorldCoffeeBigAbstraction_HPP
