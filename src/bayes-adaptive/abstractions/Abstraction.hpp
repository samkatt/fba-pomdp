#ifndef Abstraction_HPP
#define Abstraction_HPP

#include <configurations/FBAConf.hpp>
#include <bayes-adaptive/states/factored/BABNModel.hpp>
#include <bayes-adaptive/models/table/BAPOMDP.hpp>

/**
 * @brief Abstraction
 **/
class Abstraction
{

public:
    virtual ~Abstraction() = default;

    virtual bayes_adaptive::factored::BABNModel
    constructAbstractModel(bayes_adaptive::factored::BABNModel model, int k, const POMDP &domain,
                           std::vector<int> *feature_set) = 0;

    virtual int selectAbstraction() = 0;

    virtual bool isFullModel(int abstraction) const = 0; // returns true if the selected abstraction is the full model

    virtual void addReturn(int abstraction, double reward) = 0;

    virtual int printSomething() const = 0;

};

namespace factory {

/**
 * @brief Return an abstraction
 */
    std::unique_ptr<Abstraction>
    makeAbstraction(configurations::BAConf const& c);

} // namespace factory


#endif //Abstraction_HPP
