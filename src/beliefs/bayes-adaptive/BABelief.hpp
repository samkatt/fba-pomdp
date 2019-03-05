#ifndef BABELIEF_HPP
#define BABELIEF_HPP

#include "beliefs/Belief.hpp"

#include <memory>

#include "bayes-adaptive/models/table/BAPOMDP.hpp"

namespace configurations {
struct Conf;
}

namespace beliefs {

/**
 * @brief A belief for the bayes-adaptive problems
 *
 * Defines the interface if state estimators want to be
 * used in bayes-adaptive problems
 **/
class BABelief : public Belief
{

public:
    ~BABelief() override = default;

    /**
     * @brief sets starting *domain* state estimation
     *
     * NOTE: will not affec the counts / learnable
     * aspects that are stored in the states
     **/
    virtual void resetDomainStateDistribution(BAPOMDP const& domain) = 0;
};

} // namespace beliefs

namespace factory {

std::unique_ptr<beliefs::BABelief> makeBABelief(configurations::Conf const& c);

} // namespace factory

#endif // BABELIEF_HPP
