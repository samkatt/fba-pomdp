#ifndef ACTION_HPP
#define ACTION_HPP

#include <string>
#include <vector>
#include <cassert>

/**
 * @brief A move that the agent can do in a state
 **/
class Action
{
public:
    virtual ~Action()         = default;
    virtual void index(std::string i) = 0;

    /**
     * @brief gets the index
     *
     * May be overriden by derived class if calculated on spot
     **/
    virtual std::string index() const = 0;

    /**
     * @brief gets state information in string form
     *
     * May be overwritten by derived classes.
     * Returns index string by default.
     **/
    virtual std::string toString() const = 0;

    virtual std::vector<int> getFeatureValues() const = 0;
};

#include "utils/IndexedElements.hpp"
using IndexAction = indexing::IndexedElement<Action>;

#endif // ACTION_HPP
