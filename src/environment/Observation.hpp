#ifndef OBSERVATION_HPP
#define OBSERVATION_HPP

#include <cassert>

#include <string>

/**
 * @brief An observation from the environment perceived by the agent
 **/
class Observation
{

public:
    virtual ~Observation()    = default;
    virtual void index(int i) = 0;

    /**
     * @brief gets the index
     *
     * May be overriden by derived class if calculated on spot
     **/
    virtual int index() const = 0;

    /**
     * @brief gets state information in string form
     *
     * May be overwritten by derived classes.
     * Returns index string by default.
     **/
    virtual std::string toString() const = 0;
};

#include "utils/IndexedElements.hpp"
using IndexObservation = indexing::IndexedElement<Observation>;

#endif // OBSERVATION_HPP
