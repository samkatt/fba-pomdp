#ifndef HISTORY_HPP
#define HISTORY_HPP

#include <cassert>
#include <cstddef>
#include <vector>

class Environment;
class POMDP;

class Action;
class Observation;

/**
 * @brief A history of agent - environment interaction: sequence of <actions, observations>
 **/
class History
{
public:
    /**
     * @brief a single agent - environment interaction
     **/
    struct Interaction
    {
        Interaction(Action const* a, Observation const* o);

        Action const* action;
        Observation const* observation;
    };

    void add(Action const* a, Observation const* o);
    void clear(POMDP const& action_owner, Environment const& observation_owner);

    std::vector<Interaction>::iterator begin();
    std::vector<Interaction>::iterator end();
    std::vector<Interaction>::reference back();

    std::vector<Interaction>::const_iterator begin() const;
    std::vector<Interaction>::const_iterator end() const;
    std::vector<Interaction>::const_reference back() const;

    size_t length() const;

    Interaction operator[](int i) const;

private:
    std::vector<Interaction> _history = {};
};

#endif // HISTORY_HPP
