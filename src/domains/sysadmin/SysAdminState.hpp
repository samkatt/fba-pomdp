#ifndef SYSADMINSTATE_HPP
#define SYSADMINSTATE_HPP

#include "environment/State.hpp"

#include <string>

namespace domains {

/**
 * @brief A state in the sysadmin problem
 **/
class SysAdminState : public State
{
public:
    SysAdminState(int i, int num_computers);

    /**
     * @brief returns whether computer n is operational
     **/
    bool isOperational(int n) const;

    /**
     * @brief returns the number of operational computers
     **/
    int numOperationalComputers() const;

    /*** state interface ***/
    void index(std::string i) final;
    std::string index() const final;
    std::vector<int> getFeatureValues() const final;

    /**
     * @brief {1 0 1 0} depending on whether operating or not
     **/
    std::string toString() const final;

private:
    int _num_computers;
    int _num_operational_computers;

    IndexState _index;
};

} // namespace domains

#endif // SYSADMINSTATE_HPP
