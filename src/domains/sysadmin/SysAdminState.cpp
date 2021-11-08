#include "SysAdminState.hpp"

#include "utils/random.hpp"

/**
 * @brief used in constructor to quickly initiate state
 **/
int countSetBits(int n)
{
    auto count = 0;
    while (n != 0)
    {
        n &= (n - 1);
        count++;
    }
    return count;
}

namespace domains {

SysAdminState::SysAdminState(int i, int num_computers) :
        _num_computers(num_computers),
        _num_operational_computers(countSetBits(i)),
        _index(std::to_string(i))
{
}

bool SysAdminState::isOperational(int n) const
{
    return ((std::stoi(index()) >> n) & 0x1) != 0;
}

int SysAdminState::numOperationalComputers() const
{
    return _num_operational_computers;
}

void SysAdminState::index(std::string i)
{
    _index.index(i);
}

std::string SysAdminState::index() const
{
    return _index.index();
}

std::string SysAdminState::toString() const
{
    std::string result = (isOperational(0)) ? "{1 " : "{0 ";

    for (auto c = 1; c < _num_computers - 1; ++c) { result += (isOperational(c)) ? "1 " : "0 "; }

    if (_num_computers > 1)
    {
        result += (isOperational(_num_computers - 1)) ? "1}" : "0}";
    } else
    {
        result += "}";
    }

    return result;
}

std::vector<int> SysAdminState::getFeatureValues() const {
    // TODO implement
    return std::vector<int>();
}

} // namespace domains
