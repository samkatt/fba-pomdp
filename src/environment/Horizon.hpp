#ifndef HORIZON_HPP
#define HORIZON_HPP

#include <cassert>

/**
 * @brief Horizon of a problem
 **/
class Horizon
{
public:
    explicit Horizon(int v);

    int toInt() const;

private:
    int const _value;
};

#endif // HORIZON_HPP
