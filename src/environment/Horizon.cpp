#include "Horizon.hpp"

Horizon::Horizon(int v) : _value(v)
{
    assert(_value > 0);
}

int Horizon::toInt() const
{
    return _value;
}
