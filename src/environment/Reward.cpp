#include "Reward.hpp"

Reward::Reward(double v) : _v(v) {}

void Reward::set(double v)
{
    _v = v;
}

double Reward::toDouble() const
{
    return _v;
}
