#include "Statistic.hpp"

namespace utils {

void Statistic::add(double v)
{
    _count++;

    auto const delta = v - _mean;
    _mean += delta / _count;

    auto const delta_2 = v - _mean;
    _m2 += delta * delta_2;
}

double Statistic::mean() const
{
    return _mean;
}

double Statistic::count() const
{
    return _count;
}

double Statistic::var() const
{

    if (_count < 2)
    {
        return 0;
    }

    return _m2 / (_count - 1);
}

double Statistic::stder() const
{
    if (_count < 2)
    {
        return 0;
    }

    return sqrt(var() / _count);
}

} // namespace utils
