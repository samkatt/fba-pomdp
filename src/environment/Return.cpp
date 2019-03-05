#include "Return.hpp"

#include "environment/Discount.hpp"
#include "environment/Reward.hpp"

void Return::add(Reward const& r, Discount const& d)
{
    _val += r.toDouble() * d.toDouble();
}

double Return::toDouble() const
{
    return _val;
}
