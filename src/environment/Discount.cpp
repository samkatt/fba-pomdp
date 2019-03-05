#include "Discount.hpp"

Discount::Discount(double v) : _accumulated_discount(1), _discount(v)
{
    assert(_discount > 0 && _discount <= 1);
}

void Discount::increment()
{
    _accumulated_discount *= _discount;
}

double Discount::toDouble() const
{
    return _accumulated_discount;
}
