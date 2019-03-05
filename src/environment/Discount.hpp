#ifndef DISCOUNT_HPP
#define DISCOUNT_HPP

#include <cassert>

/**
 * @brief Discount of a problem
 **/
class Discount
{
public:
    explicit Discount(double v);

    /**
     * @brief increments the discount with 1 timestep
     **/
    void increment();

    /**
     * @brief returns accumulated discount
     **/
    double toDouble() const;

private:
    double _accumulated_discount;

    // the actual discount does not change over time
    double const _discount;
};

#endif // DISCOUNT_HPP
