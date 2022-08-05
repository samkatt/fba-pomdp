#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

#include <cassert>
#include <cstddef>
#include <vector>

namespace utils {

/**
 * @brief The categorical distribution
 **/
struct categoricalDistr
{
public:
    explicit categoricalDistr(size_t size, float init = 0);

    explicit categoricalDistr(std::vector<float> const& distr);

    /**
     * @brief returns the probability of element i
     *
     * @param i the element of which to return the probability of (0 <= i < size)
     *
     * @return probability (0 <= p <= 1)
     */
    float prob(size_t i) const;

    /**
     * @brief Sets the raw (probability) value of elemnt i to v
     *
     * @param i what element to set the probability of (0 <= i < size)
     * @param v the probability to set (0 <= v <= 1)
     */
    void setRawValue(size_t i, float v);

    /**
     * @brief samples an element from the distribution
     *
     * @return an element 0 <= i < size according to this distribution
     */
    unsigned int sample() const;

private:
    std::vector<float> _values;
    double _total;
};

} // namespace utils

#endif // DISTRIBUTIONS_HPP
