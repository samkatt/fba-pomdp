#ifndef STATISTIC_HPP
#define STATISTIC_HPP

#include <cmath>

namespace utils {

/**
 * @brief Contains the statistic information of a value
 **/
struct Statistic
{
public:
    /**
     * @brief updates this with new statistic
     **/
    void add(double v);

    double mean() const;
    double count() const;
    double var() const;
    double stder() const;

private:
    int _count   = 0;
    double _mean = 0;
    double _m2   = 0;
};

} // namespace utils

#endif // STATISTIC_HPP
