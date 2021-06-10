#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>

namespace rnd {

/**
 * @brief initiates this (should be called once per program)
 **/
void initiate();

/**
 * @brief sets generator seed
 **/
void seed(std::string& seed_str);

/**
 * @brief returns reference to the random number generator
 **/
std::mt19937& rng();

bool boolean();

/**
 * @brief returns random number between 0 and 1
 **/
double uniform_rand01();

/**
 * @brief returns a unsigned long between 0 and 32bits big
 **/
unsigned long randomLong32();

/**
 * @brief returns a random integer (between min and max) GENERATOR
 *
 * to call and get random int: distribution(random::rng());
 **/
std::uniform_int_distribution<int> integerDistribution(int min, int max);

/**
 * @brief returns a random int between (including) min and (excluding!) max
 *
 * This function is slow, please consider using integerDistribution
 * if you need multiple within the same range
 */
int slowRandomInt(int min, int max);

namespace normal {

/**
 * @brief returns the normal cumulative distribution function of x, given mean and standard
 *deviation
 **/
double cdf(double x, double mu, double std);

} // namespace normal

namespace math {
/**
 * @brief custom lgamma, returns 0 for x < 1
 *
 * Allows taking gammas for the dirichlet distribution that typically
 * assumes the parameters (counts) > 1
 **/
double logGamma(double x);
} // namespace math

namespace sample {

/**
 * @brief samples from a gamma distribution
 **/
double gamma(double shape);

namespace Dir {

enum SAMPLETYPE { Regular, Expected };
using sampleMethod = int(float const* dir, int n);

/**
 * @brief samples from a multinominal of n elements and total probability
 *
 * Expects: T to be float or double
 **/
template<typename T>
int sampleFromMult(T* mult, size_t n, double total_prob)
{

    // get sample using uniform distribution scaled by the total
    auto const p = uniform_rand01() * total_prob;
    auto sum     = mult[0];

    // sample value from multinominal
    for (size_t i = 1; i < n; ++i)
    {
        if (p < sum)
        {
            return i - 1;
        }

        sum += mult[i];
    }
    assert(p < sum + 0.01); // TODO try to fix abstract bapomcp such that this does not occur?

    return n - 1;
}

/**
 * @brief samples from a multinominal sampled from a dirichlet
 **/
int sampleFromSampledMult(float const* dir, int n);

/**
 * @brief samples from dirichlet using its expectation / maximum likelyhood
 **/
int sampleFromExpectedMult(float const* dir, int n);

using sampleMultinominal = std::vector<float>(float const* dir, int n);

/**
 * @brief returns expected multinomial distr given dir of n elements
 **/
std::vector<float> expectedMult(float const* dir, int n);

/**
 * @brief samples mult from dirichlet of n elements
 **/
std::vector<float> sampleMult(float const* dir, int n);

} // namespace Dir
} // namespace sample
} // namespace rnd

#endif // RANDOM_HPP
