#include "random.hpp"

#include "easylogging++.h"

namespace rnd {

/**** random lookup tables (set in initiate()) ****/
unsigned long random_unsigned_long[128];
double random_double_wn[128], random_double_fn[128];

std::mt19937 _rng;

std::bernoulli_distribution bernoulli_distribution(0.5); // random bool generator
std::uniform_real_distribution<double> uniform_probability_distribution(0, 1);

std::uniform_int_distribution<unsigned long> uniform_long32_distribution(0, 2147483647);

#if ULONG_MAX == 4294967295ul
unsigned long long2unsignedLong(unsigned long x)
{
    return (unsigned long)x;
}
unsigned long makeUnsignedLong(double x)
{
    return (unsigned long)x;
}
signed long randomLong()
{
    randomLong32();
}
#else
unsigned long long2unsignedLong(unsigned long x)
{
    return (x & 0xfffffffful);
}
unsigned long makeUnsignedLong(double x)
{
    return (unsigned long)x;
}
signed long randomLong()
{
    auto x = randomLong32();
    return (x < 0x80000000ul ? (unsigned long)x : -1 * (0x80000000ul - (x & 0x7ffffffful)));
}
#endif

void initiate()
{
    _rng.seed(time(nullptr));

    // setup random lookup tables
    double tn = 3.442619855899;

    double const m1 = 2147483648.0, // 2^31
        vn = 9.91256303526217e-3, q = vn / exp(-.5 * tn * tn);

    random_unsigned_long[0] = makeUnsignedLong((tn / q) * m1);
    random_unsigned_long[1] = 0;

    random_double_wn[0]   = q / m1;
    random_double_wn[127] = tn / m1;

    random_double_fn[0]   = 1.;
    random_double_fn[127] = exp(-.5 * tn * tn);

    for (auto i = 126; i > 0; --i)
    {
        double const dn             = sqrt(-2 * log(vn / tn + exp(-.5 * tn * tn)));
        random_unsigned_long[i + 1] = makeUnsignedLong((dn / tn) * m1);
        random_double_fn[i]         = exp(-.5 * dn * dn);
        random_double_wn[i]         = dn / m1;
        tn                          = dn;
    }
}

void seed(std::string& seed_str)
{
    std::seed_seq seed(seed_str.begin(), seed_str.end());

    LOG(INFO) << "Random seed " << seed_str;

    _rng.seed(seed);
}

std::mt19937& rng()
{
    return _rng;
}

bool boolean()
{
    return bernoulli_distribution(_rng);
}

unsigned long randomLong32()
{
    return uniform_long32_distribution(_rng);
}

double uniform_rand01()
{
    return uniform_probability_distribution(_rng);
}

std::uniform_int_distribution<int> integerDistribution(int min, int max)
{
    assert(min <= max);
    return std::uniform_int_distribution<int>(min, max - 1);
}

int slowRandomInt(int min, int max)
{
    assert(min <= max);
    return min + (int)floor(uniform_rand01() * (double)(max - min));
}

namespace normal {
double const two_squared = sqrt(2);
double cdf(double x, double mu, double std)
{
    return .5 + .5 * erf((x - mu) / (std * two_squared));
}

} // namespace normal

namespace math {
double logGamma(double x)
{
    if (x < 1)
    {
        return 0;
    }

    return lgamma(x);
}
} // namespace math

namespace sample {
/**** consts used in random computations ****/
double const r = 3.442620, // starting of the right tail
    r_inverse  = 0.2904764;

/**
 * @brief generates variates after rejection from NROR.
 **/
double normalRejectFix(long h, unsigned long i)
{
    double y;

    while (true)
    {
        auto x = (double)h * random_double_wn[i];

        // base strip
        if (i == 0)
        {
            do {
                x = -log(uniform_rand01()) * r_inverse;
                y = -log(uniform_rand01());
            } while (y + y < x * x);

            return ((h > 0) ? r + x : -r - x);
        }

        // handle wedges of other strips
        if (random_double_fn[i] + uniform_rand01() * (random_double_fn[i - 1] - random_double_fn[i])
            < exp(-.5 * x * x))
            return x;

        // start all over
        h = randomLong();
        i = h & 127;
        if (long2unsignedLong(std::abs(h)) < random_unsigned_long[i])
            return (double)h * random_double_wn[i];
    }
}

/**
 * @brief returns a random normal
 **/
double randomNormal()
{
    long const h = randomLong(), i = h & 127;

    return ((unsigned long)std::abs(h) < random_unsigned_long[i]) ? (double)h * random_double_wn[i]
                                                                  : normalRejectFix(h, i);
}

double gamma(double shape)
{
    if (shape < 1.)
        return gamma(shape + 1) * pow(uniform_rand01(), 1 / shape);

    double x, v;
    double const d = shape - 1. / 3., c = 1. / sqrt(9. * d);

    for (;;)
    {
        do {
            x = randomNormal();
            v = 1.0 + c * x;
        } while (v <= 0.0);

        v = v * v * v;

        double const u = uniform_rand01(), x2 = x * x;

        if (u < 1.0 - 0.0331 * x2 * x2)
            return d * v;
        if (log(u) < .5 * x2 + d * (1. - v + log(v)))
            return d * v;
    }
}

namespace Dir {

int sampleFromSampledMult(float const* dir, int n)
{
    assert(n > 0);

    static std::vector<double> probs(0);
    probs.clear();
    probs.reserve(n);

    auto rand_gamma = gamma(dir[0]), gamma_sum = rand_gamma;

    probs.emplace_back(rand_gamma);
    // sample multinominal
    for (auto i = 1; i < n; ++i)
    {
        rand_gamma = gamma(dir[i]);

        probs.emplace_back(rand_gamma);
        gamma_sum += rand_gamma;
    }

    // if this fails, I may need to
    // catch very small probabilities
    assert(gamma_sum > 1e-300);

    return sampleFromMult(&probs[0], n, gamma_sum);
}

int sampleFromExpectedMult(float const* dir, int n)
{
    assert(n > 0);

    double total = dir[0];

    // get total
    for (auto i = 1; i < n; ++i) total += dir[i];

    assert(total > 0);
    return sampleFromMult(&dir[0], n, total);
}

std::vector<float> expectedMult(float const* dir, int n)
{
    assert(n > 0);

    auto res = std::vector<float>();
    res.reserve(n);

    auto sum = dir[0];
    for (auto i = 1; i < n; ++i)
    {
        sum += dir[i];
        assert(dir[i] >= 0);
    }

    if (sum <= 1e-300) // total is too small, so return 0 distribution
    {
        return std::vector<float>(n);
    }

    for (auto i = 0; i < n; ++i) { res.emplace_back(dir[i] / sum); }

    return res;
}

std::vector<float> sampleMult(float const* dir, int n)
{
    auto res = std::vector<float>();
    res.reserve(n);

    float sum = 0;
    // initiate random gamma values
    for (auto i = 0; i < n; ++i)
    {
        assert(dir[i] >= 0);

        res.emplace_back(gamma(dir[i]));
        sum += res[i];
    }

    // if this fails, I may need to
    // catch very small probabilities
    assert(sum > 1e-300);

    // normalize
    for (auto& x : res) x = x / sum;

    return res;
}

} // namespace Dir
} // namespace sample
} // namespace rnd
