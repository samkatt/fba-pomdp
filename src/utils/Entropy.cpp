#include "Entropy.hpp"

#include <cmath>
#include <numeric>

namespace ent {

double H(std::vector<int>& histogram, double base)
{
    double ci_logci = 0.;
    for (auto ci : histogram)
    {
        if (ci > 0)
        {
            ci_logci += ci * log2(ci);
        }
    }
    auto c   = std::accumulate(histogram.begin(), histogram.end(), 0.);
    double H = log2(c) - ci_logci / c;
    return H / log2(base);
}

std::vector<double> dH(std::vector<int>& histogram, double base)
{
    double H = ent::H(histogram, base);
    auto n   = histogram.size();

    std::vector<double> ci_logci(n, 0), _1pci_log1pci(n);
    double sum_ci_logci = 0;
    int c               = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        c += histogram[i];
        if (histogram[i] > 0)
        {
            ci_logci[i] = histogram[i] * log2(histogram[i]);
        }
        sum_ci_logci += ci_logci[i];
        _1pci_log1pci[i] = (histogram[i] + 1) * log2(histogram[i] + 1);
    }

    std::vector<double> dH(n);
    for (unsigned int k = 0; k < n; k++)
    {
        auto dHk = log2(c + 1) - (sum_ci_logci - ci_logci[k] + _1pci_log1pci[k]) / (c + 1);
        dH[k]    = dHk / log2(base) - H;
    }

    return dH;
}

} // namespace ent
