#ifndef ENTROPY_HPP
#define ENTROPY_HPP

#include <vector>

namespace ent {

double H(std::vector<int>& histogram, double base = 2);

std::vector<double> dH(std::vector<int>& histogram, double base = 2);

} // namespace ent

#endif // ENTROPY_HPP
