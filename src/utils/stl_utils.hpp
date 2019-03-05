#ifndef STL_UTILS_HPP
#define STL_UTILS_HPP

#include <string>
#include <vector>

namespace utils { namespace stl {

template<typename C>
std::string toString(C const& container)
{
    assert(!container.empty());

    std::string out = "{";

    for (auto const& e : container) { out += std::to_string(e) + ","; }

    out.back() = '}';

    return out;
}

}} // namespace utils::stl

#endif // STL_UTILS_HPP
