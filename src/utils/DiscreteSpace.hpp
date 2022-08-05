#ifndef DISCRETESPACE_HPP
#define DISCRETESPACE_HPP

#include <cassert>
#include <cstddef>
#include <vector>

namespace utils {

/**
 * @brief A discrete space that contains element of value 0 ... size
 *
 * Assumes templated class can be initiated with single int argument.
 * Works specifically for IndexedElement<> classes, but for any
 * class that can be initiated with just an integer and has some range from
 * 0...n can be represted in this way.
 **/
template<typename T>
class DiscreteSpace
{

public:
    explicit DiscreteSpace(int n)
    {

        assert(n > 0);

        // generated n elements with values 0...n
        _elements.reserve(n);
        for (auto i = 0; i < n; ++i) { _elements.emplace_back(new T(i)); }
    }

    ~DiscreteSpace()
    {

        // clean up all elements created with new
        for (auto e : _elements) { delete e; }
    }

    T const* get(int n) const
    {
        assert(n >= 0);
        assert(static_cast<size_t>(n) < _elements.size());

        return _elements[n];
    };

private:
    std::vector<T const*> _elements = {};
};

} // namespace utils

#endif // DISCRETESPACE_HPP
