#include "Terminal.hpp"

Terminal::Terminal(bool v) : _v(v) {}

bool Terminal::terminated() const
{
    return _v;
}
