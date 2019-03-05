#ifndef DOMAIN_SIZE_HPP
#define DOMAIN_SIZE_HPP

struct Domain_Size
{
    Domain_Size(int s, int a, int o) : _S(s), _A(a), _O(o) {}

    // action-, observation- & state space size
    int _S, _A, _O;
};

#endif // DOMAIN_SIZE_HPP
