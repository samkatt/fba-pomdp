#ifndef TERMINAL_HPP
#define TERMINAL_HPP

/**
 * @brief Whether a transition was terminal or not
 **/
class Terminal
{
public:
    explicit Terminal(bool v);

    bool terminated() const;

private:
    bool _v;
};

#endif // TERMINAL_HPP
