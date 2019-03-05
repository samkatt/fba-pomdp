#ifndef RETURN_HPP
#define RETURN_HPP

class Reward;
class Discount;

/**
 * @brief The return (accumulated discounted rewards) of an episode
 **/
class Return
{
public:
    Return() = default;
    explicit Return(double v) : _val(v) {}

    void add(Reward const& r, Discount const& d);

    double toDouble() const;

private:
    double _val{0};
};

#endif // RETURN_HPP
