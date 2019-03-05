#ifndef REWARD_HPP
#define REWARD_HPP

/**
 * @brief The reward of an action
 **/
class Reward
{
public:
    explicit Reward(double v);

    void set(double v);

    double toDouble() const;

private:
    double _v;
};

#endif // REWARD_HPP
