#ifndef DOMAIN_FEATURE_SIZE_HPP
#define DOMAIN_FEATURE_SIZE_HPP

#include <utility>
#include <vector>

/**
 * @brief describes the size (action, observation, state) of a factored domain
 **/
struct Domain_Feature_Size
{

public:
    Domain_Feature_Size(
        // cppcheck-suppress passedByValue
        std::vector<int> state_feature_size,
        // cppcheck-suppress passedByValue
        std::vector<int> observation_feature_size) :
            _S(std::move(state_feature_size)),
            _O(std::move(observation_feature_size))
    {
    }

    std::vector<int> _S, _O;
};

#endif // DOMAIN_FEATURE_SIZE_HPP
