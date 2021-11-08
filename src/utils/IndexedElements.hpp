#ifndef INDEXEDELEMENT_HPP
#define INDEXEDELEMENT_HPP

#include <string>
#include <vector>

namespace indexing {

/**
 * @brief A state purely defined by a single index
 **/
template<typename T>
class IndexedElement : public T
{
public:
    explicit IndexedElement(std::string i) :
            _index(std::move(i)){
            };

    /***** T implementation *****/
    void index(std::string i) final { _index = i; };

    std::string index() const final { return _index; }

    std::string toString() const final { return "(" + _index + ")"; }

    virtual std::vector<int> getFeatureValues() const final { return {std::stoi(_index)}; };  //{  throw "should not be called here"; };

private:
    std::string _index;
};

} // namespace indexing

#endif // INDEXEDELEMENT_HPP
