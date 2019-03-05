#ifndef INDEXEDELEMENT_HPP
#define INDEXEDELEMENT_HPP

#include <string>

namespace indexing {

/**
 * @brief A state purely defined by a single index
 **/
template<typename T>
class IndexedElement : public T
{
public:
    explicit IndexedElement(int i) :
            _index(i){

            };

    /***** T implementation *****/
    void index(int i) final { _index = i; };

    int index() const final { return _index; }

    std::string toString() const final { return "(" + std::to_string(_index) + ")"; }

private:
    int _index;
};

} // namespace indexing

#endif // INDEXEDELEMENT_HPP
