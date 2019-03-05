#ifndef COFFEEPROBLEMINDICES_HPP
#define COFFEEPROBLEMINDICES_HPP

namespace domains {

enum ACTION { GetCoffee, CheckCoffee };
enum OBSERVATION { Want_Coffee, Not_Want_Coffee };

// masks to infer the state variables
enum CoffeeStateMask {
    RAINS        = 0x01 << 0,
    UMBRELLA     = 0x01 << 1,
    WET          = 0x01 << 2,
    HAS_COFEE    = 0x01 << 3,
    WANTS_COFFEE = 0x01 << 4
};

} // namespace domains

#endif // COFFEEPROBLEMINDICES_HPP
