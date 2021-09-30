#include <domains/POMDP.hpp>
#include "Abstraction.hpp"

#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBigAbstraction.hpp"


namespace factory {

    std::unique_ptr<Abstraction>
    makeAbstraction(configurations::BAConf const &c) {

        auto const d = c.domain_conf.domain;

        if (d == "gridworldcoffeebig")
            return std::unique_ptr<Abstraction>(
                    new abstractions::GridWorldCoffeeBigAbstraction(c));

        throw "incorrect domain provided, " + d + " is not supported as factored BA-POMDP";
    }

}
