#include <domains/POMDP.hpp>
#include "Abstraction.hpp"

#include "domains/gridworld-coffee-trap-big/GridWorldCoffeeBigAbstraction.hpp"
#include "domains/collision-avoidance-big/CollisionAvoidanceBigAbstraction.hpp"


namespace factory {

    std::unique_ptr<Abstraction>
    makeAbstraction(configurations::BAConf const &c) {

        auto const d = c.domain_conf.domain;

        if (d == "gridworldcoffeebig") {
            return std::unique_ptr<Abstraction>(
                    new abstractions::GridWorldCoffeeBigAbstraction(c));

        } else if (d == "random-collision-avoidance-big" || d == "centered-collision-avoidance-big") {
            return std::unique_ptr<Abstraction>(
                    new abstractions::CollisionAvoidanceBigAbstraction(c));
        } else {
            return nullptr;
        }

        throw "incorrect domain provided, " + d + " is not supported as factored BA-POMDP";
    }

}
