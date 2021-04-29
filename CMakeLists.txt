# project information
cmake_minimum_required(VERSION 2.8)
project("Partially Observable Reinforcement Learning")

# flags
set(COMPILE_FLAGS "-pedantic -Werror -Wall -Wextra -Wshadow -Weffc++")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} ${COMPILE_FLAGS}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# allow c++ 11 syntax
set(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_STANDARD 11)

# gather sources
set(SRC
    "src/beliefs/Belief.cpp"
    "src/beliefs/particle_filters/FlatFilter.cpp"
    "src/beliefs/particle_filters/ImportanceSampler.cpp"
    "src/beliefs/particle_filters/RejectionSampling.cpp"
    "src/beliefs/particle_filters/WeightedFilter.cpp"
    "src/beliefs/point_estimation/PointEstimation.cpp"
    "src/configurations/ArgumentParser.cpp"
    "src/configurations/BeliefConf.cpp"
    "src/configurations/Conf.cpp"
    "src/configurations/DomainConf.cpp"
    "src/configurations/PlannerConf.cpp"
    "src/domains/POMDP.cpp"
    "src/domains/agr/AGR.cpp"
    "src/domains/coffee/CoffeeProblem.cpp"
    "src/domains/coffee/CoffeeProblemAction.cpp"
    "src/domains/coffee/CoffeeProblemObservation.cpp"
    "src/domains/coffee/CoffeeProblemState.cpp"
    "src/domains/collision-avoidance/CollisionAvoidance.cpp"
    "src/domains/dummy/DummyDomain.cpp"
    "src/domains/dummy/FactoredDummyDomain.cpp"
    "src/domains/dummy/LinearDummyDomain.cpp"
    "src/domains/gridworld/GridWorld.cpp"
    "src/domains/gridworld-coffee-trap/GridWorldCoffee.cpp"
    "src/domains/sysadmin/SysAdmin.cpp"
    "src/domains/sysadmin/SysAdminState.cpp"
    "src/domains/tiger/FactoredTiger.cpp"
    "src/domains/tiger/Tiger.cpp"
    "src/environment/Discount.cpp"
    "src/environment/Environment.cpp"
    "src/environment/History.cpp"
    "src/environment/Horizon.cpp"
    "src/environment/Return.cpp"
    "src/environment/Reward.cpp"
    "src/environment/Terminal.cpp"
    "src/experiments/Episode.cpp"
    "src/experiments/PlanningExperiment.cpp"
    "src/planners/Planner.cpp"
    "src/planners/mcts/MCTSTreeNodes.cpp"
    "src/planners/mcts/POUCT.cpp"
    "src/planners/random/RandomPlanner.cpp"
    "src/planners/ts/TSPlanner.cpp"
    "src/utils/Entropy.cpp"
    "src/utils/Statistic.cpp"
    "src/utils/distributions.cpp"
    "src/utils/index.cpp"
    "src/utils/random.cpp"
    )

set(BA_SRC
    "src/bayes-adaptive/models/factored/FBADomainExtension.cpp"
    "src/bayes-adaptive/models/factored/FBAPOMDP.cpp"
    "src/bayes-adaptive/models/table/BADomainExtension.cpp"
    "src/bayes-adaptive/models/table/BAPOMDP.cpp"
    "src/bayes-adaptive/priors/BAPOMDPPrior.cpp"
    "src/bayes-adaptive/priors/FBAPOMDPPrior.cpp"
    "src/bayes-adaptive/states/BAState.cpp"
    "src/bayes-adaptive/states/factored/BABNModel.cpp"
    "src/bayes-adaptive/states/factored/DBNNode.cpp"
    "src/bayes-adaptive/states/factored/FBAPOMDPState.cpp"
    "src/bayes-adaptive/states/table/BAFlatModel.cpp"
    "src/bayes-adaptive/states/table/BAPOMDPState.cpp"
    "src/beliefs/bayes-adaptive/BABelief.cpp"
    "src/beliefs/bayes-adaptive/BAImportanceSampling.cpp"
    "src/beliefs/bayes-adaptive/BAPointEstimation.cpp"
    "src/beliefs/bayes-adaptive/BARejectionSampling.cpp"
    "src/beliefs/bayes-adaptive/NestedBelief.cpp"
    "src/beliefs/bayes-adaptive/factored/MHNIPS2018.cpp"
    "src/beliefs/bayes-adaptive/factored/MHwithinGibbs.cpp"
    "src/beliefs/bayes-adaptive/factored/ReinvigoratingRejectionSampling.cpp"
    "src/beliefs/bayes-adaptive/factored/StructureIncubatorSampling.cpp"
    "src/beliefs/bayes-adaptive/prototypes/CheatingReinvigoration.cpp"
    "src/configurations/BAConf.cpp"
    "src/configurations/FBAConf.cpp"
    "src/domains/collision-avoidance/CollisionAvoidanceBAExtension.cpp"
    "src/domains/collision-avoidance/CollisionAvoidanceFBAExtension.cpp"
    "src/domains/collision-avoidance/CollisionAvoidancePriors.cpp"
    "src/domains/dummy/DummyDomainBAExtension.cpp"
    "src/domains/dummy/FactoredDummyDomainBAExtension.cpp"
    "src/domains/dummy/FactoredDummyDomainFBAExtension.cpp"
    "src/domains/dummy/FactoredDummyDomainPriors.cpp"
    "src/domains/gridworld/GridWorldBAExtension.cpp"
    "src/domains/gridworld/GridWorldBAPriors.cpp"
    "src/domains/gridworld/GridWorldFBAExtension.cpp"
    "src/domains/gridworld-coffee-trap/GridWorldCoffeeBAExtension.cpp"
    "src/domains/gridworld-coffee-trap/GridWorldCoffeeBAPriors.cpp"
    "src/domains/gridworld-coffee-trap/GridWorldCoffeeFBAExtension.cpp"
    "src/domains/sysadmin/SysAdminBAExtension.cpp"
    "src/domains/sysadmin/SysAdminFBAExtension.cpp"
    "src/domains/sysadmin/SysAdminFactoredPrior.cpp"
    "src/domains/sysadmin/SysAdminFlatPrior.cpp"
    "src/domains/tiger/FactoredTigerBAExtension.cpp"
    "src/domains/tiger/FactoredTigerFBAExtension.cpp"
    "src/domains/tiger/FactoredTigerPriors.cpp"
    "src/domains/tiger/TigerBAExtension.cpp"
    "src/domains/tiger/TigerPriors.cpp"
    "src/experiments/BAPOMDPExperiment.cpp"
    "src/planners/bayes-adaptive/BAPlanner.cpp"
    "src/planners/bayes-adaptive/BATSPlanner.cpp"
    "src/planners/bayes-adaptive/RBAPOUCT.cpp"
    )

set(TEST_SRC
    "test/bayes-adaptive/BAPOMDPStateTest.cpp"
    "test/bayes-adaptive/BAPOMDPTest.cpp"
    "test/beliefs/bayes-adaptive/BAImportanceSamplingTest.cpp"
    "test/beliefs/bayes-adaptive/BAPointEstimationTest.cpp"
    "test/beliefs/bayes-adaptive/BARejectionSamplingTest.cpp"
    "test/bayes-adaptive/factored/DBNNodeTest.cpp"
    "test/domains/domain_extensions/GridWorldBAExtensionTests.cpp"
    "test/domains/domain_extensions/TigerBAExtensionTest.cpp"
    "test/domains/domain_extensions/CollisionAvoidanceBAExtensionTests.cpp"
    "test/domains/domain_extensions/SysAdminBAExtensionTest.cpp"
    "test/domains/priors/CollisionAvoidancePriorTests.cpp"
    "test/domains/priors/SysAdminPriorTest.cpp"
    "test/bayes-adaptive/factored/FBAPOMDPStateTest.cpp"
    "test/beliefs/ImportanceSamplingTest.cpp"
    "test/beliefs/PointEstimationTest.cpp"
    "test/beliefs/RejectionSamplingTest.cpp"
    "test/config/ConfigTest.cpp"
    "test/domains/CoffeeProblemTest.cpp"
    "test/domains/CollisionAvoidanceTest.cpp"
    "test/domains/DummyDomainTest.cpp"
    "test/domains/domain_extensions/DummyDomainBAExtensionTests.cpp"
    "test/domains/FactoredDummyDomainTest.cpp"
    "test/domains/GridWorldTest.cpp"
    "test/domains/SysAdminTest.cpp"
    "test/domains/TigerTest.cpp"
    "test/environment/BasicTest.cpp"
    "test/utils/StatisticTest.cpp"
    "test/domains/domain_extensions/FactoredDummyDomainBAExtensionTests.cpp"
    "test/domains/priors/FactoredDummyDomainPriorTests.cpp"
    "test/domains/priors/TigerPriorTest.cpp"
    "test/utils/indexTest.cpp"
    "test/utils/randomTest.cpp"
    )

include_directories(
    src/
    test/
    SYSTEM includes/ # SYSTEM allows compilation despite warnings in includes
    )

# define project and tests
add_executable(planning "src/planning.cpp" ${SRC})
add_executable(bapomdp "src/bapomdp.cpp" ${SRC} ${BA_SRC})
add_executable(fbapomdp "src/fbapomdp.cpp" ${SRC} ${BA_SRC})
add_executable(tests "test/test.cpp" ${SRC} ${BA_SRC} ${TEST_SRC})

add_custom_target(cppcheck
    COMMAND cppcheck --enable=all --project=compile_commands.json --inline-suppr --suppressions-list=../.cppcheck.suppressions --output-file=/tmp/cppcheck.log
    && echo "output written to /tmp/cppecheck.log"
    && less /tmp/cppcheck.log)

add_custom_target(clang-format
    python ${CMAKE_CURRENT_SOURCE_DIR}/run-clang-format.py -r ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${CMAKE_CURRENT_SOURCE_DIR}/test/
    && echo "no formatting problems found"
    )


add_custom_target(docs
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}
    && doxygen .doxygen.config
    && ln -s ${CMAKE_CURRENT_SOURCE_DIR}/doc ${CMAKE_CURRENT_BINARY_DIR}/doc
    && echo "documentation is reachable in doc/html/index.html" 
    )

# add boost library
find_package(Boost REQUIRED COMPONENTS program_options)
target_link_libraries(planning ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(bapomdp ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(fbapomdp ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(tests ${Boost_PROGRAM_OPTIONS_LIBRARY})
