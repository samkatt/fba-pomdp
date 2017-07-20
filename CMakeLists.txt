# project information
cmake_minimum_required(VERSION 2.8) 
project("Partially Observable Reinforcement Learning")

# flags 
set(COMPILE_FLAGS "-pedantic -Werror -Wall -Wextra -Wconversion -Wundef -Wunreachable-code -Wunused -ftrapv -Wunreachable-code")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} ${COMPILE_FLAGS}")

# allow c++ 11 syntax
set(CMAKE_CXX_FLAGS "--std=c++11")
set(CMAKE_CXX_STANDARD 11)

# gather sources
set(SRC 
    "src/configurations/ArgumentParser.cpp"
    )

set(TEST_SRC 
    )

include_directories(
    src/
    test/
    SYSTEM includes/ # SYSTEM will compile despite warnings in includes
    )

# define project and tests
add_executable(po-rl "src/main.cpp" ${SRC})
add_executable(po-rl-tests "test/test.cpp" ${TEST_SRC})

# add boost library
find_package(Boost REQUIRED COMPONENTS program_options)
target_link_libraries(po-rl ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(po-rl-tests ${Boost_PROGRAM_OPTIONS_LIBRARY})
