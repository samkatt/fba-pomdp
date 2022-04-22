# (Factored) Bayes-Adaptive POMDP

- written by Sammie Katt (katt.s at husky dot neu dot edu)
- find it @ https://github.com/samkatt/fba-pomdp

A code base to run (Bayes-Adaptive) reinforcement learning experiments on
partially observable domains. This project is meant for reinforcement learning
researchers to compare different methods. It contains various different
environments to test the methods on, of which all partially observable and
discrete. Note that this project has mostly been written for personal use,
research, and thus may lack the documentation that one would typically expect
from open source projects.

## Related research:

- [BA-POMCP paper ICML 2017](http://proceedings.mlr.press/v70/katt17a/katt17a.pdf)
- [FBA-POMDP paper AAMAS 2019](http://ifaamas.org/Proceedings/aamas2019/pdfs/p7.pdf)

## Installation

```console
mkdir wherever/you/want && cd wherever/you/want
cmake -DCMAKE_BUILD_TYPE=Release /path/to/root/of/this/project
make
```

Requirements:

- [Boost](https://www.boost.org/)
- [Doxygen](https://www.doxygen.nl/index.html) for documentation

## Usage

### To run planning algorithms (no learning)

```console
./planning -D episodic-tiger -v 2 -f results.txt
cat results.txt
./planning --help
```

### To run learning approaches

Tabular BA-POMDP:

```console
./bapomdp --help
```

Or Factored BA-POMDP:

```console
./fbapomdp --help
```

### Plotting and Processing results

See analysis/README.md

## Documentation

After installation to generate the documentation in the 'doc' folder, run

```console
cd to/the/build/directory
make docs
```

## Development

### Using IDEs

Cmake provides some support for creating project files for various IDEs. Cf. cmake --help

E.g., cmake -G <generator-name> should be able to do:

```
  Unix Makefiles                  = Generates standard UNIX makefiles.
  Ninja                           = Generates build.ninja files.
  Watcom WMake                    = Generates Watcom WMake makefiles.
  CodeBlocks - Ninja              = Generates CodeBlocks project files.
  CodeBlocks - Unix Makefiles     = Generates CodeBlocks project files.
  CodeLite - Ninja                = Generates CodeLite project files.
  CodeLite - Unix Makefiles       = Generates CodeLite project files.
  Sublime Text 2 - Ninja          = Generates Sublime Text 2 project files.
  Sublime Text 2 - Unix Makefiles = Generates Sublime Text 2 project files.
  Kate - Ninja                    = Generates Kate project files.
  Kate - Unix Makefiles           = Generates Kate project files.
  Eclipse CDT4 - Ninja            = Generates Eclipse CDT 4.0 project files.
  Eclipse CDT4 - Unix Makefiles   = Generates Eclipse CDT 4.0 project files.
```

### Debugging & Profiling

CMake uses different build types (`CMAKE_BUILD_TYPE`, "Typical values include
`Debug`, `Release`, `RelWithDebInfo` and `MinSizeRel`, but custom build types
can also be defined."), for which you can set different flags.

For instance you can do:

```
cmake -DCMAKE_CXX_FLAGS_DEBUG="-O0 -ggdb -p" -DCMAKE_BUILD_TYPE=Debug /path/to/root/of/this/project
make VERBOSE=1
```

Alternatively, you can put these commands directly in `CMakeLists.txt.` E.g.:

```
set(CMAKE_CXX_FLAGS_DEBUG "put your flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL "put your flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "put your flags")
set(CMAKE_CXX_FLAGS_RELEASE "put your flags")
```

```
set(CMAKE_C_FLAGS_DEBUG "put your flags")
set(CMAKE_C_FLAGS_MINSIZEREL "put your flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "put your flags")
set(CMAKE_C_FLAGS_RELEASE "put your flags")
```

### maintenance

- formatting
  - `make clang-format`
- static analysis
  - `scan-build make`
  - `make ccpcheck`
  - `python run-clang-tidy.py -checks=clang-analyzer-*,cppcoreguidlines-*,misc-*,modernize-*,performance-*,readability-*,-readability-named-parameter -header-filter=src/`
- dynamic analysis (and running tests)
  - `valgrind ./tests` (do not forget to first compile with `-DCMAKE_BUILD_TYPE=Debug`)
