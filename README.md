# (Factored) Bayes-Adaptive POMDP

* written by Sammie Katt (katt.s at husky dot neu dot edu)
* find it @ git@bitbucket.org:samdjo/po-rl.git

A code base to run (Bayes-Adaptive) reinforcement learning experiments on
partially observable domains. This project is meant for reinforcement learning
researchers to compare different methods. It contains various different
environments to test the methods on, of which all partially observable and
discrete. Note that this project has mostly been written for personal use,
research, and thus may lack the documentation that one would typically expect
from open source projects.

## Installation

```console
mkdir wherever/you/want && cd wherever/you/want
cmake -DCMAKE_BUILD_TYPE=Release /path/to/root/of/this/project
make
```

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

``` console
cd to/the/build/directory
make docs
```

## Development

### TODO

* automate clang-tidy static analysis

### maintenance
* formatting
    - ``` make clang-format ```
* static analysis
    - ``` scan-build make ```
    - ``` make ccpcheck ```
    - ``` python run-clang-tidy.py -checks=clang-analyzer-*,cppcoreguidlines-*,misc-*,modernize-*,performance-*,readability-*,-readability-named-parameter -header-filter=src/ ```
* dynamic analysis (and running tests)
    - ``` valgrind ./tests ``` (do not forget to first compile with ``` -DCMAKE_BUILD_TYPE=Debug ```)
