[![Build Status](https://dev.azure.com/blosc/caterva/_apis/build/status/Blosc.cat4py?branchName=master)](https://dev.azure.com/blosc/caterva/_build/latest?definitionId=1&branchName=master)
[![codecov](https://codecov.io/gh/Blosc/cat4py/branch/master/graph/badge.svg)](https://codecov.io/gh/Blosc/cat4py)
<!--- ![](https://img.shields.io/azure-devops/coverage/blosc/caterva/1.svg)--->

# cat4py

Python wrapper for Caterva.  Still on development.

## Clone repo and submodules

```sh
$ git clone https://github.com/Blosc/cat4py
$ git submodule init
$ git submodule update --recursive --remote 
```

## Development workflow

### Compile

```sh
$ CFLAGS='' python setup.py build_ext -i
```

**Please note**: If the CFLAGS environment variable is not passed, Anaconda Python (maybe other distributions too) will inject their own paths there. As a result, it will find possible incompatible headers/libs for Blosc, LZ4 or Zstd.  I understand packagers trying to re-use shared libraries in their setups, but this can create issues when normal users try to compile extensions by themselves.

Compiling the extension implies re-compiling C-Blosc2 and Caterva sources everytime, so a trick for accelerating the process during the development process is to direct the compiler to not optimize the code:

```sh
$ CFLAGS=-O0 python setup.py build_ext -i
```

### Run tests

```sh
$ PYTHONPATH=. pytest
```

### Run example

```sh
$ PYTHONPATH=. python examples/ex_persistency.py
```
