# VortexAD

<!---
[![Python](https://img.shields.io/pypi/pyversions/VortexAD)](https://img.shields.io/pypi/pyversions/VortexAD)
[![Pypi](https://img.shields.io/pypi/v/VortexAD)](https://pypi.org/project/VortexAD/)
[![Coveralls Badge][13]][14]
[![PyPI version][10]][11]
[![PyPI Monthly Downloads][12]][11]
-->

[![GitHub Actions Test Badge](https://github.com/LSDOlab/VortexAD/actions/workflows/actions.yml/badge.svg)](https://github.com/VortexAD/VortexAD/actions)
[![Forks](https://img.shields.io/github/forks/LSDOlab/VortexAD.svg)](https://github.com/LSDOlab/VortexAD/network)
[![Issues](https://img.shields.io/github/issues/LSDOlab/VortexAD.svg)](https://github.com/LSDOlab/VortexAD/issues)

A general potential flow solver repository.

![777_Cp_streamlines](/docs/src/images/777_Cp_streamlines.png "Title displayed")

Available solvers:

|     | Fixed wake | Prescribed wake | Free wake |
|:---:|:----------:|:---------------:|:---------:|
| VLM |      WIP   |         No      |      No   |
|  PM |      Yes   |        WIP      |     WIP   |

Table guide:
- **Yes**: solver works as is
- **WIP**: solver was written in a previous version and is being ported over
- **No**: solver has not been written but will be explored

<!-- A template repository for LSDOlab projects

This repository serves as a template for all LSDOlab projects with regard to documentation, testing and hosting of open-source code.
Note that template users need to edit the README badge definitions for their respective packages.

*README.md file contains high-level information about your package: it's purpose, high-level instructions for installation and usage.* -->

# Installation

## Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/LSDOlab/VortexAD.git
```
If you want users to install a specific branch, run
```sh
pip install git+https://github.com/LSDOlab/VortexAD.git@branch
```

<!-- **Enabled by**: `packages=find_packages()` in the `setup.py` file. -->

## Installation instructions for developers
To install `VortexAD`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/LSDOlab/VortexAD.git
pip install -e ./VortexAD
```

# For Developers
For details on documentation, refer to the README in `docs` directory.

For details on testing/pull requests, refer to the README in `tests` directory.

# License
This project is licensed under the terms of the **GNU Lesser General Public License v3.0**.
