# Welcome to VortexAD

<!-- This page describes conceptually the purpose of your package at a high-level.
Start with a one sentence description of your package.
For example, "This repository serves as a template for all LSDOlab projects with regard to documentation, testing and hosting of open-source code."
Include figures from the relevant paper and citation. -->

VortexAD is a general purpose potential flow solver library for aerodynamic analysis. VortexAD is built on CSDL, a graph-based modeling framework also developed at the LSDO Lab.

![777_Cp_streamlines](/src/images/777_Cp_streamlines.png "777 Cp + streamlines")


# Solver toolbox:
VortexAD aims to serve as a general potential flow solver library. The table below shows the available solvers and modes.

|     | Fixed wake | Prescribed wake | Free wake |
|:---:|:----------:|:---------------:|:---------:|
| VLM |      WIP   |         No      |      No   |
|  PM |      Yes   |        WIP      |     WIP   |

Table guide:
- **Yes**: solver works as is
- **WIP**: solver was written in a previous version and is being ported over
- **No**: solver has not been written but will be explored


# Cite us
```none
@inbook{doi:10.2514/6.2025-3021,
        author = {Luca Scotzniovsky and John T. Hwang},
        title = {A Fast, Memory-Efficient Panel Method for Large-Scale Multidisciplinary Design Optimization Under Uncertainty Using Graph-Based Modeling},
        booktitle = {AIAA AVIATION FORUM AND ASCEND 2025},
        chapter = {},
        pages = {},
        doi = {10.2514/6.2025-3021},
        URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2025-3021},
        eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2025-3021}
}
```

<!-- @inproceedings{scotzniovsky2025fast,
  title={A fast, memory-efficient panel method for large-scale multidisciplinary design optimization under uncertainty using graph-based modeling},
  author={Scotzniovsky, Luca and Hwang, John T},
  booktitle={AIAA AVIATION FORUM AND ASCEND 2025},
  pages={3021},
  year={2025}
} -->


# Brought to you by

![LSDO Lab image](/src/images/lsdolab.png "LSDO Lab image")



<!-- Remove/add custom pages from/to toc as per your package's requirement -->

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
src/tutorials
<!-- src/custom_1
src/custom_2 -->
src/examples
src/api
```

<!-- <img src="/src/images/lsdolab.png" alt="LSDO Lab logo" width="200"/> -->