# NACA0012 optimization

This directory shows an example of a lift-constrained induced drag minimization of a rectangular wing with a NACA0012 airfoil.

The run script is `run_optimization.py`. 

The optimization uses taper ratio, span, root twist and tip twist to optimize the induced drag. The root chord is fixed at 1 meter.

## Packages to install
- [CSDL](https://github.com/LSDOlab/CSDL_alpha)
- [lsdo_geo](https://github.com/LSDOlab/lsdo_geo)
- [lsdo_function_spaces](https://github.com/LSDOlab/lsdo_function_spaces)
- [VortexAD](https://github.com/LSDOlab/VortexAD)
- [modopt](https://github.com/LSDOlab/modopt)
- PySLSQP (via `pip install pyslsqp`)
- the `h5web` extension to VSCode (or your IDE of choice) is recommended to look at the optimization data
    - this extension allows you to look at the data interactively

## Post-processing functionalities
In our lab, we store optimization data in `hdf5` files.
The post-processing pipeline has a simple way of extracting this data and storing it in a more user-friendly file format.

Post-processing steps:
- Run `extract_data.py` with the proper file/case name in line 10. This will convert the optimization data into a `.pkl` file format. The data can be looked at by running the `check_data.py` file.
- Extract sectional data by running using `gen_sectional_data.py`. This is not fully generalized yet;  many of the details here are hard-coded.
- Some spanwise properties can be plotted using `plot_sectional_data.py` to view spanwise lift and twist distributions.