# Post-processing the NACA0012 optimization results

The files here are a generic guideline to how to post-process the optimization results.
The order in which each file should be called, along with general info, is shown below.
This specific post-processing pipeline is used to generate sectional distributions (such as lift and twist), 
as well as sectional pressure distributions.

1. extract_data.py
    - This file extracts the data from the hdf5 file and stores it into a pickle format, which is more user friendly to most python users.
2. check_data.py
    - This is an optional file the user can run to take a look at optimization data.
    - The data loaded from the pickle file is stored in a dictionary. Use the .keys() attribute to deduce variable names, which line up with the optimization script.
3. gen_sectional_data.py
    - This file runs most of the forward evaluation of the optimization, but loads the converged solution.
    - This file extracts sectional inflow angle and lift.
4. plot_sectional_data.py
    - This file plots the data gathered from the previous script.
5. make_animations.py (BETA)
    - This file generates an animation of the solution across the optimization history, compared to the initial design.