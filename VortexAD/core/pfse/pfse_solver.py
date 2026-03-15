import numpy as np
import csdl_alpha as csdl
import ozone

from VortexAD.core.pfse.pfse_ode_function import pfse_ode_function

def pfse_solver(orig_mesh_dict, solver_options_dict):

    dt                  = solver_options_dict['dt']
    nt                  = solver_options_dict['nt']
    store_state_history = solver_options_dict['store_state_history']
    reuse_AIC           = solver_options_dict['reuse_AIC']
    compressibility     = solver_options_dict['compressibility']
    rho                 = solver_options_dict['rho']
    sos                 = solver_options_dict['sos']
    ref_area            = solver_options_dict['ref_area']
    ref_chord           = solver_options_dict['ref_chord']
    moment_ref          = solver_options_dict['moment_reference']
    free_wake           = solver_options_dict['free_wake']
    ROM                 = solver_options_dict['ROM']
