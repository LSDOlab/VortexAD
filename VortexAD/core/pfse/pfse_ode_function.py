import numpy as np
import csdl_alpha as csdl


def pfse_ode_function(orig_mesh_dict, solver_options_dict, nt, dt, ode_states, reuse_vars=False):
    '''
    Docstring for pfse_ode_function
    
    :param orig_mesh_dict: Description
    :param solver_options_dict: Description
    :param nt: Description
    :param dt: Description
    :param ode_states: Description
    :param reuse_vars: Description
    '''
    batch_size = solver_options_dict['partition_size']