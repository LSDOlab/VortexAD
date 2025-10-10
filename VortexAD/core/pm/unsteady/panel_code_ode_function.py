import numpy as np
import csdl_alpha as csdl

from VortexAD.core.pm.pre_processor import pre_processor

def panel_code_ode_function(orig_mesh_dict, solver_options_dict, ode_states):
    rho             = solver_options_dict['rho']
    compressibility = solver_options_dict['compressibility']
    sos             = solver_options_dict['sos']
    Cp_cutoff       = solver_options_dict['Cp cutoff']
    BC              = solver_options_dict['BC']
    mesh_mode       = solver_options_dict['mesh_mode']
    partition_size  = solver_options_dict['partition_size']
    iterative       = solver_options_dict['iterative']
    ROM             = solver_options_dict['ROM']
    reuse_AIC       = solver_options_dict['reuse_AIC']
    ref_area        = solver_options_dict['ref_area']
    ref_chord       = solver_options_dict['ref_chord']
    moment_ref      = solver_options_dict['moment_reference']
    drag_type       = solver_options_dict['drag_type']

    if isinstance(rho, float):
        rho = csdl.Variable(value=np.array([rho]))
    elif isinstance(rho, list):
        rho = csdl.Variable(value=np.array(rho))

    if mesh_mode == 'structured':
        surface_0 = list(orig_mesh_dict.keys())[0]
        num_nodes = orig_mesh_dict[surface_0]['nodal_velocity'].shape[0]
    
    elif mesh_mode == 'unstructured':
        num_nodes = orig_mesh_dict['nodal_velocity'].shape[0]

    print('running pre-processing')
    mesh_dict = pre_processor(orig_mesh_dict, mode=mesh_mode, constant_geometry=reuse_AIC)



    return outputs, d_dt