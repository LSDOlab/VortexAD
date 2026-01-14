import numpy as np
import csdl_alpha as csdl

from VortexAD.core.vlm.pre_processor import pre_processor
from VortexAD.core.vlm.gamma_solver import gamma_solver
from VortexAD.core.vlm.post_processor import post_processor

def steady_vlm_solver(orig_mesh_dict, solver_options_dict):
    # num_nodes = solver_options_dict['num_nodes']
    num_nodes = solver_options_dict['num_nodes']

    alpha_ML = solver_options_dict['alpha_ML']

    mesh_dict, vectorized_mesh_dict = pre_processor(orig_mesh_dict)

    gamma = gamma_solver(num_nodes, mesh_dict)

    surface_output_dict, total_output_dict = post_processor(num_nodes, mesh_dict, gamma, alpha_ML=alpha_ML)
    surface_output_dict['gamma'] = gamma
    surface_output_dict['wake_vortex_mesh'] = mesh_dict['surface_0']['wake_vortex_mesh']
    surface_output_dict['net_gamma'] = surface_output_dict['net_gamma'][0]

    return surface_output_dict #, total_output_dict