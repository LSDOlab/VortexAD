import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.pm.source_doublet.steady_source_doublet_solver import source_doublet_solver

def steady_panel_solver(orig_mesh_dict, solver_options_dict):

    boundary_condition = solver_options_dict['BC']
    higher_order = solver_options_dict['higher_order']

    if higher_order:
        pass
        # outputs = linear_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, rho)
        # output_dict = outputs[0]
        # mesh_dict = outputs[1]
        # mu = outputs[2]
        # sigma = outputs[3]

        # return output_dict, mesh_dict, mu, sigma
    
    else:
        outputs = source_doublet_solver(orig_mesh_dict, solver_options_dict)
        output_dict = outputs[0]
        mesh_dict = outputs[1]
        mu = outputs[2]
        sigma = outputs[3]

        return output_dict, mesh_dict