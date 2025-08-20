import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.pm.pre_processor import pre_processor
from VortexAD.core.pm.source_doublet.mu_sigma_solver import mu_sigma_solver
# from VortexAD.core.panel_method.steady.mu_sigma_solver_iterative import mu_sigma_solver_iterative
from VortexAD.core.pm.post_processor import post_processor, unstructured_post_processor

# def source_doublet_solver(exp_orig_mesh_dict, num_nodes, mesh_mode, constant_geometry, M_inf, rho, batch_size, Cp_cutoff, boundary_condition, patch_flag, iterative=False, warm_start=None, ROM=False, ref_point=np.zeros(3)):
def source_doublet_solver(orig_mesh_dict, solver_options_dict):
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

    if mesh_mode == 'structured':
        surface_0 = list(orig_mesh_dict.keys())[0]
        num_nodes = orig_mesh_dict[surface_0]['nodal_velocity'].shape[0]
    
    elif mesh_mode == 'unstructured':
        num_nodes = orig_mesh_dict['nodal_velocity'].shape[0]

    print('running pre-processing')
    mesh_dict = pre_processor(orig_mesh_dict, mode=mesh_mode, constant_geometry=reuse_AIC)

    print('solving for doublet strengths')
    if not iterative:
        # sigma = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode, bc=boundary_condition)
        # return sigma
    
        # AIC_mu = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode, bc=boundary_condition)
        # return AIC_mu
    
        # mu, sigma, wake_dict, AIC_mu, AIC_sigma, AIC_mu_orig = mu_sigma_solver(num_nodes, mesh_dict, mode=mesh_mode, bc=boundary_condition)
        mu, sigma, wake_dict, AIC_mu, AIC_sigma, RHS, AIC_mu_orig = mu_sigma_solver(
            num_nodes, mesh_dict, mode=mesh_mode, batch_size=partition_size, 
            bc=BC, ROM=ROM, constant_geometry=reuse_AIC
        )
        # return mu
    else:
        pass
        # mu, sigma, wake_dict = mu_sigma_solver_iterative(num_nodes, mesh_dict, mode=mesh_mode, batch_size=batch_size, bc=boundary_condition, warm_start=warm_start, ROM=ROM)
        # mu = mu.reshape((1,) + mu.shape)
        # sigma = sigma.reshape((1,) + sigma.shape)
    # return mu
    print('running post-processor')
    if mesh_mode == 'structured':
        output_dict = post_processor(mesh_dict, mu, sigma, num_nodes, rho, Cp_cutoff)
    elif mesh_mode == 'unstructured':
        output_dict = unstructured_post_processor(
            mesh_dict, mu, sigma, num_nodes, compressibility, 
            rho, Cp_cutoff, reuse_AIC, ref_point=moment_ref,
            ref_area=ref_area, ref_chord=ref_chord
        )

    output_dict['wake_dict'] = wake_dict
    if not iterative:
        output_dict['AIC_mu'] = AIC_mu
        output_dict['AIC_mu_orig'] = AIC_mu_orig
        output_dict['AIC_sigma'] = AIC_sigma
        output_dict['RHS'] = RHS
    
    else:
        output_dict['AIC_mu'] = 1
        output_dict['AIC_mu_orig'] = 1
        output_dict['AIC_sigma'] = 1
        output_dict['RHS'] = 1


    return output_dict, mesh_dict, mu, sigma