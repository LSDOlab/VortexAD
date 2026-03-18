import numpy as np
import csdl_alpha as csdl

from VortexAD.core.pm.source_doublet.compute_source_strength import compute_source_strength
from VortexAD.core.pfse.functions.compute_linear_system_terms import compute_linear_system_terms


def solve_linear_system(num_nodes, solver_options_dict, pm_mesh_dict, vlm_vectorized_dict, mu_w):
    '''
    We generate the AIC matrices and the proper RHS matrix-vector products here.
    We have three non-vectorizable components to handle
    - Tri-elements of panel method
    - Quad-elements of panel method
    - VLM vortex rings
    This means we will generate our AIC matrix in 9 specific regions

    For efficiency, we shouldn't generate the dense AICs for sigma and the wake doublets.
    We should use matvec products so we don't need to store the giant matrices in memory.

    For the V*n
    '''
    partition_size  = solver_options_dict['partition_size']
    pm_BC           = solver_options_dict['panel_method_BC']

    num_pm_panels = pm_mesh_dict['num_cells']
    
    # computing source strength
    sigma = compute_source_strength(
        pm_mesh_dict,
        num_nodes,
        num_pm_panels,
        mesh_mode='unstructured',
    )

    # computing VLM surface BC
    vlm_panel_normal = vlm_vectorized_dict['panel_normal']
    vlm_coll_vel = vlm_vectorized_dict['collocation_velocity']
    num_vlm_panels = vlm_coll_vel.shape[1]
    VLM_BC = csdl.sum(vlm_coll_vel[0,:]*vlm_panel_normal[0,:], axes=(1,))
    num_tot_panels = num_pm_panels + num_vlm_panels
    bc_VR = csdl.Variable(value=np.zeros((num_nodes, num_tot_panels,)))
    bc_VR = bc_VR.set(csdl.slice[0,num_pm_panels:], VLM_BC) # this BC for PM panels is zero
    # bc_VR is the VLM RHS: -csdl.dot(V,n)

    AIC_mu, RHS_sigma, RHS_w = compute_linear_system_terms(
        pm_mesh_dict,
        vlm_vectorized_dict,
        sigma,
        mu_w,
        batch_size=partition_size,
        bc=pm_BC,
    )

    # full RHS
    RHS = -RHS_w - RHS_sigma - bc_VR
    # RHS_w is the wake-induced potential and normal velocity
    # RHS_sigma is the source-induced potential and normal velocity

    # solving linear system
    mu = csdl.solve_linear(AIC_mu[0,:,:], RHS[0,:])
    mu = mu.reshape((1,) + mu.shape, 'i->ia')

    output_dict = {
        'mu': mu,
        'sigma': sigma,
    }

    return output_dict