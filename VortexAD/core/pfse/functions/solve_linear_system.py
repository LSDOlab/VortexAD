import numpy as np
import csdl_alpha as csdl

from VortexAD.core.pm.source_doublet.compute_source_strength import compute_source_strength
from VortexAD.core.pfse.functions.compute_linear_system_terms import compute_linear_system_terms


def solve_linear_system(num_nodes, solver_options_dict, pm_mesh_dict, vlm_vectorized_dict, wake_mesh_dict, mu_w):
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
    pm_BC           = solver_options_dict['BC_PM']

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
        wake_mesh_dict,
        sigma,
        mu_w,
        num_tot_panels=num_tot_panels,
        batch_size=partition_size,
        bc=pm_BC,
    )

    # full RHS
    # RHS = -RHS_sigma - bc_VR
    # RHS_w is the wake-induced potential and normal velocity
    # RHS_sigma is the source-induced potential and normal velocity
    if pm_BC == 'Neumann':
        # adjust RHS to include sigma as part of the flow tangency condition
        bc_VR = bc_VR.set(csdl.slice[0,:num_pm_panels], -sigma[0,:])
        # NOTE: might need a negative sign on the sigma here since we subtract it down below
        # the flow tangency RHS is -dot(V,n)
        # bc_VR computes dot(V,n) and then we add a negative sign when computing the RHS
        # sigma = -dot(V,n) already, so we add a negative sign to match the sign convention

    # RHS = -RHS_w - RHS_sigma - bc_VR
    RHS = -(RHS_w + RHS_sigma + bc_VR)


    if pm_BC == 'Dirichlet':
        # solving linear system
        mu = csdl.solve_linear(AIC_mu[0,:,:], RHS[0,:])
        mu = mu.reshape((1,) + mu.shape)
    elif pm_BC == 'Neumann':
        
        ind = 0
        mu_0 = 0
        v = AIC_mu[0,:,ind]
        A_bar = csdl.Variable(value=np.zeros((AIC_mu.shape[1], AIC_mu.shape[1]-1)))
        A_bar = A_bar.set(csdl.slice[:,ind:], AIC_mu[0,:,(ind+1):])
        if ind != 0:
            A_bar = A_bar.set(csdl.slice[:,:ind], AIC_mu[0,:,:ind])
        A_bar_T = A_bar.T()
        lin_sys_red = csdl.matmat(A_bar_T, A_bar)
        RHS_red = csdl.matvec(A_bar_T, RHS[0,:]-mu_0*v)
        mu_red = csdl.solve_linear(lin_sys_red, RHS_red)

        mu = csdl.Variable(value=np.zeros((1, AIC_mu.shape[1])))
        mu = mu.set(csdl.slice[0,ind], mu_0)
        mu = mu.set(csdl.slice[0,(ind+1):], mu_red[ind:])
        if ind != 0:
            mu = mu.set(csdl.slice[0,:ind], mu_red[:ind])

    output_dict = {
        'mu': mu,
        'sigma': sigma,
    }

    return output_dict