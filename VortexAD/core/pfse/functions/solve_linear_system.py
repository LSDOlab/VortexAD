import numpy as np
import csdl_alpha as csdl

from VortexAD.core.pm.source_doublet.compute_source_strength import compute_source_strength
from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel
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

    induced_vel_4_sigma = compute_ind_vel_4_sigma(
        pm_mesh_dict,
        wake_mesh_dict, # this is the vectorized wake mesh dictionary
        mu_w,
        batch_size=partition_size
    )
    sigma = sigma - induced_vel_4_sigma # negative sign is intentional

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

    # RHS = - (RHS_sigma + bc_VR)
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

def compute_ind_vel_4_sigma(pm_mesh_dict, wake_mesh_dict, mu_w, batch_size):
    # evaluation point info (PM panel centers + normal)
    eval_point = pm_mesh_dict['panel_center_mod']
    eval_normal = pm_mesh_dict['panel_normal']

    # velocity inducing info (from wake)
    wake_panel_corners = wake_mesh_dict['panel_corners']
    vc = wake_mesh_dict['vortex_core_radius']
    # print(mu_w.shape)
    num_wake_panels = mu_w.shape[1]

    batch_size_surf = batch_size
    batch_size_surf = 2
    if batch_size is None:
        batch_size_surf = num_wake_panels

    wake_ind_vel_batch_func = csdl.experimental.batch_function(
        wake_ind_vel_batched,
        batch_size=batch_size_surf,
        batch_dims=[1,1,None, None]
    )

    ind_vel_4_sigma = wake_ind_vel_batch_func(
        eval_point,
        eval_normal,
        wake_panel_corners,
        mu_w,
        vc=1.e-4,
    )

    ind_vel_4_sigma = ind_vel_4_sigma.reshape((1, np.prod(ind_vel_4_sigma.shape)))
    
    return ind_vel_4_sigma

def wake_ind_vel_batched(coll_point, coll_normal, panel_corners, mu_w, vc=None):
    num_nodes = coll_point.shape[0]
    num_eval_pts = coll_point.shape[1]
    num_induced_pts = panel_corners.shape[1]
    num_interactions = num_eval_pts*num_induced_pts
    num_corners = panel_corners.shape[2]

    expanded_shape = (num_nodes, num_eval_pts, num_induced_pts, num_corners, 3)
    vectorized_shape = (num_nodes, num_interactions, num_corners, 3)

    # ============ expanding across columns ============
    coll_point_exp = csdl.expand(coll_point, expanded_shape, 'ijk->ijabk')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    # ============ expanding across rows ============
    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'ijkl->iajkl')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    num_edges = num_corners

    vc_exp_vec = vc
    if isinstance(vc, csdl.Variable):
        vc_exp = csdl.expand(vc, (num_nodes, num_eval_pts, num_induced_pts, num_corners), 'ijk->iajk')
        vc_exp_vec = vc_exp.reshape((num_nodes, num_interactions, num_corners))
        vc_list = [vc_exp_vec[:,:,i] for i in range(num_edges)]
    else:
        vc_list = [vc]*num_edges

    AIC_vel_vec_list = []
    for i in range(num_edges-1):
        asdf = compute_vortex_line_ind_vel(
            panel_corners_exp_vec[:,:,i], 
            panel_corners_exp_vec[:,:,i+1], 
            coll_point_exp_vec[:,:,0], 
            mode='wake', 
            vc=vc_list[i]
        )
        AIC_vel_vec_list.append(asdf)
    asdf = compute_vortex_line_ind_vel(
        panel_corners_exp_vec[:,:,-1], 
        panel_corners_exp_vec[:,:,0], 
        coll_point_exp_vec[:,:,0], 
        mode='wake', 
        vc=vc_list[-1]
    )
    AIC_vel_vec_list.append(asdf)
    AIC_vel_vec = sum(AIC_vel_vec_list)

    AIC_vel = AIC_vel_vec.reshape((1, num_eval_pts, num_induced_pts, 3))
    AIC_vel_normal_proj = csdl.einsum(AIC_vel, coll_normal, action='ijak,ijk->ija')
    ind_vel = csdl.einsum(AIC_vel_normal_proj, mu_w, action='ijk,ik->ij')
    print(num_eval_pts)
    print(num_induced_pts)
    print(AIC_vel_normal_proj.shape)
    print(AIC_vel.shape)
    ind_vel = ind_vel.reshape((num_nodes, num_eval_pts))
    return ind_vel