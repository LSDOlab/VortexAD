import numpy as np
import csdl_alpha as csdl

from VortexAD.core.elements.source import compute_source_influence_new
from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel

def compute_wake_velocity(vectorized_mesh_dict, pm_mesh_dict, vlm_vectorized_dict, total_wake_mesh_dict, 
                          batch_size, x_w, mu, sigma, mu_w, free_wake=False, vc=1.e-6):

    nodal_velocity = vectorized_mesh_dict['nodal_velocity']
    TE_node_indices = vectorized_mesh_dict['TE_node_indices']
    TE_velocity = nodal_velocity[:,TE_node_indices,:]
    num_TE_pts = len(TE_node_indices)
    num_wake_pts = x_w.shape[1]
    wake_vel = TE_velocity.expand((1, num_TE_pts, int(num_wake_pts/num_TE_pts), 3), 'ijk->ijak')
    wake_vel = wake_vel.reshape((1, num_wake_pts, 3))

    if free_wake:
        ind_vel = compute_free_wake_velocity(
            pm_mesh_dict,
            vlm_vectorized_dict,
            total_wake_mesh_dict,
            batch_size,
            mu,
            sigma,
            mu_w,
            x_w,
            vc_body=vc
        )
        wake_vel = wake_vel + ind_vel

    return wake_vel

def compute_free_wake_velocity(pm_mesh_dict, vlm_vectorized_dict, total_wake_mesh_dict, batch_size, mu, sigma, mu_w, x_w, vc_body):
    '''
    evaluating at the wake nodes (which is vectorized)
    Code block has 2 sections:
    - panel method (tri and quad doublets and sources)
        - there would be a loop here over cell type
    - vlm induced velocities (via vectorized mesh dict)

    As a result, there are 6 terms that induce velocities:
    We divide the velocity computation into four groupings:
    - each grouping constitutes a call to a batching functin
    - Triangular PM elements
        - sources and doublets
    - Quad PM elements
        - sources and doublets
    - VLM surface doublets/vortex rings
    - wake doublets/vortex rings

    We can use two batching functions for computing induced velocities.
    One will be for doublets, other for sources
    A\mu + B\sigma + C\mu_w = x_w
    A \in (nw, np)
    B \in (nw, np_pm)
    C \in (nw, nwp)
    '''
    # eval_pt = x_w.reshape((1,) + x_w.shape)
    eval_pt = x_w
    num_wake_pts = x_w.shape[1]

    # region Panel method surface induced free-wake velocities
    cells = pm_mesh_dict['cell_point_indices'] # keys are cell types, entries are points for each cell
    cell_types = list(cells.keys())
    cell_adjacency_types = pm_mesh_dict['cell_adjacency'] # keys are cell types, entries are adjacent cell indices
    num_cells_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]
    batch_size_surf = batch_size
    if batch_size is None:
        batch_size_surf = num_wake_pts

    surf_induced_vel_batch_func = csdl.experimental.batch_function(
        PM_free_wake_vel_batched,
        # batch_size=batch_size,
        batch_size=batch_size_surf,
        batch_dims=[1]+[None]*10
    )

    start_j, stop_j = 0, 0
    PM_doublet_ind_vel_list = []
    PM_source_ind_vel_list = []
    AIC_sigma_list = []
    for j, cell_type_j in enumerate(cell_types):
        num_cells_j = num_cells_per_type[j]
        stop_j += num_cells_j

        start_stop = [start_j, stop_j]

        coll_point = pm_mesh_dict['panel_center_' + cell_type_j] # (nn, num_tot_panels, 3)
        panel_corners = pm_mesh_dict['panel_corners_' + cell_type_j] # (nn, num_tot_panels, 3, 3) 
        panel_x_dir = pm_mesh_dict['panel_x_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
        panel_y_dir = pm_mesh_dict['panel_y_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
        panel_normal = pm_mesh_dict['panel_normal_' + cell_type_j] # (nn, num_tot_panels, 3)
        S = pm_mesh_dict['S_' + cell_type_j]
        SL = pm_mesh_dict['SL_' + cell_type_j]
        SM = pm_mesh_dict['SM_' + cell_type_j]

        mu_cell_type = mu[:,start_j:stop_j]
        sigma_cell_type = sigma[:,start_j:stop_j]

        doublet_ind_vel, source_ind_vel = surf_induced_vel_batch_func(
            eval_pt, 
            coll_point,
            panel_corners,
            panel_x_dir,
            panel_y_dir,
            panel_normal,
            S,
            SL,
            SM,
            mu_cell_type,
            sigma_cell_type,
            vc=vc_body
        )
        # print('===')
        # print(AIC_sigma.shape)
        # print(doublet_ind_vel.shape)
        PM_doublet_ind_vel_list.append(doublet_ind_vel)
        PM_source_ind_vel_list.append(source_ind_vel)
        # AIC_sigma_list.append(AIC_sigma.reshape((1, num_wake_pts, num_cells_j, 3)))
        start_j += num_cells_j
    # exit()
    PM_doublet_ind_vel = sum(PM_doublet_ind_vel_list)
    PM_source_ind_vel = sum(PM_source_ind_vel_list)
    # endregion

    # region VLM surface induced free-wake velocities
    batch_size_surf = batch_size
    # eval_pt = x_w.reshape((1,) + x_w.shape)
    num_wake_pts = eval_pt.shape[1]
    if batch_size is None:
        batch_size_surf = num_wake_pts

    surf_induced_vel_batch_func = csdl.experimental.batch_function(
        doublet_induced_vel_batched,
        # batch_size=batch_size,
        batch_size=batch_size_surf,
        batch_dims=[1]+[None]*2
    )

    # coll_point = vlm_vectorized_dict['panel_centers']
    panel_corners = vlm_vectorized_dict['panel_corners']
    num_body_panels = panel_corners.shape[1]

    VLM_ind_vel = surf_induced_vel_batch_func(
        eval_pt,
        panel_corners,
        mu[:,stop_j:],
        vc=vc_body # constant core model on the body
    )
    # endregion

    # region wake-induced free-wake velocities
    batch_size_wake = batch_size
    if batch_size is None:
        batch_size_wake = num_wake_pts

    wake_induced_vel_batch_func = csdl.experimental.batch_function(
        doublet_induced_vel_batched,
        # batch_size=batch_size,
        batch_size=batch_size_wake,
        batch_dims=[1]+[None]*3
    )

    wake_panel_corners = total_wake_mesh_dict['panel_corners']
    vc_wake = total_wake_mesh_dict['vortex_core_radius']

    wake_ind_vel = wake_induced_vel_batch_func(
        eval_pt,
        wake_panel_corners,
        mu_w,
        vc_wake # csdl variable with finite core model
    )
    # endregion

    ind_vel = PM_doublet_ind_vel+PM_source_ind_vel+VLM_ind_vel+wake_ind_vel
    ind_vel = ind_vel.reshape((1, num_wake_pts, 3))

    return ind_vel

def PM_free_wake_vel_batched(coll_point, panel_center, panel_corners, panel_x_dir, panel_y_dir,
                        panel_normal, S_j, SL_j, SM_j, mu, sigma, vc):
    
    num_nodes = coll_point.shape[0]
    num_eval_pts = coll_point.shape[1]
    num_induced_pts = panel_center.shape[1]
    num_interactions = num_eval_pts*num_induced_pts
    num_corners = panel_corners.shape[2]

    expanded_shape = (num_nodes, num_eval_pts, num_induced_pts, num_corners, 3)
    vectorized_shape = (num_nodes, num_interactions, num_corners, 3)

    # ============ expanding across columns ============
    coll_point_exp = csdl.expand(coll_point, expanded_shape, 'ijk->ijabk')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    # ============ expanding across rows ============
    coll_point_j_exp = csdl.expand(panel_center, expanded_shape, 'ijk->iajbk')
    coll_point_j_exp_vec = coll_point_j_exp.reshape(vectorized_shape)

    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'ijkl->iajkl')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    panel_x_dir_exp = csdl.expand(panel_x_dir, expanded_shape, 'ijk->iajbk')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)
    panel_y_dir_exp = csdl.expand(panel_y_dir, expanded_shape, 'ijk->iajbk')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)
    panel_normal_exp = csdl.expand(panel_normal, expanded_shape, 'ijk->iajbk')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    S_j_exp = csdl.expand(S_j, expanded_shape[:-1] , 'ijk->iajk')
    S_j_exp_vec = S_j_exp.reshape(vectorized_shape[:-1])

    SL_j_exp = csdl.expand(SL_j, expanded_shape[:-1], 'ijk->iajk')
    SL_j_exp_vec = SL_j_exp.reshape(vectorized_shape[:-1])

    SM_j_exp = csdl.expand(SM_j, expanded_shape[:-1], 'ijk->iajk')
    SM_j_exp_vec = SM_j_exp.reshape(vectorized_shape[:-1])

    a = coll_point_exp_vec - panel_corners_exp_vec + 1.e-12 # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_j_exp_vec + 1.e-12  # RcJ - RcK
    sum_ind = len(a.shape) - 1

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_exp_vec, axes=(sum_ind,)) # normal projection of CP
    # print(A.shape)
    B = csdl.Variable(shape=A.shape, value=0.)
    B = B.set(csdl.slice[:,:,:-1], value=A[:,:,1:])
    B = B.set(csdl.slice[:,:,-1], value=A[:,:,0])

    BL = csdl.Variable(shape=AL.shape, value=0.)
    BL = BL.set(csdl.slice[:,:,:-1], value=BL[:,:,1:])
    BL = BL.set(csdl.slice[:,:,-1], value=BL[:,:,0])

    BM = csdl.Variable(shape=AM.shape, value=0.)
    BM = BM.set(csdl.slice[:,:,:-1], value=AM[:,:,1:])
    BM = BM.set(csdl.slice[:,:,-1], value=AM[:,:,0])

    A1 = AM*SL_j_exp_vec - AL*SM_j_exp_vec

    # additional expansions for the (3,) dimension for velocity
    A = A.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    AM = AM.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    B = B.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    BM = BM.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    SL_j_exp_vec = SL_j_exp_vec.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    SM_j_exp_vec = SM_j_exp_vec.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    A1 = A1.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    PN = PN.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
    S_j_exp_vec = S_j_exp_vec.expand(panel_normal_exp_vec.shape, 'ijk->ijka')

    A_list = [A[:,:,ind] for ind in range(num_corners)]
    AM_list = [AM[:,:,ind] for ind in range(num_corners)]
    B_list = [B[:,:,ind] for ind in range(num_corners)]
    BM_list = [BM[:,:,ind] for ind in range(num_corners)]
    SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(num_corners)]
    SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(num_corners)]
    A1_list = [A1[:,:,ind] for ind in range(num_corners)]
    PN_list = [PN[:,:,ind] for ind in range(num_corners)]
    S_list = [S_j_exp_vec[:,:,ind] for ind in range(num_corners)]

    AIC_sigma_vec = compute_source_influence_new(
        A_list, 
        AM_list, 
        B_list, 
        BM_list, 
        SL_list, 
        SM_list, 
        A1_list, 
        PN_list, 
        S_list,
        panel_x_dir_exp_vec[:,:,0,:],
        panel_y_dir_exp_vec[:,:,0,:],
        panel_normal_exp_vec[:,:,0,:],
        mode='velocity'
    )
    AIC_sigma = AIC_sigma_vec.reshape((1, num_eval_pts, num_induced_pts, 3))
    # source_ind_vel = csdl.matvec(AIC_sigma, sigma)
    source_ind_vel = csdl.einsum(AIC_sigma, sigma, action='ijkl,ik->ijl')

    num_edges = panel_corners.shape[2]
    vc_exp_vec = vc
    if isinstance(vc, csdl.Variable):
        vc_exp = csdl.expand(vc, (num_nodes, num_eval_pts, num_induced_pts, num_corners), 'ijk->iajk')
        vc_exp_vec = vc_exp.reshape((num_nodes, num_interactions, num_corners))
        vc_list = [vc_exp_vec[:,:,i] for i in range(num_edges)]
    else:
        vc_list = [vc]*num_edges
    AIC_mu_list = []
    for i in range(num_edges-1):
        asdf = compute_vortex_line_ind_vel(
            panel_corners_exp_vec[:,:,i], 
            panel_corners_exp_vec[:,:,i+1], 
            coll_point_exp_vec[:,:,0], 
            mode='wake', 
            vc=vc_list[i]
        )
        AIC_mu_list.append(asdf)
    asdf = compute_vortex_line_ind_vel(
        panel_corners_exp_vec[:,:,-1], 
        panel_corners_exp_vec[:,:,0], 
        coll_point_exp_vec[:,:,0], 
        mode='wake', 
        vc=vc_list[-1]
    )
    AIC_mu_list.append(asdf)
    AIC_mu_vec = sum(AIC_mu_list)

    AIC_mu = AIC_mu_vec.reshape((1, num_eval_pts, num_induced_pts, 3))
    # doublet_ind_vel = csdl.matvec(AIC_mu, mu)
    doublet_ind_vel = csdl.einsum(AIC_mu, mu, action='ijkl,ik->ijl')

    return doublet_ind_vel, source_ind_vel
    # return doublet_ind_vel, source_ind_vel, AIC_sigma

def doublet_induced_vel_batched(coll_point, panel_corners, mu, vc):

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
    ind_vel = csdl.einsum(AIC_vel, mu, action='ijkl,ik->ijl')
    # ind_vel = csdl.Variable(value=np.zeros((num_eval_pts, 3)))
    # for i in range(3):
    #     ind_vel = ind_vel.set(
    #         csdl.slice[:,i],
    #         csdl.sum(AIC_vel_vec[:,i]*mu)
    #     )
    return ind_vel