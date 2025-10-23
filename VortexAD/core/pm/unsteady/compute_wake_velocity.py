import numpy as np
import csdl_alpha as csdl

from VortexAD.core.elements.source import compute_source_influence_new
from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel

def compute_wake_velocity(mesh_dict, wake_mesh_dict, batch_size, mu, sigma, mu_w, free_wake=False, vc=1.e-6):

    x_w = wake_mesh_dict['wake_mesh']
    num_nodes = x_w.shape[0]
    num_wake_pts = x_w.shape[1]
    # wake_vel = csdl.Variable(value=np.zeros(num_nodes, num_wake_pts, 3))

    mesh_nodal_vel = mesh_dict['nodal_velocity']
    TE_node_indices = mesh_dict['TE_node_indices']
    num_TE_pts = TE_node_indices.shape[0]
    TE_vel = mesh_nodal_vel[:, list(TE_node_indices), :]

    wake_connectivity = wake_mesh_dict['wake_connectivity']
    # wake_vel = TE_vel.expand((1,) + wake_connectivity.shape[:-1] + (3,), 'ijk->ijak')
    wake_vel = TE_vel.expand((1, num_TE_pts, int(num_wake_pts/num_TE_pts), 3), 'ijk->ijak')
    wake_vel = wake_vel.reshape((1, num_wake_pts, 3))

    wake_vel_vars = {}

    if free_wake:
        ind_vel, free_wake_vars = compute_free_wake_velocity(mesh_dict, wake_mesh_dict, batch_size, mu, sigma, mu_w, vc)
        ind_vel = ind_vel.reshape((1, num_wake_pts, 3))
        wake_vel = wake_vel + ind_vel
        # wake_vel = wake_vel

        wake_vel_vars = {
            # 'AIC_fw_mu': free_wake_vars['AIC_fw_mu'],
            'AIC_fw_sigma': free_wake_vars['AIC_fw_sigma'],
            # 'AIC_fw_mu_w': free_wake_vars['AIC_fw_mu_w'],
        }

    return wake_vel, wake_vel_vars

def compute_free_wake_velocity(mesh_dict, wake_mesh_dict, batch_size, mu, sigma, mu_w, vc):
    x_w = wake_mesh_dict['wake_mesh']
    num_nodes = x_w.shape[0]
    num_wake_pts = x_w.shape[1]
    # ind_vel = csdl.Variable(value=np.zeros(num_nodes, num_wake_pts, 3))
    # use a similar loop structure to the AIC, where the outer is the x_w
    # inner will require a loop around the cell types

    cells = mesh_dict['cell_point_indices'] # keys are cell types, entries are points for each cell
    cell_types = list(cells.keys())
    cell_adjacency_types = mesh_dict['cell_adjacency'] # keys are cell types, entries are adjacent cell indices
    num_cells_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]
    num_tot_panels = sum(num_cells_per_type)

    batch_size_surf = batch_size
    if batch_size is None:
        batch_size_surf = num_wake_pts

    surf_induced_vel_batch_func = csdl.experimental.batch_function(
        surf_induced_vel_batched,
        # batch_size=batch_size,
        batch_size=batch_size_surf,
        batch_dims=[1]+[None]*11
    )

    x_w = wake_mesh_dict['wake_mesh']

    start_j, stop_j = 0, 0
    doublet_ind_vel_list = []
    source_ind_vel_list = []
    AIC_sigma_list = []
    for j, cell_type_j in enumerate(cell_types):
        num_cells_j = num_cells_per_type[j]
        stop_j += num_cells_j

        start_stop = [start_j, stop_j]

        coll_point = mesh_dict['panel_center_' + cell_type_j] # (nn, num_tot_panels, 3)
        panel_corners = mesh_dict['panel_corners_' + cell_type_j] # (nn, num_tot_panels, 3, 3) 
        panel_x_dir = mesh_dict['panel_x_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
        panel_y_dir = mesh_dict['panel_y_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
        panel_normal = mesh_dict['panel_normal_' + cell_type_j] # (nn, num_tot_panels, 3)
        S = mesh_dict['S_' + cell_type_j]
        SL = mesh_dict['SL_' + cell_type_j]
        SM = mesh_dict['SM_' + cell_type_j]

        mu_cell_type = mu[:,start_j:stop_j]
        sigma_cell_type = sigma[:,start_j:stop_j]

        doublet_ind_vel, source_ind_vel, AIC_sigma = surf_induced_vel_batch_func(
            x_w, 
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
            vc
        )
        # print('===')
        # print(AIC_sigma.shape)
        # print(doublet_ind_vel.shape)
        doublet_ind_vel_list.append(doublet_ind_vel)
        source_ind_vel_list.append(source_ind_vel)
        AIC_sigma_list.append(AIC_sigma.reshape((1, num_wake_pts, num_cells_j, 3)))
        start_j += num_cells_j
    # exit()
    doublet_ind_vel = sum(doublet_ind_vel_list)
    source_ind_vel = sum(source_ind_vel_list)
    # AIC_sigma = sum(AIC_sigma_list)

    batch_size_wake = batch_size
    if batch_size is None:
        batch_size_wake = num_wake_pts

    wake_induced_vel_batch_func = csdl.experimental.batch_function(
        wake_induced_vel_batched,
        # batch_size=batch_size,
        batch_size=batch_size_wake,
        batch_dims=[1]+[None]*2
    )

    panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, np_w, 4, 3)

    wake_ind_vel = wake_induced_vel_batch_func(
        x_w, 
        panel_corners_w,
        mu_w,
        vc=vc
    )

    ind_vel = doublet_ind_vel + source_ind_vel + wake_ind_vel
    # ind_vel = wake_ind_vel # NOTE: source velocity produces nan
    # ind_vel = source_ind_vel # NOTE: source velocity produces nan

    wake_vel_vars = {
        # 'AIC_fw_mu': free_wake_vars['AIC_fw_mu'],
        'AIC_fw_sigma': AIC_sigma,
        # 'AIC_fw_mu_w': free_wake_vars['AIC_fw_mu_w'],
    }

    return ind_vel, wake_vel_vars

def surf_induced_vel_batched(coll_point, panel_center, panel_corners, panel_x_dir, panel_y_dir,
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

    a = coll_point_exp_vec - panel_corners_exp_vec # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_j_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    A = csdl.norm(a+1.e-12, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
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
    AIC_mu_list = []
    for i in range(num_edges-1):
        asdf = compute_vortex_line_ind_vel(panel_corners_exp_vec[:,:,i], panel_corners_exp_vec[:,:,i+1], coll_point_exp_vec[:,:,0], mode='wake', vc=vc)
        AIC_mu_list.append(asdf)
    asdf = compute_vortex_line_ind_vel(panel_corners_exp_vec[:,:,-1], panel_corners_exp_vec[:,:,0], coll_point_exp_vec[:,:,0], mode='wake', vc=vc)
    AIC_mu_list.append(asdf)
    AIC_mu_vec = sum(AIC_mu_list)

    AIC_mu = AIC_mu_vec.reshape((1, num_eval_pts, num_induced_pts, 3))
    # doublet_ind_vel = csdl.matvec(AIC_mu, mu)
    doublet_ind_vel = csdl.einsum(AIC_mu, mu, action='ijkl,ik->ijl')

    return doublet_ind_vel, source_ind_vel, AIC_sigma

def wake_induced_vel_batched(coll_point, panel_corners, mu_w, vc):
    # print(vc)
    # exit()
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

    num_edges = panel_corners.shape[2]
    AIC_mu_wake_list = []
    for i in range(num_edges-1):
        asdf = compute_vortex_line_ind_vel(panel_corners_exp_vec[:,:,i], panel_corners_exp_vec[:,:,i+1], coll_point_exp_vec[:,:,0], mode='wake', vc=vc)
        AIC_mu_wake_list.append(asdf)
    asdf = compute_vortex_line_ind_vel(panel_corners_exp_vec[:,:,-1], panel_corners_exp_vec[:,:,0], coll_point_exp_vec[:,:,0], mode='wake', vc=vc)
    AIC_mu_wake_list.append(asdf)
    AIC_mu_wake_vec = sum(AIC_mu_wake_list)

    AIC_mu_wake = AIC_mu_wake_vec.reshape((1, num_eval_pts, num_induced_pts, 3))
    # wake_doublet_ind_vel = csdl.matvec(AIC_mu_wake, mu_w)
    wake_doublet_ind_vel = csdl.einsum(AIC_mu_wake, mu_w, action='ijkl,ik->ijl')

    return wake_doublet_ind_vel