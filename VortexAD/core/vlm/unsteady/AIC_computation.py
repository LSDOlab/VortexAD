import numpy as np
import csdl_alpha as csdl

from VortexAD.core.vlm.velocity_computations import compute_induced_velocity

from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel

def compute_AIC(mesh_dict, total_mesh_dict, solver_options_dict, eval_pt='collocation'):
    ROM = solver_options_dict['ROM']
    batch_size = solver_options_dict['partition_size']
    if ROM: # column batching of base AIC for matvec product
        batch_dims = [None]*2+[1]*8
    elif not ROM: # traditional row batching
        batch_dims = [1]*2+[None]*1

    # building surface AIC matrices
    # points of influence:
    if eval_pt == 'collocation':
        body_eval_pts = total_mesh_dict['panel_centers']
    elif eval_pt == 'force_eval':
        body_eval_pts = total_mesh_dict['force_eval_points']
    # body_panel_centers = total_mesh_dict['panel_centers']
    body_panel_normal = total_mesh_dict['panel_normal']

    # panels doing the influence
    body_panel_corners = total_mesh_dict['panel_corners']

    num_body_panels = body_panel_corners.shape[1]
    batch_size_surf = batch_size
    if batch_size is None:
        batch_size_surf = num_body_panels

    AIC_batch_func = csdl.experimental.batch_function(
        compute_aic_batched,
        batch_size=batch_size_surf,
        batch_dims=batch_dims
    )

    AIC_body_vec = AIC_batch_func(
        body_eval_pts,
        body_panel_normal,
        body_panel_corners,
    )
    # print(AIC_body_vec.shape)
    # exit()

    AIC_body = AIC_body_vec.reshape((num_body_panels, num_body_panels))

    # wake computation
    wake_panel_corners = total_mesh_dict['wake_corners']
    num_wake_panels = wake_panel_corners.shape[1]
    batch_size_wake = batch_size
    if batch_size is None:
        batch_size_wake = num_body_panels

    AIC_wake_batch_func = csdl.experimental.batch_function(
        compute_aic_batched,
        batch_size=batch_size_wake,
        batch_dims=batch_dims
    )

    AIC_wake_vec = AIC_wake_batch_func(
        body_eval_pts,
        body_panel_normal,
        wake_panel_corners,
        vc=1.e-6
    )
    # print(AIC_wake_vec.shape)
    # exit()

    AIC_wake = AIC_wake_vec.reshape((num_body_panels, num_wake_panels))
    # AIC_wake = AIC_wake_vec[:,0,:]
    # AIC_wake = AIC_wake_vec.reshape((num_wake_panels, num_tot_panels)).T()

    AIC_mat_list = [AIC_body, AIC_wake]
    return AIC_mat_list

def compute_aic_batched(coll_point, normal_vec_eval, panel_corners, vc=None):
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

    normal_vec_eval_exp = csdl.expand(normal_vec_eval, expanded_shape, 'ijk->ijabk')
    normal_vec_eval_exp_vec = normal_vec_eval_exp.reshape(vectorized_shape)

    # ============ expanding across rows ============
    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'ijkl->iajkl')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    num_edges = num_corners

    AIC_vel_vec_list = []
    for  i in range(num_edges-1):
        asdf = compute_vortex_line_ind_vel(
            panel_corners_exp_vec[:,:,i], 
            panel_corners_exp_vec[:,:,i+1], 
            coll_point_exp_vec[:,:,0], 
            mode='wake', 
            vc=vc
        )
        AIC_vel_vec_list.append(asdf)
    asdf = compute_vortex_line_ind_vel(
        panel_corners_exp_vec[:,:,-1], 
        panel_corners_exp_vec[:,:,0], 
        coll_point_exp_vec[:,:,0], 
        mode='wake', 
        vc=vc
    )
    AIC_vel_vec_list.append(asdf)
    AIC_vel_vec = sum(AIC_vel_vec_list)

    expanded_shape_proj = (num_nodes, num_eval_pts, num_induced_pts, 3)
    vectorized_shape_proj = (num_nodes, num_interactions, 3)

    normal_vec_eval_exp = csdl.expand(normal_vec_eval, expanded_shape_proj, 'ijk->ijak')
    normal_vec_eval_exp_vec = normal_vec_eval_exp.reshape(vectorized_shape_proj)

    AIC_vec = csdl.sum(normal_vec_eval_exp_vec*AIC_vel_vec, axes=(2,))
    
    return AIC_vec