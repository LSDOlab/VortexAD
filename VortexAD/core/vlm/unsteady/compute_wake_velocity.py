import numpy as np
import csdl_alpha as csdl

from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel
from VortexAD.core.vlm.unsteady.AIC_computation import compute_AIC

def compute_wake_velocity(mesh_dict, vectorized_mesh_dict, batch_size, x_w, gamma, gamma_w, free_wake=False, vc=1.e-6):

    bound_vortex_mesh = vectorized_mesh_dict['bound_vortex_mesh']
    nodal_velocity =  vectorized_mesh_dict['nodal_velocity']
    TE_node_indices = vectorized_mesh_dict['TE_node_indices']
    # x_w = vectorized_mesh_dict['wake_mesh']
    TE_velocity = nodal_velocity[:,TE_node_indices,:]
    num_TE_pts = len(TE_node_indices)
    num_wake_pts = x_w.shape[0]

    wake_vel = TE_velocity.expand((1, num_TE_pts, int(num_wake_pts/num_TE_pts), 3), 'ijk->ijak')
    wake_vel = wake_vel.reshape((num_wake_pts, 3))

    if free_wake:
        ind_vel = compute_free_wake_velocity(mesh_dict, vectorized_mesh_dict, batch_size, x_w, gamma, gamma_w, vc)
        wake_vel = wake_vel + ind_vel
    return wake_vel

def compute_free_wake_velocity(mesh_dict, vectorized_mesh_dict, batch_size, x_w, gamma, gamma_w, vc_body):

    batch_size_surf = batch_size
    if batch_size is None:
        batch_size_surf = num_wake_pts

    surf_induced_vel_batch_func = csdl.experimental.batch_function(
        induced_vel_batched,
        # batch_size=batch_size,
        batch_size=batch_size_surf,
        batch_dims=[1]+[None]*2
    )

    coll_point = vectorized_mesh_dict['panel_centers']
    eval_pt = x_w.reshape((1,) + x_w.shape)
    num_wake_pts = eval_pt.shape[1]
    panel_corners = vectorized_mesh_dict['panel_corners']
    num_body_panels = panel_corners.shape[1]

    body_ind_vel = surf_induced_vel_batch_func(
        eval_pt,
        panel_corners,
        gamma,
        vc=vc_body # constant core model on the body
    )

    batch_size_wake = batch_size
    if batch_size is None:
        batch_size_wake = num_wake_pts

    wake_induced_vel_batch_func = csdl.experimental.batch_function(
        induced_vel_batched,
        # batch_size=batch_size,
        batch_size=batch_size_wake,
        batch_dims=[1]+[None]*3
    )

    wake_panel_corners = vectorized_mesh_dict['wake_corners']
    vc_wake = vectorized_mesh_dict['wake_core_radius']

    wake_ind_vel = wake_induced_vel_batch_func(
        eval_pt,
        wake_panel_corners,
        gamma_w,
        vc_wake # csdl variable with finite core model
    )
    
    ind_vel = body_ind_vel + wake_ind_vel
    ind_vel = ind_vel.reshape((num_wake_pts, 3)) # removing stacking dimension + num_nodes initially

    return ind_vel

def induced_vel_batched(coll_point, panel_corners, gamma, vc):
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
    AIC_vel_vec = sum(AIC_vel_vec_list)[0,:]

    ind_vel = csdl.Variable(value=np.zeros((num_eval_pts, 3)))
    for i in range(3):
        ind_vel = ind_vel.set(
            csdl.slice[:,i],
            csdl.sum(AIC_vel_vec[:,i]*gamma)
        )

    return ind_vel