import numpy as np
import csdl_alpha as csdl

def trefftz_plane_drag(mesh_dict, wake_mesh_dict, mu, sigma, mu_w, rho, constant_geometry):
    num_nodes = mu.shape[0]
    num_geom_nodes = num_nodes
    if constant_geometry:
        num_geom_nodes = 1
    TE_node_indices = mesh_dict['TE_node_indices']
    TE_edges = mesh_dict['TE_edges']
    ns = len(TE_node_indices)
    wake_mesh = wake_mesh_dict['wake_mesh']
    TE = wake_mesh[:,:ns,:]
    wake_end = wake_mesh[:,-ns:,:]
    wake_connectivity = wake_mesh_dict['wake_connectivity']

    pc = wake_mesh_dict['panel_corners']

    panel_widths = (pc[:,:,2,1] + pc[:,:,3,1] - pc[:,:,0,1] - pc[:,:,1,1])/4 
    panel_widths = (panel_widths**2)**0.5

    TE_panel_corners = pc[:,:ns] # trailing-edge-adjacent panel corners (includes the TE)
    WE_panel_corners = pc[:,-ns:] # wake end panel corners

    upstream_edge_center = (TE_panel_corners[:,:,0] + TE_panel_corners[:,:,3])/2
    downstream_edge_center = (WE_panel_corners[:,:,0] + WE_panel_corners[:,:,3])/2

    TPFW = 0.75 # trefftz plane location as a fraction of the wake
    # recommended to be 0.5. at 1, it is evaluated at the wake end
    # eval_pts = wake_end*TPFW + TE*(1-TPFW) # should have num_nodes embedded into it
    eval_pts = downstream_edge_center*TPFW + upstream_edge_center*(1-TPFW)

    # computing induced velocities
    if constant_geometry:
        nn_ind_array = np.arange(num_nodes).tolist()
        with csdl.experimental.enter_loop(vals=[nn_ind_array]) as loop_builder:
            n = loop_builder.get_loop_indices()
    
            w_loop = compute_vertical_induced_velocity(
                eval_pts,
                mesh_dict,
                wake_mesh_dict,
                mu[n,:].reshape((1,) + mu.shape[1:]),
                sigma[n,:].reshape((1,) + sigma.shape[1:]),
                mu_w[n,:].reshape((1,) + mu_w.shape[1:])
            )
            w_ind = w_loop[0,:]
        w = loop_builder.add_stack(w_ind)
        loop_builder.finalize()
    else:
        w = compute_vertical_induced_velocity(
            eval_pts,
            mesh_dict,
            wake_mesh_dict,
            mu,
            sigma,
            mu_w
        )

    # computing potential jump across span
    num_TE_edges = len(TE_edges)
    mu_w_shape = mu_w.shape
    num_wake_elements = mu_w_shape[1]
    if num_wake_elements > num_TE_edges: # means there is more than 1 wake row
        dPhi_span = mu_w[:,:num_TE_edges] # all the same, take the first row
    else:
        dPhi_span = mu_w

    # adjusting for more than 1 wake rows
    wake_rows = wake_connectivity.shape[0]
    if wake_rows > 1: # NOTE: CHECK TO MAKE SURE THE SHAPES HERE ARE CORRECT RELATIVE TO WAKE CONNECTIVITY
        panel_widths = panel_widths.reshape((num_nodes, wake_rows, num_TE_edges))
        wake_panel_width = csdl.average(panel_widths, axes=(1,))
        dPhi_span = mu_w[:,:num_TE_edges] # all the same, take the first row
    else:
        wake_panel_width = panel_widths
        dPhi_span = mu_w

    if constant_geometry:
        wake_panel_width = wake_panel_width[0,:].expand((num_nodes, num_TE_edges), 'i->ji')

    # print(dPhi_span.shape)
    # print(wake_panel_width.shape)
    # trefftz plane integral
    TPI_integrand = dPhi_span*w*wake_panel_width
    # print(TPI_integrand.shape)
    TPI = csdl.sum(TPI_integrand, axes=(1,))
    # print(TPI.shape)
    # trefftz plane induced drag
    D_Trefftz = rho/2*TPI # no negative sign here because of panel normal orientation
    return D_Trefftz

def compute_vertical_induced_velocity(eval_pts, mesh_dict, wake_mesh_dict, mu, sigma, mu_w):
    from VortexAD.core.pm.unsteady.compute_wake_velocity import surf_induced_vel_batched, wake_induced_vel_batched

    batch_size = 1
    num_nodes = mu.shape[0]
    num_eval_pts = eval_pts.shape[1]
    # ind_vel = csdl.Variable(value=np.zeros(num_nodes, num_eval_pts, 3))
    # use a similar loop structure to the AIC, where the outer is the x_w
    # inner will require a loop around the cell types

    cells = mesh_dict['cell_point_indices'] # keys are cell types, entries are points for each cell
    cell_types = list(cells.keys())
    cell_adjacency_types = mesh_dict['cell_adjacency'] # keys are cell types, entries are adjacent cell indices
    num_cells_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]
    num_tot_panels = sum(num_cells_per_type)

    batch_size_surf = batch_size
    if batch_size is None:
        batch_size_surf = num_eval_pts

    surf_induced_vel_batch_func = csdl.experimental.batch_function(
        surf_induced_vel_batched,
        # batch_size=batch_size,
        batch_size=batch_size_surf,
        batch_dims=[1]+[None]*10
    )
    vc_body = 1.e-6
    start_j, stop_j = 0, 0
    doublet_ind_vel_list = []
    source_ind_vel_list = []
    AIC_sigma_list = []
    for j, cell_type_j in enumerate(cell_types):
        num_cells_j = num_cells_per_type[j]
        stop_j += num_cells_j

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
            eval_pts, 
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
        doublet_ind_vel_list.append(doublet_ind_vel)
        source_ind_vel_list.append(source_ind_vel)
        AIC_sigma_list.append(AIC_sigma.reshape((1, num_eval_pts, num_cells_j, 3)))
        start_j += num_cells_j
    # exit()
    doublet_ind_vel = sum(doublet_ind_vel_list)
    source_ind_vel = sum(source_ind_vel_list)
    # AIC_sigma = sum(AIC_sigma_list)

    batch_size_wake = batch_size
    if batch_size is None:
        batch_size_wake = num_eval_pts

    wake_induced_vel_batch_func = csdl.experimental.batch_function(
        wake_induced_vel_batched,
        # batch_size=batch_size,
        batch_size=batch_size_wake,
        batch_dims=[1]+[None]*3
    )

    panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, np_w, 4, 3)
    vc_wake = wake_mesh_dict['wake_core_radius']

    wake_ind_vel = wake_induced_vel_batch_func(
        eval_pts, 
        panel_corners_w,
        mu_w,
        1.e-6
    )

    ind_vel = doublet_ind_vel + source_ind_vel + wake_ind_vel
    # ind_vel = wake_ind_vel # NOTE: source velocity needs checking
    # ind_vel = source_ind_vel+doublet_ind_vel # NOTE: source velocity needs checking

    wake_vel_vars = {
        # 'AIC_fw_mu': free_wake_vars['AIC_fw_mu'],
        'AIC_fw_sigma': AIC_sigma,
        # 'AIC_fw_mu_w': free_wake_vars['AIC_fw_mu_w'],
    }

    w_ind_vel = ind_vel.reshape(num_nodes, num_eval_pts, 3)[:,:,2]

    return w_ind_vel