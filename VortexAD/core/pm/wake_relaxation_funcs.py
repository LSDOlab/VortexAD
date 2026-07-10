import numpy as np
import csdl_alpha as csdl

def update_wake_mesh_params(mesh_dict, wake_mesh_dict):
    '''
    This function just repeats the fixed wake representation operations.
    '''

    TE_node_indices = mesh_dict['TE_node_indices']
    TE_edges = mesh_dict['TE_edges'] # each entry is a tuple with two entries
    # the two entries are the two mesh point indices defining the edge

    num_TE_edges = len(TE_edges)

    wake_connectivity = wake_mesh_dict['wake_connectivity']

    wake_mesh = wake_mesh_dict['wake_mesh']
    num_nodes = wake_mesh.shape[0]
    nc_w = wake_mesh_dict['nc']
    ns = wake_mesh_dict['ns']

    p1_ind = [int(x) for x in list(wake_connectivity[:,:,0].flatten())]
    p2_ind = [int(x) for x in list(wake_connectivity[:,:,1].flatten())]
    p3_ind = [int(x) for x in list(wake_connectivity[:,:,2].flatten())]
    p4_ind = [int(x) for x in list(wake_connectivity[:,:,3].flatten())]

    nn_loop_vals = [np.arange(num_nodes).tolist()]
    loop_vals = [p1_ind, p2_ind, p3_ind, p4_ind]
    with csdl.experimental.enter_loop(vals=nn_loop_vals) as nn_loop_builder:
        n = nn_loop_builder.get_loop_indices()
        with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
            i,j,k,l = loop_builder.get_loop_indices()
            p1 = wake_mesh[n,i,:]
            p2 = wake_mesh[n,j,:]
            p3 = wake_mesh[n,k,:]
            p4 = wake_mesh[n,l,:]
        
        p1 = loop_builder.add_stack(p1)
        p2 = loop_builder.add_stack(p2)
        p3 = loop_builder.add_stack(p3)
        p4 = loop_builder.add_stack(p4)
        loop_builder.finalize()
    
    p1 = nn_loop_builder.add_stack(p1)
    p2 = nn_loop_builder.add_stack(p2)
    p3 = nn_loop_builder.add_stack(p3)
    p4 = nn_loop_builder.add_stack(p4)
    nn_loop_builder.finalize()

    Rc = (p1+p2+p3+p4)/4.
    wake_mesh_dict['panel_center'] = Rc

    # panel_corners = csdl.Variable(value=np.zeros((num_nodes, (nc_w-1)*(ns-1), 4, 3)))
    panel_corners = csdl.Variable(value=np.zeros((num_nodes, (nc_w-1)*num_TE_edges, 4, 3)))
    panel_corners = panel_corners.set(csdl.slice[:,:,0,:], value=p1)
    panel_corners = panel_corners.set(csdl.slice[:,:,1,:], value=p2)
    panel_corners = panel_corners.set(csdl.slice[:,:,2,:], value=p3)
    panel_corners = panel_corners.set(csdl.slice[:,:,3,:], value=p4)
    wake_mesh_dict['panel_corners'] = panel_corners

    D1 = p3-p1
    D2 = p4-p2

    D1D2_cross = csdl.cross(D1, D2, axis=2)
    D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(2,))
    panel_area = D1D2_cross_norm/2.
    wake_mesh_dict['panel_area'] = panel_area

    normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'ij->ija')

    m_dir = (p3+p4)/2. - Rc
    m_norm = csdl.norm(m_dir, axes=(2,))
    m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'ij->ija')
    l_vec = csdl.cross(m_vec, normal_vec, axis=2)

    panel_x_dir = l_vec
    panel_y_dir = m_vec
    panel_normal = normal_vec

    wake_mesh_dict['panel_x_dir'] = panel_x_dir
    wake_mesh_dict['panel_y_dir'] = panel_y_dir
    wake_mesh_dict['panel_normal'] = panel_normal

    # s = csdl.Variable(shape=(panel_corners.shape[0],) + panel_corners.shape[2:], value=0.)
    s = csdl.Variable(shape=panel_corners.shape, value=0.)
    s = s.set(csdl.slice[:,:,:-1,:], value=panel_corners[:,:,1:,:] - panel_corners[:,:,:-1,:])
    s = s.set(csdl.slice[:,:,-1,:], value=panel_corners[:,:,0,:] - panel_corners[:,:,-1,:])

    l_exp = csdl.expand(l_vec, s.shape, 'ijl->ijal')
    m_exp = csdl.expand(m_vec, s.shape, 'ijl->ijal')
    
    S = csdl.norm(s+1.e-6, axes=(3,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
    SL = csdl.sum(s*l_exp+1.e-6, axes=(3,))
    SM = csdl.sum(s*m_exp+1.e-6, axes=(3,))

    # S = csdl.norm(s+1.e-12, axes=(3,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
    # SL = csdl.sum(s*l_exp, axes=(3,))
    # SM = csdl.sum(s*m_exp, axes=(3,))

    wake_mesh_dict['S'] = S
    wake_mesh_dict['SL'] = SL
    wake_mesh_dict['SM'] = SM
    
    return wake_mesh_dict

def compute_wake_relaxation_vel(mesh_dict, wake_mesh_dict, solver_options_dict, mu, sigma, mu_w):
    '''
    This function computes the induced velocity for wake points using a 
    free-wake-like approach
    We have to recompute the AIC matrices here, now evaluated at the wake nodes.
    We can reuse the functions from the linear system step.
    '''
    from VortexAD.core.pm.unsteady.compute_wake_velocity import compute_free_wake_velocity
    free_wake_vel, _ = compute_free_wake_velocity(
        mesh_dict=mesh_dict,
        wake_mesh_dict=wake_mesh_dict,
        batch_size=solver_options_dict['partition_size'],
        mu=mu,
        sigma=sigma,
        mu_w=mu_w,
        vc_body=solver_options_dict['core_radius']
    )
    return free_wake_vel

def compute_wake_relaxation_influence(mesh_dict, wake_mesh_dict, mu_w):
    '''
    This function computes the mat-vec product of the wake induced
    potential and wake doublet strengths for the RHS of the linear system
    '''
    cells = mesh_dict['cell_point_indices'] # keys are cell types, entries are points for each cell
    cell_types = list(cells.keys())
    cell_adjacency_types = mesh_dict['cell_adjacency'] # keys are cell types, entries are adjacent cell indices
    num_cells_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]
    num_tot_panels = sum(num_cells_per_type)

    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']
    num_wake_panels = wake_mesh_dict['num_panels']
    num_nodes=1
    AIC_mu_wake = csdl.Variable(shape=(num_nodes, num_tot_panels, num_wake_panels), value=0.)

    batch_size = 1
    
    AIC_batch_func = csdl.experimental.batch_function(
        compute_matvec_batched,
        batch_size=batch_size,
        batch_dims=[1]*2+[None]*9
        # batch_dims=[None]+[1]*8
    )

    coll_point_eval = mesh_dict['panel_center_mod']
    normal_vec_eval = mesh_dict['panel_normal']
    num_surf_panels = coll_point_eval.shape[1]

    # wake influence outside of the inner loop
    # compute the AIC wake matrix here (reduced shape of (num_panels,num_wake_panels))

    panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, np_w, 4, 3)
    coll_point_w = wake_mesh_dict['panel_center'] # (nn, np_w, 3)
    panel_x_dir_w = wake_mesh_dict['panel_x_dir'] # (nn, np_w, 3)
    panel_y_dir_w = wake_mesh_dict['panel_y_dir'] # (nn, np_w, 3)
    panel_normal_w = wake_mesh_dict['panel_normal'] # (nn, np_w, 3)
    S_w = wake_mesh_dict['S']
    SL_w = wake_mesh_dict['SL']
    SM_w = wake_mesh_dict['SM']

    RHS_wake = AIC_batch_func(
        coll_point_eval,
        normal_vec_eval,
        coll_point_w,
        panel_corners_w,
        panel_x_dir_w,
        panel_y_dir_w,
        panel_normal_w,
        S_w,
        SL_w,
        SM_w,
        mu_w
    )
    print(RHS_wake.shape)
    RHS_wake = RHS_wake.reshape((1,num_surf_panels))
    # AIC_mu_wake = AIC_mu_wake.set(csdl.slice[:,start_i:stop_i,:], wake_doublet_influence)

    # wake_conn = wake_mesh_dict['wake_connectivity']
    # nws, nwTE = wake_conn.shape[1], wake_conn.shape[0]
    # AIC_mu_wake_reshape = AIC_mu_wake.reshape(
        # (num_nodes, num_tot_panels, nws, nwTE)
    # )
    return RHS_wake

def compute_matvec_batched(coll_point_eval, normal_vec_eval, coll_point_w, panel_corners_w, 
                           panel_x_dir_w, panel_y_dir_w, panel_normal_w, S_w, SL_w, SM_w, mu_w):
    from VortexAD.core.pm.source_doublet.mu_sigma_solver import compute_aic_batched

    wake_doublet_influence = compute_aic_batched(
        coll_point_eval,
        normal_vec_eval,
        coll_point_w,
        panel_corners_w,
        panel_x_dir_w,
        panel_y_dir_w,
        panel_normal_w,
        S_w,
        SL_w,
        SM_w,
        mode='wake',
        BC='Dirichlet'
    )
    print(wake_doublet_influence.shape)
    print(mu_w.shape)
    # RHS_wake = csdl.einsum(wake_doublet_influence, mu_w, action='ijk,ik->ij')
    RHS_wake = csdl.sum(wake_doublet_influence*mu_w)

    return RHS_wake

