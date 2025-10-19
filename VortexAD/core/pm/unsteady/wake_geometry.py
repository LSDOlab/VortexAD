import numpy as np
import csdl_alpha as csdl

def wake_geometry(num_nodes, mesh_dict, wake_points, wake_connectivity):
    num_wake_pts = wake_points.shape[1]
    nt = wake_connectivity.shape[0]+1
    nc_w = nt
    num_TE_edges = wake_connectivity.shape[1]
    num_wake_panels = (nc_w-1)*num_TE_edges
    wake_mesh_dict = {}

    wake_mesh = csdl.Variable(value=np.zeros((1, num_wake_pts, 3)))
    wake_mesh = wake_points
    wake_mesh_dict['wake_mesh'] = wake_mesh

    wake_mesh_dict['num_panels'] = num_wake_panels

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

    panel_corners = csdl.Variable(value=np.zeros((num_nodes, num_wake_panels, 4, 3)))
    panel_corners = panel_corners.set(csdl.slice[:,:,0,:], value=p1)
    panel_corners = panel_corners.set(csdl.slice[:,:,1,:], value=p2)
    panel_corners = panel_corners.set(csdl.slice[:,:,2,:], value=p3)
    panel_corners = panel_corners.set(csdl.slice[:,:,3,:], value=p4)
    wake_mesh_dict['panel_corners'] = panel_corners
    
    D1 = p3-p1
    D2 = p4-p2

    D1D2_cross = csdl.cross(D1, D2, axis=2)
    D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(2,)) + 1.e-12
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