import numpy as np
import csdl_alpha as csdl

def compute_vlm_pm_params_vectorized(mesh_dict, vectorized_mesh_dict):
    surface_names = list(mesh_dict.keys())

    # getting surface shapes
    surf_names = list(mesh_dict.keys())
    surf_points = []
    surf_panels = []
    surf_nc = []
    surf_ns = []
    num_tot_points = 0
    num_tot_panels = 0

    for name in surf_names:
        mesh = mesh_dict[name]['mesh']

        num_nodes = mesh.shape[0]
        nc, ns = mesh.shape[1], mesh.shape[2]

        num_points = nc*ns
        num_panels = (nc-1)*(ns-1)

        surf_nc.append(nc)
        surf_ns.append(ns)

        surf_points.append(num_points)
        surf_panels.append(num_panels)

        num_tot_points += num_points
        num_tot_panels += num_panels

    

    # terms that are used for panel method interactions

    # others (either already computed or needed by VLM)
    # bvm_all = csdl.Variable(value=np.zeros(base_shape + (3,))) # bound_vortex_mesh
    # nodal_velocity = csdl.Variable(value=np.zeros(base_shape + (3,)))
    # panel_centers = csdl.Variable(value=np.zeros(base_shape) + (3,))
    # panel_corners = csdl.Variable(value=np.zeros(base_shape + (4, 3)))
    # panel_normal = csdl.Variable(value=np.zeros(base_shape) + (3,))
    # force_eval_pts = csdl.Variable(value=np.zeros(base_shape + (3,)))
    # bound_vec_velocity = csdl.Variable(value=np.zeros(base_shape + (3,)))
    # bound_vec = csdl.Variable(value=np.zeros(base_shape + (3,)))
    # coll_vel = csdl.Variable(value=np.zeros(base_shape + (3,)))
    # panel_areas = csdl.Variable(value=np.zeros(base_shape + (3,)))

    cs_panels, ce_panels = 0, 0
    cs_points, ce_points = 0, 0

    TE_node_indices = []
    TE_offset = 0

    '''
    Loop here does two things:
    - computes remaining panel method parameters
    - vectorizes remaining parameters

    Things we do NOT need to recompute:
    - bound vortex mesh
    - panel centers
    - panel corners
    - panel vectors (x_dir, y_dir, normal)
    - nodal velocity
    - collocation velocity
    - force eval points
    - bound vector + velocity at bound vector
    - panel areas

    PM parameters we need to VECTORIZE:
    - panel x-dir
    - panel y-dir
    
    PM parameters we NEED to compute:
    - S
    - SL
    - SM
    '''

    base_shape = (num_nodes, num_tot_panels)

    # parameters computed in the VLM preprocessor that need to be fully vectorized
    panel_x_dir = csdl.Variable(value=np.zeros((base_shape) + (3,)))
    panel_y_dir = csdl.Variable(value=np.zeros((base_shape) + (3,)))

    # New panel method parameters
    S_total = csdl.Variable(value=np.zeros((base_shape) + (4,)))
    SL_total = csdl.Variable(value=np.zeros((base_shape) + (4,)))
    SM_total = csdl.Variable(value=np.zeros((base_shape) + (4,)))
    
    for i, surf_name in enumerate(surface_names):
        surf_name = surf_names[i]
        nc = surf_nc[i]
        ns = surf_ns[i]
        num_points = surf_points[i]
        num_panels = surf_panels[i]

        ce_panels += num_panels
        ce_points += num_points

        # getting TE indices for vectorized grid points
        asdf = list(np.arange(ns))
        surf_TE_node_indices = [(val+1)*nc - 1 + TE_offset for val in asdf]
        TE_node_indices.extend(surf_TE_node_indices)
        TE_offset += num_points


        # COMPUTING S, SL, SM
        # from VLM preprocessor
        panel_corners = mesh_dict[surf_name]['bound_vortex_panel_corners']
        normal_vec = mesh_dict[surf_name]['bd_normal_vec']
        Rc = mesh_dict[surf_name]['collocation_points']
        S3 = (panel_corners[:,:,:,2,:]+panel_corners[:,:,:,3,:])/2

        m_dir = S3 - Rc
        m_norm = csdl.norm(m_dir, axes=(3,))
        m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'jkl->jkla')
        l_vec = csdl.cross(m_vec, normal_vec, axis=3)

        s = csdl.Variable(shape=panel_corners.shape, value=0.)
        s = s.set(csdl.slice[:,:,:,:-1,:], value=panel_corners[:,:,:,1:,:] - panel_corners[:,:,:,:-1,:])
        s = s.set(csdl.slice[:,:,:,-1,:], value=panel_corners[:,:,:,0,:] - panel_corners[:,:,:,-1,:])

        l_exp = csdl.expand(l_vec, panel_corners.shape, 'jklm->jklam')
        m_exp = csdl.expand(m_vec, panel_corners.shape, 'jklm->jklam')
        
        S = csdl.norm(s, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0 --> added to the equations instead
        # S = csdl.norm(s, axes=(5,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
        SL = csdl.sum(s*l_exp, axes=(4,))
        SM = csdl.sum(s*m_exp, axes=(4,))

        mesh_dict[surf_name]['panel_x_dir'] = l_vec
        mesh_dict[surf_name]['panel_y_dir'] = m_vec
        
        mesh_dict[surf_name]['S'] = S
        mesh_dict[surf_name]['SL'] = SL
        mesh_dict[surf_name]['SM'] = SM

        # additional vectorizations

        panel_x_dir = panel_x_dir.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[surf_name]['panel_x_dir'].reshape((num_nodes, num_panels, 3))
        )

        panel_y_dir = panel_y_dir.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[surf_name]['panel_y_dir'].reshape((num_nodes, num_panels, 3))
        )

        S_total = S_total.set(
            csdl.slice[:,cs_panels:ce_panels],
            S.reshape((num_nodes, num_panels, 4))
        )

        SL_total = SL_total.set(
            csdl.slice[:,cs_panels:ce_panels],
            SL.reshape((num_nodes, num_panels, 4))
        )

        SM_total = SM_total.set(
            csdl.slice[:,cs_panels:ce_panels],
            SM.reshape((num_nodes, num_panels, 4))
        )

        cs_panels += num_panels
        cs_points += num_points

    vectorized_mesh_dict['panel_x_dir'] = panel_x_dir
    vectorized_mesh_dict['panel_y_dir'] = panel_y_dir
    vectorized_mesh_dict['S'] = S_total
    vectorized_mesh_dict['SL'] = SL_total
    vectorized_mesh_dict['SM'] = SM_total

    return mesh_dict, vectorized_mesh_dict