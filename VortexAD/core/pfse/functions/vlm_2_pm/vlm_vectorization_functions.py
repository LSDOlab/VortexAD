import numpy as np
import csdl_alpha as csdl

def vectorize_vlm_variables(mesh_dict):
    '''
    Variables that need to be vectorized for vlm:
    panel_corners
    panel_center
    panel_x_dir
    panel_y_dir
    panel_normal
    S
    SL
    SM

    '''
    surf_names = list(mesh_dict.keys)
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
    
    base_shape = (num_nodes, num_tot_panels)

    # terms that are used for panel method interactions
    panel_corners = csdl.Variable(value=np.zeros(base_shape + (4, 3)))
    panel_centers = csdl.Variable(value=np.zeros(base_shape) + (3,))
    panel_x_dir = csdl.Variable(value=np.zeros(base_shape) + (3,))
    panel_y_dir = csdl.Variable(value=np.zeros(base_shape) + (3,))
    panel_normal = csdl.Variable(value=np.zeros(base_shape) + (3,))
    S = csdl.Variable(value=np.zeros(base_shape) + (4,))
    SL = csdl.Variable(value=np.zeros(base_shape) + (4,))
    SM = csdl.Variable(value=np.zeros(base_shape) + (4,))

    # others (either already computed or needed by VLM)
    bvm_all = csdl.Variable(value=np.zeros(base_shape + (3,))) # bound_vortex_mesh
    nodal_velocity = csdl.Variable(value=np.zeros(base_shape + (3,)))
    force_eval_pts = csdl.Variable(value=np.zeros(base_shape + (3,)))
    coll_vel = csdl.Variable(value=np.zeros(base_shape + (3,)))
    bound_vec_velocity = csdl.Variable(value=np.zeros(base_shape + (3,)))
    bound_vec = csdl.Variable(value=np.zeros(base_shape + (3,)))
    panel_areas = csdl.Variable(value=np.zeros(base_shape + (3,)))

    cs_panels, ce_panels = 0, 0
    cs_points, ce_points = 0, 0

    TE_node_indices = []
    TE_offset = 0

    for i, name in enumerate(surf_names):
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

        # needed for panel method interactions
        panel_corners = panel_corners.set(
            csdl.slice[:,cs_panels:ce_panels],
            mesh_dict[surf_name]['panel_corners'].reshape((num_nodes, num_panels, 4, 3))
        )

        panel_centers = panel_centers.set(
            csdl.slice[:,cs_panels:ce_panels],
            mesh_dict[surf_name]['panel_centers'].reshape((num_nodes, num_panels, 3))
        )

        panel_x_dir = panel_x_dir.set(
            csdl.slice[:,cs_panels:ce_panels],
            mesh_dict[surf_name]['panel_x_dir'].reshape((num_nodes, num_panels, 3))
        )

        panel_y_dir = panel_y_dir.set(
            csdl.slice[:,cs_panels:ce_panels],
            mesh_dict[surf_name]['panel_y_dir'].reshape((num_nodes, num_panels, 3))
        )

        panel_normal = panel_normal.set(
            csdl.slice[:,cs_panels:ce_panels],
            mesh_dict[surf_name]['panel_normal'].reshape((num_nodes, num_panels, 3))
        )

        S = S.set(
            csdl.slice[:,cs_panels:ce_panels],
            mesh_dict[surf_name]['S'].reshape((num_nodes, num_panels, 4))
        )

        SL = SL.set(
            csdl.slice[:,cs_panels:ce_panels],
            mesh_dict[surf_name]['SL'].reshape((num_nodes, num_panels, 4))
        )

        SM = SM.set(
            csdl.slice[:,cs_panels:ce_panels],
            mesh_dict[surf_name]['SM'].reshape((num_nodes, num_panels, 4))
        )

        # computing other VLM variables
        mesh = mesh_dict[surf_name]['mesh']

        bound_vortex_mesh = csdl.Variable(shape=mesh.shape, value=0.)
        bound_vortex_mesh = bound_vortex_mesh.set(csdl.slice[:,:-1,:,:], value=(3*mesh[:,:-1,:,:] + mesh[:,1:,:,:])/4)
        bound_vortex_mesh = bound_vortex_mesh.set(csdl.slice[:,-1,:,:], value=mesh[:,-1,:,:] + (mesh[:,-1,:,:] - mesh[:,-2,:,:])/4)
        mesh_dict[surf_name]['bound_vortex_mesh'] = bound_vortex_mesh










        bvm_all = bvm_all.set(
            csdl.slice[:,cs_points:ce_points,:],
            mesh_dict[surf_name]['bound_vortex_mesh'].reshape(num_nodes, num_points, 3)
        )

        nodal_velocity = nodal_velocity.set(
            csdl.slice[:, cs_points:ce_points,:],
            mesh_dict[surf_name]['nodal_velocity'].reshape(num_nodes, num_points, 3)
        )

        force_eval_pts = force_eval_pts.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[surf_name]['force_eval_points'].reshape(num_nodes, num_panels, 3)
        )

        coll_vel = coll_vel.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[surf_name]['collocation_velocity'].reshape(num_nodes, num_panels, 3)
        )

        bound_vec_velocity = bound_vec_velocity.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[surf_name]['bound_vector_velocity'].reshape(num_nodes, num_panels, 3)
        )

        bound_vec = bound_vec.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[surf_name]['bound_vec'].reshape(num_nodes, num_panels, 3)
        )

        panel_areas = panel_areas.set(
            csdl.slice[:,cs_panels:ce_panels],
            mesh_dict[surf_name]['panel_area'].reshape((num_nodes, num_panels))
        )



    vlm_vectorized_dict = {

        'TE_node_indices': TE_node_indices,
    }

    return vlm_mesh_dict, vlm_vectorized_dict