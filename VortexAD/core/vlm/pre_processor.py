import numpy as np 
import csdl_alpha as csdl

def pre_processor(mesh_dict):
    mesh_names = mesh_dict.keys()

    surf_points = []
    surf_panels = []
    surf_nc = []
    surf_ns = []
    total_points = 0
    total_panels = 0
    for key in mesh_names:
        # mesh = csdl.Variable(name=key+'_mesh', value=mesh_dict[key]['mesh'])
        mesh = mesh_dict[key]['mesh']
        # NOTE: mesh has shape (num_nodes, nc, ns, 3)
        num_nodes = mesh.shape[0]
        mesh_dict[key]['nc'] = nc = mesh.shape[1]
        mesh_dict[key]['ns'] = ns = mesh.shape[2]

        num_points = nc*ns
        num_panels = (nc-1)*(ns-1)

        mesh_dict[key]['num_panels'] = num_panels

        surf_nc.append(nc)
        surf_ns.append(ns)

        surf_points.append(num_points)
        surf_panels.append(num_panels)
        total_points += num_points
        total_panels += num_panels

        # bound vortex grid computation
        print(mesh.shape)
        bound_vortex_mesh = csdl.Variable(shape=mesh.shape, value=0.)
        bound_vortex_mesh = bound_vortex_mesh.set(csdl.slice[:,:-1,:,:], value=(3*mesh[:,:-1,:,:] + mesh[:,1:,:,:])/4)
        bound_vortex_mesh = bound_vortex_mesh.set(csdl.slice[:,-1,:,:], value=mesh[:,-1,:,:] + (mesh[:,-1,:,:] - mesh[:,-2,:,:])/4)

        mesh_dict[key]['bound_vortex_mesh'] = bound_vortex_mesh

        # collocation point computation (center of the vortex rings defined by vortex mesh)
        p1_bd = bound_vortex_mesh[:,:-1, :-1, :]
        p2_bd = bound_vortex_mesh[:,:-1, 1:, :]
        p3_bd = bound_vortex_mesh[:,1:, 1:, :]
        p4_bd = bound_vortex_mesh[:,1:, :-1, :]

        # p1_bd = bound_vortex_mesh[:,:-1, :-1, :]
        # p2_bd = bound_vortex_mesh[:,1:, :-1, :]
        # p3_bd = bound_vortex_mesh[:,1:, 1:, :]
        # p4_bd = bound_vortex_mesh[:,:-1, 1:, :]

        panel_corners = csdl.Variable(shape=(num_nodes, nc-1, ns-1, 4, 3), value=0.)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,0,:], p1_bd)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,1,:], p2_bd)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,2,:], p3_bd)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,3,:], p4_bd)

        mesh_dict[key]['bound_vortex_panel_corners'] = panel_corners


        collocation_points = (p1_bd + p2_bd + p3_bd + p4_bd)/4.
        mesh_dict[key]['collocation_points'] = collocation_points

        force_eval_pts = (p1_bd + p2_bd)/2.
        mesh_dict[key]['force_eval_points'] = force_eval_pts

        # panel area and normal vector computation (NOTE: CHECK IF WE NEED TO USE THE MESH OR BOUND VORTEX GRID)
        p1 = mesh[:, :-1, :-1, :]
        p2 = mesh[:, :-1, 1:, :]
        p3 = mesh[:, 1:, 1:, :]
        p4 = mesh[:, 1:, :-1, :]

        # panel diagonal vectors
        A = p3_bd - p1_bd
        B = p2_bd - p4_bd
        # B = p4_bd - p2_bd
        normal_dir = csdl.cross(A, B, axis=3)
        panel_area = csdl.norm(normal_dir, axes=(3,)) / 2.

        # vector normalization
        normal_vec = normal_dir/(csdl.expand(panel_area*2, out_shape=normal_dir.shape, action='ijk->ijka') + 1.e-12)

        mesh_dict[key]['panel_area'] = panel_area
        mesh_dict[key]['bd_normal_vec'] = normal_vec

        wetted_area = csdl.sum(panel_area, axes=(1,2))
        mesh_dict[key]['wetted_area'] = wetted_area

        bound_vec = p2_bd - p1_bd
        mesh_dict[key]['bound_vec'] = bound_vec # NO NEED TO NORMALIZE BECAUSE WE NEED THE MAGNITUDE

        # VELOCITY COMPUTATIONS FOR COLLOCATION POINT AND BOUND VECTORS
        nodal_velocity = mesh_dict[key]['nodal_velocity'] # at the nodes of the mesh

        v1 = nodal_velocity[:, :-1, :-1, :]
        v2 = nodal_velocity[:, :-1, 1:, :]
        v3 = nodal_velocity[:, 1:, 1:, :]
        v4 = nodal_velocity[:, 1:, :-1, :]

        # v1 = nodal_velocity[:, :-1, :-1, :]
        # v2 = nodal_velocity[:, 1:, :-1, :]
        # v3 = nodal_velocity[:, 1:, 1:, :]
        # v4 = nodal_velocity[:, :-1, 1:, :]

        coll_point_velocity = 0.75*(v1+v2)/2. + 0.25*(v3+v4)/2.
        coll_vel_flag = mesh_dict[key]['coll_vel_flag']
        if coll_vel_flag:
            coll_point_velocity += mesh_dict[key]['coll_vel']

        mesh_dict[key]['collocation_velocity'] = coll_point_velocity
        mesh_dict[key]['bound_vector_velocity'] = 0.25*(v1+v2)/2. + 0.75*(v3+v4)/2.

        # computing MAC of surface
        num_half_span = int((ns+1)/2)
        chord_dist = mesh[:,-1,:num_half_span,0] - mesh[:,0,:num_half_span,0]
        avg_chord_dist = chord_dist[:,1:] - chord_dist[:,:-1]
        el_width = mesh[:,0,1:num_half_span,1] - mesh[:,0,:num_half_span-1,1]

        MAC = 2/wetted_area*csdl.sum(
            avg_chord_dist**2*el_width,
            axes=(1,)
        )
        
        mesh_dict[key]['MAC'] = MAC

    #  assembling vectorized variables
    num_surfaces = len(surf_panels)
    surf_names = list(mesh_dict.keys())
    num_tot_points = sum(surf_points)
    num_tot_panels = sum(surf_panels)

    # initializing vectorized  csdl variables
    bvm_all = csdl.Variable(value=np.zeros((num_nodes, num_tot_points, 3))) # bound_vortex_mesh


    nodal_velocity = csdl.Variable(value=np.zeros((num_nodes, num_tot_points, 3)))
    panel_centers = csdl.Variable(value=np.zeros((num_nodes, num_tot_panels, 3)))
    force_eval_pts = csdl.Variable(value=np.zeros((num_nodes, num_tot_panels, 3)))
    panel_normal = csdl.Variable(value=np.zeros((num_nodes, num_tot_panels, 3)))
    coll_vel = csdl.Variable(value=np.zeros((num_nodes, num_tot_panels, 3)))
    bound_vec_velocity = csdl.Variable(value=np.zeros((num_nodes, num_tot_panels, 3)))
    bound_vec = csdl.Variable(value=np.zeros((num_nodes, num_tot_panels, 3)))
    panel_corners = csdl.Variable(value=np.zeros((num_nodes, num_tot_panels, 4, 3)))

    cs_panels, ce_panels = 0, 0
    cs_points, ce_points = 0, 0

    TE_node_indices = []
    TE_offset = 0

    for i in range(num_surfaces):
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
        
        bvm_all = bvm_all.set(
            csdl.slice[:,cs_points:ce_points,:],
            mesh_dict[surf_name]['bound_vortex_mesh'].reshape(num_nodes, num_points, 3)
        )

        nodal_velocity = nodal_velocity.set(
            csdl.slice[:, cs_points:ce_points,:],
            mesh_dict[surf_name]['nodal_velocity'].reshape(num_nodes, num_points, 3)
        )

        panel_centers = panel_centers.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[surf_name]['collocation_points'].reshape(num_nodes, num_panels, 3)
        )
        force_eval_pts = force_eval_pts.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[surf_name]['force_eval_points'].reshape(num_nodes, num_panels, 3)
        )
        panel_normal = panel_normal.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[surf_name]['bd_normal_vec'].reshape(num_nodes, num_panels, 3)
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
        panel_corners = panel_corners.set(
            csdl.slice[:,cs_panels:ce_panels,:,:],
            mesh_dict[surf_name]['bound_vortex_panel_corners'].reshape(num_nodes, num_panels, 4, 3)
        )

        cs_panels += num_panels
        cs_points += num_points
    
    vectorized_mesh_dict = {
        'bound_vortex_mesh': bvm_all,
        'nodal_velocity': nodal_velocity,
        'panel_centers': panel_centers,
        'panel_corners': panel_corners,
        'panel_normal': panel_normal,
        'force_eval_pts': force_eval_pts,
        'bound_vec_velocity': bound_vec_velocity,
        'bound_vec': bound_vec,
        'collocation_velocity': coll_vel,


        'TE_node_indices': TE_node_indices,
    }

    return mesh_dict, vectorized_mesh_dict