import numpy as np
import csdl_alpha as csdl

def fixed_wake_representation(mesh_dict, num_nodes, wake_propagation_dt=100., mesh_mode='structured', constant_geometry=False):
    # wake propagation dt: time elapsed to propagate wake back (dx = V_inf*dt)
    if mesh_mode == 'structured':
        surface_names = list(mesh_dict.keys())
        wake_mesh_dict = {}

        for i, surface_name in enumerate(surface_names):
            
            surf_wake_mesh_dict = {}
            nc_w = 2 # only one panel for now
            surface_mesh = mesh_dict[surface_name]['mesh'] # (nn, nc, ns, 3)
            mesh_velocity = mesh_dict[surface_name]['nodal_velocity']

            ns = surface_mesh.shape[2]
            
            TE = (surface_mesh[:,0,:,:] + surface_mesh[:,-1,:,:])/2.
            # wake_end = TE + mesh_velocity[:,-1,:,:]*wake_propagation_dt
            wake_end = TE + 500

            wake_mesh = csdl.Variable(value=np.zeros((num_nodes, 2, ns, 3))) # only 2 nodes in the "chordwise" direction
            wake_mesh = wake_mesh.set(csdl.slice[:,0,:,:], value=TE)
            wake_mesh = wake_mesh.set(csdl.slice[:,1,:,:], value=wake_end)


            surf_wake_mesh_dict['mesh'] = wake_mesh
            surf_wake_mesh_dict['nc'], surf_wake_mesh_dict['ns'] = nc_w, ns
            surf_wake_mesh_dict['num_panels'] = (nc_w-1)*(ns-1)
            surf_wake_mesh_dict['num_points'] = nc_w*ns


            # computing wake parameters

            # mesh has shape (nn, 2, ns, 3) bc there's only ONE chordwise panel
            R1 = wake_mesh[:,:-1,:-1,:]
            R2 = wake_mesh[:,1:,:-1,:]
            R3 = wake_mesh[:,1:,1:,:]
            R4 = wake_mesh[:,:-1,1:,:]

            Rc = (R1+R2+R3+R4)/4.
            panel_center = Rc
            surf_wake_mesh_dict['panel_center'] = panel_center

            panel_corners = csdl.Variable(value=np.zeros((num_nodes, nc_w-1, ns-1, 4, 3)))
            panel_corners = panel_corners.set(csdl.slice[:,:,:,0,:], value=R1)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,1,:], value=R2)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,2,:], value=R3)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,3,:], value=R4)
            surf_wake_mesh_dict['panel_corners'] = panel_corners

            D1 = R3-R1
            D2 = R4-R2

            D1D2_cross = csdl.cross(D1, D2, axis=3)
            D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(3,))
            panel_area = D1D2_cross_norm/2.
            surf_wake_mesh_dict['panel_area'] = panel_area

            normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'ijk->ijka')

            m_dir = (R3+R4)/2. - Rc
            m_norm = csdl.norm(m_dir, axes=(3,))
            m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'ijk->ijka')
            l_vec = csdl.cross(m_vec, normal_vec, axis=3)

            panel_x_dir = l_vec
            panel_y_dir = m_vec
            panel_normal = normal_vec

            surf_wake_mesh_dict['panel_x_dir'] = panel_x_dir
            surf_wake_mesh_dict['panel_y_dir'] = panel_y_dir
            surf_wake_mesh_dict['panel_normal'] = panel_normal

            # s = csdl.Variable(shape=(panel_corners.shape[0],) + panel_corners.shape[2:], value=0.)
            s = csdl.Variable(shape=panel_corners.shape, value=0.)
            s = s.set(csdl.slice[:,:,:,:-1,:], value=panel_corners[:,:,:,1:,:] - panel_corners[:,:,:,:-1,:])
            s = s.set(csdl.slice[:,:,:,-1,:], value=panel_corners[:,:,:,0,:] - panel_corners[:,:,:,-1,:])

            l_exp = csdl.expand(l_vec, s.shape, 'ijkl->ijkal')
            m_exp = csdl.expand(m_vec, s.shape, 'ijkl->ijkal')
            
            S = csdl.norm(s+1.e-12, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
            SL = csdl.sum(s*l_exp+1.e-12, axes=(4,))
            SM = csdl.sum(s*m_exp+1.e-12, axes=(4,))

            surf_wake_mesh_dict['S'] = S
            surf_wake_mesh_dict['SL'] = SL
            surf_wake_mesh_dict['SM'] = SM
            
            wake_mesh_dict[surface_name] = surf_wake_mesh_dict

    elif mesh_mode == 'unstructured':
        # NOTE: We have omitted the "per-surface" looping here
        # This is difficult to do for unstructured meshes, so we will revisit in the future
        # In the future, we can use sublists to denote individual surfaces
        # Number of sublists = number of surfaces
        # Length of each sublist = number of TE nodes for each surface
        wake_mesh_dict = {}

        TE_node_indices = mesh_dict['TE_node_indices']
        TE_edges = mesh_dict['TE_edges'] # each entry is a tuple with two entries
        # the two entries are the two mesh point indices defining the edge

        mesh = mesh_dict['points']
        nodal_vel = mesh_dict['nodal_velocity']
        if constant_geometry:
            num_nodes = 1

        ns = len(TE_node_indices)
        num_TE_edges = len(TE_edges)
        nc_w = 2
        TE = mesh[:,list(TE_node_indices),:]
        
        if constant_geometry: # propagating back at 0 aoa essentially
            wake_disp = np.zeros(TE.shape)
            wake_disp[:,:,0] = 500.
            wake_end = TE + wake_disp
        else: # account for aoa in wake
            TE_vel = nodal_vel[:,list(TE_node_indices),:]
            wake_end = TE + TE_vel*wake_propagation_dt

        # creating unstructured wake mesh using TE points
        TE_node_ind_zeroed = list(np.arange(ns)) # corresponds to indices in TE and TE_vel
        TE_edges_zeroed = []
        for i in range(num_TE_edges):
            edge = TE_edges[i]
            new_edge = []
            for j in range(2):
                ind = np.where(TE_node_indices == edge[j])[0][0]
                new_edge.append(ind)

            TE_edges_zeroed.append(tuple(new_edge))

        wake_connectivity = np.array([[
            edge[0],
            edge[0]+ns,
            edge[1]+ns,
            edge[1]
        ] for edge in TE_edges_zeroed])

        wake_mesh = csdl.Variable(value=np.zeros((num_nodes, ns*nc_w, 3)))
        wake_mesh = wake_mesh.set(csdl.slice[:,:ns,:], value=TE)
        wake_mesh = wake_mesh.set(csdl.slice[:,ns:,:], value=wake_end)

        wake_mesh_dict['mesh'] = wake_mesh
        wake_mesh_dict['nc'], wake_mesh_dict['ns'] = nc_w, ns
        # wake_mesh_dict['num_panels'] = (nc_w-1)*(ns-1)
        wake_mesh_dict['num_panels'] = (nc_w-1)*num_TE_edges
        wake_mesh_dict['num_points'] = nc_w*ns

        # p1 = wake_mesh[:,list(wake_connectivity[:,0]),:]
        # p2 = wake_mesh[:,list(wake_connectivity[:,1]),:]
        # p3 = wake_mesh[:,list(wake_connectivity[:,2]),:]
        # p4 = wake_mesh[:,list(wake_connectivity[:,3]),:]

        p1_ind = [int(x) for x in list(wake_connectivity[:,0])]
        p2_ind = [int(x) for x in list(wake_connectivity[:,1])]
        p3_ind = [int(x) for x in list(wake_connectivity[:,2])]
        p4_ind = [int(x) for x in list(wake_connectivity[:,3])]

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
