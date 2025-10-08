import numpy as np 
import csdl_alpha as csdl

def pre_processor(mesh_dict, mode='structured', constant_geometry=False):
    
    if mode == 'structured':
        surface_names = list(mesh_dict.keys())
        for i, surf_name in enumerate(surface_names):

            mesh = mesh_dict[surf_name]['mesh']
            mesh_shape = mesh.shape # (nn, nc, ns, 3)
            nc, ns = mesh_shape[1], mesh_shape[2]

            mesh_dict[surf_name]['num_panels'] = (nc-1)*(ns-1)
            mesh_dict[surf_name]['nc'] = nc
            mesh_dict[surf_name]['ns'] = ns

            R1 = mesh[:,:-1,:-1,:]
            R2 = mesh[:,1:,:-1,:]
            R3 = mesh[:,1:,1:,:]
            R4 = mesh[:,:-1,1:,:]
        
            S1 = (R1+R2)/2.
            S2 = (R2+R3)/2.
            S3 = (R3+R4)/2.
            S4 = (R4+R1)/2.

            Rc = (R1+R2+R3+R4)/4.
            mesh_dict[surf_name]['panel_center'] = Rc

            panel_corners = csdl.Variable(shape=(Rc.shape[:-1] + (4,3)), value=0.)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,0,:], value=R1)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,1,:], value=R2)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,2,:], value=R3)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,3,:], value=R4)
            mesh_dict[surf_name]['panel_corners'] = panel_corners

            D1 = R3-R1
            D2 = R4-R2

            D1D2_cross = csdl.cross(D1, D2, axis=3)
            D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(3,))
            panel_area = D1D2_cross_norm/2.
            mesh_dict[surf_name]['panel_area'] = panel_area

            normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'jkl->jkla')
            mesh_dict[surf_name]['panel_normal'] = normal_vec

            panel_center_mod = Rc - normal_vec*1.e-6
            # panel_center_mod = Rc 
            mesh_dict[surf_name]['panel_center_mod'] = panel_center_mod

            m_dir = S3 - Rc
            m_norm = csdl.norm(m_dir, axes=(3,))
            m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'jkl->jkla')
            l_vec = csdl.cross(m_vec, normal_vec, axis=3)
            # this also tells us that normal_vec = cross(l_vec, m_vec)

            mesh_dict[surf_name]['panel_x_dir'] = l_vec
            mesh_dict[surf_name]['panel_y_dir'] = m_vec

            rot_mat = csdl.Variable(value=np.zeros(normal_vec.shape + (3,))) # taken from dissertation of Pranav Prashant Ladkat, Pg. 26 eq. 4.5 
            rot_mat = rot_mat.set(csdl.slice[:,:,:,:,0], value=l_vec)
            rot_mat = rot_mat.set(csdl.slice[:,:,:,:,1], value=m_vec)
            rot_mat = rot_mat.set(csdl.slice[:,:,:,:,2], value=normal_vec)
            mesh_dict[surf_name]['rot_mat'] = rot_mat # rotation matrix transforms panel coordinates to global coordinates

            SMP = csdl.norm((S2)/2 - Rc, axes=(3,))
            SMQ = csdl.norm((S3)/2 - Rc, axes=(3,)) # same as m_norm

            mesh_dict[surf_name]['SMP'] = SMP
            mesh_dict[surf_name]['SMQ'] = SMQ

            s = csdl.Variable(shape=panel_corners.shape, value=0.)
            s = s.set(csdl.slice[:,:,:,:-1,:], value=panel_corners[:,:,:,1:,:] - panel_corners[:,:,:,:-1,:])
            s = s.set(csdl.slice[:,:,:,-1,:], value=panel_corners[:,:,:,0,:] - panel_corners[:,:,:,-1,:])

            l_exp = csdl.expand(l_vec, panel_corners.shape, 'jklm->jklam')
            m_exp = csdl.expand(m_vec, panel_corners.shape, 'jklm->jklam')
            
            S = csdl.norm(s, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0 --> added to the equations instead
            # S = csdl.norm(s, axes=(5,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
            SL = csdl.sum(s*l_exp, axes=(4,))
            SM = csdl.sum(s*m_exp, axes=(4,))

            mesh_dict[surf_name]['S'] = S
            mesh_dict[surf_name]['SL'] = SL
            mesh_dict[surf_name]['SM'] = SM

            delta_coll_point = csdl.Variable(Rc.shape[:-1] + (4,2), value=0.)
            delta_coll_point = delta_coll_point.set(csdl.slice[:,1:,:,0,0], value=csdl.sum((Rc[:,:-1,:,:]-Rc[:,1:,:,:])*l_vec[:,1:,:,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,1:,:,0,1], value=csdl.sum((Rc[:,:-1,:,:]-Rc[:,1:,:,:])*m_vec[:,1:,:,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:-1,:,1,0], value=csdl.sum((Rc[:,1:,:,:]-Rc[:,:-1,:,:])*l_vec[:,:-1,:,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:-1,:,1,1], value=csdl.sum((Rc[:,1:,:,:]-Rc[:,:-1,:,:])*m_vec[:,:-1,:,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,2,0], value=csdl.sum((Rc[:,:,:-1,:]-Rc[:,:,1:,:])*l_vec[:,:,1:,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,2,1], value=csdl.sum((Rc[:,:,:-1,:]-Rc[:,:,1:,:])*m_vec[:,:,1:,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,3,0], value=csdl.sum((Rc[:,:,1:,:]-Rc[:,:,:-1,:])*l_vec[:,:,:-1,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,3,1], value=csdl.sum((Rc[:,:,1:,:]-Rc[:,:,:-1,:])*m_vec[:,:,:-1,:], axes=(3,)))

            # # setting deltas for panels wrapping around TE to zero
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,0,:,:,:], value=0.)

            mesh_dict[surf_name]['delta_coll_point'] = delta_coll_point

            nodal_vel = mesh_dict[surf_name]['nodal_velocity']
            mesh_dict[surf_name]['nodal_cp_velocity'] = (
                nodal_vel[:,:-1,:-1,:]+nodal_vel[:,:-1,1:,:]+\
                nodal_vel[:,1:,1:,:]+nodal_vel[:,1:,:-1,:]) / 4.

            # computing planform area
            panel_width_spanwise = csdl.norm((mesh[:,:,1:,:] - mesh[:,:,:-1,:]), axes=(3,))
            avg_panel_width_spanwise = csdl.average(panel_width_spanwise, axes=(1,)) # num_nodes, ns - 1
            surface_TE = (mesh[:,-1,:-1,:] + mesh[:,0,:-1,:] + mesh[:,-1,1:,:] + mesh[:,0,1:,:])/4
            surface_LE = (mesh[:,int((nc-1)/2),:-1,:] + mesh[:,int((nc-1)/2),1:,:])/2 # num_nodes, ns - 1, 3

            chord_spanwise = csdl.norm(surface_TE - surface_LE, axes=(2,)) # num_nodes,  ns - 1

            planform_area = csdl.sum(chord_spanwise*avg_panel_width_spanwise, axes=(1,))
            mesh_dict[surf_name]['planform_area'] = planform_area
            
    elif mode == 'unstructured':
        mesh = mesh_dict['points'] # num_nodes, num_panels, 3
        cell_point_indices = mesh_dict['cell_point_indices']
        cell_adjacency = mesh_dict['cell_adjacency']
        mesh_shape = mesh.shape
        num_nodes = mesh_shape[0]
        if constant_geometry:
            num_nodes = 1
        else:
            num_nodes=mesh.shape[0]
        num_panels = cell_point_indices.shape[0]

        # ==== WITH DUPLICATE INDICES ====
        # p1 = mesh[:,list(cell_point_indices[:,0]),:]
        # p2 = mesh[:,list(cell_point_indices[:,1]),:]
        # p3 = mesh[:,list(cell_point_indices[:,2]),:]

        # ==== USING CSDL FRANGE ====
        panel_indices_np_int = list(np.arange(cell_point_indices.shape[0]))
        p1_indices_np_int = list(cell_point_indices[:,0])
        p2_indices_np_int = list(cell_point_indices[:,1])
        p3_indices_np_int = list(cell_point_indices[:,2])

        panel_indices = [int(x) for x in panel_indices_np_int]
        p1_indices = [int(x) for x in p1_indices_np_int]
        p2_indices = [int(x) for x in p2_indices_np_int]
        p3_indices = [int(x) for x in p3_indices_np_int]

        # p1 = csdl.Variable(shape=(num_nodes, cell_point_indices.shape[0], 3), value=0.)
        # p2 = csdl.Variable(shape=(num_nodes, cell_point_indices.shape[0], 3), value=0.)
        # p3 = csdl.Variable(shape=(num_nodes, cell_point_indices.shape[0], 3), value=0.)

        # for cell_ind, ind1, ind2, ind3 in csdl.frange(vals=(panel_indices, p1_indices, p2_indices, p3_indices)):
        #     p1 = p1.set(csdl.slice[:,cell_ind,:], value=mesh[:,ind1,:])
        #     p2 = p2.set(csdl.slice[:,cell_ind,:], value=mesh[:,ind2,:])
        #     p3 = p3.set(csdl.slice[:,cell_ind,:], value=mesh[:,ind3,:])

        # ==== USING STACK VIA LOOP BUILDER ====
        nn_loop_vals = [np.arange(num_nodes).tolist()]
        loop_vals = [p1_indices, p2_indices, p3_indices]
        with csdl.experimental.enter_loop(vals=nn_loop_vals) as nn_loop_builder:
            n = nn_loop_builder.get_loop_indices()
            with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
                j,k,l = loop_builder.get_loop_indices()
                p1 = mesh[n,j,:]
                p2 = mesh[n,k,:]
                p3 = mesh[n,l,:]

            p1 = loop_builder.add_stack(p1)
            p2 = loop_builder.add_stack(p2)
            p3 = loop_builder.add_stack(p3)
            loop_builder.finalize()

        p1 = nn_loop_builder.add_stack(p1)
        p2 = nn_loop_builder.add_stack(p2)
        p3 = nn_loop_builder.add_stack(p3)
        nn_loop_builder.finalize()

        # p1 = p1.reshape((num_nodes, num_panels, 3))
        # p2 = p2.reshape((num_nodes, num_panels, 3))
        # p3 = p3.reshape((num_nodes, num_panels, 3))
        
        panel_center = (p1+p2+p3)/3.
        mesh_dict['panel_center'] = panel_center

        panel_corners = csdl.Variable(shape=panel_center.shape[:-1] + (3,3), value=0.) # (3,3) is 3 points, 3 dimensions
        panel_corners = panel_corners.set(csdl.slice[:,:,0,:], value=p1)
        panel_corners = panel_corners.set(csdl.slice[:,:,1,:], value=p2)
        panel_corners = panel_corners.set(csdl.slice[:,:,2,:], value=p3)
        mesh_dict['panel_corners'] = panel_corners

        a = csdl.norm(p2-p1, axes=(2,))
        b = csdl.norm(p3-p2, axes=(2,))
        c = csdl.norm(p1-p3, axes=(2,))

        s = (a+b+c)/2.
        panel_area = (s*(s-a)*(s-b)*(s-c))**0.5
        mesh_dict['panel_area'] = panel_area

        m12 = (p1+p2)/2.
        m23 = (p2+p3)/2.
        m31 = (p3+p1)/2.

        l_vec = m12 - panel_center
        l_vec = l_vec / csdl.expand(csdl.norm(l_vec, axes=(2,)), l_vec.shape, 'jk->jka')

        normal_vec = csdl.cross(l_vec, m23-panel_center, axis=2)
        normal_vec = normal_vec / csdl.expand(csdl.norm(normal_vec, axes=(2,)), l_vec.shape, 'jk->jka')

        m_vec = csdl.cross(normal_vec, l_vec, axis=2)

        mesh_dict['panel_x_dir'] = l_vec
        mesh_dict['panel_y_dir'] = m_vec
        mesh_dict['panel_normal'] = normal_vec

        panel_center_mod = panel_center - normal_vec*0.001
        # panel_center_mod = panel_center 
        mesh_dict['panel_center_mod'] = panel_center_mod

        rot_mat = csdl.Variable(value=np.zeros(normal_vec.shape + (3,))) # taken from dissertation of Pranav Prashant Ladkat, Pg. 26 eq. 4.5 
        rot_mat = rot_mat.set(csdl.slice[:,:,:,0], value=l_vec)
        rot_mat = rot_mat.set(csdl.slice[:,:,:,1], value=m_vec)
        rot_mat = rot_mat.set(csdl.slice[:,:,:,2], value=normal_vec)
        mesh_dict['rot_mat'] = rot_mat # rotation matrix transforms panel coordinates to global coordinates

        s = csdl.Variable(shape=panel_corners.shape, value=0.)
        s = s.set(csdl.slice[:,:,:-1,:], value=panel_corners[:,:,1:,:] - panel_corners[:,:,:-1,:])
        s = s.set(csdl.slice[:,:,-1,:], value=panel_corners[:,:,0,:] - panel_corners[:,:,-1,:])

        l_exp = csdl.expand(l_vec, panel_corners.shape, 'klm->klam')
        m_exp = csdl.expand(m_vec, panel_corners.shape, 'klm->klam')
        
        S = csdl.norm(s, axes=(3,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0 --> added to the equations instead
        # S = csdl.norm(s, axes=(5,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
        SL = csdl.sum(s*l_exp, axes=(3,))
        SM = csdl.sum(s*m_exp, axes=(3,))

        mesh_dict['S'] = S
        mesh_dict['SL'] = SL
        mesh_dict['SM'] = SM

        # ==== USING CSDL FRANGE ====
        cp_delta_1_ind_np_int = list(cell_adjacency[:,0])
        cp_delta_2_ind_np_int = list(cell_adjacency[:,1])
        cp_delta_3_ind_np_int = list(cell_adjacency[:,2])

        cp_delta_1_ind = [int(x) for x in cp_delta_1_ind_np_int]
        cp_delta_2_ind = [int(x) for x in cp_delta_2_ind_np_int]
        cp_delta_3_ind = [int(x) for x in cp_delta_3_ind_np_int]

        # cp_deltas = csdl.Variable(shape=panel_corners.shape, value=0.)
        # for cell_ind, ind1, ind2, ind3 in csdl.frange(vals=(panel_indices, cp_delta_1_ind, cp_delta_2_ind, cp_delta_3_ind)):
        #     cp_deltas = cp_deltas.set(csdl.slice[:,cell_ind,0,:], value=panel_center[:,ind1,:] - panel_center[:,cell_ind,:])
        #     cp_deltas = cp_deltas.set(csdl.slice[:,cell_ind,1,:], value=panel_center[:,ind2,:] - panel_center[:,cell_ind,:])
        #     cp_deltas = cp_deltas.set(csdl.slice[:,cell_ind,2,:], value=panel_center[:,ind3,:] - panel_center[:,cell_ind,:])
        
        # ==== USING STACK VIA LOOP BUILDER ====
        loop_vals = [panel_indices, cp_delta_1_ind, cp_delta_2_ind, cp_delta_3_ind]
        with csdl.experimental.enter_loop(vals=nn_loop_vals) as nn_loop_builder:
            n = nn_loop_builder.get_loop_indices()
            with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
                i,a,b,c = loop_builder.get_loop_indices()
                cp_delta_1 = panel_center[n,a,:] - panel_center[n,i,:]
                cp_delta_2 = panel_center[n,b,:] - panel_center[n,i,:]
                cp_delta_3 = panel_center[n,c,:] - panel_center[n,i,:]
            cp_delta_1 = loop_builder.add_stack(cp_delta_1)
            cp_delta_2 = loop_builder.add_stack(cp_delta_2)
            cp_delta_3 = loop_builder.add_stack(cp_delta_3)
            loop_builder.finalize()

        cp_delta_1 = nn_loop_builder.add_stack(cp_delta_1)
        cp_delta_2 = nn_loop_builder.add_stack(cp_delta_2)
        cp_delta_3 = nn_loop_builder.add_stack(cp_delta_3)
        nn_loop_builder.finalize()
        # cp_delta_1 = cp_delta_1.reshape((num_nodes, num_panels, 3))
        # cp_delta_2 = cp_delta_2.reshape((num_nodes, num_panels, 3))
        # cp_delta_3 = cp_delta_3.reshape((num_nodes, num_panels, 3))

        cp_deltas = csdl.Variable(shape=panel_corners.shape, value=0.)
        cp_deltas = cp_deltas.set(csdl.slice[:,:,0,:], value=cp_delta_1)
        cp_deltas = cp_deltas.set(csdl.slice[:,:,1,:], value=cp_delta_2)
        cp_deltas = cp_deltas.set(csdl.slice[:,:,2,:], value=cp_delta_3)


        cell_deltas = csdl.Variable(shape=panel_corners.shape[:-1] + (2,), value=0.) # each cell has 3 deltas, with 2 dimensions (l,m)
        cell_deltas = cell_deltas.set(csdl.slice[:,:,0,0], value=csdl.sum(cp_deltas[:,:,0,:]*l_vec, axes=(2,)))
        cell_deltas = cell_deltas.set(csdl.slice[:,:,0,1], value=csdl.sum(cp_deltas[:,:,0,:]*m_vec, axes=(2,)))
        cell_deltas = cell_deltas.set(csdl.slice[:,:,1,0], value=csdl.sum(cp_deltas[:,:,1,:]*l_vec, axes=(2,)))
        cell_deltas = cell_deltas.set(csdl.slice[:,:,1,1], value=csdl.sum(cp_deltas[:,:,1,:]*m_vec, axes=(2,)))
        cell_deltas = cell_deltas.set(csdl.slice[:,:,2,0], value=csdl.sum(cp_deltas[:,:,2,:]*l_vec, axes=(2,)))
        cell_deltas = cell_deltas.set(csdl.slice[:,:,2,1], value=csdl.sum(cp_deltas[:,:,2,:]*m_vec, axes=(2,)))

        upper_TE_cells = mesh_dict['upper_TE_cells']
        lower_TE_cells = mesh_dict['lower_TE_cells']
        num_TE_cells = len(upper_TE_cells)
        upper_loc_list, lower_loc_list = [], []

        for i in range(num_TE_cells):
            upper_cell_ind, lower_cell_ind = upper_TE_cells[i], lower_TE_cells[i]
            upper_cell_neighbors = cell_adjacency[upper_cell_ind]
            lower_cell_neighbors = cell_adjacency[lower_cell_ind]

            upper_loc = np.where(lower_cell_neighbors == upper_cell_ind)[0][0]
            lower_loc = np.where(upper_cell_neighbors == lower_cell_ind)[0][0]

            upper_loc_list.append(upper_loc)
            lower_loc_list.append(lower_loc)

        cell_deltas = cell_deltas.set(csdl.slice[:,list(upper_TE_cells),lower_loc_list,:], value=0.)
        cell_deltas = cell_deltas.set(csdl.slice[:,list(lower_TE_cells),upper_loc_list,:], value=0.)

        mesh_dict['delta_coll_point'] = cell_deltas
        # NOTE: CHECK IF AXIS ON THESE LINES ABOVE SHOULD BE 2 OR 3

        nodal_vel = mesh_dict['nodal_velocity']
        # num_nodes = nodal_vel.shape[0]
        nn_loop_vals = np.arange(nodal_vel.shape[0]).tolist()
        # ==== WITH DUPLICATE NODES ====
        # v1 = nodal_vel[:,list(cell_point_indices[:,0]),:]
        # v2 = nodal_vel[:,list(cell_point_indices[:,1]),:]
        # v3 = nodal_vel[:,list(cell_point_indices[:,2]),:]

        # ==== USING CSDL FRANGE ====
        # v1 = csdl.Variable(shape=(num_nodes, cell_point_indices.shape[0], 3), value=0.)
        # v2 = csdl.Variable(shape=(num_nodes, cell_point_indices.shape[0], 3), value=0.)
        # v3 = csdl.Variable(shape=(num_nodes, cell_point_indices.shape[0], 3), value=0.)

        # for cell_ind, ind1, ind2, ind3 in csdl.frange(vals=(panel_indices, p1_indices, p2_indices, p3_indices)):
        #     v1 = v1.set(csdl.slice[:,cell_ind,:], value=nodal_vel[:,ind1,:])
        #     v2 = v2.set(csdl.slice[:,cell_ind,:], value=nodal_vel[:,ind2,:])
        #     v3 = v3.set(csdl.slice[:,cell_ind,:], value=nodal_vel[:,ind3,:])

        # ==== USING STACK VIA LOOP BUILDER ====
        loop_vals = [p1_indices, p2_indices, p3_indices]
        with csdl.experimental.enter_loop(vals=[nn_loop_vals]) as nn_loop_builder:
            n = nn_loop_builder.get_loop_indices()
            with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
                i,j,k = loop_builder.get_loop_indices()
                v1 = nodal_vel[n,i,:]
                v2 = nodal_vel[n,j,:]
                v3 = nodal_vel[n,k,:]
            v1 = loop_builder.add_stack(v1)
            v2 = loop_builder.add_stack(v2)
            v3 = loop_builder.add_stack(v3)
            loop_builder.finalize()
        v1 = nn_loop_builder.add_stack(v1)
        v2 = nn_loop_builder.add_stack(v2)
        v3 = nn_loop_builder.add_stack(v3)
        nn_loop_builder.finalize()
        # v1 = v1.reshape((num_nodes, num_panels, 3))
        # v2 = v2.reshape((num_nodes, num_panels, 3))
        # v3 = v3.reshape((num_nodes, num_panels, 3))
        
        mesh_dict['coll_point_velocity'] = (v1+v2+v3)/3.
    return mesh_dict