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
        nodal_vel = mesh_dict['nodal_velocity']
        cells = mesh_dict['cell_point_indices']
        cell_types = list(cells.keys())
        cell_adjacency_types = mesh_dict['cell_adjacency']

        mesh_shape = mesh.shape
        num_nodes = mesh_shape[0]
        if constant_geometry:
            num_nodes = 1
        else:
            num_nodes=mesh.shape[0]

        '''
        with mixed elements:
        - assemble the variables for each element type separately
        - concatenate the cell_type to the key name

        once each has been set up, we need to also combine the variables for:
        - normal vectors
        - panel local in-plane vectors
        - panel centers
        - collocation velocity
        - panel area
        '''

        for i, cell_type in enumerate(cell_types):
            cell_point_indices = np.array(cells[cell_type])
            if cell_type == 'triangle':
                num_corners = 3
            elif cell_type == 'quad':
                num_corners = 4

            panel_indices_np_int = list(np.arange(len(cell_point_indices)))
            p1_indices_np_int = list(cell_point_indices[:,0])
            p2_indices_np_int = list(cell_point_indices[:,1])
            p3_indices_np_int = list(cell_point_indices[:,2])

            panel_indices = [int(x) for x in panel_indices_np_int]
            p1_indices = [int(x) for x in p1_indices_np_int]
            p2_indices = [int(x) for x in p2_indices_np_int]
            p3_indices = [int(x) for x in p3_indices_np_int]
            
            nn_loop_vals = [np.arange(num_nodes).tolist()]
            loop_vals = [p1_indices, p2_indices, p3_indices]
            if cell_type == 'quad':
                p4_indices_np_int = list(cell_point_indices[:,3])
                p4_indices = [int(x) for x in p4_indices_np_int]
                loop_vals.append(p4_indices)
            
            with csdl.experimental.enter_loop(vals=nn_loop_vals) as nn_loop_builder:
                n = nn_loop_builder.get_loop_indices()
                with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
                    loop_ind_var = loop_builder.get_loop_indices()
                    p_list_inner = [mesh[n, ind, :] for ind in loop_ind_var]
                
                p_list_outer = [loop_builder.add_stack(p) for p in p_list_inner]
                loop_builder.finalize()

            p_list = [nn_loop_builder.add_stack(p) for p in p_list_outer]
            nn_loop_builder.finalize()

            panel_center = sum(p_list) / num_corners
            mesh_dict['panel_center_' + cell_type] = panel_center

            panel_corners = csdl.Variable(shape=panel_center.shape[:-1] + (num_corners,3), value=0.) # (3,3) is 3 points, 3 dimensions
            for j in range(num_corners):
                panel_corners = panel_corners.set(csdl.slice[:,:,j,:], value=p_list[j])
            mesh_dict['panel_corners_' + cell_type] = panel_corners

            if cell_type == 'triangle':
                m12 = (p_list[0]+p_list[1])/2.
                m23 = (p_list[1]+p_list[2])/2.
                m31 = (p_list[2]+p_list[0])/2.
                l_vec = m12 - panel_center
                l_vec = l_vec / csdl.expand(csdl.norm(l_vec, axes=(2,)), l_vec.shape, 'jk->jka')
                normal_vec = csdl.cross(l_vec, m23-panel_center, axis=2)
                normal_vec_norm = csdl.norm(normal_vec, axes=(2,))
                normal_vec = normal_vec / csdl.expand(normal_vec_norm, l_vec.shape, 'jk->jka')
                m_vec = csdl.cross(normal_vec, l_vec, axis=2)

                AB = p_list[1] - p_list[0]
                AC = p_list[2] - p_list[0]
                panel_area = 0.5 * csdl.norm(
                    csdl.cross(AB, AC, axis=2),
                    axes=(2,)
                )


            elif cell_type == 'quad':
                D1 = p_list[2] - p_list[0]
                D2 = p_list[3] - p_list[1]
                D1D2_cross = csdl.cross(D1, D2, axis=2)
                D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(2,))
                panel_area = D1D2_cross_norm/2.
                normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'jk->jka')

                S3 = (p_list[2] + p_list[3])/2.

                m_dir = S3 - panel_center
                m_norm = csdl.norm(m_dir, axes=(2,))
                m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'jk->jka')
                l_vec = csdl.cross(m_vec, normal_vec, axis=2)

            panel_center_mod = panel_center - normal_vec*0.001

            mesh_dict['panel_center_mod_' + cell_type] = panel_center_mod
            mesh_dict['panel_area_' + cell_type] = panel_area
            mesh_dict['panel_x_dir_' + cell_type] = l_vec
            mesh_dict['panel_y_dir_' + cell_type] = m_vec
            mesh_dict['panel_normal_' + cell_type] = normal_vec

            s = s = csdl.Variable(shape=panel_corners.shape, value=0.)
            s = s.set(csdl.slice[:,:,:-1,:], value=panel_corners[:,:,1:,:] - panel_corners[:,:,:-1,:])
            s = s.set(csdl.slice[:,:,-1,:], value=panel_corners[:,:,0,:] - panel_corners[:,:,-1,:])

            l_exp = csdl.expand(l_vec, panel_corners.shape, 'klm->klam')
            m_exp = csdl.expand(m_vec, panel_corners.shape, 'klm->klam')

            S = csdl.norm(s, axes=(3,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0 --> added to the equations instead
            # S = csdl.norm(s, axes=(5,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
            SL = csdl.sum(s*l_exp, axes=(3,))
            SM = csdl.sum(s*m_exp, axes=(3,))

            mesh_dict['S_' + cell_type] = S
            mesh_dict['SL_' + cell_type] = SL
            mesh_dict['SM_' + cell_type] = SM

            loop_vals = [p1_indices, p2_indices, p3_indices]
            if cell_type == 'quad':
                loop_vals.append(p4_indices)
            nn_loop_vals = np.arange(nodal_vel.shape[0]).tolist()
            with csdl.experimental.enter_loop(vals=[nn_loop_vals]) as nn_loop_builder:
                n = nn_loop_builder.get_loop_indices()
                with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
                    loop_ind_var = loop_builder.get_loop_indices()
                    v_list_inner = [nodal_vel[n,j,:] for j in range(num_corners)]
                    # v1 = nodal_vel[n,i,:]
                    # v2 = nodal_vel[n,j,:]
                    # v3 = nodal_vel[n,k,:]
                v_list_outer = [loop_builder.add_stack(val) for val in v_list_inner]
                # v1 = loop_builder.add_stack(v1)
                # v2 = loop_builder.add_stack(v2)
                # v3 = loop_builder.add_stack(v3)
                loop_builder.finalize()
            v_list = [nn_loop_builder.add_stack(val) for val in v_list_outer]
            # v1 = nn_loop_builder.add_stack(v1)
            # v2 = nn_loop_builder.add_stack(v2)
            # v3 = nn_loop_builder.add_stack(v3)
            nn_loop_builder.finalize()

            mesh_dict['coll_point_velocity_' + cell_type] = sum(v_list) / num_corners

        # CP DELTA CALCULATION MIGHT NEED TO BE DONE IN A SEPARATE LOOP
        num_cell_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]
        num_cells = sum(num_cell_per_type)

        panel_normal = csdl.Variable(value=np.zeros((num_nodes, num_cells, 3)))
        panel_x_dir = csdl.Variable(value=np.zeros(panel_normal.shape))
        panel_y_dir = csdl.Variable(value=np.zeros(panel_normal.shape))
        panel_center = csdl.Variable(value=np.zeros(panel_normal.shape))
        panel_area = csdl.Variable(value=np.zeros(panel_normal.shape[:-1]))
        coll_point_velocity = csdl.Variable(value=np.zeros((nodal_vel.shape[0],) + panel_normal.shape[1:]))
        print(nodal_vel.shape)

        start, stop = 0, 0
        for i, cell_type in enumerate(cell_types):
            stop += num_cell_per_type[i]
            panel_normal = panel_normal.set(csdl.slice[:,start:stop,:], mesh_dict['panel_normal_'+cell_type])
            panel_x_dir = panel_x_dir.set(csdl.slice[:,start:stop,:], mesh_dict['panel_x_dir_'+cell_type])
            panel_y_dir = panel_y_dir.set(csdl.slice[:,start:stop,:], mesh_dict['panel_y_dir_'+cell_type])
            panel_center = panel_center.set(csdl.slice[:,start:stop,:], mesh_dict['panel_center_'+cell_type])
            coll_point_velocity = coll_point_velocity.set(csdl.slice[:,start:stop,:], mesh_dict['coll_point_velocity_'+cell_type])
            panel_area = panel_area.set(csdl.slice[:,start:stop], mesh_dict['panel_area_'+cell_type])


            start += num_cell_per_type[i]

        mesh_dict['panel_normal'] = panel_normal
        mesh_dict['panel_x_dir'] = panel_x_dir
        mesh_dict['panel_y_dir'] = panel_y_dir
        mesh_dict['panel_center'] = panel_center
        mesh_dict['coll_point_velocity'] = coll_point_velocity
        mesh_dict['panel_area'] = panel_area

        start, stop = 0, 0
        for i, cell_type in enumerate(cell_types):
            cell_point_indices = np.array(cells[cell_type])
            if cell_type == 'triangle':
                num_corners = 3
            elif cell_type == 'quad':
                num_corners = 4

            cell_adjacency = np.array(cell_adjacency_types[cell_type])
            num_panels_cell_type = len(cell_adjacency)
            stop += num_panels_cell_type

            panel_indices_np_int = list(np.arange(len(cell_point_indices)) + start)
            panel_indices = [int(x) for x in panel_indices_np_int]
            # NOTE: we add "start" to this to signify the shift in panel indices with different types
            # this is only needed with mixed grids

            cp_delta_1_ind_np_int = list(cell_adjacency[:,0])
            cp_delta_2_ind_np_int = list(cell_adjacency[:,1])
            cp_delta_3_ind_np_int = list(cell_adjacency[:,2])

            cp_delta_1_ind = [int(x) for x in cp_delta_1_ind_np_int]
            cp_delta_2_ind = [int(x) for x in cp_delta_2_ind_np_int]
            cp_delta_3_ind = [int(x) for x in cp_delta_3_ind_np_int]

            loop_vals = [panel_indices, cp_delta_1_ind, cp_delta_2_ind, cp_delta_3_ind]
            if cell_type == 'quad':
                cp_delta_4_ind_np_int = list(cell_adjacency[:,3])
                cp_delta_4_ind = [int(x) for x in cp_delta_4_ind_np_int]
                loop_vals.append(cp_delta_4_ind)

            nn_loop_vals = [np.arange(num_nodes).tolist()]
            with csdl.experimental.enter_loop(vals=nn_loop_vals) as nn_loop_builder:
                n = nn_loop_builder.get_loop_indices()
                with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
                    loop_ind_var = loop_builder.get_loop_indices()
                    cp_delta_inner = [
                        panel_center[n,loop_ind_var[1+j],:] - panel_center[n,loop_ind_var[0],:]
                        for j in range(num_corners)
                    ]
                    # cp_delta_1 = panel_center[n,loop_ind_var[1],:] - panel_center[n,loop_ind_var[0],:]
                    # cp_delta_2 = panel_center[n,loop_ind_var[2],:] - panel_center[n,loop_ind_var[0],:]
                    # cp_delta_3 = panel_center[n,loop_ind_var[3],:] - panel_center[n,loop_ind_var[0],:]
                cp_delta_outer = [loop_builder.add_stack(val) for val in cp_delta_inner]
                # cp_delta_1 = loop_builder.add_stack(cp_delta_1)
                # cp_delta_2 = loop_builder.add_stack(cp_delta_2)
                # cp_delta_3 = loop_builder.add_stack(cp_delta_3)
                loop_builder.finalize()

            # cp_delta_1 = nn_loop_builder.add_stack(cp_delta_1)
            # cp_delta_2 = nn_loop_builder.add_stack(cp_delta_2)
            # cp_delta_3 = nn_loop_builder.add_stack(cp_delta_3)
            cp_delta_list = [nn_loop_builder.add_stack(val) for val in cp_delta_outer]
            nn_loop_builder.finalize()
            
            panel_corners_cell = mesh_dict['panel_corners_'+cell_type]
            cp_deltas = csdl.Variable(shape=panel_corners_cell.shape, value=0.)
            for j in range(num_corners):
                cp_deltas = cp_deltas.set(csdl.slice[:,:,j,:], value=cp_delta_list[j])
            
            cell_deltas = csdl.Variable(shape=panel_corners_cell.shape[:-1] + (2,), value=0.) # deltas only in l and m directions
            l_vec_asdf = mesh_dict['panel_x_dir_'+cell_type]
            m_vec_asdf = mesh_dict['panel_y_dir_'+cell_type]
            for j in range(num_corners):
                cell_deltas = cell_deltas.set(csdl.slice[:,:,j,0], value=csdl.sum(cp_deltas[:,:,j,:]*l_vec_asdf, axes=(2,)))
                cell_deltas = cell_deltas.set(csdl.slice[:,:,j,1], value=csdl.sum(cp_deltas[:,:,j,:]*m_vec_asdf, axes=(2,)))

            mesh_dict['delta_coll_point_' + cell_type] = cell_deltas
            start += num_panels_cell_type

        upper_TE_cells = mesh_dict['upper_TE_cells']
        lower_TE_cells = mesh_dict['lower_TE_cells']
        num_TE_cells = len(upper_TE_cells)
        upper_loc_list, lower_loc_list = [], []

        cells_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]

        # inner list is the delta to set to zero for TE elements (tri: [0,2] and quad: [0,3])
        upper_TE_cells_types = [[] for i in cells_per_type]
        lower_TE_cells_types = [[] for i in cells_per_type]
        lower_loc_list = [[] for i in cells_per_type] # inner list represents the indices for the cell types
        upper_loc_list = [[] for i in cells_per_type] # inner list represents the indices for the cell types

        for i in range(num_TE_cells):
            upper_cell_ind, lower_cell_ind = upper_TE_cells[i], lower_TE_cells[i]

            
            start_j, stop_j = 0, 0
            for j, cell_type_j in enumerate(cell_types):
                cell_adjacency = np.array(cell_adjacency_types[cell_type_j])
                stop_j += cells_per_type[j]

                # NOTE: THIS IS WRONG --> WE NEED TO CHECK IF THE upper_cell_ind and lower_cell_ind
                # ARE IN THE RANGE OF PANEL INDICES CORRESPONDING TO THIS CELL TYPE
                # start <= upper_cell_ind < stop to continue loop (& same for lower_cell_ind)
                # check_upper = np.where(cell_adjacency == upper_cell_ind)[0]
                # check_lower = np.where(cell_adjacency == lower_cell_ind)[0]
                # if len(check_upper):
                if upper_cell_ind <= start_j or upper_cell_ind > stop_j:
                    start_j += cells_per_type[j]
                    continue # ignores this panel for this cell type

                upper_cell_neighbors = cell_adjacency[upper_cell_ind-start_j]
                lower_cell_neighbors = cell_adjacency[lower_cell_ind-start_j]
                upper_loc = np.where(lower_cell_neighbors == upper_cell_ind)[0][0]
                lower_loc = np.where(upper_cell_neighbors == lower_cell_ind)[0][0]

                upper_TE_cells_types[j].append(upper_cell_ind-start_j)
                lower_TE_cells_types[j].append(lower_cell_ind-start_j)
                upper_loc_list[j].append(upper_loc)
                lower_loc_list[j].append(lower_loc)
                
                start_j += cells_per_type[j]
        
        for i, cell_type in enumerate(cell_types):
            # print(upper_TE_cells_types)
            # print(lower_loc_list)
            cell_deltas_type = mesh_dict['delta_coll_point_' + cell_type]
            cell_deltas_type = cell_deltas_type.set(
                csdl.slice[:,upper_TE_cells_types[i], lower_loc_list[i],:],
                value=0.
            )
            cell_deltas_type = cell_deltas_type.set(
                csdl.slice[:,lower_TE_cells_types[i], upper_loc_list[i],:],
                value=0.
            )
            mesh_dict['delta_coll_point_' + cell_type] = cell_deltas_type

    return mesh_dict