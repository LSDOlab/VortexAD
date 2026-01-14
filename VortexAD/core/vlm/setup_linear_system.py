import csdl_alpha as csdl 

from VortexAD.core.vlm.velocity_computations import compute_normal_velocity, compute_induced_velocity
from VortexAD.core.vlm.compute_AIC import compute_AIC

def setup_linear_system(num_nodes, mesh_dict):

    AIC = compute_AIC(num_nodes, mesh_dict, eval_pt_name='collocation_points')

    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)

    num_total_panels = 0
    for key in mesh_dict.keys():
        ns, nc = mesh_dict[key]['ns'], mesh_dict[key]['nc']
        num_total_panels += (ns-1)*(nc-1)

    RHS = csdl.Variable(shape=(num_nodes, num_total_panels), name='RHS', value=0.)
    start_panel_counter = 0
    stop_panel_counter = 0
    for i in range(num_surfaces):
        surface_name = surface_names[i] # THIS IS THE SURFACE OF INTEREST
        ns_i, nc_i = mesh_dict[surface_name]['ns'], mesh_dict[surface_name]['nc']
        np_surf_i = (ns_i-1)*(nc_i-1)
        bd_vortex_normals_i = mesh_dict[surface_name]['bd_normal_vec'] # (num_nodes, nc-1, ns-1, 3)
        collocation_velocity = mesh_dict[surface_name]['collocation_velocity'] # (num_nodes, nc-1, ns-1, 3)

        # SETTING BOUNDARY CONDITION
        stop_panel_counter += np_surf_i
        # surface_bc = compute_normal_velocity(V_inf, bd_vortex_normals_i)
        surface_bc = compute_normal_velocity(collocation_velocity, bd_vortex_normals_i)
        RHS = RHS.set(csdl.slice[:, start_panel_counter:stop_panel_counter], csdl.reshape(-surface_bc, (num_nodes, np_surf_i)))
        start_panel_counter += np_surf_i

    return AIC, RHS

def setup_linear_system_old(num_nodes, mesh_dict, V_inf):
    '''
    Assemble the AIC matrix and RHS boundary condition here
    '''
    surface_names = list(mesh_dict.keys())
    print(surface_names)
    num_surfaces = len(surface_names)
    print(num_surfaces)

    num_total_panels = 0
    for key in mesh_dict.keys():
        ns, nc = mesh_dict[key]['ns'], mesh_dict[key]['nc']
        num_total_panels += (ns-1)*(nc-1)

    RHS = csdl.Variable(shape=(num_nodes, num_total_panels), name='RHS', value=0.)
    AIC = csdl.Variable(shape=(num_nodes, num_total_panels, num_total_panels,), name='AIC', value=0.)

    '''
    What's needed here:
    - bound vortex grid/mesh
    - collocation points

    What we want to find:
    - induced velocity from all panels on panel i
    '''
    start_panel_counter = 0
    stop_panel_counter = 0
    for i in range(num_surfaces):
        surface_name = surface_names[i] # THIS IS THE SURFACE OF INTEREST
        ns_i, nc_i = mesh_dict[surface_name]['ns'], mesh_dict[surface_name]['nc']
        np_surf_i = (ns_i-1)*(nc_i-1)
        bd_vortex_normals_i = mesh_dict[surface_name]['bd_normal_vec'] # (num_nodes, ns-1, nc-1, 3)
        collocation_points_i = mesh_dict[surface_name]['collocation_points'] # (num_nodes, ns-1, nc-1, 3)
        bound_vortex_mesh = mesh_dict[surface_name]['bound_vortex_mesh'] # (num_nodes, ns-1, nc-1, 3)

        # p1_bd = bound_vortex_mesh[:, :-1, :-1, :]
        # p2_bd = bound_vortex_mesh[:, :-1, 1:, :]
        # p3_bd = bound_vortex_mesh[:, 1:, 1:, :]
        # p4_bd = bound_vortex_mesh[:, 1:, :-1, :]

        # bd_vortex_normals_vec = csdl.reshape(bd_vortex_normals, shape=(num_nodes, np_surf, 3))
        collocation_points_vec = csdl.reshape(collocation_points_i, shape=(num_nodes, np_surf_i, 3))

        # p1_bd_vec = csdl.reshape(p1_bd, (num_nodes, np_surf, 3))
        # p2_bd_vec = csdl.reshape(p2_bd, (num_nodes, np_surf, 3))
        # p3_bd_vec = csdl.reshape(p3_bd, (num_nodes, np_surf, 3))
        # p4_bd_vec = csdl.reshape(p4_bd, (num_nodes, np_surf, 3))

        # SETTING BOUNDARY CONDITION
        stop_panel_counter += np_surf_i
        surface_bc = compute_normal_velocity(V_inf, bd_vortex_normals_i)
        RHS = RHS.set(csdl.slice[:, start_panel_counter:stop_panel_counter], csdl.reshape(surface_bc, (num_nodes, np_surf_i)))

        # getting interactions of looping surface j ON main surface i
        # means we evaluate induced velocity at collocation point of surface i
        # based on the influence from the bound vortices from surface j
        start_panel_counter_j = 0
        stop_panel_counter_j = 0
        for j in range(num_surfaces):
            surface_name_j = surface_names[j]
            ns_j, nc_j = mesh_dict[surface_name_j]['ns'], mesh_dict[surface_name_j]['nc']
            np_surf_j = (ns_j-1)*(nc_j-1)

            bound_vortex_mesh_j = mesh_dict[surface_name_j]['bound_vortex_mesh']
            bd_vortex_normals_j = mesh_dict[surface_name_j]['bd_normal_vec']

            # VECTORIZE AND EXPAND EVERYTHING TO SHAPE (num_nodes, np_surf_j*np_surf_i) (+ (3,) if needed)
            num_interactions = np_surf_i*np_surf_j

            p1_bd_grid = bound_vortex_mesh_j[:, :-1, :-1, :]
            p2_bd_grid = bound_vortex_mesh_j[:, :-1, 1:, :]
            p3_bd_grid = bound_vortex_mesh_j[:, 1:, 1:, :]
            p4_bd_grid = bound_vortex_mesh_j[:, 1:, :-1, :]

            p1_bd = csdl.expand(p1_bd_grid, (num_nodes, nc_j-1, ns_j-1, np_surf_i, 3), 'ijkl->ijkal')
            p2_bd = csdl.expand(p2_bd_grid, (num_nodes, nc_j-1, ns_j-1, np_surf_i, 3), 'ijkl->ijkal')
            p3_bd = csdl.expand(p3_bd_grid, (num_nodes, nc_j-1, ns_j-1, np_surf_i, 3), 'ijkl->ijkal')
            p4_bd = csdl.expand(p4_bd_grid, (num_nodes, nc_j-1, ns_j-1, np_surf_i, 3), 'ijkl->ijkal')

            interaction_shape = (num_nodes, num_interactions, 3)

            # reshape starts with the right-most dimensions. So, this vectorizes panel 0 np_surf_i times, then panel 1, ...
            p1_bd_vec = p1_bd.reshape(interaction_shape)
            p2_bd_vec = p2_bd.reshape(interaction_shape)
            p3_bd_vec = p3_bd.reshape(interaction_shape)
            p4_bd_vec = p4_bd.reshape(interaction_shape)

            # use csdl for loop to iteratively expand over one variable (different expansion from csdl.expand)
            coll_point_i_exp = csdl.Variable(shape=(num_nodes, np_surf_j, nc_i-1, ns_i-1, 3), value=0.)
            bd_normal_vec_i_exp = csdl.Variable(shape=(num_nodes, np_surf_j, nc_i-1, ns_i-1, 3), value=0.)
            for k in csdl.frange((np_surf_j)):
                coll_point_i_exp = coll_point_i_exp.set(csdl.slice[:,k,:,:,:], value=collocation_points_i)
                bd_normal_vec_i_exp = bd_normal_vec_i_exp.set(csdl.slice[:,k,:,:,:], value=bd_vortex_normals_i)

            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions, 3))
            
            v_i_12 = compute_induced_velocity(p1_bd_vec, p2_bd_vec, coll_point_i_exp_vec)
            v_i_23 = compute_induced_velocity(p2_bd_vec, p3_bd_vec, coll_point_i_exp_vec)
            v_i_34 = compute_induced_velocity(p3_bd_vec, p4_bd_vec, coll_point_i_exp_vec)
            v_i_41 = compute_induced_velocity(p4_bd_vec, p1_bd_vec, coll_point_i_exp_vec)

            v_induced = v_i_12 + v_i_23 + v_i_34 + v_i_41

            v_induced_grid = v_induced.reshape((num_nodes, np_surf_j, np_surf_i, 3))

            # NOTE: ADD THE WAKE INFLUENCE HERE TO RESOLVE KUTTA CONDITION
            # WAKE IS ACCESSED WITH mesh_dict[surface_name]['wake_vortex_mesh']
            # treat exactly like the bound vortices of loop j
            wake_vortex_mesh_j  = mesh_dict[surface_name_j]['wake_vortex_mesh']
            nc_w_j, ns_w_j = wake_vortex_mesh_j.shape[1], wake_vortex_mesh_j.shape[2]
            np_wake_j = (nc_w_j-1)*(ns_w_j-1)

            num_wake_interactions = np_surf_i*np_wake_j
            wake_interaction_shape = (num_nodes, num_wake_interactions, 3)

            p1_w_grid = wake_vortex_mesh_j[:, :-1, :-1, :]
            p2_w_grid = wake_vortex_mesh_j[:, :-1, 1:, :]
            p3_w_grid = wake_vortex_mesh_j[:, 1:, 1:, :]
            p4_w_grid = wake_vortex_mesh_j[:, 1:, :-1, :]

            p1_w = csdl.expand(p1_w_grid, (num_nodes, nc_w_j-1, ns_w_j-1, np_surf_i, 3), 'ijkl->ijkal')
            p2_w = csdl.expand(p2_w_grid, (num_nodes, nc_w_j-1, ns_w_j-1, np_surf_i, 3), 'ijkl->ijkal')
            p3_w = csdl.expand(p3_w_grid, (num_nodes, nc_w_j-1, ns_w_j-1, np_surf_i, 3), 'ijkl->ijkal')
            p4_w = csdl.expand(p4_w_grid, (num_nodes, nc_w_j-1, ns_w_j-1, np_surf_i, 3), 'ijkl->ijkal')

            p1_w_vec = p1_w.reshape(wake_interaction_shape)
            p2_w_vec = p2_w.reshape(wake_interaction_shape)
            p3_w_vec = p3_w.reshape(wake_interaction_shape)
            p4_w_vec = p4_w.reshape(wake_interaction_shape)

            coll_point_i_wake_exp = csdl.Variable(shape=(num_nodes, np_wake_j, nc_i-1, ns_i-1, 3), value=0.)
            for k in csdl.frange((np_wake_j)):
                coll_point_i_wake_exp = coll_point_i_wake_exp.set(csdl.slice[:,k,:,:,:], value=collocation_points_i)

            coll_point_i_wake_exp_vec = coll_point_i_wake_exp.reshape(wake_interaction_shape)

            v_i_12_w = compute_induced_velocity(p1_w_vec, p2_w_vec, coll_point_i_wake_exp_vec)
            v_i_23_w = compute_induced_velocity(p2_w_vec, p3_w_vec, coll_point_i_wake_exp_vec)
            v_i_34_w = compute_induced_velocity(p3_w_vec, p4_w_vec, coll_point_i_wake_exp_vec)
            v_i_41_w = compute_induced_velocity(p4_w_vec, p1_w_vec, coll_point_i_wake_exp_vec)

            v_induced_wake = v_i_12_w + v_i_23_w + v_i_34_w + v_i_41_w
            v_induced_wake_grid = v_induced_wake.reshape((num_nodes, np_wake_j, np_surf_i, 3))
            wake_influence_grid = csdl.Variable(shape=v_induced_grid.shape, value=0.)
            stop_panel_counter_j += np_surf_j
            start_wake_ind = stop_panel_counter_j - np_wake_j
            wake_influence_grid = wake_influence_grid.set(csdl.slice[:,start_wake_ind:stop_panel_counter_j,start_panel_counter:stop_panel_counter], value=v_induced_wake_grid)

            bd_normal_vec_i_exp_grid = bd_normal_vec_i_exp.reshape((num_nodes, np_surf_j, np_surf_i, 3))

            total_v_induced_grid = v_induced_grid+wake_influence_grid

            normal_induced_vel = csdl.sum(total_v_induced_grid*bd_normal_vec_i_exp_grid, axes=(3,)) # (num_nodes, num_panels_j, num_panels_i)

            

            


            
            AIC = AIC.set(csdl.slice[:, start_panel_counter_j:stop_panel_counter_j, start_panel_counter:stop_panel_counter], value=normal_induced_vel)
            # AIC[:, start_panel_counter_j:stop_panel_counter_j, start_panel_counter:stop_panel_counter] = normal_induced_vel
            start_panel_counter_j += np_surf_j
        start_panel_counter += np_surf_i

    
    
    '''
    NOTE: csdl for loops work differently from python ones
    - the indices during slicing MUST be the same within the for loop
    - looping iterator is not an integer but a CSDL variable
        - we can't use .get(i) or var[i] unless the operation is done on a CSDL variable
        - ex: surface_names[i] would fail
    
    To capture the interactions across surfaces, we need two loops:
    - outer python for-loop to assemble indices, etc.
        - inner csdl for-loop to assign values to compute interactions
    '''

    return AIC, RHS