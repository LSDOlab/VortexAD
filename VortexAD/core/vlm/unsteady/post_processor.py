import numpy as np
import csdl_alpha as csdl

from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel

def post_processor(mesh_dict, vectorized_mesh_dict, solver_options_dict, gamma, gamma_w):
    rho = solver_options_dict['rho']
    # getting vectorized net circulation indices
    gamma_ind_pos, gamma_ind_neg = get_net_gamma_indices(mesh_dict)

    # computing net circulation
    net_gamma = csdl.Variable(value=np.zeros(gamma.shape))
    net_gamma = net_gamma.set(csdl.slice[:], gamma)
    net_gamma = net_gamma.set(
        csdl.slice[gamma_ind_neg],
        gamma[gamma_ind_pos] - gamma[gamma_ind_neg]
    )

    bd_vec_induced_vel = compute_bd_vec_induced_vel(
        mesh_dict,
        vectorized_mesh_dict,
        solver_options_dict,
        gamma,
        gamma_w,
        vc=solver_options_dict['core_radius']
    )

    bd_vec_fs_velocity = vectorized_mesh_dict['bound_vec_velocity'][0,:]
    bound_vec = vectorized_mesh_dict['bound_vec'][0,:]
    force_eval_pts = vectorized_mesh_dict['force_eval_pts'][0,:]

    total_velocity = bd_vec_fs_velocity + bd_vec_induced_vel

    net_gamma_exp = net_gamma.expand(gamma.shape + (3,), 'i->ia')
    panel_force = rho * csdl.cross(total_velocity, bound_vec, axis=1) * net_gamma_exp
    ref_point = solver_options_dict['moment_reference']
    if not isinstance(ref_point, csdl.Variable):
        ref_point = csdl.Variable(value=ref_point)
    ref_point_exp = ref_point.expand(panel_force.shape, 'i->ai')
    moment_arm = force_eval_pts - ref_point_exp
    panel_moment = csdl.cross(moment_arm, panel_force, axis=1)

    # computing lift and drag
    alpha = csdl.arctan(bd_vec_fs_velocity[:,2]/bd_vec_fs_velocity[:,0])
    cosa = csdl.cos(alpha)
    sina = csdl.sin(alpha)

    panel_forces_x = panel_force[:,0]
    panel_forces_z = panel_force[:,2]

    panel_lift = panel_forces_z*cosa - panel_forces_x*sina
    panel_drag = panel_forces_z*sina + panel_forces_x*cosa

    total_force = csdl.sum(panel_force, axes=(0,))
    total_moment = csdl.sum(panel_moment, axes=(0,))
    total_lift = csdl.sum(panel_lift)
    total_drag = csdl.sum(panel_drag)

    ref_area = solver_options_dict['ref_area']
    ref_chord = solver_options_dict['ref_chord']

    V_inf_panels = csdl.norm(bd_vec_fs_velocity, axes=(1,))
    V_inf = csdl.average(V_inf_panels)

    total_CL = total_lift/(0.5*rho*V_inf**2*ref_area)
    total_CDi = total_drag/(0.5*rho*V_inf**2*ref_area)
    total_CM = total_moment/(0.5*rho*V_inf**2*ref_area*ref_chord)

    output_dict = {
        'panel_force': panel_force,
        'panel_moment': panel_moment,
        'panel_lift': panel_lift,
        'panel_drag': panel_drag,

        'total_force': total_force,
        'total_moment': total_moment,
        'total_lift': total_lift,
        'total_drag': total_drag,

        'total_CL': total_CL,
        'total_CDi': total_CDi,
        'total_CM': total_CM,

        'net_gamma': net_gamma
    }

    surface_output_dict = compute_surface_outputs(mesh_dict, vectorized_mesh_dict, output_dict, solver_options_dict)

    return output_dict, surface_output_dict

def get_net_gamma_indices(mesh_dict):
    '''
    Function to collect indices for net gamma calculation
    On the surface, we compute a net gamma between vortex rings
    Bar LE panel, we use gamma_{i} - gamma{i+1} because the vortex
    rings overlap
    '''
    mesh_names = mesh_dict.keys()

    # instantiating lists to compute net gamma
    gamma_ind_pos = [] # the positive index
    gamma_ind_neg = [] # the negative index
    surf_offset = 0
    for name in mesh_names:
        nc = mesh_dict[name]['nc']
        ns = mesh_dict[name]['ns']
        num_panels = mesh_dict[name]['num_panels']

        nc_p = nc-1
        ns_p = ns-1

        grid_indices = np.arange(num_panels).reshape((nc_p, ns_p))

        pos_ind_surf = list(grid_indices[1:,:].flatten() + surf_offset)
        neg_ind_surf = list(grid_indices[:-1,:].flatten() + surf_offset)

        gamma_ind_pos.extend(pos_ind_surf)
        gamma_ind_neg.extend(neg_ind_surf)

        surf_offset += num_panels

    return gamma_ind_pos, gamma_ind_neg

def compute_bd_vec_induced_vel(mesh_dict, vectorized_mesh_dict, solver_options_dict, gamma, gamma_w, vc):
    eval_pts = vectorized_mesh_dict['force_eval_pts']
    body_panel_normal = vectorized_mesh_dict['panel_normal']  # NOTE: CHECK IF THIS IS SUPPOSED TO BE A DIFFERENT NORMAL VECTOR, NOT THE SAME AS THE COLLOCATION ONE
    batch_size = solver_options_dict['partition_size']
    batch_dims = [1, None, None]

    # body panel influence
    body_panel_corners = vectorized_mesh_dict['panel_corners']

    num_body_panels = body_panel_corners.shape[1]
    batch_size_surf = batch_size
    if batch_size is None:
        batch_size_surf = num_body_panels

    AIC_batch_func = csdl.experimental.batch_function(
        batched_induced_vel,
        batch_size=batch_size_surf,
        batch_dims=batch_dims
    )

    body_ind_vel = AIC_batch_func(
        eval_pts,
        body_panel_corners,
        gamma,
        vc=vc
    )

    # wake panel influence
    wake_panel_corners = vectorized_mesh_dict['wake_corners']

    num_wake_panels = wake_panel_corners.shape[1]
    batch_size_wake = batch_size
    if batch_size is None:
        batch_size_wake = num_wake_panels

    AIC_batch_func = csdl.experimental.batch_function(
        batched_induced_vel,
        batch_size=batch_size_wake,
        batch_dims=batch_dims
    )

    wake_ind_vel = AIC_batch_func(
        eval_pts,
        wake_panel_corners,
        gamma_w,
        vc=vc
    )

    ind_vel = body_ind_vel + wake_ind_vel
    ind_vel = ind_vel.reshape((num_body_panels, 3)) # removing stacking dimension + num_nodes initially

    return ind_vel

def batched_induced_vel(eval_pt, panel_corners, gamma, vc=1.e-6):
    num_nodes = eval_pt.shape[0]
    num_eval_pts = eval_pt.shape[1]
    num_induced_pts = panel_corners.shape[1]
    num_interactions = num_eval_pts*num_induced_pts
    num_corners = panel_corners.shape[2]
    
    expanded_shape = (num_nodes, num_eval_pts, num_induced_pts, num_corners, 3)
    vectorized_shape = (num_nodes, num_interactions, num_corners, 3)

    # ============ expanding across columns ============
    eval_point_exp = csdl.expand(eval_pt, expanded_shape, 'ijk->ijabk')
    eval_point_exp_vec = eval_point_exp.reshape(vectorized_shape)

    # ============ expanding across rows ============
    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'ijkl->iajkl')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    num_edges = num_corners

    AIC_vel_vec_list = []
    for  i in range(num_edges-1):
        asdf = compute_vortex_line_ind_vel(
            panel_corners_exp_vec[:,:,i], 
            panel_corners_exp_vec[:,:,i+1], 
            eval_point_exp_vec[:,:,0], 
            mode='wake', 
            vc=vc
        )
        AIC_vel_vec_list.append(asdf)
    asdf = compute_vortex_line_ind_vel(
        panel_corners_exp_vec[:,:,-1], 
        panel_corners_exp_vec[:,:,0], 
        eval_point_exp_vec[:,:,0], 
        mode='wake', 
        vc=vc
    )
    AIC_vel_vec_list.append(asdf)
    AIC_vel_vec = sum(AIC_vel_vec_list)[0,:]

    ind_vel = csdl.Variable(value=np.zeros((num_eval_pts, 3)))
    for i in range(3):
        ind_vel = ind_vel.set(
            csdl.slice[:,i],
            csdl.sum(AIC_vel_vec[:,i]*gamma)
        )

    return ind_vel

def compute_surface_outputs(mesh_dict, vectorized_mesh_dict, output_dict, solver_options_dict):
    surface_output_dict = {}
    mesh_names = mesh_dict.keys()

    rho = solver_options_dict['rho']
    
    pcs, pce = 0, 0 # panel counter start and end
    for name in mesh_names:
        nc, ns = mesh_dict[name]['nc'], mesh_dict[name]['ns']
        num_panels = mesh_dict[name]['num_panels']
        pce += num_panels
        
        panel_areas = mesh_dict[name]['panel_area']
        surf_area = csdl.sum(panel_areas)
        MAC = mesh_dict[name]['MAC']

        panel_L = output_dict['panel_lift'][pcs:pce]
        panel_Di = output_dict['panel_drag'][pcs:pce]
        panel_M = output_dict['panel_moment'][pcs:pce,:]

        surf_L = csdl.sum(panel_L)
        surf_Di = csdl.sum(panel_Di)
        surf_M = csdl.sum(panel_M, axes=(0,))

        bd_vec_fs_velocity = mesh_dict[name]['bound_vector_velocity'][0,:]
        V_inf_panels = csdl.norm(bd_vec_fs_velocity, axes=(1,))
        V_inf = csdl.average(V_inf_panels)

        surf_CL = surf_L / (0.5*rho*V_inf**2*surf_area)
        surf_CDi = surf_Di / (0.5*rho*V_inf**2*surf_area)
        surf_CM = surf_M / (0.5*rho*V_inf**2*surf_area*MAC)

        sub_dict = {
            'L': surf_L,
            'Di': surf_Di,
            'M': surf_M,

            'CL': surf_CL,
            'CDi': surf_CDi,
            'CM': surf_CM,
        }

        '''
        TODO:
        - compute center of pressure?
        - compute sectional center of pressure?
        '''

        surface_output_dict[name] = sub_dict
        pcs += num_panels
        
    return surface_output_dict