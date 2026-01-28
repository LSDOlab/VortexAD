import numpy as np
import csdl_alpha as csdl
import ozone

from VortexAD.core.vlm.unsteady.vlm_ode_function import vlm_ode_function
from VortexAD.core.vlm.unsteady.post_processor import unsteady_post_processor

def unsteady_vlm_solver(orig_mesh_dict, solver_options_dict):

    dt                  = solver_options_dict['dt']
    nt                  = solver_options_dict['nt']
    store_state_history = solver_options_dict['store_state_history']
    reuse_AIC           = solver_options_dict['reuse_AIC']
    compressibility     = solver_options_dict['compressibility']
    rho                 = solver_options_dict['rho']
    sos                 = solver_options_dict['sos']
    ref_area            = solver_options_dict['ref_area']
    ref_chord           = solver_options_dict['ref_chord']
    moment_ref          = solver_options_dict['moment_reference']
    free_wake           = solver_options_dict['free_wake']
    ROM                 = solver_options_dict['ROM']

    # coll_vel_flag       = orig_mesh_dict['coll_vel_flag']
    # coll_vel            = orig_mesh_dict['collocation_velocity']


    solver_options_dict['ROM_orig'] = ROM # saving original
    if ROM: # NOTE: this only works for precomputed basis vectors with POD
        # for Krylov-subspace, we will need to adjust this
        UT, U = ROM[0], ROM[1]
        ROM_shape = UT.shape

    if isinstance(rho, float):
        rho = csdl.Variable(value=np.array([rho]))
    elif isinstance(rho, list):
        rho = csdl.Variable(value=np.array(rho))

    reuse_vars = None
    if reuse_AIC: # not recomputing AIC stuff

        AIC_mu, AIC_sigma = ...
        reuse_vars = {
            'AIC_mu': AIC_mu,
            'AIC_sigma': AIC_sigma
        }
    
    def ode_function(ozone_vars:ozone.ODEVars, reuse_vars=False):
        x_w = ozone_vars.states['x_w']
        gamma_w = ozone_vars.states['gamma_w']

        solver_options_dict['time'] = ozone_vars.dynamic_parameters['time']
        solver_options_dict['time_in_wake'] = ozone_vars.dynamic_parameters['time_in_wake']

        for i in range(num_meshes):
            mesh_name = mesh_names[i]
            orig_mesh_dict[mesh_name]['mesh'] = ozone_vars.dynamic_parameters[mesh_name]
            orig_mesh_dict[mesh_name]['nodal_velocity'] = ozone_vars.dynamic_parameters[mesh_name+'_vel']
            coll_vel_flag = orig_mesh_dict[mesh_name]['coll_vel_flag']
            if coll_vel_flag:
                orig_mesh_dict[mesh_name]['coll_vel'] = ozone_vars.dynamic_parameters[mesh_name+'_coll_vel']

        outputs, d_dt = vlm_ode_function(
            orig_mesh_dict,
            solver_options_dict,
            nt,
            dt,
            ode_states=[x_w.reshape(x_w.shape[1:]), gamma_w.reshape(gamma_w.shape[1:])],
            reuse_vars=reuse_vars
        )

        dxw_dt, dgamma_dt = d_dt[0], d_dt[1]
        ozone_vars.d_states['x_w'] = dxw_dt
        ozone_vars.d_states['gamma_w'] = dgamma_dt

        gamma = outputs['gamma']
        # CL = outputs['total_CL']
        # CDi = outputs['total_CDi']
        steady_panel_force = outputs['steady_panel_force']
        net_gamma = outputs['net_gamma']

        panel_centers = outputs['panel_centers']
        panel_areas = outputs['panel_areas']
        force_eval_pts = outputs['force_eval_pts']
        bound_vec_velocity = outputs['bound_vec_velocity']
        panel_normal = outputs['panel_normal']
        wake_corners = outputs['wake_corners']
        wake_core_radius = outputs['wake_core_radius']

        AIC = outputs['AIC']
        AIC_w = outputs['AIC_w']
        RHS = outputs['RHS']
        BC = outputs['BC']
        wake_influence = outputs['wake_influence']
        dissipation_deriv = outputs['dissipation_deriv']

        ozone_vars.profile_outputs['dxw_dt'] = dxw_dt.reshape((1,)+dxw_dt.shape)

        ozone_vars.profile_outputs['gamma'] = gamma
        ozone_vars.profile_outputs['gamma_w'] = gamma_w
        # ozone_vars.profile_outputs['CL'] = CL
        # ozone_vars.profile_outputs['CDi'] = CDi
        ozone_vars.profile_outputs['steady_panel_force'] = steady_panel_force
        ozone_vars.profile_outputs['net_gamma'] = net_gamma
        ozone_vars.profile_outputs['panel_centers'] = panel_centers
        ozone_vars.profile_outputs['panel_areas'] = panel_areas
        ozone_vars.profile_outputs['force_eval_pts'] = force_eval_pts
        ozone_vars.profile_outputs['bound_vec_velocity'] = bound_vec_velocity
        ozone_vars.profile_outputs['panel_normal'] = panel_normal
        ozone_vars.profile_outputs['wake_corners'] = wake_corners
        ozone_vars.profile_outputs['wake_core_radius'] = wake_core_radius

        ozone_vars.profile_outputs['AIC'] = AIC
        ozone_vars.profile_outputs['AIC_w'] = AIC_w
        ozone_vars.profile_outputs['RHS'] = RHS
        ozone_vars.profile_outputs['BC'] = BC
        ozone_vars.profile_outputs['wake_influence'] = wake_influence
        ozone_vars.profile_outputs['dissipation_deriv'] = dissipation_deriv

        # for name in mesh_names:
        #     ozone_vars.profile_outputs[f'CL_surf_{name}'] = outputs[f'CL_surf_{name}']
        #     ozone_vars.profile_outputs[f'CDi_surf_{name}'] = outputs[f'CDi_surf_{name}']
    
    approach = ozone.approaches.TimeMarching()
    ode_problem = ozone.ODEProblem(ozone.methods.ForwardEuler(), approach)

    mesh_names = list(orig_mesh_dict.keys())
    num_meshes = len(mesh_names)

    meshes = [orig_mesh_dict[name]['mesh'] for name in mesh_names]
    mesh_velocities = [orig_mesh_dict[name]['nodal_velocity'] for name in mesh_names]
    
    time_array = np.arange(0,nt*dt,dt)
    ode_problem.add_dynamic_parameter('time', csdl.Variable(value=time_array))

    time_in_wake = np.zeros((nt, nt))
    for i in range(1,nt):
        time_in_wake[i,:i] = time_array[:i]
        # time_in_wake[i,:i] = time_array[1:i+1]

    time_in_wake_var = csdl.Variable(value=time_in_wake)
    ode_problem.add_dynamic_parameter('time_in_wake',time_in_wake_var)

    nc_list, ns_list = [], []
    ns_panels_list = []
    for i in range(num_meshes):
        mesh_name = mesh_names[i]
        ode_problem.add_dynamic_parameter(mesh_name, meshes[i])
        ode_problem.add_dynamic_parameter(mesh_name + '_vel', mesh_velocities[i])

        nc_list.append(meshes[i].shape[1])
        ns_list.append(meshes[i].shape[2])
        ns_panels_list.append(meshes[i].shape[2]-1)

        coll_vel_flag = orig_mesh_dict[mesh_name]['coll_vel_flag']
        if coll_vel_flag:
            ode_problem.add_dynamic_parameter(mesh_name + '_coll_vel', orig_mesh_dict[mesh_name]['coll_vel'])
    
    num_wake_panels = sum(ns_panels_list) * (nt-1)
    num_wake_nodes = sum(ns_list) * nt

    x_w_0 = csdl.Variable(value=np.zeros((num_wake_nodes,3)))

    start, stop = 0, 0
    for i in range(num_meshes):
        ns = meshes[i].shape[2]
        mesh_TE = meshes[i][0,-1,:]
        mesh_last_two = meshes[i][0,-2:,:]
        bdvtx_TE = 1.25*mesh_last_two[1,:] - 0.25*mesh_last_two[0,:]
        mesh_wake_nodes = ns*nt
        stop += mesh_wake_nodes
        mesh_TE_exp = bdvtx_TE.expand((nt, ns, 3), 'ij->aij').reshape((mesh_wake_nodes, 3))

        x_w_0 = x_w_0.set(csdl.slice[start:stop,:], mesh_TE_exp)
        start += mesh_wake_nodes

    gamma_w_0 = csdl.Variable(value=np.zeros((num_wake_panels)))

    ode_problem.add_state('x_w', x_w_0, store_history = store_state_history)
    ode_problem.add_state('gamma_w', gamma_w_0, store_history = store_state_history)

    step_vector = np.ones(nt-1)*dt
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0., step_vector=step_vector))
    ode_problem.set_function(
        ode_function,
        reuse_vars=reuse_vars
    )

    ode_outputs = ode_problem.solve()

    gamma_w = ode_outputs.states['gamma_w']
    x_w = ode_outputs.states['x_w']

    dxw_dt = ode_outputs.profile_outputs['dxw_dt']

    gamma = ode_outputs.profile_outputs['gamma']
    # CL = ode_outputs.profile_outputs['CL']
    # CDi = ode_outputs.profile_outputs['CDi']
    steady_panel_force = ode_outputs.profile_outputs['steady_panel_force']
    net_gamma = ode_outputs.profile_outputs['net_gamma']

    AIC = ode_outputs.profile_outputs['AIC']
    AIC_w = ode_outputs.profile_outputs['AIC_w']
    RHS = ode_outputs.profile_outputs['RHS']
    BC = ode_outputs.profile_outputs['BC']
    wake_influence = ode_outputs.profile_outputs['wake_influence']
    dissipation_deriv = ode_outputs.profile_outputs['dissipation_deriv']

    panel_centers = ode_outputs.profile_outputs['panel_centers']
    panel_areas = ode_outputs.profile_outputs['panel_areas']
    force_eval_pts = ode_outputs.profile_outputs['force_eval_pts']
    bound_vec_velocity = ode_outputs.profile_outputs['bound_vec_velocity']

    panel_normal = ode_outputs.profile_outputs['panel_normal']
    wake_corners = ode_outputs.profile_outputs['wake_corners']
    wake_core_radius = ode_outputs.profile_outputs['wake_core_radius']


    

    output_dict = {
        'meshes': meshes,
        'mesh_names': mesh_names,
        'gamma': gamma,
        'gamma_w': gamma_w,
        'x_w': x_w,

        'dxw_dt': dxw_dt,

        # needed for post-processing
        'steady_panel_force': steady_panel_force,
        'net_gamma': net_gamma,
        'panel_centers': panel_centers,
        'panel_normal': panel_normal,
        'panel_areas': panel_areas,
        'force_eval_pts': force_eval_pts,
        'bound_vec_velocity': bound_vec_velocity,

        # others
        'wake_corners': wake_corners,
        'wake_core_radius': wake_core_radius,

        'AIC': AIC,
        'AIC_w': AIC_w,
        'RHS': RHS,
        'BC': BC,
        'wake_influence': wake_influence,
        'dissipation_deriv': dissipation_deriv,
    }

    output_dict, surface_output_dict = unsteady_post_processor(
        output_dict,
        solver_options_dict,
        gamma
    )

    surf_CL = [surface_output_dict[name]['CL'] for name in mesh_names]
    surf_CDi = [surface_output_dict[name]['CDi'] for name in mesh_names]

    output_dict['surf_CL'] = surf_CL
    output_dict['surf_CDi'] = surf_CDi

    return output_dict