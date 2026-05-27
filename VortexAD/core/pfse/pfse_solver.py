import numpy as np
import csdl_alpha as csdl
import ozone

from VortexAD.core.pfse.pfse_ode_function import pfse_ode_function

from VortexAD.core.pm.unsteady.post_processor import unsteady_post_processor as PM_postproc
from VortexAD.core.vlm.unsteady.post_processor import unsteady_post_processor as VLM_postproc

def pfse_solver(pm_orig_mesh_dict, vlm_orig_mesh_dict, solver_options_dict):

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
    dissipation_flag    = solver_options_dict['dissipation']

    if isinstance(rho, float):
        rho = csdl.Variable(value=np.array([rho]))
    elif isinstance(rho, list):
        rho = csdl.Variable(value=np.array(rho))

    reuse_vars = None
    def ode_function(ozone_vars:ozone.ODEVars, reuse_vars=False):
        x_w = ozone_vars.states['x_w']
        mu_w = ozone_vars.states['mu_w']

        # dynamic parameters for the WAKE
        solver_options_dict['time'] = ozone_vars.dynamic_parameters['time']
        solver_options_dict['time_in_wake'] = ozone_vars.dynamic_parameters['time_in_wake']
        solver_options_dict['velocity_activation'] = ozone_vars.dynamic_parameters['velocity_activation']
        solver_options_dict['kutta_activation'] = ozone_vars.dynamic_parameters['kutta_activation']
        if dissipation_flag:
            solver_options_dict['dissipation_activation'] = ozone_vars.dynamic_parameters['dissipation_activation']

        # pm mesh dynamic parameters
        pm_orig_mesh_dict['points'] = ozone_vars.dynamic_parameters['points']
        pm_orig_mesh_dict['nodal_velocity'] = ozone_vars.dynamic_parameters['nodal_velocity']
        coll_vel_flag = pm_orig_mesh_dict['coll_vel_flag']
        if coll_vel_flag:
            pm_orig_mesh_dict['collocation_velocity'] = ozone_vars.dynamic_parameters['coll_vel']

        # vlm mesh dynamic parameters
        vlm_meshes = list(vlm_orig_mesh_dict.keys())
        num_meshes = len(vlm_meshes)
        for i in range(num_meshes):
            mesh_name = mesh_names[i]
            vlm_orig_mesh_dict[mesh_name]['mesh'] = ozone_vars.dynamic_parameters[mesh_name]
            vlm_orig_mesh_dict[mesh_name]['nodal_velocity'] = ozone_vars.dynamic_parameters[mesh_name+'_vel']
            coll_vel_flag = vlm_orig_mesh_dict[mesh_name]['coll_vel_flag']
            if coll_vel_flag:
                vlm_orig_mesh_dict[mesh_name]['coll_vel'] = ozone_vars.dynamic_parameters[mesh_name+'_coll_vel']
        

        outputs, pm_outputs, vlm_outputs, d_dt = pfse_ode_function(
            pm_orig_mesh_dict,
            vlm_orig_mesh_dict,
            solver_options_dict,
            nt,
            dt, 
            ode_states=[x_w.reshape(x_w.shape[1:]), mu_w.reshape(mu_w.shape[1:])],
            reuse_vars=reuse_vars,
        )
        dxw_dt, dmuw_dt = d_dt[0], d_dt[1]
        ozone_vars.d_states['x_w'] = dxw_dt
        ozone_vars.d_states['mu_w'] = dmuw_dt

        mu = outputs['mu']
        # sigma = outputs['sigma']
        ozone_vars.profile_outputs['mu'] = mu

        # panel method post-processing outputs
        ozone_vars.profile_outputs['sigma'] = pm_outputs['sigma']
        ozone_vars.profile_outputs['panel_normal_PM'] = pm_outputs['panel_normal']
        ozone_vars.profile_outputs['panel_area_PM'] = pm_outputs['panel_area']
        ozone_vars.profile_outputs['panel_center_PM'] = pm_outputs['panel_center']
        ozone_vars.profile_outputs['nodal_cp_velocity_PM'] = pm_outputs['nodal_cp_velocity']
        ozone_vars.profile_outputs['Cp_static'] = pm_outputs['Cp_static']
        ozone_vars.profile_outputs['ql'] = pm_outputs['ql']
        ozone_vars.profile_outputs['qm'] = pm_outputs['qm']
        ozone_vars.profile_outputs['qn'] = pm_outputs['qn']

        # vlm post-processing outputs
        ozone_vars.profile_outputs['net_gamma'] = vlm_outputs['net_gamma']
        ozone_vars.profile_outputs['steady_panel_force_VLM'] = vlm_outputs['steady_panel_force']
        ozone_vars.profile_outputs['panel_areas_VLM'] = vlm_outputs['panel_areas']
        ozone_vars.profile_outputs['panel_normal_VLM'] = vlm_outputs['panel_normal']
        ozone_vars.profile_outputs['force_eval_pts'] = vlm_outputs['force_eval_pts']
        ozone_vars.profile_outputs['bound_vec_velocity'] = vlm_outputs['bound_vec_velocity']
        


    # setting up ODE approach
    approach = ozone.approaches.TimeMarching()
    ode_problem = ozone.ODEProblem(ozone.methods.ForwardEuler(), approach)

    # region setting up wake activation dynamic parameters
    time_array = np.arange(0,nt*dt,dt)
    ode_problem.add_dynamic_parameter('time', csdl.Variable(value=time_array))

    time_in_wake = np.zeros((nt, nt))
    for i in range(1,nt):
        time_in_wake[i,-i:] = time_array[1:(i+1)]
        # time_in_wake[i,:i] = time_array[1:i+1]
    
    velocity_activation = np.zeros((nt, nt))
    for i in range(nt):
        # velocity_activation[i,:(i+1)] = 1.
        velocity_activation[i,-(i+1):] = 1.

    kutta_activation = np.zeros((nt, nt-1))
    for i in range(0,nt-1):
        kutta_activation[i,-(i+1)] = 1.

    time_in_wake_var = csdl.Variable(value=time_in_wake)
    ode_problem.add_dynamic_parameter('time_in_wake',time_in_wake_var)

    vel_activation_var = csdl.Variable(value=velocity_activation)
    ode_problem.add_dynamic_parameter('velocity_activation', vel_activation_var)

    kutta_activation_var = csdl.Variable(value=kutta_activation)
    ode_problem.add_dynamic_parameter('kutta_activation', kutta_activation_var)

    if dissipation_flag:
        dissipation_activation = np.zeros((nt, nt-1))
        for i in range(1,nt):
            dissipation_activation[i,-i:] = 1
            # dissipation_activation[i,:] = diss_val
        dissipation_activation_var = csdl.Variable(value=dissipation_activation)
        ode_problem.add_dynamic_parameter('dissipation_activation', dissipation_activation_var)
    # endregion

    # region setting up mesh and velocity dynamic parameters
    # panel method
    ode_problem.add_dynamic_parameter('points', pm_orig_mesh_dict['points'])
    ode_problem.add_dynamic_parameter('nodal_velocity', pm_orig_mesh_dict['nodal_velocity'])
    coll_vel_flag = pm_orig_mesh_dict['coll_vel_flag']
    if coll_vel_flag:
        ode_problem.add_dynamic_parameter('coll_vel', pm_orig_mesh_dict['collocation_velocity'])

    # VLM
    mesh_names = list(vlm_orig_mesh_dict.keys())
    num_meshes = len(mesh_names)

    meshes = [vlm_orig_mesh_dict[name]['mesh'] for name in mesh_names]
    mesh_velocities = [vlm_orig_mesh_dict[name]['nodal_velocity'] for name in mesh_names]

    nc_list, ns_list = [], []
    ns_panels_list = []
    for i in range(num_meshes):
        mesh_name = mesh_names[i]
        ode_problem.add_dynamic_parameter(mesh_name, meshes[i])
        ode_problem.add_dynamic_parameter(mesh_name + '_vel', mesh_velocities[i])

        nc_list.append(meshes[i].shape[1])
        ns_list.append(meshes[i].shape[2])
        ns_panels_list.append(meshes[i].shape[2]-1)

        coll_vel_flag = vlm_orig_mesh_dict[mesh_name]['coll_vel_flag']
        if coll_vel_flag:
            ode_problem.add_dynamic_parameter(mesh_name + '_coll_vel', vlm_orig_mesh_dict[mesh_name]['coll_vel'])

    
    num_wake_panels = sum(ns_panels_list) * (nt-1)
    num_wake_nodes = sum(ns_list) * nt
    # endregion

    # region setting up initial conditions
    # panel method IC
    TE_node_indices = pm_orig_mesh_dict['TE_node_indices']
    num_TE_pts = len(TE_node_indices)
    TE_edges = pm_orig_mesh_dict['TE_edges']
    num_TE_panels = len(TE_edges)
    num_wake_panels_PM = num_TE_panels * (nt-1)

    mu_w_0_PM = csdl.Variable(value=np.zeros((num_wake_panels_PM))) # wake doublet initial condition
    points = pm_orig_mesh_dict['points']
    TE_pts = points[:, list(TE_node_indices), :]
    if TE_pts.shape[0] == 1:
        x_w_0_PM = csdl.expand(TE_pts[0,:], TE_pts.shape[1:], 'ij->aij')
    else:
        # x_w_0 = TE_pts.reshape((np.prod(TE_pts.shape[:2]),3))
        x_w_0_PM = TE_pts[::-1,:,:].reshape((np.prod(TE_pts.shape[:2]),3))
        # NOTE: this reversal is to reorder the initial condition points for actuating geometries

    # VLM IC
    num_wake_panels = sum(ns_panels_list) * (nt-1)
    num_wake_nodes = sum(ns_list) * nt
    mu_w_0_VLM = csdl.Variable(value=np.zeros((num_wake_panels)))
    x_w_0_VLM = csdl.Variable(value=np.zeros((num_wake_nodes,3)))

    start, stop = 0, 0
    # this for loop uses the TE across all timesteps as an IC
    for i in range(num_meshes):
        # meshes[i] has shape (nt, nc, ns, 3)
        ns = meshes[i].shape[2]
        mesh_last_two = meshes[i][:,-2:,:] # shape of (nt, 2, ns, 3)
        bdvtx_TE = 1.25*mesh_last_two[:,1,:] - 0.25*mesh_last_two[:,0,:] # shape of (nt, ns, 3)
        mesh_wake_nodes = ns*nt
        stop += mesh_wake_nodes
        # mesh_TE_exp = bdvtx_TE.reshape((mesh_wake_nodes, 3))
        mesh_TE_exp = bdvtx_TE[::-1,:].reshape((mesh_wake_nodes, 3))
        # this reversal is correct; the last entry in the wake array is the first to shed

        x_w_0_VLM = x_w_0_VLM.set(csdl.slice[start:stop,:], mesh_TE_exp)
        start += mesh_wake_nodes

    x_w_0 = csdl.concatenate((x_w_0_PM, x_w_0_VLM), axis=0)
    mu_w_0 = csdl.concatenate((mu_w_0_PM, mu_w_0_VLM), axis=0)

    ode_problem.add_state('x_w', x_w_0, store_history = store_state_history)
    ode_problem.add_state('mu_w', mu_w_0, store_history = store_state_history)

    # endregion

    step_vector = np.ones(nt-1)*dt
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0., step_vector=step_vector))
    ode_problem.set_function(
        ode_function,
        reuse_vars=reuse_vars
    )

    ode_outputs = ode_problem.solve()

    mu_w = ode_outputs.states['mu_w']
    x_w = ode_outputs.states['x_w']

    mu = ode_outputs.profile_outputs['mu']
    sigma = ode_outputs.profile_outputs['sigma']

    # panel method outputs
    panel_normal_PM = ode_outputs.profile_outputs['panel_normal_PM']
    panel_area_PM = ode_outputs.profile_outputs['panel_area_PM']
    panel_center_PM = ode_outputs.profile_outputs['panel_center_PM']
    nodal_cp_velocity_PM = ode_outputs.profile_outputs['nodal_cp_velocity_PM']
    Cp_static = ode_outputs.profile_outputs['Cp_static']
    ql = ode_outputs.profile_outputs['ql']
    qm = ode_outputs.profile_outputs['qm']
    qn = ode_outputs.profile_outputs['qn']
    # wake_vel = ode_outputs.profile_outputs['wake_vel']

    # VLM outputs
    meshes
    net_gamma = ode_outputs.profile_outputs['net_gamma']
    steady_panel_force = ode_outputs.profile_outputs['steady_panel_force_VLM']
    panel_areas = ode_outputs.profile_outputs['panel_areas_VLM']
    panel_normal = ode_outputs.profile_outputs['panel_normal_VLM']
    force_eval_pts = ode_outputs.profile_outputs['force_eval_pts']
    bound_vec_velocity = ode_outputs.profile_outputs['bound_vec_velocity']


    
    # PM post-processing
    upp_mesh_dict = {
        'panel_normal': panel_normal_PM,
        'panel_area': panel_area_PM,
        'panel_center': panel_center_PM,
        'coll_point_velocity': nodal_cp_velocity_PM, # NOTE: CHECK AND FIX THIS (doesn't include actuation velocity)
        # '': ,
        # '': ,
    }
    num_PM_panels = panel_normal_PM.shape[1]
    mu_PM = mu[:,:num_PM_panels]
    PM_output_dict = {
        'mesh': points,
        # 'mu': mu,
        # 'x_w': x_w,
        # 'mu_w': mu_w,
        'Cp_static': Cp_static,
        'ql': ql,
        'qm': qm,
        'qn': qn,
        # 'wake_vel': wake_vel,
        # 'AIC_mu': AIC_mu,
        # 'AIC_sigma': AIC_sigma,
        # 'AIC_mu_wake': AIC_mu_wake,
    }

    num_nodes = 1 # not sure why this is the case, this was what the unsteady panel method did
    PM_output_dict = PM_postproc(upp_mesh_dict, PM_output_dict, mu_PM, num_nodes, dt, nt, 
                                compressibility=compressibility, rho=rho, constant_geometry=reuse_AIC, 
                                ref_point=moment_ref, ref_area=ref_area, ref_chord=ref_chord, sos=sos)
    
    # VLM post-processing
    gamma = mu[:,num_PM_panels:]
    VLM_meshes = meshes
    VLM_mesh_names = mesh_names
    VLM_output_dict = {
        'meshes': VLM_meshes, 
        'mesh_names': VLM_mesh_names,

        'net_gamma': net_gamma,
        'steady_panel_force': steady_panel_force,
        'panel_areas': panel_areas,
        'panel_normal': panel_normal,
        'force_eval_pts': force_eval_pts,
        'bound_vec_velocity': bound_vec_velocity,
        # '': ,
        # '': ,
        # '': ,
    }

    VLM_output_dict, VLM_surf_output_dict = VLM_postproc(
        VLM_output_dict,
        solver_options_dict,
        gamma
    )

    # PM outputs ONLY
    Cp = PM_output_dict['Cp']

    # getting TOTAL quantities
    total_lift = PM_output_dict['L'] + VLM_output_dict['total_lift']
    total_drag = PM_output_dict['Di'] + VLM_output_dict['total_drag']
    total_force = PM_output_dict['F'] + VLM_output_dict['total_force']
    total_moment = PM_output_dict['M'] + VLM_output_dict['total_moment']

    # concatenating variables that are vectorized ACROSS PANELS
    panel_lift = csdl.concatenate(
        (PM_output_dict['L_panel'], VLM_output_dict['panel_lift']), 
        axis=1
    )
    panel_drag = csdl.concatenate(
        (PM_output_dict['Di_panel'], VLM_output_dict['panel_drag']), 
        axis=1
    )
    panel_forces = csdl.concatenate(
        (PM_output_dict['panel_forces'], VLM_output_dict['panel_force']), 
        axis=1
    )
    panel_moments = csdl.concatenate(
        (PM_output_dict['panel_moments'], VLM_output_dict['panel_moment']), 
        axis=1
    )

    num_vlm_surf = len(VLM_mesh_names)
    num_tot_surf = 1+num_vlm_surf
    # initializing surface quantities
    surface_CL = csdl.Variable(shape=(nt,num_tot_surf), value=0.)
    surface_CDi = csdl.Variable(shape=(nt,num_tot_surf), value=0.)
    surface_CM = csdl.Variable(shape=(nt,num_tot_surf,3), value=0.)
    surface_L = csdl.Variable(shape=(nt,num_tot_surf), value=0.)
    surface_Di = csdl.Variable(shape=(nt,num_tot_surf), value=0.)
    surface_M = csdl.Variable(shape=(nt,num_tot_surf,3), value=0.)

    # setting panel method value
    surface_CL = surface_CL.set(csdl.slice[:,0], PM_output_dict['CL'])
    surface_CDi = surface_CDi.set(csdl.slice[:,0], PM_output_dict['CDi'])
    surface_CM = surface_CM.set(csdl.slice[:,0,:], PM_output_dict['CM'])
    surface_L = surface_L.set(csdl.slice[:,0], PM_output_dict['L'])
    surface_Di = surface_Di.set(csdl.slice[:,0], PM_output_dict['Di'])
    surface_M = surface_M.set(csdl.slice[:,0,:], PM_output_dict['M'])

    # setting VLM values via loop through surfaces
    vlm_surf_names = list(VLM_surf_output_dict.keys())
    for i in range(num_vlm_surf):
        surf_name = vlm_surf_names[i]
        surface_CL = surface_CL.set(csdl.slice[:,i+1], VLM_surf_output_dict[surf_name]['CL'])
        surface_CDi = surface_CDi.set(csdl.slice[:,i+1], VLM_surf_output_dict[surf_name]['CDi'])
        surface_CM = surface_CM.set(csdl.slice[:,i+1,:], VLM_surf_output_dict[surf_name]['CM'])
        surface_L = surface_L.set(csdl.slice[:,i+1], VLM_surf_output_dict[surf_name]['L'])
        surface_Di = surface_Di.set(csdl.slice[:,i+1], VLM_surf_output_dict[surf_name]['Di'])
        surface_M = surface_M.set(csdl.slice[:,i+1,:], VLM_surf_output_dict[surf_name]['M'])

    output_dict = {
        
        'mu': mu,
        'sigma': sigma,
        'mu_w': mu_w,
        'x_w': x_w,
        'Cp': Cp,
        'Cp_static': Cp_static,

        # total quantities
        'total_lift': total_lift,
        'total_drag': total_drag,
        'total_force': total_force,
        'total_moment': total_moment,
        # '': ,
        
        # surface quantities
        'surf_CL': surface_CL,
        'surf_CDi': surface_CDi,
        'surf_CM': surface_CM,
        'surf_L': surface_L,
        'surf_Di': surface_Di,
        'surf_M': surface_M,
        
        
        # per panel quantities
        'panel_lift': panel_lift,
        'panel_drag': panel_drag,
        'panel_forces': panel_forces,
        'panel_moments': panel_moments,

        # others
        'steady_panel_force_VLM': steady_panel_force,
        'net_gamma_VLM': net_gamma,
        'panel_areas_VLM': panel_areas,
        # '': ,
        # '': ,
        # '': ,
    }

    return output_dict

'''
Notes:
outputs that make sense to concatenate:
- force terms:
    - lift and drag (num_surfaces)
    - coefficients (num_surfaces)
    - panel forces and moments
    - panel lift and drag
'''