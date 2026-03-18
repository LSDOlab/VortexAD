import numpy as np
import csdl_alpha as csdl
import ozone

from VortexAD.core.pfse.pfse_ode_function import pfse_ode_function

def pfse_solver(orig_mesh_dict, solver_options_dict):

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

        outputs, d_dt = pfse_ode_function(
            orig_mesh_dict,
            solver_options_dict,
            nt,
            dt, 
            ode_states=[x_w.reshape(x_w.shape[1:]), mu_w.reshape(mu_w.shape[1:])],
            reuse_vars=reuse_vars,
        )
        dxw_dt, dmuw_dt = d_dt[0], d_dt[1]
        ozone_vars.d_states['x_w'] = dxw_dt
        ozone_vars.d_states['mu_w'] = dmuw_dt


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
    ode_problem.add_dynamic_parameter('points', orig_mesh_dict['points'])
    ode_problem.add_dynamic_parameter('nodal_velocity', orig_mesh_dict['nodal_velocity'])
    if coll_vel_flag:
        ode_problem.add_dynamic_parameter('coll_vel', orig_mesh_dict['collocation_velocity'])

    # VLM
    mesh_names = list(orig_mesh_dict.keys())
    num_meshes = len(mesh_names)

    meshes = [orig_mesh_dict[name]['mesh'] for name in mesh_names]
    mesh_velocities = [orig_mesh_dict[name]['nodal_velocity'] for name in mesh_names]

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
    # endregion

    # region setting up initial conditions
    # panel method IC
    TE_node_indices = orig_mesh_dict['TE_node_indices']
    num_TE_pts = len(TE_node_indices)
    TE_edges = orig_mesh_dict['TE_edges']
    num_TE_panels = len(TE_edges)
    num_wake_panels_PM = num_TE_panels * (nt-1)

    mu_w_0_PM = csdl.Variable(value=np.zeros((num_wake_panels_PM))) # wake doublet initial condition
    points = orig_mesh_dict['points']
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

    x_w_0 = csdl.concatenate((x_w_0_PM, x_w_0_VLM), axis=1)
    mu_w_0 = csdl.concatenate((mu_w_0_PM, mu_w_0_VLM), axis=1)

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

