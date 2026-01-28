import numpy as np
import csdl_alpha as csdl

def wake_pre_processor(solver_options_dict, mesh_dict, vectorized_mesh_dict, ode_states, nt):
    mesh_names = list(mesh_dict.keys())
    num_surfaces = len(mesh_names)
    x_w = ode_states[0]
    gamma_w = ode_states[1]

    wake_points = []
    wake_panels = []
    wake_nc = []
    wake_ns = []
    total_points = []
    total_panels = [] 

    # setting up time & vortex core stuff
    # NOTE: DO  THIS AHEAD OF TIME ONCE IN THE ODE FUNCTION
    # time_deficit = csdl.Variable(value=np.arange(0,nt*dt,dt)[::-1]*-1) + time
    # time_in_wake = csdl.maximum(time_deficit, csdl.Variable(value=np.zeros(time_deficit.shape)), rho=100.)

    rc0 = solver_options_dict['core_radius']
    nu = solver_options_dict['nu']
    time_in_wake = solver_options_dict['time_in_wake'][0,:] # removing num_nodes
    vc_parameters = solver_options_dict['vc_parameters']
    alpha = vc_parameters[0]
    a1 = vc_parameters[1]
    bqs = vc_parameters[2]

    gamma_dummy = 0 # removing dependence from gamma
    delta_nu = nu + a1*gamma_dummy

    rc = (rc0**2 + 4*alpha*delta_nu*time_in_wake)**0.5

    start_wpc, stop_wpc = 0, 0 # wake point counter

    for key in mesh_names:
        mesh = mesh_dict[key]['mesh']
        ns = mesh_dict[key]['ns']
        ns = mesh.shape[2]

        num_wake_points = ns*nt
        num_wake_panels = (ns-1)*(nt-1)
        wake_points.append(num_wake_points)
        wake_panels.append(num_wake_panels)
        stop_wpc += num_wake_points

        x_w_grid = x_w[start_wpc:stop_wpc].reshape((nt, ns, 3))
        wake_corners = csdl.Variable(value=np.zeros((nt-1, ns-1, 4, 3)))
        wake_corners = wake_corners.set(csdl.slice[:,:,0,:], x_w_grid[:-1, :-1, :])
        wake_corners = wake_corners.set(csdl.slice[:,:,1,:], x_w_grid[:-1, 1:, :])
        wake_corners = wake_corners.set(csdl.slice[:,:,2,:], x_w_grid[1:, 1:, :])
        wake_corners = wake_corners.set(csdl.slice[:,:,3,:], x_w_grid[1:, :-1, :])

        # wake_corners = wake_corners.set(csdl.slice[:,:,0,:], x_w_grid[:-1, :-1, :])
        # wake_corners = wake_corners.set(csdl.slice[:,:,1,:], x_w_grid[1:, :-1, :])
        # wake_corners = wake_corners.set(csdl.slice[:,:,2,:], x_w_grid[1:, 1:, :])
        # wake_corners = wake_corners.set(csdl.slice[:,:,3,:], x_w_grid[:-1, 1:, :])

        mesh_dict[key]['wake_corners'] = wake_corners

        rc_exp = csdl.expand(rc, (nt, ns-1), 'i->ia')

        vortex_core_radius = csdl.Variable(value=np.zeros(wake_corners.shape[:-1]))
        vortex_core_radius = vortex_core_radius.set(csdl.slice[:,:,0], rc_exp[:-1,:]) # point 0 to 1 based on wake corners above
        vortex_core_radius = vortex_core_radius.set(csdl.slice[:,:,1], rc_exp[1:,:]) # point 1 to 2 based on wake corners above
        vortex_core_radius = vortex_core_radius.set(csdl.slice[:,:,2], rc_exp[1:,:]) # point 2 to 3 based on wake corners above
        vortex_core_radius = vortex_core_radius.set(csdl.slice[:,:,3], rc_exp[1:,:]) # point 3 to 0 based on wake corners above

        mesh_dict[key]['wake_core_radius'] = vortex_core_radius

        start_wpc += num_wake_points
    
    num_tot_wake_panels = sum(wake_panels)
    num_nodes = 1 # HARDCODED BC UNSTEADY SOLVER SOLVES ONE GEOMETRY
    total_wake_corners = csdl.Variable(value=np.zeros((num_nodes, num_tot_wake_panels, 4, 3))) 
    vortex_core_radius = csdl.Variable(value=np.zeros((num_nodes, num_tot_wake_panels, 4)))

    cs_panels, ce_panels = 0, 0 # panel counter
    for i in range(num_surfaces):
        key = mesh_names[i]
        num_wake_panels = wake_panels[i]
        ce_panels += num_wake_panels

        total_wake_corners = total_wake_corners.set(
            csdl.slice[:,cs_panels:ce_panels,:,:],
            mesh_dict[key]['wake_corners'].reshape((1, num_wake_panels, 4, 3))
        )

        vortex_core_radius = vortex_core_radius.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            mesh_dict[key]['wake_core_radius'].reshape((1, num_wake_panels, 4))
        )

        cs_panels += num_wake_panels
    
    vectorized_mesh_dict['wake_corners'] = total_wake_corners
    vectorized_mesh_dict['wake_core_radius'] = vortex_core_radius

    return mesh_dict, vectorized_mesh_dict