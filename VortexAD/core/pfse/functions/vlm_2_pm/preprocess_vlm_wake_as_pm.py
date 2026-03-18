import numpy as np
import csdl_alpha as csdl

def preprocess_vlm_wake_as_pm(solver_options_dict, mesh_dict, vectorized_mesh_dict, ode_states, nt):
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

        mesh_dict[key]['num_panels'] = (nc-1)*(ns-1)
        mesh_dict[key]['nc'] = nc
        mesh_dict[key]['ns'] = ns

        # WE DON'T NEED THE BOUND VORTEX MESH FOR THE WAKE
        # bound_vortex_mesh = csdl.Variable(shape=mesh.shape, value=0.)
        # bound_vortex_mesh = bound_vortex_mesh.set(csdl.slice[:,:-1,:,:], value=(3*mesh[:,:-1,:,:] + mesh[:,1:,:,:])/4)
        # bound_vortex_mesh = bound_vortex_mesh.set(csdl.slice[:,-1,:,:], value=mesh[:,-1,:,:] + (mesh[:,-1,:,:] - mesh[:,-2,:,:])/4)

        R1 = mesh[:,:-1,:-1,:]
        R2 = mesh[:,1:,:-1,:]
        R3 = mesh[:,1:,1:,:]
        R4 = mesh[:,:-1,1:,:]
        
        S1 = (R1+R2)/2.
        S2 = (R2+R3)/2.
        S3 = (R3+R4)/2.
        S4 = (R4+R1)/2.
        
        Rc = (R1+R2+R3+R4)/4.
        mesh_dict[key]['panel_center'] = Rc

        x_w_grid = x_w[start_wpc:stop_wpc].reshape((nt, ns, 3))
        wake_corners = csdl.Variable(value=np.zeros((nt-1, ns-1, 4, 3)))
        # wake_corners = wake_corners.set(csdl.slice[:,:,0,:], x_w_grid[:-1, :-1, :])
        # wake_corners = wake_corners.set(csdl.slice[:,:,1,:], x_w_grid[:-1, 1:, :])
        # wake_corners = wake_corners.set(csdl.slice[:,:,2,:], x_w_grid[1:, 1:, :])
        # wake_corners = wake_corners.set(csdl.slice[:,:,3,:], x_w_grid[1:, :-1, :])

        wake_corners = wake_corners.set(csdl.slice[:,:,0,:], x_w_grid[:-1, :-1, :])
        wake_corners = wake_corners.set(csdl.slice[:,:,1,:], x_w_grid[1:, :-1, :])
        wake_corners = wake_corners.set(csdl.slice[:,:,2,:], x_w_grid[1:, 1:, :])
        wake_corners = wake_corners.set(csdl.slice[:,:,3,:], x_w_grid[:-1, 1:, :])

        mesh_dict[key]['wake_corners'] = wake_corners

        D1 = R3-R1
        D2 = R4-R2

        D1D2_cross = csdl.cross(D1, D2, axis=3)
        D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(3,))
        panel_area = D1D2_cross_norm/2.
        mesh_dict[key]['panel_area'] = panel_area

        normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'jkl->jkla')
        mesh_dict[key]['panel_normal'] = normal_vec

        m_dir = S3 - Rc
        m_norm = csdl.norm(m_dir, axes=(3,))
        m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'jkl->jkla')
        l_vec = csdl.cross(m_vec, normal_vec, axis=3)

        panel_center_mod = Rc # for flat s
        mesh_dict[key]['panel_center_mod'] = panel_center_mod

        mesh_dict[key]['panel_x_dir'] = l_vec
        mesh_dict[key]['panel_y_dir'] = m_vec

        s = csdl.Variable(shape=panel_corners.shape, value=0.)
        s = s.set(csdl.slice[:,:,:,:-1,:], value=panel_corners[:,:,:,1:,:] - panel_corners[:,:,:,:-1,:])
        s = s.set(csdl.slice[:,:,:,-1,:], value=panel_corners[:,:,:,0,:] - panel_corners[:,:,:,-1,:])

        l_exp = csdl.expand(l_vec, panel_corners.shape, 'jklm->jklam')
        m_exp = csdl.expand(m_vec, panel_corners.shape, 'jklm->jklam')
        
        S = csdl.norm(s, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0 --> added to the equations instead
        # S = csdl.norm(s, axes=(5,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
        SL = csdl.sum(s*l_exp, axes=(4,))
        SM = csdl.sum(s*m_exp, axes=(4,))

        mesh_dict[key]['S'] = S
        mesh_dict[key]['SL'] = SL
        mesh_dict[key]['SM'] = SM



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