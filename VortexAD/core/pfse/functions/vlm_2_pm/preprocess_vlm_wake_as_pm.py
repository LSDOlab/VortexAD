import numpy as np
import csdl_alpha as csdl

def preprocess_vlm_wake_as_pm(solver_options_dict, vlm_mesh_dict, pm_wake_mesh_dict, x_w, nt):
    '''
    This function does a couple of things:
    - computes the parameters needed for both panel method and VLM interactions
    - vectorizes these parameters ACROSS THE ENTIRE WAKE
        - panel method wake + VLM wake
    '''
    
    mesh_names = list(vlm_mesh_dict.keys())
    num_vlm_surfaces = len(mesh_names)
    x_w_vlm = x_w
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

    wake_mesh_dict = {name: {} for name in mesh_names}
    num_nodes = 1
    for key in mesh_names:
        surf_mesh = vlm_mesh_dict[key]['mesh']
        ns = surf_mesh.shape[2]
        # x_w_grid = x_w_vlm[start_wpc:stop_wpc].reshape((nt, ns, 3))

        num_wake_points = ns*nt
        num_wake_panels = (ns-1)*(nt-1)
        wake_points.append(num_wake_points)
        wake_panels.append(num_wake_panels)
        stop_wpc += num_wake_points

        mesh = x_w_vlm[:,start_wpc:stop_wpc,:].reshape((num_nodes, nt, ns, 3))
        x_w_grid = mesh

        wake_mesh_dict[key]['num_panels'] = (nt-1)*(ns-1)
        wake_mesh_dict[key]['nc'] = nt
        wake_mesh_dict[key]['ns'] = ns

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
        wake_mesh_dict[key]['panel_center'] = Rc

        wake_corners = csdl.Variable(value=np.zeros((num_nodes, nt-1, ns-1, 4, 3)))
        # wake_corners = wake_corners.set(csdl.slice[:,:,0,:], x_w_grid[:-1, :-1, :])
        # wake_corners = wake_corners.set(csdl.slice[:,:,1,:], x_w_grid[:-1, 1:, :])
        # wake_corners = wake_corners.set(csdl.slice[:,:,2,:], x_w_grid[1:, 1:, :])
        # wake_corners = wake_corners.set(csdl.slice[:,:,3,:], x_w_grid[1:, :-1, :])

        wake_corners = wake_corners.set(csdl.slice[:,:,:,0,:], x_w_grid[:,:-1, :-1, :])
        wake_corners = wake_corners.set(csdl.slice[:,:,:,1,:], x_w_grid[:,1:, :-1, :])
        wake_corners = wake_corners.set(csdl.slice[:,:,:,2,:], x_w_grid[:,1:, 1:, :])
        wake_corners = wake_corners.set(csdl.slice[:,:,:,3,:], x_w_grid[:,:-1, 1:, :])

        wake_mesh_dict[key]['wake_corners'] = wake_corners

        D1 = R3-R1
        D2 = R4-R2

        D1D2_cross = csdl.cross(D1, D2, axis=3)
        D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(3,)) + 1.e-12
        panel_area = D1D2_cross_norm/2.
        wake_mesh_dict[key]['panel_area'] = panel_area

        normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'jkl->jkla')
        wake_mesh_dict[key]['panel_normal'] = normal_vec

        m_dir = S3 - Rc
        m_norm = csdl.norm(m_dir, axes=(3,))
        m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'jkl->jkla')
        l_vec = csdl.cross(m_vec, normal_vec, axis=3)

        # panel_center_mod = Rc # for flat s
        # wake_mesh_dict[key]['panel_center_mod'] = panel_center_mod

        wake_mesh_dict[key]['panel_x_dir'] = l_vec
        wake_mesh_dict[key]['panel_y_dir'] = m_vec

        s = csdl.Variable(shape=wake_corners.shape, value=0.)
        s = s.set(csdl.slice[:,:,:,:-1,:], value=wake_corners[:,:,:,1:,:] - wake_corners[:,:,:,:-1,:])
        s = s.set(csdl.slice[:,:,:,-1,:], value=wake_corners[:,:,:,0,:] - wake_corners[:,:,:,-1,:])

        l_exp = csdl.expand(l_vec, wake_corners.shape, 'jklm->jklam')
        m_exp = csdl.expand(m_vec, wake_corners.shape, 'jklm->jklam')

        S = csdl.norm(s+1.e-6, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0 --> added to the equations instead
        # S = csdl.norm(s, axes=(5,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
        SL = csdl.sum(s*l_exp+1.e-6, axes=(4,))
        SM = csdl.sum(s*m_exp+1.e-6, axes=(4,))

        wake_mesh_dict[key]['S'] = S
        wake_mesh_dict[key]['SL'] = SL
        wake_mesh_dict[key]['SM'] = SM

        rc_exp = csdl.expand(rc, (nt, ns-1), 'i->ia')

        vortex_core_radius = csdl.Variable(value=np.zeros(wake_corners.shape[:-1]))
        vortex_core_radius = vortex_core_radius.set(csdl.slice[0,:,:,0], rc_exp[:-1,:]) # point 0 to 1 based on wake corners above
        vortex_core_radius = vortex_core_radius.set(csdl.slice[0,:,:,1], rc_exp[1:,:]) # point 1 to 2 based on wake corners above
        vortex_core_radius = vortex_core_radius.set(csdl.slice[0,:,:,2], rc_exp[1:,:]) # point 2 to 3 based on wake corners above
        vortex_core_radius = vortex_core_radius.set(csdl.slice[0,:,:,3], rc_exp[1:,:]) # point 3 to 0 based on wake corners above

        wake_mesh_dict[key]['wake_core_radius'] = vortex_core_radius

        start_wpc += num_wake_points
    
    nvwp = sum(wake_panels) # num vlm wake panels
    TE_edges = pm_wake_mesh_dict['TE_edges']
    TE_node_indices = pm_wake_mesh_dict['TE_node_indices']
    ns_panels_pm_wake = len(TE_edges) # number of spanwise panels in pm wake
    ns_pm_wake = len(TE_node_indices)
    npwp = (nt-1)*(ns_panels_pm_wake)
    npwn = nt*ns_pm_wake
    num_tot_wake_panels = npwp + nvwp
    num_nodes = 1 # HARDCODED BC UNSTEADY SOLVER SOLVES ONE GEOMETRY

    base_shape = (num_nodes, num_tot_wake_panels)
    total_panel_center = csdl.Variable(value=np.zeros((base_shape) + (3,)))
    total_panel_corners = csdl.Variable(value=np.zeros((base_shape) + (4,3)))
    total_panel_x_dir = csdl.Variable(value=np.zeros((base_shape) + (3,)))
    total_panel_y_dir = csdl.Variable(value=np.zeros((base_shape) + (3,)))
    total_panel_normal = csdl.Variable(value=np.zeros((base_shape) + (3,)))
    total_S = csdl.Variable(value=np.zeros((base_shape) + (4,)))
    total_SL = csdl.Variable(value=np.zeros((base_shape) + (4,)))
    total_SM = csdl.Variable(value=np.zeros((base_shape) + (4,)))
    vortex_core_radius = csdl.Variable(value=np.zeros((num_nodes, num_tot_wake_panels, 4)))
        # total_wake_corners = csdl.Variable(value=np.zeros((num_nodes, num_tot_wake_panels, 4, 3))) 

    total_panel_center = total_panel_center.set(csdl.slice[:,:npwp], pm_wake_mesh_dict['panel_center'])
    total_panel_corners = total_panel_corners.set(csdl.slice[:,:npwp], pm_wake_mesh_dict['panel_corners'])
    total_panel_x_dir = total_panel_x_dir.set(csdl.slice[:,:npwp], pm_wake_mesh_dict['panel_x_dir'])
    total_panel_y_dir = total_panel_y_dir.set(csdl.slice[:,:npwp], pm_wake_mesh_dict['panel_y_dir'])
    total_panel_normal = total_panel_normal.set(csdl.slice[:,:npwp], pm_wake_mesh_dict['panel_normal'])
    total_S = total_S.set(csdl.slice[:,:npwp], pm_wake_mesh_dict['S'])
    total_SL = total_SL.set(csdl.slice[:,:npwp], pm_wake_mesh_dict['SL'])
    total_SM = total_SM.set(csdl.slice[:,:npwp], pm_wake_mesh_dict['SM'])
    vortex_core_radius = vortex_core_radius.set(csdl.slice[:,:npwp], pm_wake_mesh_dict['wake_core_radius'])

    pco = npwp # offsets for counters
    nco = npwn  # offsets for counters
    cs_panels, ce_panels = pco, pco # panel counter
    cs_nodes, ce_nodes = nco, nco


    for i, name in enumerate(mesh_names):
        key = mesh_names[i]
        num_wake_panels = wake_panels[i]
        ce_panels += num_wake_panels

        total_panel_corners = total_panel_corners.set(
            csdl.slice[:,cs_panels:ce_panels,:,:],
            wake_mesh_dict[key]['wake_corners'].reshape((1, num_wake_panels, 4, 3))
        )
        total_panel_center = total_panel_center.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            wake_mesh_dict[key]['panel_center'].reshape((1, num_wake_panels, 3))
        )
        total_panel_x_dir = total_panel_x_dir.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            wake_mesh_dict[key]['panel_x_dir'].reshape((1, num_wake_panels, 3))
        )
        total_panel_y_dir = total_panel_y_dir.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            wake_mesh_dict[key]['panel_y_dir'].reshape((1, num_wake_panels, 3))
        )
        total_panel_normal = total_panel_normal.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            wake_mesh_dict[key]['panel_normal'].reshape((1, num_wake_panels, 3))
        )
        total_S = total_S.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            wake_mesh_dict[key]['S'].reshape((1, num_wake_panels, 4))
        )
        total_SL = total_SL.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            wake_mesh_dict[key]['SL'].reshape((1, num_wake_panels, 4))
        )
        total_SM = total_SM.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            wake_mesh_dict[key]['SM'].reshape((1, num_wake_panels, 4))
        )
        vortex_core_radius = vortex_core_radius.set(
            csdl.slice[:,cs_panels:ce_panels,:],
            wake_mesh_dict[key]['wake_core_radius'].reshape((1, num_wake_panels, 4))
        )

        cs_panels += num_wake_panels
    
    vectorized_wake_dict = {
        'panel_center': total_panel_center,
        'panel_corners': total_panel_corners,
        'panel_x_dir': total_panel_x_dir,
        'panel_y_dir': total_panel_y_dir,
        'panel_normal': total_panel_normal,
        'S': total_S,
        'SL': total_SL,
        'SM': total_SM,
        'vortex_core_radius': vortex_core_radius,
    }

    return wake_mesh_dict, vectorized_wake_dict