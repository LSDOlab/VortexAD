import numpy as np
import csdl_alpha as csdl

# panel method imports
from VortexAD.core.pm.pre_processor import pre_processor as pm_preproc
from VortexAD.core.pm.unsteady.wake_geometry import wake_geometry as pm_wake_preproc
from VortexAD.core.pm.unsteady.post_processor import steady_pressure_computation as pm_postproc
# from VortexAD.core.pm.unsteady.AIC_computation import AIC_computation
# from VortexAD.core.pm.unsteady.compute_wake_velocity import compute_wake_velocity

# vortex lattice method imports
from VortexAD.core.vlm.pre_processor import pre_processor as vlm_preproc # not needed bc we need the pm preprocessor
from VortexAD.core.vlm.unsteady.wake_pre_processor import wake_pre_processor as vlm_wake_preproc # not needed bc we need the pm preprocessor
from VortexAD.core.vlm.unsteady.post_processor import compute_steady_forces as vlm_postproc
# from VortexAD.core.vlm.unsteady.gamma_solver import gamma_solver
# from VortexAD.core.vlm.unsteady.compute_wake_velocity import compute_wake_velocity

# potential flow solver environment imports
from VortexAD.core.pfse.functions.vlm_2_pm.preprocess_vlm_surf_as_pm import compute_vlm_pm_params_vectorized
from VortexAD.core.pfse.functions.vlm_2_pm.preprocess_vlm_wake_as_pm import preprocess_vlm_wake_as_pm
from VortexAD.core.pfse.functions.vlm_2_pm.vlm_vectorization_functions import vectorize_vlm_variables
from VortexAD.core.pfse.functions.solve_linear_system import solve_linear_system
from VortexAD.core.pfse.functions.compute_wake_velocity import compute_wake_velocity



def pfse_ode_function(pm_orig_mesh_dict, vlm_orig_mesh_dict, solver_options_dict, nt, dt, ode_states, reuse_vars=False):
    '''
    Docstring for pfse_ode_function
    
    :param orig_mesh_dict: Description
    :param solver_options_dict: Description
    :param nt: Description
    :param dt: Description
    :param ode_states: Description
    :param reuse_vars: Description

    Structure of ODE function
    - preprocessor of panel method
    - wake preprocessor for panel method
    - preprocessor of VLM
    - wake preprocessor for VLM
    - vectorize data for interactions
    - solve for doublet/vortex strengths
        - generate AIC matrix
        - make RHS
    - compute free-wake velocities
    - separate into panel method and VLM data
    - steady force computation

    '''
    Cp_cutoff       = solver_options_dict['Cp cutoff']
    BC              = solver_options_dict['BC_PM']
    # mesh_mode       = solver_options_dict['mesh_mode']
    # partition_size  = solver_options_dict['partition_size']
    # iterative       = solver_options_dict['iterative']
    # ROM             = solver_options_dict['ROM']
    reuse_AIC       = solver_options_dict['reuse_AIC']
    free_wake       = solver_options_dict['free_wake']
    vc              = solver_options_dict['core_radius']
    batch_size      = solver_options_dict['partition_size']

    x_w     = ode_states[0]
    mu_w    = ode_states[1]
    tot_wake_pts = x_w.shape[0]
    tot_wake_panels = mu_w.shape[0]
    num_nodes = 1
    x_w = x_w.expand((num_nodes,) + x_w.shape, 'ij->aij')
    mu_w = mu_w.expand((num_nodes,) + mu_w.shape, 'i->ai')


    pm_TE_node_indices = pm_orig_mesh_dict['TE_node_indices']
    num_TE_nodes = pm_TE_node_indices.shape[0]

    pm_TE_edges = pm_orig_mesh_dict['TE_edges']
    num_TE_edges = len(pm_TE_edges)

    npwn = num_TE_nodes * nt # num PM wake nodes
    npwe = (num_TE_nodes-1) * (nt-1) # num PM wake elements
    

    x_w_pm = x_w[:,:npwn,:]
    x_w_vlm = x_w[:,npwn:,:]

    mu_w_pm = mu_w[:,:npwe]
    mu_w_vlm = mu_w[:,npwe:]

    '''
    MAKE DICTIONARIES FOR THE PM AND VLM INPUTS AND OUTPUTS
    PASS THESE INTO THE SPECIALIZED FUNCTIONS
    THEN VECTORIZE BEFORE OR AFTER FOR THE LINEAR SYSTEM SOLVE AND
    FOR THE FREE WAKE COMPUTATION
    '''
    # Preprocessors for panel method surface and wake
    pm_mesh_dict = pm_preproc(pm_orig_mesh_dict, mode='unstructured', constant_geometry=reuse_AIC, bc=BC)
    pm_wake_connectivity = pm_orig_mesh_dict['wake_connectivity']
    pm_wake_mesh_dict = pm_wake_preproc(
        num_nodes, 
        pm_orig_mesh_dict, 
        solver_options_dict, 
        x_w_pm, 
        pm_wake_connectivity
    )

    # Preprocessors for VLM surface and wake
    vlm_mesh_dict, vlm_vectorized_mesh_dict = vlm_preproc(vlm_orig_mesh_dict)


    # DO VLM VECTORIZATION HERE
    vlm_mesh_dict, vlm_vectorized_mesh_dict = compute_vlm_pm_params_vectorized(
        vlm_mesh_dict, 
        vlm_vectorized_mesh_dict
    )

    vlm_wake_dict, vectorized_wake_dict = preprocess_vlm_wake_as_pm(
        solver_options_dict,
        vlm_mesh_dict, 
        pm_wake_mesh_dict,
        x_w_vlm,
        nt
    )

    # offset for npwe is b/c the variable contains panel method wake data as well
    vlm_vectorized_mesh_dict['wake_corners'] = vectorized_wake_dict['panel_corners'][:,npwe:,:]
    vlm_vectorized_mesh_dict['wake_core_radius'] = vectorized_wake_dict['vortex_core_radius'][:,npwe:,:]

    # vlm_mesh_dict, vlm_vectorized_mesh_dict = preprocess_vlm_wake_as_pm(
    #     solver_options_dict,
    #     vlm_mesh_dict, 
    #     vlm_vectorized_mesh_dict,
    #     ode_states,
    #     nt
    # )

    output_dict = solve_linear_system(
        num_nodes, 
        solver_options_dict, 
        pm_mesh_dict, 
        vlm_vectorized_mesh_dict, 
        vectorized_wake_dict,
        mu_w, 
    )

    mu = output_dict['mu']
    sigma = output_dict['sigma']

    # segmenting mu and mu_w into pm and vlm surfaces (NOTE: CHECK)
    num_pm_panels = sigma.shape[1]
    mu_pm = mu[:,:num_pm_panels]
    mu_vlm = mu[:,num_pm_panels:]

    PM_TE_nodes = pm_mesh_dict['TE_node_indices']
    num_PM_TE_nodes = len(PM_TE_nodes)
    PM_TE_edges = pm_mesh_dict['TE_edges']
    num_PM_TE_el = len(PM_TE_edges)
    mu_w_pm = mu_w[:,:num_PM_TE_el*(nt-1)]
    mu_w_vlm = mu_w[:,num_PM_TE_el*(nt-1):]

    # Un-vectorize here between 

    pm_output_dict = pm_postproc(
        pm_mesh_dict,
        mu_pm,
        sigma,
        num_nodes,
        constant_geometry=reuse_AIC,
        Cp_cutoff=Cp_cutoff
    )

    vlm_output_dict = vlm_postproc(
        vlm_mesh_dict,
        vlm_vectorized_mesh_dict,
        solver_options_dict,
        mu_vlm[0,:],
        mu_w_vlm[0,:]
    )

    vectorized_mesh_dict = compute_vectorized_dict(
        pm_mesh_dict,
        vlm_mesh_dict
    )

    wake_vel = compute_wake_velocity(
        vectorized_mesh_dict, 
        pm_mesh_dict, 
        vlm_vectorized_mesh_dict, 
        vectorized_wake_dict, 
        batch_size, 
        x_w, 
        mu, 
        sigma, 
        mu_w, 
        free_wake, 
        vc
    )

    # initializing activation arrays
    dissipation_flag = solver_options_dict['dissipation']
    vc_parameters = solver_options_dict['vc_parameters']
    bqs = vc_parameters[2]

    # dissipation_deriv = csdl.exp(-bqs*time_in_wake)

    time_in_wake = solver_options_dict['time_in_wake'][0] # removing num_nodes
    velocity_activation = solver_options_dict['velocity_activation'][0] # removing num_nodes
    kutta_activation = solver_options_dict['kutta_activation'][0] # removing num_nodes
    if dissipation_flag:
        dissipation_activation = solver_options_dict['dissipation_activation'][0] # removing num_nodes
    vde = csdl.exp(-bqs*time_in_wake)  # dissipation effect

    # computing derivatives
    dmuw_dt = csdl.Variable(value=np.zeros(mu_w.shape))
    dxw_dt = csdl.Variable(value=np.zeros(x_w.shape))

    bps, bpe = 0, 0
    wps, wpe = 0, 0 # wake panel start/end
    wns, wne = 0, 0 # wake node start/end

    vlm_mesh_names = list(vlm_mesh_dict.keys())

    surf_list = [pm_mesh_dict['points']] + [
        vlm_mesh_dict[name]['mesh'] for name in vlm_mesh_names
    ]
    num_vlm = len(vlm_mesh_dict.keys())
    surf_type_list = ['PM'] + ['VLM']*num_vlm
    num_surfaces = len(surf_type_list)

    # here we treat the PM surface as a separate surface in this loop
    for i in range(num_surfaces):
        surf = surf_list[i] # should be a mesh of some kind
        surf_type = surf_type_list[i]
        if surf_type == 'PM':
            ns = num_PM_TE_nodes
            num_surf_bp = pm_mesh_dict['panel_center'].shape[1]
            num_surf_wp = num_PM_TE_el*(nt-1) # first term in product is number of TE nodes
            num_surf_wn = ns*nt
        elif surf_type == 'VLM':
            nc = surf.shape[1]
            ns = surf.shape[2]
            num_surf_bp = (ns-1)*(nc-1)
            num_surf_wp = (ns-1)*(nt-1)
            num_surf_wn = ns*nt

        bpe += num_surf_bp
        wpe += num_surf_wp
        wne += num_surf_wn

        mu_surf = mu[0,bps:bpe]
        mu_w_surf = mu_w[0,wps:wpe].reshape((nt-1, ns-1))
        wake_vel_surf = wake_vel[0, wns:wne].reshape((nt, ns, 3))
        # x_w_surf = x_w[0,wns:wne].reshape((nt,ns, 3))

        dmuw_dt_surf_shape = (nt-1, ns-1)

        if surf_type == 'PM':
            upper_TE_cell_ind = pm_mesh_dict['upper_TE_cells']
            lower_TE_cell_ind = pm_mesh_dict['lower_TE_cells']
            delta_mu_TE = mu_surf[upper_TE_cell_ind] - mu_surf[lower_TE_cell_ind]
            mu_TE_KC = delta_mu_TE
        elif surf_type == 'VLM':
            mu_surf = mu_surf.reshape((nc-1, ns-1))
            mu_TE_KC = mu_surf[-1,:]

        KC_deriv = mu_TE_KC/dt
        KC_deriv_exp = csdl.expand(
            KC_deriv,
            dmuw_dt_surf_shape,
            'a->ia'
        )
        KC_activation = csdl.expand(
            kutta_activation,
            dmuw_dt_surf_shape,
            'i->ia'
        )
        dmuw_dt_surf_no_diss = KC_deriv_exp*KC_activation
        dmuw_dt_surf = dmuw_dt_surf_no_diss

        if dissipation_flag:

            diss_activation_exp = csdl.expand(
                dissipation_activation,
                dmuw_dt_surf_no_diss.shape,
                'i->ia'
            )
            # dissipation_deriv = diss_activation_exp*(csdl.exp(-bqs*dt)-1)*gamma_w_surf/dt
            dissipation_deriv = diss_activation_exp*mu_w_surf*(-bqs)

            dmuw_dt_surf += dissipation_deriv

        vel_act_surf = csdl.expand(
            velocity_activation,
            (nt,ns,3),
            'i->iab'
        )
        dxw_dt_surf = wake_vel_surf*vel_act_surf

        dmuw_dt =  dmuw_dt.set(
            csdl.slice[0,wps:wpe],
            dmuw_dt_surf.reshape((num_surf_wp,))
        )

        dxw_dt = dxw_dt.set(
            csdl.slice[0,wns:wne],
            dxw_dt_surf.reshape((num_surf_wn, 3))
        )

        bps += num_surf_bp
        wps += num_surf_wp
        wns += num_surf_wn
        
    d_dt = [dxw_dt, dmuw_dt]

    outputs = {
        'mu': mu,
        # 'CL_PM': pm_output_dict['CL']
    }

    pm_outputs = {
        'sigma': sigma,
        'panel_normal': pm_mesh_dict['panel_normal'],
        'panel_area': pm_mesh_dict['panel_area'],
        'panel_center': pm_mesh_dict['panel_center'],
        'nodal_cp_velocity': pm_mesh_dict['coll_point_velocity'],
        'Cp_static': pm_output_dict['Cp_static'],
        'ql': pm_output_dict['ql'],
        'qm': pm_output_dict['qm'],
        'qn': pm_output_dict['qn'],
    }

    num_vlm_panels = mu_vlm.shape[1]
    vlm_outputs = {

        'steady_panel_force': vlm_output_dict['steady_panel_force'].reshape((1,num_vlm_panels,3)),
        'net_gamma': vlm_output_dict['net_gamma'].reshape((1,num_vlm_panels)),

        'panel_centers': vlm_vectorized_mesh_dict['panel_centers'],
        'panel_areas': vlm_vectorized_mesh_dict['panel_areas'],
        'force_eval_pts': vlm_vectorized_mesh_dict['force_eval_pts'],
        'bound_vec_velocity': vlm_vectorized_mesh_dict['bound_vec_velocity'],
        'panel_normal': vlm_vectorized_mesh_dict['panel_normal'],
        'wake_corners': vlm_vectorized_mesh_dict['wake_corners'],
        'wake_core_radius': vlm_vectorized_mesh_dict['wake_core_radius'],

        # 'AIC': AIC.reshape((1,) + AIC.shape),
        # 'AIC_w': AIC_w.reshape((1,) + AIC_w.shape),
        # 'RHS': RHS.reshape((1,) + RHS.shape),
        # 'BC': BC.reshape((1,) + BC.shape),
        # 'wake_influence': wake_influence.reshape((1,) + wake_influence.shape),
        # 'dissipation_deriv': vde.reshape((1,)+vde.shape)
    }

    return outputs, pm_outputs, vlm_outputs, d_dt




def compute_vectorized_dict(pm_dict, vlm_dict):

    pm_points = pm_dict['points']
    num_pm_pts = pm_points.shape[1]
    surf_names = list(vlm_dict.keys())
    num_vlm_pts = 0
    for name in surf_names:
        nc, ns = vlm_dict[name]['nc'], vlm_dict[name]['ns']
        num_vlm_pts += nc*ns

    
    num_points = num_pm_pts + num_vlm_pts
    nodal_velocity = csdl.Variable(value=np.zeros((1, num_points, 3)))

    nodal_velocity = nodal_velocity.set(
        csdl.slice[:,:num_pm_pts],
        pm_dict['nodal_velocity']
    )
    TE_node_indices = list(pm_dict['TE_node_indices'])

    npp = num_pm_pts
    start, stop = npp, npp
    TE_offset = npp

    for i, name in enumerate(surf_names):
        nc, ns = vlm_dict[name]['nc'], vlm_dict[name]['ns']

        num_vlm_surf_pts = nc*ns
        stop += num_vlm_surf_pts
        nodal_velocity = nodal_velocity.set(
            csdl.slice[:,start:stop],
            vlm_dict[name]['nodal_velocity'].reshape((1, nc*ns, 3))
        )

        asdf = list(np.arange(ns))
        surf_TE_node_indices = [(val+1)*nc - 1 + TE_offset for val in asdf]
        TE_node_indices.extend(surf_TE_node_indices)
        TE_offset += num_vlm_surf_pts

        start += num_vlm_surf_pts

    vectorized_dict = {
        'nodal_velocity': nodal_velocity,
        'TE_node_indices': TE_node_indices,
    }
    return vectorized_dict