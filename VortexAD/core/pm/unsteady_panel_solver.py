import numpy as np
import csdl_alpha as csdl
import ozone

from VortexAD.core.pm.unsteady.panel_code_ode_function import panel_code_ode_function
from VortexAD.core.pm.unsteady.post_processor import unsteady_post_processor

def unsteady_panel_solver(orig_mesh_dict, solver_options_dict):

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
    drag_type           = solver_options_dict['drag_type']
    free_wake           = solver_options_dict['free_wake']

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
        mu_w = ozone_vars.states['mu_w']

        # mesh_list = []
        # mesh_vel_list = []
        # for i in range(len(mesh)):
        #     mesh_list.append(ozone_vars.dynamic_parameters[f'surface_{i}_mesh'])
        #     mesh_vel_list.append(ozone_vars.dynamic_parameters[f'surface_{i}_vel'])

        # outputs, d_dt = panel_code_ode_function(
        #     mesh_list,
        #     mesh_vel_list,
        #     ode_states=[x_w.reshape(x_w.shape[1:]), mu_w.reshape(mu_w.shape[1:])],
        #     nt=nt,
        #     dt=dt,
        #     mesh_mode=mesh_mode,
        #     mode=mode,
        #     free_wake=free_wake,
        #     vc=vc
        # )
        orig_mesh_dict['points'] = ozone_vars.dynamic_parameters['points']
        orig_mesh_dict['nodal_velocity'] = ozone_vars.dynamic_parameters['nodal_velocity']
        outputs, d_dt = panel_code_ode_function(
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

        mu = outputs['mu']
        panel_area = outputs['panel_area']
        panel_normal = outputs['panel_normal']
        panel_center = outputs['panel_center']
        nodal_cp_velocity = outputs['nodal_cp_velocity']
        Cp_static = outputs['Cp_static']
        ql = outputs['ql']
        qm = outputs['qm']
        qn = outputs['qn']
        # coll_pt_velocity = outputs['coll_pt_velocity']
        # planform_area = outputs['planform_area']
        AIC_mu = outputs['AIC_mu']
        AIC_sigma = outputs['AIC_sigma']
        AIC_mu_wake = outputs['AIC_mu_wake']

        ozone_vars.profile_outputs['mu'] = mu
        ozone_vars.profile_outputs['panel_area'] = panel_area
        ozone_vars.profile_outputs['panel_normal'] = panel_normal
        ozone_vars.profile_outputs['panel_center'] = panel_center
        ozone_vars.profile_outputs['nodal_cp_velocity'] = nodal_cp_velocity
        ozone_vars.profile_outputs['Cp_static'] = Cp_static
        ozone_vars.profile_outputs['ql'] = ql
        ozone_vars.profile_outputs['qm'] = qm
        ozone_vars.profile_outputs['qn'] = qn
        # ozone_vars.profile_outputs['coll_pt_velocity'] = coll_pt_velocity
        # ozone_vars.profile_outputs['planform_area'] = planform_area

        ozone_vars.profile_outputs['AIC_mu'] = AIC_mu
        ozone_vars.profile_outputs['AIC_sigma'] = AIC_sigma
        ozone_vars.profile_outputs['AIC_mu_wake'] = AIC_mu_wake

        if free_wake:
            # ozone_vars.profile_outputs['AIC_fw_mu'] = AIC_mu
            ozone_vars.profile_outputs['AIC_fw_sigma'] = outputs['AIC_fw_sigma']
            # ozone_vars.profile_outputs['AIC_fw_mu_w'] = AIC_mu_wake


    TE_node_indices = orig_mesh_dict['TE_node_indices']
    num_TE_pts = len(TE_node_indices)
    TE_edges = orig_mesh_dict['TE_edges']
    num_TE_panels = len(TE_edges)
    num_wake_panels = num_TE_panels * (nt-1)

    mu_w_0 = csdl.Variable(value=np.zeros((num_wake_panels))) # wake doublet initial condition
    points = orig_mesh_dict['points']
    TE_pts = points[:, list(TE_node_indices), :]
    if TE_pts.shape[0] == 1:
        x_w_0 = csdl.expand(TE_pts[0,:], TE_pts.shape[1:], 'ij->aij')
    else:
        x_w_0 = TE_pts.reshape((np.prod(TE_pts.shape[:2]),3))
        # x_w_0_shift = csdl.Variable(value=np.zeros(x_w_0.shape))
        # x_w_0_shift = x_w_0_shift.set(
        #     csdl.slice[:,0], value=0.001
        # )
        # x_w_0 = x_w_0 + x_w_0_shift

    # x_w_0 = csdl.Variable(value=np.zeros(nt, num_TE_pts)) 
    # x_w_0 = TE_pts # wake position initial condition
    

    approach = ozone.approaches.TimeMarching()
    ode_problem = ozone.ODEProblem(ozone.methods.ForwardEuler(), approach)

    ode_problem.add_state('x_w', x_w_0, store_history=store_state_history)
    ode_problem.add_state('mu_w', mu_w_0, store_history=store_state_history)

    # for i in range(len(mesh)):
    #     ode_problem.add_dynamic_parameter(f'surface_{i}_mesh', mesh[i])
    #     ode_problem.add_dynamic_parameter(f'surface_{i}_vel', mesh_velocity[i])

    ode_problem.add_dynamic_parameter('points', orig_mesh_dict['points'])
    ode_problem.add_dynamic_parameter('nodal_velocity', orig_mesh_dict['nodal_velocity'])

    step_vector = np.ones(nt-1)*dt
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0., step_vector=step_vector))
    ode_problem.set_function(
        ode_function,
        reuse_vars=reuse_vars
    )

    ode_outputs = ode_problem.solve()

    mu = ode_outputs.profile_outputs['mu']
    panel_normal = ode_outputs.profile_outputs['panel_normal']
    nodal_cp_velocity = ode_outputs.profile_outputs['nodal_cp_velocity']
    # coll_point_velocity = ode_outputs.profile_outputs['coll_point_velocity']
    panel_area = ode_outputs.profile_outputs['panel_area']
    panel_center = ode_outputs.profile_outputs['panel_center']
    Cp_static = ode_outputs.profile_outputs['Cp_static']
    ql = ode_outputs.profile_outputs['ql']
    qm = ode_outputs.profile_outputs['qm']
    qn = ode_outputs.profile_outputs['qn']
    # planform_area = ode_outputs.profile_outputs['planform_area']
    AIC_mu = ode_outputs.profile_outputs['AIC_mu']
    AIC_sigma = ode_outputs.profile_outputs['AIC_sigma']
    AIC_mu_wake = ode_outputs.profile_outputs['AIC_mu_wake']


    mu_w = ode_outputs.states['mu_w']
    x_w = ode_outputs.states['x_w']

    # unsteady pressure computation here (the dmu_dt term)
    upp_mesh_dict = {
        'panel_normal': panel_normal,
        'panel_area': panel_area,
        'panel_center': panel_center,
        'coll_point_velocity': nodal_cp_velocity, # NOTE: CHECK AND FIX THIS (doesn't include actuation velocity)
        # '': ,
        # '': ,
    }
    num_nodes = 1
    output_dict = {
        'mesh': points,
        'mu': mu,
        'x_w': x_w,
        'mu_w': mu_w,
        'Cp_static': Cp_static,
        'ql': ql,
        'qm': qm,
        'qn': qn,
        'AIC_mu': AIC_mu,
        'AIC_sigma': AIC_sigma,
        'AIC_mu_wake': AIC_mu_wake,
    }

    if free_wake:
        # output_dict['AIC_fw_mu'] = ode_outputs.profile_outputs['AIC_fw_mu']
        output_dict['AIC_fw_sigma'] = ode_outputs.profile_outputs['AIC_fw_sigma']
        # output_dict['AIC_fw_mu_w'] = ode_outputs.profile_outputs['AIC_fw_mu_w']

    output_dict = unsteady_post_processor(upp_mesh_dict, output_dict, mu, num_nodes, dt, nt, 
                                          compressibility=compressibility, rho=rho, constant_geometry=reuse_AIC, 
                                          ref_point=moment_ref, ref_area=ref_area, ref_chord=ref_chord, sos=sos)

    return output_dict