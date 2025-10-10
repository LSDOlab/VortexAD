import numpy as np
import csdl_alpha as csdl
import ozone

from VortexAD.core.pm.unsteady.panel_code_ode_function import panel_code_ode_function

def unsteady_panel_solver(orig_mesh_dict, solver_options_dict):

    dt                  = solver_options_dict['dt']
    nt                  = solver_options_dict['nt']
    store_state_history = solver_options_dict['store_state_history']
    vc                  = solver_options_dict['vortex_core_radius']
    free_wake           = solver_options_dict['free_wake']


    step_vector = np.ones(nt-1)*dt

    def ode_function(ozone_vars:ozone.ODEVars):
        x_w = ozone_vars.states['x_w']
        mu_w = ozone_vars.states['mu_w']

        mesh_list = []
        mesh_vel_list = []
        for i in range(len(mesh)):
            mesh_list.append(ozone_vars.dynamic_parameters[f'surface_{i}_mesh'])
            mesh_vel_list.append(ozone_vars.dynamic_parameters[f'surface_{i}_vel'])

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

        outputs, d_dt = panel_code_ode_function(
            orig_mesh_dict,
            solver_options_dict,
            ode_states=[x_w.reshape(x_w.shape[1:]), mu_w.reshape(mu_w.shape[1:])]
        )
        
        dxw_dt, dmuw_dt = d_dt[0], d_dt[1]
        ozone_vars.d_states['x_w'] = dxw_dt
        ozone_vars.d_states['mu_w'] = dmuw_dt

        mu = outputs['mu']
        Cp_static = outputs['Cp_static']
        panel_area = outputs['panel_area']
        panel_normal = outputs['panel_normal']
        nodal_cp_velocity = outputs['nodal_cp_velocity']
        coll_pt_velocity = outputs['coll_pt_velocity']
        planform_area = outputs['planform_area']

        ozone_vars.profile_outputs['mu'] = mu
        ozone_vars.profile_outputs['Cp_static'] = Cp_static
        ozone_vars.profile_outputs['panel_area'] = panel_area
        ozone_vars.profile_outputs['panel_normal'] = panel_normal
        ozone_vars.profile_outputs['nodal_cp_velocity'] = nodal_cp_velocity
        # ozone_vars.profile_outputs['coll_pt_velocity'] = coll_pt_velocity
        ozone_vars.profile_outputs['planform_area'] = planform_area

    mu_w_0 = csdl.Variable(value=np.zeros((nt, num_panels)))
    x_w_0 = csdl.Variable(value=np.zeros(nt, num_TE_pts))
    

    approach = ozone.approaches.TimeMarching()
    ode_problem = ozone.ODEProblem(ozone.methods.ForwardEuler(), approach)

    ode_problem.add_state('x_w', x_w_0, store_history=store_state_history)
    ode_problem.add_state('mu_w', mu_w_0, store_history=store_state_history)

    for i in range(len(mesh)):
        ode_problem.add_dynamic_parameter(f'surface_{i}_mesh', mesh[i])
        ode_problem.add_dynamic_parameter(f'surface_{i}_vel', mesh_velocity[i])

    ode_problem.set_timespan(ozone.timespans.StepVector(start=0., step_vector=step_vector))
    ode_problem.set_function(
        ode_function,
        # self=self,
    )

    ode_outputs = ode_problem.solve()

    
    mu = ode_outputs.profile_outputs['mu']
    Cp_static = ode_outputs.profile_outputs['Cp_static']
    panel_normal = ode_outputs.profile_outputs['panel_normal']
    nodal_cp_velocity = ode_outputs.profile_outputs['nodal_cp_velocity']
    # coll_point_velocity = ode_outputs.profile_outputs['coll_point_velocity']
    panel_area = ode_outputs.profile_outputs['panel_area']
    planform_area = ode_outputs.profile_outputs['planform_area']

    return output_dict, mesh_dict