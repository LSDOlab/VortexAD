import numpy as np
import csdl_alpha as csdl

from VortexAD.core.vlm.pre_processor import pre_processor
from VortexAD.core.vlm.unsteady.wake_pre_processor import wake_pre_processor
from VortexAD.core.vlm.unsteady.gamma_solver import gamma_solver
from VortexAD.core.vlm.unsteady.post_processor import post_processor
from VortexAD.core.vlm.unsteady.compute_wake_velocity import compute_wake_velocity

def vlm_ode_function(orig_mesh_dict, solver_options_dict, nt, dt, ode_states, reuse_vars=False):
    '''
    Docstring for vlm_ode_function
    
    :param orig_mesh_dict: Description
    :param solver_options_dict: Description
    :param nt: Description
    :param dt: Description
    :param ode_states: Description
    :param reuse_vars: Description
    '''
    batch_size = solver_options_dict['partition_size']
    free_wake = solver_options_dict['free_wake']
    vc = solver_options_dict['core_radius']
    x_w = ode_states[0]
    gamma_w = ode_states[1]

    mesh_dict, vectorized_mesh_dict = pre_processor(orig_mesh_dict)

    mesh_dict, vectorized_mesh_dict = wake_pre_processor(
        mesh_dict, 
        vectorized_mesh_dict,
        ode_states,
        nt
    )

    mesh_names = list(mesh_dict.keys())
    num_surfaces = len(mesh_names)
    # wake_meshes = []
    # wake_gamma = []
    # for i in range(num_surfaces):
    #     wake_meshes.append(ode_states[f'x_w_{i}'])
    #     wake_gamma.append(ode_states[f'gamma_w_{i}'])
    # mesh_dict['wake_meshes'] = wake_meshes
    # mesh_dict['wake_gamma'] = wake_gamma

    lin_solve_dict = gamma_solver(mesh_dict, vectorized_mesh_dict, solver_options_dict, x_w, gamma_w)

    gamma = lin_solve_dict['gamma']
    AIC = lin_solve_dict['AIC']
    AIC_w = lin_solve_dict['AIC_w']
    RHS = lin_solve_dict['RHS']
    BC = lin_solve_dict['BC']
    wake_influence = lin_solve_dict['wake_influence']

    output_dict, surface_output_dict = post_processor(mesh_dict, vectorized_mesh_dict, solver_options_dict, gamma, gamma_w)

    wake_vel = compute_wake_velocity(mesh_dict, vectorized_mesh_dict, batch_size, x_w, gamma, gamma_w, free_wake, vc)

    TE_indices = vectorized_mesh_dict['TE_node_indices']

    dgammaw_dt = csdl.Variable(value=np.zeros(gamma_w.shape))
    dxw_dt = csdl.Variable(value=np.zeros(x_w.shape))

    bps, bpe = 0, 0
    wps, wpe = 0, 0 # wake panel start/end
    wns, wne = 0, 0 # wake node start/end
    
    for i in range(num_surfaces):
        mesh_name = mesh_names[i]
        ns = mesh_dict[mesh_name]['ns']
        nc = mesh_dict[mesh_name]['nc']
        num_surf_body_panels = (ns-1)*(nc-1)
        num_surf_wake_panels = (ns-1)*(nt-1)
        num_surf_wake_nodes = ns*nt

        bpe += num_surf_body_panels
        wpe += num_surf_wake_panels
        wne += num_surf_wake_nodes

        gamma_surf = gamma[bps:bpe].reshape((nc-1, ns-1))
        gamma_w_surf = gamma_w[wps:wpe].reshape((nt-1, ns-1))
        wake_vel_surf = wake_vel[wns:wne].reshape((nt, ns, 3))
        x_w_surf = x_w[wns:wne].reshape((nt,ns, 3))

        surf_bd_vortex_mesh = mesh_dict[mesh_name]['bound_vortex_mesh'][0,:] # removing num_nodes

        dgammaw_dt_surf = csdl.Variable(value=np.zeros((nt-1, ns-1)))
        dgammaw_dt_surf = dgammaw_dt_surf.set(
            csdl.slice[0,:],
            (gamma_surf[-1,:] - gamma_w_surf[0,:])/dt
        )
        dgammaw_dt_surf = dgammaw_dt_surf.set(
            csdl.slice[1:,:],
            (gamma_w_surf[:-1,:] - gamma_w_surf[1:,:])/dt
        )

        dxw_dt_surf = csdl.Variable(value=np.zeros((nt, ns, 3)))
        dxw_dt_surf = dxw_dt_surf.set(
            csdl.slice[0,:],
            (surf_bd_vortex_mesh[-1,:] - x_w_surf[0,:])/dt
        )
        dxw_dt_surf = dxw_dt_surf.set(
            csdl.slice[1:,:],
            wake_vel_surf[1:,:] + (x_w_surf[:-1,:] - x_w_surf[1:,:])/dt
        )

        dgammaw_dt =  dgammaw_dt.set(
            csdl.slice[wps:wpe],
            dgammaw_dt_surf.reshape((num_surf_wake_panels,))
        )

        dxw_dt = dxw_dt.set(
            csdl.slice[wns:wne],
            dxw_dt_surf.reshape((num_surf_wake_nodes, 3))
        )

        bps += num_surf_body_panels
        wps += num_surf_wake_panels
        wns += num_surf_wake_nodes

    d_dt = [dxw_dt, dgammaw_dt]
    panel_force = output_dict['panel_force']
    net_gamma = output_dict['net_gamma']
    outputs = {
        'gamma': gamma.expand((1,) + gamma.shape, 'i->ai'),
        'total_CL': output_dict['total_CL'],
        'total_CDi': output_dict['total_CDi'],
        'panel_force': panel_force.reshape((1,) + panel_force.shape),
        'net_gamma': net_gamma.reshape((1,) + net_gamma.shape),

        'panel_centers': vectorized_mesh_dict['panel_centers'],
        'panel_normal': vectorized_mesh_dict['panel_normal'],
        'wake_corners': vectorized_mesh_dict['wake_corners'],

        'AIC': AIC.reshape((1,) + AIC.shape),
        'AIC_w': AIC_w.reshape((1,) + AIC_w.shape),
        'RHS': RHS.reshape((1,) + RHS.shape),
        'BC': BC.reshape((1,) + BC.shape),
        'wake_influence': wake_influence.reshape((1,) + wake_influence.shape),
    }

    for name in mesh_names:
        outputs[f'CL_surf_{name}'] = surface_output_dict[name]['CL']
        outputs[f'CDi_surf_{name}'] = surface_output_dict[name]['CDi']

    return outputs, d_dt


'''
Structure of vlm ode function:

geometry pre-processor
wake pre-processor
AIC computation
linear solve
    need PV + ROMs?
    can lump these two into one for iterative solvers

free wake computation
force computation
outputs and d_dt computation
'''