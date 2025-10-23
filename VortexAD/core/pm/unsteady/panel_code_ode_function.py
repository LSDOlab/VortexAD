import numpy as np
import csdl_alpha as csdl

from VortexAD.core.pm.unsteady.wake_geometry import wake_geometry
from VortexAD.core.pm.pre_processor import pre_processor
from VortexAD.core.pm.unsteady.AIC_computation import AIC_computation
from VortexAD.core.pm.source_doublet.compute_source_strength import compute_source_strength
from VortexAD.core.pm.unsteady.post_processor import steady_pressure_computation
from VortexAD.core.pm.unsteady.compute_wake_velocity import compute_wake_velocity

def panel_code_ode_function(orig_mesh_dict, solver_options_dict, nt, dt, ode_states, reuse_vars=False):

    Cp_cutoff       = solver_options_dict['Cp cutoff']
    BC              = solver_options_dict['BC']
    mesh_mode       = solver_options_dict['mesh_mode']
    partition_size  = solver_options_dict['partition_size']
    iterative       = solver_options_dict['iterative']
    ROM             = solver_options_dict['ROM']
    reuse_AIC       = solver_options_dict['reuse_AIC']
    free_wake       = solver_options_dict['free_wake']
    vc              = solver_options_dict['core_radius']

    # if mesh_mode == 'structured':
    #     surface_0 = list(orig_mesh_dict.keys())[0]
    #     num_nodes = orig_mesh_dict[surface_0]['nodal_velocity'].shape[0]
    
    # elif mesh_mode == 'unstructured':
    #     num_nodes = orig_mesh_dict['nodal_velocity'].shape[0]

    x_w = ode_states[0]
    mu_w = ode_states[1]
    tot_wake_pts = x_w.shape[0]
    tot_wake_panels = mu_w.shape[0]
    x_w = x_w.expand((1,) + x_w.shape, 'ij->aij')
    mu_w = mu_w.expand((1,) + mu_w.shape, 'i->ai')
    num_nodes = 1

    cell_adjacency_dict = orig_mesh_dict['cell_adjacency']
    cells_per_type = [len(cell_adjacency_dict[val]) for val in cell_adjacency_dict.keys()]
    num_panels = sum(cells_per_type)

    # wake connectivity (unstructured)
    TE_node_indices = orig_mesh_dict['TE_node_indices']
    TE_edges        = orig_mesh_dict['TE_edges']
    mesh            = orig_mesh_dict['points']

    ns = len(TE_node_indices)
    num_TE_edges = len(TE_edges)
    nc_w = nt
    TE = mesh[:,list(TE_node_indices), :]

    TE_edges_zeroed = []
    for i in range(num_TE_edges):
        edge = TE_edges[i]
        new_edge = []
        for j in range(2):
            ind = np.where(TE_node_indices == edge[j])[0][0]
            new_edge.append(ind)
        TE_edges_zeroed.append(tuple(new_edge))

    wake_connectivity = orig_mesh_dict['wake_connectivity']

    # wake_connectivity = np.array([[[
    #     edge[0] + i*ns,
    #     edge[0] + (i+1)*ns,
    #     edge[1] + (i+1)*ns,
    #     edge[1] + i*ns,
    # ] for edge in TE_edges_zeroed] for i in range(nt-1)])

    # wake_connectivity = np.array([[[
    #     edge[0] + i*ns,
    #     edge[0] + (i+1)*ns,
    #     edge[1] + (i+1)*ns,
    #     edge[1] + i*ns,
    # ] for i in range(nt-1)] for edge in TE_edges_zeroed])

    wake_mesh_dict = wake_geometry(num_nodes, orig_mesh_dict, x_w, wake_connectivity)
    wake_mesh_dict['wake_connectivity'] = wake_connectivity

    print('running pre-processing')
    if not reuse_AIC:
        mesh_dict = pre_processor(orig_mesh_dict, mode=mesh_mode, constant_geometry=reuse_AIC)
    else:
        mesh_dict = reuse_vars['mesh_dict']

    sigma = compute_source_strength(mesh_dict, num_nodes, num_panels, mesh_mode, reuse_AIC)

    AIC_matrices = AIC_computation(
        mesh_dict,
        wake_mesh_dict,
        batch_size=partition_size,
        bc=BC,
        ROM=ROM,
        constant_geometry=reuse_AIC
    )
    if not reuse_AIC:
        AIC_mu = AIC_matrices[0]
        AIC_sigma = AIC_matrices[1]
        AIC_mu_wake = AIC_matrices[2]
    else:
        AIC_mu = reuse_vars['AIC_mu']
        AIC_sigma = reuse_vars['AIC_sigma']
        AIC_mu_wake = AIC_matrices[0]
    # NOTE: with the ROMs, each of these matrices are of shape r*n, r being the reduced basis size
    # (with the exception of AIC_mu_red, which is r*r)
    # These are named the same way as the full-space ones for ease of reading the code
    
    # solve linear system here: A\mu = -B\sigma - C\mu_w

    RHS = -csdl.matvec(AIC_sigma[0,:], sigma[0,:]) - csdl.matvec(AIC_mu_wake[0,:], mu_w[0,:])

    if not ROM:
        mu = csdl.solve_linear(AIC_mu[0,:], RHS)
        mu = mu.expand((1,) + mu.shape, 'i->ai')
    else:
        UT, U = ROM[0], ROM[1]
        mu_red = csdl.solve_linear(AIC_mu[0,:], RHS) # matrices account for reduced basis in ROM
        mu = csdl.matvec(U, mu_red)
        mu = mu.expand((1,) + mu.shape, 'i->ai')

    # steady pressure computation
    output_dict = steady_pressure_computation(mesh_dict, mu, sigma, num_nodes, reuse_AIC, Cp_cutoff)

    # free wake computation 
    wake_vel, wake_vel_vars = compute_wake_velocity(mesh_dict, wake_mesh_dict, partition_size, mu, sigma, mu_w, free_wake=free_wake, vc=vc)

    # compute derivatives
    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']
    delta_mu_TE = mu[:,upper_TE_cell_ind] - mu[:,lower_TE_cell_ind]
    dxw_dt = csdl.Variable(value=np.zeros((num_nodes, tot_wake_pts, 3)))
    dmuw_dt = csdl.Variable(value=np.zeros((num_nodes,) + wake_connectivity.shape[:-1]))

    mu_wake = mu_w[0,:].reshape(wake_connectivity.shape[:2])

    dmuw_dt = dmuw_dt.set(
        csdl.slice[0,0,:],
        (delta_mu_TE[0,:] - mu_wake[0,:])/dt
    )
    dmuw_dt = dmuw_dt.set(
        csdl.slice[0,1:,:],
        (mu_wake[:-1,:]-mu_wake[1:,:])/dt
        # (mu_wake[1:,:]-mu_wake[:-1,:])/dt
    )
    
    x_w_grid = x_w.reshape((nt, ns, 3))
    wake_vel_grid = wake_vel.reshape((nt, ns, 3))

    dxw_dt = wake_vel 

    dxw_dt_grid = csdl.Variable(value=np.zeros(x_w_grid.shape))
    # dxw_dt_grid = dxw_dt_grid.set(
        # csdl.slice[0,:,],
        # wake_vel_grid[0,:,:] + (TE[0] - x_w_grid[0,:,:])/dt
        # wake_vel_grid[0,:,:] + (x_w_grid[0,:,:] - TE[0])/dt
    # )
    dxw_dt_grid = dxw_dt_grid.set(
        csdl.slice[1:,:,:],
        wake_vel_grid[1:,:,:] + (x_w_grid[:-1,:,:] - x_w_grid[1:,:,:])/dt
        # wake_vel_grid[1:,:,:] + (x_w_grid[1:,:,:] - x_w_grid[:-1,:,:])/dt
    )

    dmuw_dt = dmuw_dt.reshape((tot_wake_panels,))
    dxw_dt = dxw_dt_grid.reshape((tot_wake_pts, 3))
    d_dt = [dxw_dt, dmuw_dt]

    outputs = {
        'mu': mu,
        'panel_normal': mesh_dict['panel_normal'],
        'panel_area': mesh_dict['panel_area'],
        'panel_center': mesh_dict['panel_center'],
        'nodal_cp_velocity': mesh_dict['coll_point_velocity'],
        'Cp_static': output_dict['Cp_static'],
        'ql': output_dict['ql'],
        'qm': output_dict['qm'],
        'qn': output_dict['qn'],

        'AIC_mu': AIC_mu,
        'AIC_sigma': AIC_sigma,
        'AIC_mu_wake': AIC_mu_wake,
        'wake_vel': wake_vel
    }

    if free_wake:
        for key in wake_vel_vars:
            outputs[key] = wake_vel_vars[key]

    return outputs, d_dt