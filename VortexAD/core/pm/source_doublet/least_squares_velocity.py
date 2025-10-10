import numpy as np
import csdl_alpha as csdl 

def least_squares_velocity_old(mu_grid, delta_coll_point):
    '''
    We use the normal equations to solve for the derivative approximations, skipping the assembly of the original matrices
    A^{T}Ax   = A^{T}b becomes Cx = d; we generate C and d directly
    '''
    num_nodes= mu_grid.shape[0]
    grid_shape = mu_grid.shape[1:]
    nc_panels, ns_panels = grid_shape[0], grid_shape[1]
    num_panels = nc_panels*ns_panels
    C = csdl.Variable(shape=(num_nodes, num_panels*2, num_panels*2), value=0.)
    b = csdl.Variable((num_nodes, num_panels*2,), value=0.)

    # matrix assembly for C
    sum_dl_sq = csdl.sum(delta_coll_point[:,:,:,:,0]**2, axes=(3,))
    sum_dm_sq = csdl.sum(delta_coll_point[:,:,:,:,1]**2, axes=(3,))
    sum_dl_dm = csdl.sum(delta_coll_point[:,:,:,:,0]*delta_coll_point[:,:,:,:,1], axes=(3,))

    diag_list_dl = np.arange(start=0, stop=2*num_panels, step=2)
    diag_list_dm = diag_list_dl + 1
    # off_diag_indices = np.arange()

    C = C.set(csdl.slice[:,list(diag_list_dl), list(diag_list_dl)], value=sum_dl_sq.reshape((num_nodes, num_panels)))
    C = C.set(csdl.slice[:,list(diag_list_dm), list(diag_list_dm)], value=sum_dm_sq.reshape((num_nodes, num_panels)))
    C = C.set(csdl.slice[:,list(diag_list_dl), list(diag_list_dm)], value=sum_dl_dm.reshape((num_nodes, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO
    C = C.set(csdl.slice[:,list(diag_list_dm), list(diag_list_dl)], value=sum_dl_dm.reshape((num_nodes, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO

    # vector assembly for d
    mu_grid_exp_deltas = csdl.expand(mu_grid, mu_grid.shape + (4,), 'jkl->jkla')

    dmu = csdl.Variable(shape=(num_nodes, nc_panels, ns_panels, 4), value=0.)
    # the last dimension of size 4 is minus l, plus l, minus m, plus m
    dmu = dmu.set(csdl.slice[:,1:,:,0], value = mu_grid[:,:-1,:] - mu_grid[:,1:,:])
    dmu = dmu.set(csdl.slice[:,:-1,:,1], value = mu_grid[:,1:,:] - mu_grid[:,:-1,:])
    dmu = dmu.set(csdl.slice[:,:,1:,2], value = mu_grid[:,:,:-1] - mu_grid[:,:,1:])
    dmu = dmu.set(csdl.slice[:,:,:-1,3], value = mu_grid[:,:,1:] - mu_grid[:,:,:-1])

    dl_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,0] * dmu, axes=(3,)).reshape((num_nodes, num_panels))
    dm_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,1] * dmu, axes=(3,)).reshape((num_nodes, num_panels))

    b = b.set(csdl.slice[:,0::2], value=dl_dot_dmu)
    b = b.set(csdl.slice[:,1::2], value=dm_dot_dmu)

    dmu_d = csdl.Variable(shape=(num_nodes, num_panels*2), value=0.)
    for i in csdl.frange(num_nodes):
        dmu_d = dmu_d.set(csdl.slice[i,:], value=csdl.solve_linear(C[i,:], b[i,:]))

    # NOTE: CHANGE TO USE A LOOP INSTEAD OF GENERATING MATRIX C
    # USE THE STACK METHOD THAT MARK SUGGESTED

    ql = -dmu_d[:,0::2].reshape((num_nodes, nc_panels, ns_panels))
    qm = -dmu_d[:,1::2].reshape((num_nodes, nc_panels, ns_panels))

    return ql, qm

def least_squares_velocity(mu_grid, delta_coll_point):
    '''
    We use the normal equations to solve for the derivative approximations, skipping the assembly of the original matrices
    A^{T}Ax   = A^{T}b becomes Cx = d; we generate C and d directly
    '''
    num_nodes= mu_grid.shape[0]
    grid_shape = mu_grid.shape[1:]
    nc_panels, ns_panels = grid_shape[0], grid_shape[1]
    num_panels = nc_panels*ns_panels
    C = csdl.Variable(shape=(num_nodes, num_panels, 2, 2), value=0.)
    b = csdl.Variable((num_nodes, num_panels, 2), value=0.)

    # matrix assembly for C
    sum_dl_sq = csdl.sum(delta_coll_point[:,:,:,:,0]**2, axes=(3,)) # [0,0] entry
    sum_dm_sq = csdl.sum(delta_coll_point[:,:,:,:,1]**2, axes=(3,)) # [1,1] entry
    sum_dl_dm = csdl.sum(delta_coll_point[:,:,:,:,0]*delta_coll_point[:,:,:,:,1], axes=(3,)) # off-diag entries

    diag_list_dl = np.arange(start=0, stop=2*num_panels, step=2)
    diag_list_dm = diag_list_dl + 1

    C = C.set(csdl.slice[:,:,0,0], value=sum_dl_sq.reshape((num_nodes, num_panels)))
    C = C.set(csdl.slice[:,:,1,1], value=sum_dm_sq.reshape((num_nodes, num_panels)))
    C = C.set(csdl.slice[:,:,0,1], value=sum_dl_dm.reshape((num_nodes, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO
    C = C.set(csdl.slice[:,:,1,0], value=sum_dl_dm.reshape((num_nodes, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO

    # vector assembly for d
    mu_grid_exp_deltas = csdl.expand(mu_grid, mu_grid.shape + (4,), 'jkl->jkla')

    dmu = csdl.Variable(shape=(num_nodes, nc_panels, ns_panels, 4), value=0.)
    # the last dimension of size 4 is minus l, plus l, minus m, plus m
    dmu = dmu.set(csdl.slice[:,1:,:,0], value = mu_grid[:,:-1,:] - mu_grid[:,1:,:])
    dmu = dmu.set(csdl.slice[:,:-1,:,1], value = mu_grid[:,1:,:] - mu_grid[:,:-1,:])
    dmu = dmu.set(csdl.slice[:,:,1:,2], value = mu_grid[:,:,:-1] - mu_grid[:,:,1:])
    dmu = dmu.set(csdl.slice[:,:,:-1,3], value = mu_grid[:,:,1:] - mu_grid[:,:,:-1])

    dl_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,0] * dmu, axes=(3,)).reshape((num_nodes, num_panels))
    dm_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,1] * dmu, axes=(3,)).reshape((num_nodes, num_panels))

    b = b.set(csdl.slice[:,:,0], value=dl_dot_dmu)
    b = b.set(csdl.slice[:,:,1], value=dm_dot_dmu)

    dmu_d = csdl.Variable(value=np.zeros((num_nodes, num_panels, 2))) # components of surface gradient in axis 2
    for i in csdl.frange(num_nodes):
        for j in csdl.frange(num_panels):
            dmu_d_panel = csdl.solve_linear(C[i,j,:,:], b[i,j,:])
            dmu_d = dmu_d.set(csdl.slice[i,j,:], value=dmu_d_panel)

    # NOTE: CHANGE TO USE A LOOP INSTEAD OF GENERATING MATRIX C
    # USE THE STACK METHOD THAT MARK SUGGESTED

    ql = -dmu_d[:,:,0].reshape((num_nodes, nc_panels, ns_panels))
    qm = -dmu_d[:,:,1].reshape((num_nodes, nc_panels, ns_panels))

    return ql, qm

def unstructured_least_squares_velocity_old(mu, delta_coll_point, cell_adjacency):

    num_nodes = mu.shape[0]
    num_tot_panels = mu.shape[1]

    diag_list_dl = np.arange(start=0, stop=2*num_tot_panels, step=2)
    diag_list_dm = list(diag_list_dl + 1)
    diag_list_dl = list(diag_list_dl)

    C = csdl.Variable(shape=(num_nodes, num_tot_panels*2, num_tot_panels*2), value=0.)
    b = csdl.Variable(shape=(num_nodes, num_tot_panels*2), value=0.)

    sum_dl_sq = csdl.sum(delta_coll_point[:,:,:,0]**2, axes=(2,))
    sum_dm_sq = csdl.sum(delta_coll_point[:,:,:,1]**2, axes=(2,))
    sum_dl_dm = csdl.sum(delta_coll_point[:,:,:,0]*delta_coll_point[:,:,:,1], axes=(2,))

    C = C.set(csdl.slice[:,diag_list_dl, diag_list_dl], value=sum_dl_sq)
    C = C.set(csdl.slice[:,diag_list_dm, diag_list_dm], value=sum_dm_sq)
    C = C.set(csdl.slice[:,diag_list_dl, diag_list_dm], value=sum_dl_dm)
    C = C.set(csdl.slice[:,diag_list_dm, diag_list_dl], value=sum_dl_dm)

    dmu = csdl.Variable(shape=(num_nodes, num_tot_panels, 3), value=0.)
    dmu = dmu.set(csdl.slice[:,:,0], value=mu[:,list(cell_adjacency[:,0])] - mu)
    dmu = dmu.set(csdl.slice[:,:,1], value=mu[:,list(cell_adjacency[:,1])] - mu)
    dmu = dmu.set(csdl.slice[:,:,2], value=mu[:,list(cell_adjacency[:,2])] - mu)

    dl_dot_dmu = csdl.sum(delta_coll_point[:,:,:,0]*dmu, axes=(2,))
    dm_dot_dmu = csdl.sum(delta_coll_point[:,:,:,1]*dmu, axes=(2,))

    b = b.set(csdl.slice[:,0::2], value=dl_dot_dmu)
    b = b.set(csdl.slice[:,1::2], value=dm_dot_dmu)
    
    dmu_d = csdl.Variable(shape=(num_nodes, num_tot_panels*2), value=0.)
    for i in csdl.frange(num_nodes):
        dmu_d = dmu_d.set(csdl.slice[i,:], value=csdl.solve_linear(C[i,:,:], b[i,:]))

    ql = -dmu_d[:,0::2]
    qm = -dmu_d[:,1::2]

    return ql, qm

def unstructured_least_squares_velocity_tri(mu, delta_coll_point, cell_adjacency, constant_geometry=False):

    num_nodes = mu.shape[0]
    num_tot_panels = mu.shape[1]
    if constant_geometry:
        num_nodes_geom = 1
    else:
        num_nodes_geom = num_nodes

    diag_list_dl = np.arange(start=0, stop=2*num_tot_panels, step=2)
    diag_list_dm = list(diag_list_dl + 1)
    diag_list_dl = list(diag_list_dl)

    C = csdl.Variable(shape=(num_nodes_geom, num_tot_panels, 2, 2), value=0.)

    sum_dl_sq = csdl.sum(delta_coll_point[:,:,:,0]**2, axes=(2,))
    sum_dm_sq = csdl.sum(delta_coll_point[:,:,:,1]**2, axes=(2,))
    sum_dl_dm = csdl.sum(delta_coll_point[:,:,:,0]*delta_coll_point[:,:,:,1], axes=(2,))

    C = C.set(csdl.slice[:,:,0,0], value=sum_dl_sq.reshape((num_nodes_geom, num_tot_panels)))
    C = C.set(csdl.slice[:,:,1,1], value=sum_dm_sq.reshape((num_nodes_geom, num_tot_panels)))
    C = C.set(csdl.slice[:,:,0,1], value=sum_dl_dm.reshape((num_nodes_geom, num_tot_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO
    C = C.set(csdl.slice[:,:,1,0], value=sum_dl_dm.reshape((num_nodes_geom, num_tot_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO

    mu_delta_1_ind_np_int = list(cell_adjacency[:,0])
    mu_delta_2_ind_np_int = list(cell_adjacency[:,1])
    mu_delta_3_ind_np_int = list(cell_adjacency[:,2])
    panel_indices_np_int = list(np.arange(num_tot_panels))

    mu_delta_1_ind = [int(x) for x in mu_delta_1_ind_np_int]
    mu_delta_2_ind = [int(x) for x in mu_delta_2_ind_np_int]
    mu_delta_3_ind = [int(x) for x in mu_delta_3_ind_np_int]
    panel_indices = [int(x) for x in panel_indices_np_int]

    dmu = csdl.Variable(shape=(num_nodes, num_tot_panels, 3), value=0.)
    # ==== WITH DUPLICATE INDICES ====
    # dmu = dmu.set(csdl.slice[:,:,0], value=mu[:,list(cell_adjacency[:,0])] - mu)
    # dmu = dmu.set(csdl.slice[:,:,1], value=mu[:,list(cell_adjacency[:,1])] - mu)
    # dmu = dmu.set(csdl.slice[:,:,2], value=mu[:,list(cell_adjacency[:,2])] - mu)

    # ==== USING CSDL FRANGE ====
    # for cell_ind, ind1, ind2, ind3 in csdl.frange(vals=(panel_indices, mu_delta_1_ind, mu_delta_2_ind, mu_delta_3_ind)):
    #     dmu = dmu.set(csdl.slice[:,cell_ind,0], value=mu[:,ind1] - mu[:,cell_ind])
    #     dmu = dmu.set(csdl.slice[:,cell_ind,1], value=mu[:,ind2] - mu[:,cell_ind])
    #     dmu = dmu.set(csdl.slice[:,cell_ind,2], value=mu[:,ind3] - mu[:,cell_ind])

    # ==== USING STACK VIA LOOP BUILDER ====
    loop_vals = [panel_indices, mu_delta_1_ind, mu_delta_2_ind, mu_delta_3_ind]
    nn_ind_array = np.arange(num_nodes).tolist()
    # with csdl.experimental.enter_loop(vals=[nn_ind_array]) as nn_loop_builder:
    #     n = nn_loop_builder.get_loop_indices()
    #     with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
    #         i,j,k,l = loop_builder.get_loop_indices()
    #         dmu_1 = mu[n,j] - mu[n,i]
    #         dmu_2 = mu[n,k] - mu[n,i]
    #         dmu_3 = mu[n,l] - mu[n,i]

    #     dmu_1 = loop_builder.add_stack(dmu_1)
    #     dmu_2 = loop_builder.add_stack(dmu_2)
    #     dmu_3 = loop_builder.add_stack(dmu_3)
    #     loop_builder.finalize()

    # dmu_1 = nn_loop_builder.add_stack(dmu_1)
    # dmu_2 = nn_loop_builder.add_stack(dmu_2)
    # dmu_3 = nn_loop_builder.add_stack(dmu_3)
    # nn_loop_builder.finalize()
    # dmu_1 = dmu_1.reshape((num_nodes, num_tot_panels))
    # dmu_2 = dmu_2.reshape((num_nodes, num_tot_panels))
    # dmu_3 = dmu_3.reshape((num_nodes, num_tot_panels))
    # dmu = dmu.set(csdl.slice[:,:,0], value=dmu_1)
    # dmu = dmu.set(csdl.slice[:,:,1], value=dmu_2)
    # dmu = dmu.set(csdl.slice[:,:,2], value=dmu_3)

    with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
        i,j,k,l = loop_builder.get_loop_indices()
        dmu_1 = mu[:,j] - mu[:,i]
        dmu_2 = mu[:,k] - mu[:,i]
        dmu_3 = mu[:,l] - mu[:,i]

    dmu_1 = loop_builder.add_stack(dmu_1)
    dmu_2 = loop_builder.add_stack(dmu_2)
    dmu_3 = loop_builder.add_stack(dmu_3)
    loop_builder.finalize()
    dmu_1 = dmu_1.T().reshape((num_nodes, num_tot_panels))
    dmu_2 = dmu_2.T().reshape((num_nodes, num_tot_panels))
    dmu_3 = dmu_3.T().reshape((num_nodes, num_tot_panels))
    dmu = dmu.set(csdl.slice[:,:,0], value=dmu_1)
    dmu = dmu.set(csdl.slice[:,:,1], value=dmu_2)
    dmu = dmu.set(csdl.slice[:,:,2], value=dmu_3)

    

    b = csdl.Variable(shape=(num_nodes, num_tot_panels, 2), value=0.)
    if constant_geometry:
        with csdl.experimental.enter_loop(vals=[nn_ind_array]) as loop_builder:
            n = loop_builder.get_loop_indices()
            dl_dot_dmu = csdl.sum(delta_coll_point[0,:,:,0]*dmu[n,:], axes=(1,))
            dm_dot_dmu = csdl.sum(delta_coll_point[0,:,:,1]*dmu[n,:], axes=(1,))
        dl_dot_dmu = loop_builder.add_stack(dl_dot_dmu)
        dm_dot_dmu = loop_builder.add_stack(dm_dot_dmu)
        loop_builder.finalize()
        b = b.set(csdl.slice[:,:,0], value=dl_dot_dmu)
        b = b.set(csdl.slice[:,:,1], value=dm_dot_dmu)

    else:
        dl_dot_dmu = csdl.sum(delta_coll_point[:,:,:,0]*dmu, axes=(2,))
        dm_dot_dmu = csdl.sum(delta_coll_point[:,:,:,1]*dmu, axes=(2,))

        b = b.set(csdl.slice[:,:,0], value=dl_dot_dmu)
        b = b.set(csdl.slice[:,:,1], value=dm_dot_dmu)
    
    # ==== USING CSDL FRANGE ====
    # dmu_d = csdl.Variable(shape=(num_nodes, num_tot_panels, 2), value=0.)
    # for i in csdl.frange(num_nodes):
    #     for j in csdl.frange(num_tot_panels):
    #         dmu_d = dmu_d.set(csdl.slice[i,j,:], value=csdl.solve_linear(C[i,j,:,:], b[i,j,:]))

    # ==== USING STACK VIA LOOP BUILDER
    # panel_ind_array = np.arange(num_tot_panels).tolist()
    # with csdl.experimental.enter_loop(vals=[nn_ind_array]) as nn_loop_builder:
    #     n = nn_loop_builder.get_loop_indices()
    #     with csdl.experimental.enter_loop(vals=[panel_ind_array]) as loop_builder:
    #         i = loop_builder.get_loop_indices()
    #         if constant_geometry:
    #             dmu_d_nn = csdl.solve_linear(C[0,i,:,:], b[n,i,:])
    #         else:
    #             dmu_d_nn = csdl.solve_linear(C[n,i,:,:], b[n,i,:])
    #     dmu_d_nn = loop_builder.add_stack(dmu_d_nn)
    #     loop_builder.finalize()

    # dmu_d = nn_loop_builder.add_stack(dmu_d_nn)
    # nn_loop_builder.finalize()
    # # dmu_d = dmu_d.reshape((num_nodes, num_tot_panels, 2))

    # ql = -dmu_d[:,:,0]
    # qm = -dmu_d[:,:,1]
    
    if constant_geometry:
        C = csdl.expand(
            C.reshape(C.shape[1:]),
            (num_nodes,)+C.shape[1:],
            'ijk->aijk'
        )
    j = C[:,:,0,0]
    k = C[:,:,1,1]
    l = C[:,:,0,1]
    m = b[:,:,0]
    n = b[:,:,1]

    dmu_d_m = (n-l*m/j)/(k-l**2/j)
    dmu_d_l = (m-l*dmu_d_m)/j

    ql = -dmu_d_l
    qm = -dmu_d_m

    return ql, qm


def unstructured_least_squares_velocity(mu, delta_coll_point, cell_adjacency, start, constant_geometry=False):

    num_nodes = mu.shape[0]
    num_panels = cell_adjacency.shape[0]
    num_neighbors = cell_adjacency.shape[1]
    if constant_geometry:
        num_nodes_geom = 1
    else:
        num_nodes_geom = num_nodes

    C = csdl.Variable(shape=(num_nodes_geom, num_panels, 2, 2), value=0.)

    sum_dl_sq = csdl.sum(delta_coll_point[:,:,:,0]**2, axes=(2,))
    sum_dm_sq = csdl.sum(delta_coll_point[:,:,:,1]**2, axes=(2,))
    sum_dl_dm = csdl.sum(delta_coll_point[:,:,:,0]*delta_coll_point[:,:,:,1], axes=(2,))

    C = C.set(csdl.slice[:,:,0,0], value=sum_dl_sq.reshape((num_nodes_geom, num_panels)))
    C = C.set(csdl.slice[:,:,1,1], value=sum_dm_sq.reshape((num_nodes_geom, num_panels)))
    C = C.set(csdl.slice[:,:,0,1], value=sum_dl_dm.reshape((num_nodes_geom, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO
    C = C.set(csdl.slice[:,:,1,0], value=sum_dl_dm.reshape((num_nodes_geom, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO

    mu_delta_1_ind_np_int = list(cell_adjacency[:,0])
    mu_delta_2_ind_np_int = list(cell_adjacency[:,1])
    mu_delta_3_ind_np_int = list(cell_adjacency[:,2])
    panel_indices_np_int = list(np.arange(num_panels)+start)
    # NOTE: we add "start" to this to signify the shift in panel indices with different types
    # this is only needed with mixed grids

    mu_delta_1_ind = [int(x) for x in mu_delta_1_ind_np_int]
    mu_delta_2_ind = [int(x) for x in mu_delta_2_ind_np_int]
    mu_delta_3_ind = [int(x) for x in mu_delta_3_ind_np_int]
    panel_indices = [int(x) for x in panel_indices_np_int]

    loop_vals = [panel_indices, mu_delta_1_ind, mu_delta_2_ind, mu_delta_3_ind]

    if num_neighbors == 4: # quad element
        mu_delta_4_ind_np_int = list(cell_adjacency[:,3])
        mu_delta_4_ind = [int(x) for x in mu_delta_4_ind_np_int]
        loop_vals.append(mu_delta_4_ind)

    # ==== USING STACK VIA LOOP BUILDER ====
    nn_ind_array = np.arange(num_nodes).tolist()

    with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
        loop_ind_var = loop_builder.get_loop_indices()
        dmu_list_inner = [
            mu[:,loop_ind_var[1+i]] - mu[:,loop_ind_var[0]]
            for i in range(num_neighbors)
        ]
        # i,j,k,l = loop_builder.get_loop_indices()
        # dmu_1 = mu[:,j] - mu[:,i]
        # dmu_2 = mu[:,k] - mu[:,i]
        # dmu_3 = mu[:,l] - mu[:,i]
    
    dmu_list_0 = [loop_builder.add_stack(val) for val in dmu_list_inner]
    loop_builder.finalize()

    # dmu_1 = loop_builder.add_stack(dmu_1)
    # dmu_2 = loop_builder.add_stack(dmu_2)
    # dmu_3 = loop_builder.add_stack(dmu_3)
    # loop_builder.finalize()

    dmu_list = [val.T().reshape((num_nodes, num_panels)) for val in dmu_list_0]

    # dmu_1 = dmu_1.T().reshape((num_nodes, num_panels))
    # dmu_2 = dmu_2.T().reshape((num_nodes, num_panels))
    # dmu_3 = dmu_3.T().reshape((num_nodes, num_panels))
    dmu = csdl.Variable(shape=(num_nodes, num_panels, num_neighbors), value=0.)
    for i in range(num_neighbors):
        dmu = dmu.set(csdl.slice[:,:,i], value=dmu_list[i])

    # dmu = dmu.set(csdl.slice[:,:,0], value=dmu_1)
    # dmu = dmu.set(csdl.slice[:,:,1], value=dmu_2)
    # dmu = dmu.set(csdl.slice[:,:,2], value=dmu_3)

    b = csdl.Variable(shape=(num_nodes, num_panels, 2), value=0.)
    if constant_geometry:
        with csdl.experimental.enter_loop(vals=[nn_ind_array]) as loop_builder:
            n = loop_builder.get_loop_indices()
            dl_dot_dmu = csdl.sum(delta_coll_point[0,:,:,0]*dmu[n,:], axes=(1,))
            dm_dot_dmu = csdl.sum(delta_coll_point[0,:,:,1]*dmu[n,:], axes=(1,))
        dl_dot_dmu = loop_builder.add_stack(dl_dot_dmu)
        dm_dot_dmu = loop_builder.add_stack(dm_dot_dmu)
        loop_builder.finalize()
        b = b.set(csdl.slice[:,:,0], value=dl_dot_dmu)
        b = b.set(csdl.slice[:,:,1], value=dm_dot_dmu)

    else:
        dl_dot_dmu = csdl.sum(delta_coll_point[:,:,:,0]*dmu, axes=(2,))
        dm_dot_dmu = csdl.sum(delta_coll_point[:,:,:,1]*dmu, axes=(2,))

        b = b.set(csdl.slice[:,:,0], value=dl_dot_dmu)
        b = b.set(csdl.slice[:,:,1], value=dm_dot_dmu)
    
    if constant_geometry:
        C = csdl.expand(
            C.reshape(C.shape[1:]),
            (num_nodes,)+C.shape[1:],
            'ijk->aijk'
        )
    j = C[:,:,0,0]
    k = C[:,:,1,1]
    l = C[:,:,0,1]
    m = b[:,:,0]
    n = b[:,:,1]

    dmu_d_m = (n-l*m/j)/(k-l**2/j)
    dmu_d_l = (m-l*dmu_d_m)/j

    ql = -dmu_d_l
    qm = -dmu_d_m

    return ql, qm