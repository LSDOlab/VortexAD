import numpy as np
import csdl_alpha as csdl 

def perturbation_velocity_FD_K_P(mu_grid, panel_center_dl, panel_center_dm):
    ql = csdl.Variable(shape=mu_grid.shape, value=0.)
    qm = csdl.Variable(shape=mu_grid.shape, value=0.)

    ql = ql.set(csdl.slice[:,:,1:-1,:], value=-(mu_grid[:,:,2:,:] - mu_grid[:,:,:-2,:])/2./((panel_center_dl[:,:,1:-1,:,0]+panel_center_dl[:,:,1:-1,:,1])/2))
    ql = ql.set(csdl.slice[:,:,0,:], value=-(-3*mu_grid[:,:,0,:]+4*mu_grid[:,:,1,:]-mu_grid[:,:,2,:])/2./((panel_center_dl[:,:,0,:,0]+panel_center_dl[:,:,1,:,0])/2))
    ql = ql.set(csdl.slice[:,:,-1,:], value=-(3*mu_grid[:,:,-1,:]-4*mu_grid[:,:,-2,:]+mu_grid[:,:,-3,:])/2./((panel_center_dl[:,:,-1,:,1]+panel_center_dl[:,:,-2,:,1])/2))

    qm = qm.set(csdl.slice[:,:,:,1:-1], value=-(mu_grid[:,:,:,2:] - mu_grid[:,:,:,:-2])/2./((panel_center_dm[:,:,:,1:-1,0]+panel_center_dm[:,:,:,1:-1,1])/2))
    qm = qm.set(csdl.slice[:,:,:,0], value=-(-3*mu_grid[:,:,:,0]+4*mu_grid[:,:,:,1]-mu_grid[:,:,:,2])/2./((panel_center_dm[:,:,:,0,0]+panel_center_dm[:,:,:,1,0])/2))
    qm = qm.set(csdl.slice[:,:,:,-1], value=-(3*mu_grid[:,:,:,-1]-4*mu_grid[:,:,:,-2]+mu_grid[:,:,:,-3])/2./((panel_center_dm[:,:,:,-1,1]+panel_center_dm[:,:,:,-2,1])/2))

    return ql, qm
            

def perturbation_velocity_FD(mu_grid, dl, dm):
    ql, qm = csdl.Variable(shape=mu_grid.shape, value=0.), csdl.Variable(shape=mu_grid.shape, value=0.)
    ql = ql.set(csdl.slice[:,:,1:-1,:], value=(mu_grid[:,:,2:,:] - mu_grid[:,:,0:-2,:]) / (dl[:,:,1:-1,:,0] + dl[:,:,1:-1,:,1])) # all panels except TE
    # ql = ql.set(csdl.slice[:,:,0,:], value=(mu_grid[:,:,1,:] - mu_grid[:,:,-1,:]) / (dl[:,:,0,:,0] + dl[:,:,0,:,1])) # TE on lower surface
    ql = ql.set(csdl.slice[:,:,0,:], value=(-3*mu_grid[:,:,0,:] + 4*mu_grid[:,:,1,:] - mu_grid[:,:,2,:]) / (3*dl[:,:,1,:,0] - dl[:,:,1,:,1])) # TE on lower surface
    # ql = ql.set(csdl.slice[:,:,-1,:], value=(mu_grid[:,:,0,:] - mu_grid[:,:,-2,:]) / (dl[:,:,-1,:,0] + dl[:,:,-1,:,1])) # TE on upper surface
    ql = ql.set(csdl.slice[:,:,-1,:], value=(3*mu_grid[:,:,-1,:] - 4*mu_grid[:,:,-2,:] + mu_grid[:,:,-3,:]) / (3*dl[:,:,-2,:,1] - dl[:,:,-2,:,0])) # TE on upper surface

    qm = qm.set(csdl.slice[:,:,:,1:-1], value=(mu_grid[:,:,:,2:] - mu_grid[:,:,:,0:-2]) / (dm[:,:,:,1:-1,0] + dm[:,:,:,1:-1,1])) # all panels expect wing tips
    qm = qm.set(csdl.slice[:,:,:,0], value=(-3*mu_grid[:,:,:,0] + 4*mu_grid[:,:,:,1] - mu_grid[:,:,:,2]) / (3*dm[:,:,:,1,0] - dm[:,:,:,1,1]))
    qm = qm.set(csdl.slice[:,:,:,-1], value=(3*mu_grid[:,:,:,-1] - 4*mu_grid[:,:,:,-2] + mu_grid[:,:,:,-3]) / (3*dm[:,:,:,-2,1] - dm[:,:,:,-2,0]))

    return -ql, -qm

def least_squares_velocity(mu_grid, delta_coll_point):
    '''
    We use the normal equations to solve for the derivative approximations, skipping the assembly of the original matrices
    A^{T}Ax   = A^{T}b becomes Cx = d; we generate C and d directly
    '''
    num_nodes, nt = mu_grid.shape[0], mu_grid.shape[1]
    grid_shape = mu_grid.shape[2:]
    nc_panels, ns_panels = grid_shape[0], grid_shape[1]
    num_panels = nc_panels*ns_panels
    C = csdl.Variable(shape=(num_nodes, nt, num_panels*2, num_panels*2), value=0.)
    b = csdl.Variable((num_nodes, nt, num_panels*2,), value=0.)

    # matrix assembly for C
    sum_dl_sq = csdl.sum(delta_coll_point[:,:,:,:,:,0]**2, axes=(4,))
    sum_dm_sq = csdl.sum(delta_coll_point[:,:,:,:,:,1]**2, axes=(4,))
    sum_dl_dm = csdl.sum(delta_coll_point[:,:,:,:,:,0]*delta_coll_point[:,:,:,:,:,1], axes=(4,))

    diag_list_dl = np.arange(start=0, stop=2*num_panels, step=2)
    diag_list_dm = diag_list_dl + 1
    # off_diag_indices = np.arange()

    C = C.set(csdl.slice[:,:,list(diag_list_dl), list(diag_list_dl)], value=sum_dl_sq.reshape((num_nodes, nt, num_panels)))
    C = C.set(csdl.slice[:,:,list(diag_list_dm), list(diag_list_dm)], value=sum_dm_sq.reshape((num_nodes, nt, num_panels)))
    C = C.set(csdl.slice[:,:,list(diag_list_dl), list(diag_list_dm)], value=sum_dl_dm.reshape((num_nodes, nt, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO
    C = C.set(csdl.slice[:,:,list(diag_list_dm), list(diag_list_dl)], value=sum_dl_dm.reshape((num_nodes, nt, num_panels))) # FOR STRUCTURED GRIDS, THESE ARE ZERO

    # vector assembly for d

    dmu = csdl.Variable(shape=(num_nodes, nt, nc_panels, ns_panels, 4), value=0.)
    # the last dimension of size 4 is minus l, plus l, minus m, plus m
    dmu = dmu.set(csdl.slice[:,:,1:,:,0], value = mu_grid[:,:,:-1,:] - mu_grid[:,:,1:,:])
    dmu = dmu.set(csdl.slice[:,:,:-1,:,1], value = mu_grid[:,:,1:,:] - mu_grid[:,:,:-1,:])
    dmu = dmu.set(csdl.slice[:,:,:,1:,2], value = mu_grid[:,:,:,:-1] - mu_grid[:,:,:,1:])
    dmu = dmu.set(csdl.slice[:,:,:,:-1,3], value = mu_grid[:,:,:,1:] - mu_grid[:,:,:,:-1])

    dl_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,:,0] * dmu, axes=(4,)).reshape((num_nodes, nt, num_panels))
    dm_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,:,1] * dmu, axes=(4,)).reshape((num_nodes, nt, num_panels))

    b = b.set(csdl.slice[:,:,0::2], value=dl_dot_dmu)
    b = b.set(csdl.slice[:,:,1::2], value=dm_dot_dmu)

    dmu_d = csdl.Variable(shape=(num_nodes, nt, num_panels*2), value=0.)
    for i in csdl.frange(num_nodes):
        for j in csdl.frange(nt):
            dmu_d = dmu_d.set(csdl.slice[i,j,:], value=csdl.solve_linear(C[i,j,:], b[i,j,:]))

    ql = -dmu_d[:,:,0::2].reshape((num_nodes, nt, nc_panels, ns_panels))
    qm = -dmu_d[:,:,1::2].reshape((num_nodes, nt, nc_panels, ns_panels))

    return ql, qm

def unstructured_least_squares_velocity(mu, delta_coll_point, cell_adjacency):

    num_nodes, nt = mu.shape[:2]
    num_tot_panels = mu.shape[2]

    diag_list_dl = np.arange(start=0, stop=2*num_tot_panels, step=2)
    diag_list_dm = list(diag_list_dl + 1)
    diag_list_dl = list(diag_list_dl)

    C = csdl.Variable(shape=(num_nodes, nt, num_tot_panels*2, num_tot_panels*2), value=0.)
    b = csdl.Variable(shape=(num_nodes, nt, num_tot_panels*2), value=0.)

    sum_dl_sq = csdl.sum(delta_coll_point[:,:,:,:,0]**2, axes=(3,))
    sum_dm_sq = csdl.sum(delta_coll_point[:,:,:,:,1]**2, axes=(3,))
    sum_dl_dm = csdl.sum(delta_coll_point[:,:,:,:,0]*delta_coll_point[:,:,:,:,1], axes=(3,))

    C = C.set(csdl.slice[:,:,diag_list_dl, diag_list_dl], value=sum_dl_sq)
    C = C.set(csdl.slice[:,:,diag_list_dm, diag_list_dm], value=sum_dm_sq)
    C = C.set(csdl.slice[:,:,diag_list_dl, diag_list_dm], value=sum_dl_dm)
    C = C.set(csdl.slice[:,:,diag_list_dm, diag_list_dl], value=sum_dl_dm)

    dmu = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, 3), value=0.)
    dmu = dmu.set(csdl.slice[:,:,:,0], value=mu[:,:,list(cell_adjacency[:,0])] - mu)
    dmu = dmu.set(csdl.slice[:,:,:,1], value=mu[:,:,list(cell_adjacency[:,1])] - mu)
    dmu = dmu.set(csdl.slice[:,:,:,2], value=mu[:,:,list(cell_adjacency[:,2])] - mu)

    dl_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,0]*dmu, axes=(3,))
    dm_dot_dmu = csdl.sum(delta_coll_point[:,:,:,:,1]*dmu, axes=(3,))

    b = b.set(csdl.slice[:,:,0::2], value=dl_dot_dmu)
    b = b.set(csdl.slice[:,:,1::2], value=dm_dot_dmu)
    
    dmu_d = csdl.Variable(shape=(num_nodes, nt, num_tot_panels*2), value=0.)
    for i in csdl.frange(num_nodes):
        for j in csdl.frange(nt):
            dmu_d = dmu_d.set(csdl.slice[i,j,:], value=csdl.solve_linear(C[i,j,:,:], b[i,j,:]))

    ql = -dmu_d[:,:,0::2]
    qm = -dmu_d[:,:,1::2]

    return ql, qm