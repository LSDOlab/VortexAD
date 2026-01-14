import numpy as np 
import csdl_alpha as csdl 

def fixed_wake_representation(mesh_dict, num_panels=1):
    '''
    We are representing the wake here coming off of the trailing edge of the bound vortex grid.
    The wake will be "propagated" based on V_inf for the specified number of panels.
    '''
    surface_keys = mesh_dict.keys()
    dt = 100.
    for key in surface_keys:
        bd_vortex_grid_TE = mesh_dict[key]['bound_vortex_mesh'][:,-1,:,:]
        num_nodes = bd_vortex_grid_TE.shape[0]
        ns = bd_vortex_grid_TE.shape[1] # nc dimension gets removed so it's 1, not 2 for ns

        nodal_velocity = mesh_dict[key]['nodal_velocity']
        TE_nodal_velocity = nodal_velocity[:,-1,:,:]

        wake_vortex_mesh = csdl.Variable(shape=(num_nodes, num_panels+1, ns, 3), value=0.)
        wake_vortex_mesh = wake_vortex_mesh.set(csdl.slice[:,0,:,:], value=bd_vortex_grid_TE) 
        for i in csdl.frange(num_panels):
            # dx_i = csdl.expand(V_inf*dt*(i+1), (num_nodes, ns, 3), 'ij->ibj') + bd_vortex_grid_TE
            dx_i = TE_nodal_velocity*dt*(i+1) + bd_vortex_grid_TE # NEGATIVE SIGN WAS ADDED RECENTLY, I FORGOT TO DO THIS BEFORE
            wake_vortex_mesh = wake_vortex_mesh.set(csdl.slice[:,i+1,:,:], value=dx_i) 

        mesh_dict[key]['wake_vortex_mesh'] = wake_vortex_mesh # bound vortex TE is included here so we keep track of the PANELS

    return mesh_dict