import numpy as np 
import csdl_alpha as csdl

def compute_net_circulation(num_nodes, mesh_dict, gamma):
    # finds the net gamma 

    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)

    net_gamma_dict = {}
    start_index, stop_index = 0, 0
    for i in range(num_surfaces):
        surface_name = surface_names[i]
        ns, nc = mesh_dict[surface_name]['ns'], mesh_dict[surface_name]['nc']
        num_panels = (ns-1)*(nc-1)
        stop_index += num_panels

        surface_gamma = gamma[:, start_index:stop_index].reshape((num_nodes, nc-1, ns-1))

        surface_net_gamma = csdl.Variable(shape=surface_gamma.shape, value=0.)
        surface_net_gamma = surface_net_gamma.set(csdl.slice[:,0,:], value=surface_gamma[:,0,:])
        surface_net_gamma = surface_net_gamma.set(csdl.slice[:,1:,:], value=surface_gamma[:,1:,:] - surface_gamma[:,:-1,:])
        net_gamma_dict[surface_name] = surface_net_gamma

        start_index += num_panels

    return net_gamma_dict