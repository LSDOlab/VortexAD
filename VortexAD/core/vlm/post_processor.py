import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.vlm.compute_net_circulation import compute_net_circulation
from VortexAD.core.vlm.compute_AIC import compute_AIC
from VortexAD.core.vlm.compute_forces import compute_forces

def post_processor(num_nodes, mesh_dict, gamma, alpha_ML=None):
    output_dict = {}

    net_gamma_dict = compute_net_circulation(num_nodes, mesh_dict, gamma)
    for key in mesh_dict.keys():
        sub_dict = {}
        sub_dict['net_gamma'] = net_gamma_dict[key]
        output_dict[key] = sub_dict
        
    # compute induced velocities here at the force evaluation points 
    # NOTE: THIS IS IDENTICAL TO THE GAMMA SOLVER, REPLACING collocation_points WITH force_eval_points
    AIC_force_eval_pts = compute_AIC(num_nodes, mesh_dict, eval_pt_name='force_eval_points')
    # AIC_force_eval_pts = compute_AIC(num_nodes, mesh_dict, eval_pt_name='collocation_points', force_computation=True)
    num_total_panels = 0
    for key in mesh_dict.keys():
        ns, nc = mesh_dict[key]['ns'], mesh_dict[key]['nc']
        num_total_panels += (ns-1)*(nc-1)
    induced_vel_force_points = csdl.Variable(shape=(num_nodes, num_total_panels, 3), value=0.)
    for i in csdl.frange(num_nodes):
        for j in csdl.frange(3): # NOTE: IN THE FUTURE, CHANGE TO MAKE MORE EFFICIENT.
            induced_vel_force_points = induced_vel_force_points.set(csdl.slice[i,:,j], value=csdl.matvec(AIC_force_eval_pts[i,:,:,j], gamma[i,:]))

    start, stop = 0, 0
    for key in mesh_dict.keys():
        nc, ns = mesh_dict[key]['nc'], mesh_dict[key]['ns']
        num_panels = (nc-1)*(ns-1)
        stop += num_panels
        output_dict[key]['force_pts_v_induced'] = induced_vel_force_points[:,start:stop,:].reshape((num_nodes, nc-1, ns-1, 3))
        start += num_panels

    # NOTE: PROJECT THE MESH VELOCITIES ONTO THE TOTAL VELOCITY FOR THE FORCE CALCULATION

    # compute lift and drag
    surface_output_dict, total_output_dict = compute_forces(num_nodes, mesh_dict, output_dict, alpha_ML=alpha_ML)
    # output_dict is being populated inside of this function
    
    
    return surface_output_dict, total_output_dict





'''
Post-processing:
- compute net gamma based on bordering vortex panels
- compute induced velocities at the force evaluation points?
- compute lift and drag and the like
'''