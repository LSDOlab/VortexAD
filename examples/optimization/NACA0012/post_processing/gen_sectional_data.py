import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from geometry_function import setup_geometry_parameterization

from modopt import CSDLAlphaProblem, PySLSQP
import lsdo_geo
import VortexAD
import meshio
import pickle

import resource
import time
from datetime import datetime

lfs.num_workers=1

recorder = csdl.Recorder(inline=True)
recorder.start()

imported_function_set = lfs.import_file_patched(file_name="../rectangular_wing_naca0012_10ar.stp", parallelize=False)
geometry = lsdo_geo.Geometry(functions=imported_function_set.functions, 
                                      function_names=imported_function_set.function_names,
                                      name='imported_geometry',
                                      space=imported_function_set.space)

# geometry.plot() # plots geometry using pyvista

# region panel mesh import
mesh = meshio.read("../rectangular_wing_naca0012_10ar.msh")

points_orig = mesh.points
cells = mesh.cells
cells_dict = mesh.cells_dict
cell_adjacency_data = VortexAD.find_cell_adjacency(points=points_orig, cells=cells_dict)

points_orig = cell_adjacency_data[0] 
cells_dict = cell_adjacency_data[1] 
cell_adjacency = cell_adjacency_data[2] 
edges2cells = cell_adjacency_data[3]
points2cells = cell_adjacency_data[4]

TE_properties = VortexAD.TE_detection(points=points_orig,
                             cells=cells_dict,
                             edges2cells=edges2cells,
                             points2cells=points2cells,
                             threshold_theta=125.
                             )

upper_TE_cells = TE_properties[0] 
lower_TE_cells = TE_properties[1] 
TE_edges = TE_properties[2] 
TE_node_indices = TE_properties[3]

cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells_dict[cell_type].tolist()

projected_panel_mesh = geometry.project(points_orig, 
                                        grid_search_density_parameter=1, 
                                        newton_tolerance=1.e-10, 
                                        grid_search_density_cutoff=30,
                                        projection_tolerance=1.e-3,
                                        force_reprojection=False, 
                                        plot=False
                                        )

# project panel centers
cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells_dict[cell_type].tolist()

panel_centers_orig = np.zeros((len(combined_cells), 3))
for i, cell in enumerate(combined_cells):
    panel_centers_orig[i] = np.mean(points_orig[cell], axis=0)

projected_panel_centers = geometry.project(panel_centers_orig, 
                            grid_search_density_parameter=1,
                            newton_tolerance=1.e-10,
                            grid_search_density_cutoff=30,
                            projection_tolerance=1.e-2,
                            force_reprojection=False, 
                            plot=False,
                            )
# endregion

geometry, DV_dict = setup_geometry_parameterization(geometry)

root_twist_dv = DV_dict['root_twist_dv']
tip_twist_dv = DV_dict['tip_twist_dv']
wingspan_outer_dv = DV_dict['wingspan_outer_dv']
taper_ratio_dv = DV_dict['taper_ratio_dv']

root_chord_outer_dv = DV_dict['root_chord_outer_dv']
tip_chord_outer_dv = DV_dict['tip_chord_outer_dv']

geometry_coefficients = geometry.stack_coefficients()
geometry_coefficients.add_name('geometry_coefficients')
# geometry_coefficients.save()

planform_area = (root_chord_outer_dv + tip_chord_outer_dv) / 2 * wingspan_outer_dv
planform_area.add_name('planform_area')
planform_area_0 = planform_area.value

asdf = geometry.plot(show=False)
# deformed_geometry.plot(
#     opacity=0.5,
#     color='yellow',
#     additional_plotting_elements=[asdf]
# )

# left_wing_new = deformed_geometry.declare_component(wing_l_indices)
# left_wing_new.plot(
#     opacity=0.5,
#     color='yellow',
#     additional_plotting_elements=[[asdf[ind] for ind in wing_l_indices]],
#     show=False
# )

# endregion

fname = 'NACA0012_opt_demo'

V_inf = 50.
rho = 1.225
sos = 343.

full_fname = f'extracted_data/{fname}.pkl'
with open(full_fname, 'rb') as file:
    data = pickle.load(file)

time_ind = -1

pitch = 0.
geometry_coefficients = data['geometry_coefficients'][time_ind,:]
geometry.unstack_coefficients(csdl.Variable(value=geometry_coefficients))

coll_points = geometry.evaluate(projected_panel_centers).value
panel_mesh = geometry.evaluate(projected_panel_mesh).value
nominal_ind=0
Cp = data['Cp'][time_ind, 0,:] # num_cells, removing optimization iteration and num_nodes axes
V_vec = np.array([
    V_inf*np.cos(pitch*np.pi/180),
    0., 
    V_inf*np.sin(pitch*np.pi/180)
])

def get_spanwise_dist_data(surf_ind):

    # sort panels and CP locations
    ppc_surf = [val for val in projected_panel_centers if val[0] == surf_ind and 0.01 < val[1][0] < 0.999] 
    # projected panel centers with [surf, parametric_coord]
    ppc_surf_ind = [i for i, val in enumerate(projected_panel_centers) if val[0] == surf_ind and 0.01 < val[1][0] < 0.999] # panel index for surface
    # panel center indices

    # pressure coefficients for the surface
    Cp_surf = Cp[ppc_surf_ind]

    num_panels_0 = Cp_surf.shape[0]
    nc_panels = 20
    ns_panels = num_panels_0//nc_panels

        # panel centers for surface (reshaped to be (nc, ns, 3))
    pc_surf = geometry.evaluate(ppc_surf).value.reshape((ns_panels, nc_panels, 3))
    pc_surf = np.swapaxes(pc_surf,0,1)
    
    Cp_surf_grid = Cp_surf.reshape(ns_panels, nc_panels).T
    # transpose maybe bc of parametric grid ordering, where u is spanwise and v is chordwise
    surf_cells = [combined_cells[val] for val in ppc_surf_ind]
    # surface panel corners (reshaped to be (nc, ns, 4, 3))
    surf_panel_corners = np.array([panel_mesh[ind] for ind in surf_cells]).reshape((ns_panels, nc_panels, 4, 3))
    surf_panel_corners = np.einsum('ijkl->jikl', surf_panel_corners)
    
    p1 = surf_panel_corners[:,:,0,:]
    p2 = surf_panel_corners[:,:,1,:]
    p3 = surf_panel_corners[:,:,2,:]
    p4 = surf_panel_corners[:,:,3,:]

    pc_surf = (p1+p2+p3+p4)/4

    A = p3-p1
    B = p4-p2
    diag_cross_prod = np.cross(A, B)
    dcpn = np.linalg.norm(diag_cross_prod, axis=2)
    panel_area = dcpn/2.
    normal_dir = diag_cross_prod/np.einsum('ij,k->ijk', dcpn, np.ones((3,)))
    
    dP = -0.5*rho*Cp_surf_grid*V_inf**2
    dF = np.einsum('ij,k->ijk', dP*panel_area, np.ones((3,)))*normal_dir
    
    dFx = dF[:,:,0]
    dFz = dF[:,:,2]

    V_vec_exp = np.einsum('ij,k->ijk', np.ones((nc_panels, ns_panels)), V_vec)
    coll_vel_normal = np.sum(V_vec_exp*normal_dir, axis=2)
    coll_vel_tangent = (V_inf**2 - coll_vel_normal**2)**0.5
    
    # alpha = pitch
    # cosa, sina = np.cos(alpha*np.pi/180), np.sin(alpha*np.pi/180)
    alpha = np.arctan(coll_vel_normal/coll_vel_tangent)
    cosa, sina = np.cos(alpha), np.sin(alpha)
    
    dL = dFz*cosa - dFx*sina

    # center of each spanwise section on surface
    sec_center = np.average(pc_surf[:,:,1], axis=0)

    # width of each spanwise section on surface
    sec_width = np.average(
        np.abs(A[:,:,1]),
        axis=0
    )

    return Cp_surf_grid, pc_surf, dL, sec_width, sec_center, normal_dir

'''
geometry functions (from in front of the nose/LE)
0: left wing (+y), bottom surface
1: left wing (+y), top surface
2: left wing, bottom wing tip
3: left wing, top wing tip
4: right wing (-y), bottom surface
5: right wing (-y), top surface
6: right wing, bottom wing tip
7: right wing, top wing tip
'''

# lower_center_oml = geometry.functions[0]
# upper_center_oml = geometry.functions[1]
# lower_left_wing = geometry.functions[2]
# upper_left_wing = geometry.functions[3]
# lower_right_wing = geometry.functions[4]
# upper_right_wing = geometry.functions[5]

# upper_surf_indices = [1, 3, 5]
# lower_surf_indices = [0, 2, 4]
# num_spanwise_surfs = 3 

upper_surf_indices = [1]
lower_surf_indices = [0]
num_spanwise_surfs = 1

center_pts_u_list = []
center_pts_l_list = []

Cp_u_list = []
Cp_l_list = []

dL_u_list = []
dL_l_list = []

sectional_lift = []
col_width = []
sec_center = []

for i in range(num_spanwise_surfs):
    upper_surf = upper_surf_indices[i]
    lower_surf = lower_surf_indices[i]
    Cp_u, pc_u, dL_u, sec_width_u, sec_center_u, normal_dir_u = get_spanwise_dist_data(upper_surf)
    Cp_l, pc_l, dL_l, sec_width_l, sec_center_l, normal_dir_l = get_spanwise_dist_data(lower_surf)
    Cp_l = Cp_l[::-1,:]
    pc_l = pc_l[::-1,:]
    dL_l = dL_l[::-1,:]
    Cp_u_list.append(Cp_u)
    Cp_l_list.append(Cp_l)
    center_pts_u_list.append(pc_u)
    center_pts_l_list.append(pc_l)
    dL_u_list.append(dL_u)
    dL_l_list.append(dL_l)
    col_width.append((sec_width_u+sec_width_l)/2.)
    sec_center.append((sec_center_u+sec_center_l)/2.)
    sectional_lift.append(np.sum(dL_u+dL_l, axis=0))

sectional_lift = np.concatenate(sectional_lift)
col_width = np.concatenate(col_width)
sec_center = np.concatenate(sec_center)

Cp_u = np.concatenate(Cp_u_list, axis=1)
pc_u = np.concatenate(center_pts_u_list, axis=1)
Cp_l = np.concatenate(Cp_l_list, axis=1)
pc_l = np.concatenate(center_pts_l_list, axis=1)


dLdb = sectional_lift/col_width
import pyvista as pv
# spanwise lift plot
spanwise_lift_chart = pv.Chart2D(x_label='span position (m)', y_label='sectional lift')
spanwise_lift_chart.line(sec_center, dLdb)
# spanwise_lift_chart.line([chs, chs], [np.min(dLdb), np.max(dLdb)], style='--')
# spanwise_lift_chart.line([chs+ths, chs+ths], [np.min(dLdb), np.max(dLdb)], style='--')
spanwise_lift_chart.show()


ind = -2
Cp_upper = Cp_u[:,ind]
Cp_lower = Cp_l[:,ind]
Cp_upper_pts = pc_u[:,ind,0]
Cp_lower_pts = pc_u[:,ind,0]

# sectional pressure at ind
Cp_chart = pv.Chart2D(x_label='chordwise position (m)', y_label='Pressure coefficient')
Cp_chart.scatter(Cp_upper_pts, Cp_upper, color='blue', label='upper')
Cp_chart.scatter(Cp_lower_pts, Cp_lower, color='red', label='lower')
ymin, ymax = Cp_chart.y_axis.range

# Flip the y-axis by setting the range in reverse order
Cp_chart.y_axis.range = [ymax, ymin]
Cp_chart.show()

# sectional lift contribution across chord at ind
dL_chart = pv.Chart2D(x_label='chordwise position (m)', y_label='Lift (N)')
dL_chart.scatter(Cp_upper_pts, dL_u_list[-1][:,-1], color='blue', label='upper')
dL_chart.scatter(Cp_lower_pts, dL_l_list[-1][:,-1], color='red', label='lower')
dL_chart.show()

# exit()
output_data = {
    'sec_center': sec_center,
    'dL': [dL_u_list, dL_l_list],
    'dLdb': dLdb,
    # 'sectional_span': [chs, ths, whs],
    'Cp_u': Cp_u,
    'Cp_l': Cp_l,
    'pc_u': pc_u,
    'pc_l': pc_l,
    'pitch': pitch,
    # '': ,
    # '': ,
}

dir_name = 'sectional_data_plots'
file_name = fname

with open(f'{dir_name}/{fname}_sectional_data.pkl', 'wb') as file:
    pickle.dump(output_data, file)
exit()

def compute_sectional_twist(geom, surf_ind, nu, nv):
    # nv varies along chord either from LE to TE or in reverse
    # nu varies in spanwise direction
    num_pts = nu*nv
    u_sample = np.linspace(0, 1, nu)
    v_sample = np.linspace(0, 1, nv)

    parametric_coords = np.zeros((nu, nv, 2))
    parametric_coords[..., 0] = np.einsum('i,j->ij', u_sample, np.ones(nv))
    parametric_coords[..., 1] = np.einsum('i,j->ij', np.ones(nu), v_sample)

    parametric_coords_vec = parametric_coords.reshape((num_pts, 2))

    parametric_pts = []
    for i in range(num_pts):
        parametric_pts.append([surf_ind, parametric_coords_vec[i,:]])

    pts = geom.evaluate(parametric_pts).value.reshape(nu, nv, 3)

    camber_line = pts[:,0] - pts[:,-1]
    twist = np.arctan(camber_line[:,2]/-camber_line[:,0]) * 180/np.pi

    return twist

nu, nv = 25, 5
for ind in upper_surf_indices:
    sec_twist = compute_sectional_twist(geometry, ind, nu, nv)
    print(sec_twist)

exit()
