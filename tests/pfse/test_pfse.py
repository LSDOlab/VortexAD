import numpy as np
from VortexAD import PFSE

from VortexAD import SAMPLE_GEOMETRY_PATH
from VortexAD.utils.unstructured_grids.cell_adjacency import find_cell_adjacency, find_wake_cell_adjacency
from VortexAD.utils.unstructured_grids.TE_detection import TE_detection

from VortexAD.utils.meshing.gen_vlm_mesh import gen_vlm_mesh

import time
import matplotlib.pyplot as plt
import pickle
import meshio
import csdl_alpha as csdl

# flow parameters
V_inf = 185. # m/s
nt, dt = 10, 0.001


V_inf = 10. # m/s
nt, dt = 50, 0.01

# wing panel mesh
ref_area = 10.
mesh_filepath = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch_quad.msh'

mesh = meshio.read(
    mesh_filepath
)

points_orig = mesh.points
cells_dict_orig = mesh.cells_dict

cell_adjacency_data = find_cell_adjacency(
    points=points_orig, 
    cells=cells_dict_orig, 
    radius=1e-10
)

points = cell_adjacency_data[0]
cells = cell_adjacency_data[1]
cell_adjacency = cell_adjacency_data[2]
edges2cells = cell_adjacency_data[3]
points2cells = cell_adjacency_data[4]
num_pm_panels = cells['quad'].shape[0]

cell_types = cells.keys()
num_cells = np.sum([len(cells[cell_type]) for cell_type in cell_types])

TE_properties = TE_detection(
    points=points,
    cells=cells,
    edges2cells=edges2cells,
    threshold_theta=125
)
upper_TE_cells = TE_properties[0]
lower_TE_cells = TE_properties[1]
TE_edges = TE_properties[2]
TE_node_indices = TE_properties[3]

# wing mesh + parameters
ns, nc = 14, 5
AR = 10
c = 1
b = AR*c
# b, c = 10., 1.
mesh_orig_val = gen_vlm_mesh(ns, nc, b, c)
mesh_orig_val[:,:,1] += 20


# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()


# mesh_list = [csdl.Variable(value=actuated_prop_meshes[i,:]) for i in range(num_blades)]
# mesh_vel_list = [csdl.Variable(value=prop_nodal_velocity[i,:]) for i in range(num_blades)]
# coll_vel_list = [csdl.Variable(value=collocation_velocity[i,:]) for i in range(num_blades)]

pitch = csdl.Variable(value=np.array([5.]))


input_dict = {
    'V_inf': V_inf,
    'alpha': pitch,
    # 'V_inf': mesh_vel_list,
    # 'collocation_velocity': coll_vel_list,
    'solver_mode': 'unsteady',
    'nt': nt,
    'dt': dt,

    'partition_size': 1,

    'free_wake': True,
    # 'meshes': mesh_list,
    # 'core_radius': c*1e0,
    'dissipation': False,
    # 'core_radius': 1.e-6,
}

pfse = PFSE(input_dict)

panel_mesh_var = csdl.Variable(value=points)
panel_mesh_nt = panel_mesh_var.expand((nt, ) + points.shape, 'ij->aij')
pfse.add_thick_surface(
    mesh=panel_mesh_nt,
    connectivity=cell_adjacency_data,
    TE_properties=TE_properties,
    coll_vel=csdl.Variable(value=np.zeros(nt, num_cells))
)

mesh_orig = csdl.Variable(value=mesh_orig_val)
vlm_mesh_nt = mesh_orig.expand((nt, nc, ns, 3), 'ijk->aijk')
pfse.add_thin_surface(
    mesh=vlm_mesh_nt,
    coll_vel=csdl.Variable(value=np.zeros((nt, nc-1, ns-1, 3)))
)

pfse_output_names = [
    'mu',
    'x_w',
    'mu_w',
    'surf_CL',
    'surf_CDi',
    'surf_L',
    # 'mesh',
    # 'panel_forces'

    # VLM outputs
    'steady_panel_force_VLM',
    'net_gamma_VLM',
    'panel_areas_VLM'
]

pfse.declare_outputs(pfse_output_names)

pfse_outputs = pfse.evaluate()

mu = pfse_outputs['mu']
x_w = pfse_outputs['x_w']
mu_w = pfse_outputs['mu_w']
surf_CL = pfse_outputs['surf_CL']
surf_CDi = pfse_outputs['surf_CDi']
surf_L = pfse_outputs['surf_L']
# mesh = pfse_outputs['mesh']
# panel_forces = pfse_outputs['panel_forces']
steady_panel_force_VLM = pfse_outputs['steady_panel_force_VLM']
net_gamma_VLM = pfse_outputs['net_gamma_VLM']
panel_areas_VLM = pfse_outputs['panel_areas_VLM']

inputs = [pitch]
outputs = [mu, x_w, mu_w, vlm_mesh_nt, panel_mesh_nt]
outputs.extend([surf_CL, surf_CDi])
outputs.append(steady_panel_force_VLM)
outputs.append(net_gamma_VLM)
outputs.append(surf_L)
outputs.append(panel_areas_VLM)

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs,
    gpu=False
)

start = time.time()
sim.run()
stop = time.time()
print(f'compile + run time: {stop-start} seconds')

surf_CL_val = sim[surf_CL]
surf_CDi_val = sim[surf_CDi]
mu_val = sim[mu]
mu_w_val = sim[mu_w]
x_w_val = sim[x_w]
spf_val = sim[steady_panel_force_VLM]
surf_L_val = sim[surf_L]
ng_val = sim[net_gamma_VLM]

# num_runs = 1
# start_total = time.time()
# for i in range(num_runs):
#     start_run = time.time()
#     sim.run()
#     stop_run = time.time()
#     print(f'run time: {stop_run-start_run} seconds')
# stop_total = time.time()
# print(f'total run time for {num_runs} runs: {stop_total-start_total} seconds')
# print(f'average run time across {num_runs} runs: {(stop_total-start_total)/num_runs} seconds per run')

panel_mesh_nt_val = sim[panel_mesh_nt]
vlm_mesh_nt_val = sim[vlm_mesh_nt]
vlm_meshes_nt = [vlm_mesh_nt_val]
wake_form = 'lines'

# iso_cam = dict(
    # position=(-27.2696, -16.3214, 8.61178),
    # focal_point=(1.27338, -0.446573, 2.47714),
    # viewup=(0.0807063, 0.229700, 0.969909),
    # roll=74.8432,
    # distance=33.2317,
    # clipping_range=(21.2845, 48.3509),
# )

# iso_cam = dict(
    # pos=(-36.3434, -23.8661, 23.8387),
    # focal_point=(2.33444, -0.267801, 6.10705),
    # viewup=(0.231323, 0.313005, 0.921150),
    # roll=69.0273,
    # distance=48.6545,
    # clipping_range=(37.9502, 60.1797),
# )

# iso cam for testing
iso_cam = dict(
    pos=(-21.3394, -35.5395, 29.9108),
    focal_point=(3.87856, -0.619522, 7.28538),
    viewup=(0.265880, 0.381579, 0.885272),
    roll=56.7716,
    distance=48.6545,
    clipping_range=(35.4764, 64.9839),
)
if False:
    pfse.plot_unsteady(
        panel_mesh_nt_val,
        vlm_meshes_nt,
        x_w_val,
        mu_val,
        mu_w_val,
        wake_form=wake_form,
        interactive=False,
        camera=iso_cam,
        name=f'rect_wing_pm_vlm_nt_{nt}' + f'_{wake_form}',
        fps=10
    )

# uvlm import
with open('uvlm_sample_data.pkl', 'rb') as file:
    uvlm_data = pickle.load(file)

mu_uvlm = uvlm_data['mu']
mu_w_uvlm = uvlm_data['mu_w']
x_w_uvlm = uvlm_data['x_w']
CL_uvlm = uvlm_data['CL']
CDi_uvlm = uvlm_data['CDi']
L_uvlm = uvlm_data['L']
spf_uvlm = uvlm_data['steady_panel_force']
ng_uvlm = uvlm_data['net_gamma']

# panel method data import
with open('PM_data_rect_wing.pkl', 'rb') as file:
    upm_data = pickle.load(file)

mu_upm = upm_data['mu']
mu_w_upm = upm_data['mu_w']
x_w_upm = upm_data['x_w']
CL_upm = upm_data['CL']
CDi_upm = upm_data['CDi']
L_upm = upm_data['L']

# mu error
mu_import = np.concatenate((mu_upm, mu_uvlm), axis=1)
mu_delta = mu_import - mu_val
mu_error = np.linalg.norm(mu_delta)/np.linalg.norm(mu_import)

# UPM mu error
mu_upm_pfse = mu_val[:,:num_pm_panels]
mu_upm_delta = mu_upm-mu_upm_pfse
mu_upm_error = np.linalg.norm(mu_upm_delta)/np.linalg.norm(mu_upm)

# UVLM mu error
mu_uvlm_pfse = mu_val[:,num_pm_panels:]
mu_uvlm_delta = mu_uvlm-mu_uvlm_pfse
mu_uvlm_error = np.linalg.norm(mu_uvlm_delta)/np.linalg.norm(mu_uvlm)

# CL error
CL_upm_delta = CL_upm - surf_CL_val[:,0]
CL_upm_error = np.linalg.norm(CL_upm_delta)/np.linalg.norm(CL_upm)

CL_uvlm_delta = CL_uvlm - surf_CL_val[:,1]
CL_uvlm_error = np.linalg.norm(CL_uvlm_delta)/np.linalg.norm(CL_uvlm)

# CDi error
CDi_upm_delta = CDi_upm - surf_CDi_val[:,0]
CDi_upm_error = np.linalg.norm(CDi_upm_delta)/np.linalg.norm(CDi_upm)

CDi_uvlm_delta = CDi_uvlm - surf_CDi_val[:,1]
CDi_uvlm_error = np.linalg.norm(CDi_uvlm_delta)/np.linalg.norm(CDi_uvlm)