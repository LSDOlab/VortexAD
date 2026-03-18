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
import csdl

# flow parameters
V_inf = 185. # m/s
nt, dt = 50, 0.001
pitch_angle = 0.

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


# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()


# mesh_list = [csdl.Variable(value=actuated_prop_meshes[i,:]) for i in range(num_blades)]
# mesh_vel_list = [csdl.Variable(value=prop_nodal_velocity[i,:]) for i in range(num_blades)]
# coll_vel_list = [csdl.Variable(value=collocation_velocity[i,:]) for i in range(num_blades)]

pitch = csdl.Variable(value=np.array([0.]))


input_dict = {
    # 'V_inf': 10.,
    # 'alpha': pitch,
    # 'V_inf': mesh_vel_list,
    # 'collocation_velocity': coll_vel_list,
    'solver_mode': 'unsteady',
    'nt': nt,
    'dt': dt,

    'partition_size': 1,

    'free_wake': True,
    # 'meshes': mesh_list,
    'core_radius': c*1e0,
    'dissipation': True,
    # 'core_radius': 1.e-6,
}

pfse = PFSE(input_dict)

panel_mesh_var = csdl.Variable(value=points)
pfse.add_thick_surface(
    mesh=panel_mesh_var.expand((nt, points.shape[0]), 'i->ai'),
    connectivity=[],
    coll_vel = csdl.Variable(value=np.zeros(nt, num_cells))
)

mesh_orig = csdl.Variable(value=mesh_orig_val)
pfse.add_thin_surface(
    mesh=mesh_orig.expand((nt, nc, ns), 'ij->aij'),
    coll_vel=csdl.Variable(np.zeros((nt, nc-1, ns-1)))
)

pfse_output_names = [
    'mu',
    'x_w',
    'mu_w',
    'mesh',
    'panel_forces'
]

pfse.declare_outputs(pfse_output_names)

pfse_outputs = pfse.evaluate()

mu = pfse_outputs['mu']
x_w = pfse_outputs['x_w']
mu_w = pfse_outputs['mu_w']
mesh = pfse_outputs['mesh']
panel_forces = pfse_outputs['panel_forces']

inputs = [pitch]
outputs = [mu, x_w, mu_w, mesh, panel_forces]

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

num_runs = 1
start_total = time.time()
for i in range(num_runs):
    start_run = time.time()
    sim.run()
    stop_run = time.time()
    print(f'run time: {stop_run-start_run} seconds')
stop_total = time.time()
print(f'total run time for {num_runs} runs: {stop_total-start_total} seconds')
print(f'average run time across {num_runs} runs: {(stop_total-start_total)/num_runs} seconds per run')