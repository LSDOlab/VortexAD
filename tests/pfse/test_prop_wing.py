import numpy as np
from VortexAD import PFSE

from VortexAD import SAMPLE_GEOMETRY_PATH
from VortexAD.utils.unstructured_grids.cell_adjacency import find_cell_adjacency, find_wake_cell_adjacency
from VortexAD.utils.unstructured_grids.TE_detection import TE_detection

from VortexAD.utils.meshing.gen_prop_mesh import gen_prop_mesh

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

# propeller mesh + parameters
RPM = 850.
RPM2omega = (2*np.pi) / 60.
omega = RPM * RPM2omega

radius = 2.
chord = 0.2
twist = 0.
num_blades = 2
nr = 5

nondim_r = np.linspace(0.2,1,nr)
pitch = 16
diam_test = 24
twist_dist = np.arctan(pitch/(np.pi*diam_test*nondim_r))*180/np.pi

prop_meshes = gen_prop_mesh(
    radius=radius, 
    chord=chord, 
    # twist=twist, 
    twist=twist_dist, 
    num_blades=num_blades, 
    num_radial=nr, 
    direction='forward',
    plot=False
)
# exit()
pms = prop_meshes.shape[1:]

actuated_prop_meshes = np.zeros((num_blades, nt) + pms)
prop_nodal_velocity = np.zeros((num_blades, nt) + pms)
collocation_velocity = np.zeros((num_blades, nt, pms[0]-1, pms[1]-1, 3))
time_vec = np.linspace(0, nt*dt, nt)
omega_vector = -omega*np.array([1., 0., 0.])
# omega_vector = omega*np.array([1., 0., 0.])
for i in range(nt):
    dtheta = time_vec[i] * omega

    # rotated meshes
    rot_mat = np.zeros((3,3))
    rot_mat[0,0] = 1
    rot_mat[1,1] = rot_mat[2,2] = np.cos(dtheta)
    rot_mat[1,2] = np.sin(dtheta)
    rot_mat[2,1] = -np.sin(dtheta)
    # rot_mat[1,2] = -np.sin(dtheta)
    # rot_mat[2,1] = np.sin(dtheta)

    asdf = np.einsum('ij,abcj->abci', rot_mat, prop_meshes)

    actuated_prop_meshes[:,i,:] = asdf

    collocation_points = (asdf[:,:-1,:-1,:]+asdf[:,1:,:-1,:]+asdf[:,1:,1:,:]+asdf[:,:-1,1:,:])/4

    ref_point = np.array([0., 0., 0.])
    vel_arm_collocation = collocation_points - ref_point

    coll_vel_t = np.cross(omega_vector, vel_arm_collocation)
    
    collocation_velocity[:,i,:] = coll_vel_t

    vel_arm = asdf - ref_point
    nodal_vel_t = np.cross(omega_vector, vel_arm)
    # prop_nodal_velocity[:,i,:] = nodal_vel_t


prop_nodal_velocity[:,:,:,:,0] = -V_inf


# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()


mesh_list = [csdl.Variable(value=actuated_prop_meshes[i,:]) for i in range(num_blades)]
mesh_vel_list = [csdl.Variable(value=prop_nodal_velocity[i,:]) for i in range(num_blades)]
coll_vel_list = [csdl.Variable(value=collocation_velocity[i,:]) for i in range(num_blades)]

pitch = csdl.Variable(value=np.array([0.]))


input_dict = {
    # 'V_inf': 10.,
    # 'alpha': pitch,
    'V_inf': mesh_vel_list,
    'collocation_velocity': coll_vel_list,
    'solver_mode': 'unsteady',
    'nt': nt,
    'dt': dt,

    'partition_size': 1,

    'free_wake': True,
    'meshes': mesh_list,
    'core_radius': chord*1e0,
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

for i in range(2):
    pfse.add_thin_surface(
        mesh=csdl.Variable(actuated_prop_meshes[i,:]),
        coll_vel=csdl.Variable(collocation_velocity[i,:])
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