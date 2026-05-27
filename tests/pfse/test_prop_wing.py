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
import csdl_alpha as csdl

# flow parameters
V_inf = 185. # m/s
# nt, dt = 50, 0.001
nt, dt = 100, 0.0005
pitch_angle = 5

# wing panel mesh
ref_area = 10.
mesh_filepath = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch_quad.msh'
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch_fine_mix.msh' # quads?

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
    threshold_theta=125,
    points2cells=points2cells
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
twist = 30.
num_blades = 2
nr = 5

nondim_r = np.linspace(0.2,1,nr)
pitch = 16
diam_test = 24
twist_dist = np.arctan(pitch/(np.pi*diam_test*nondim_r))*180/np.pi

prop_meshes = gen_prop_mesh(
    radius=radius, 
    chord=chord, 
    twist=twist, 
    # twist=twist_dist, 
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

actuated_prop_meshes[:,:,:,:,0] -= 1
# actuated_prop_meshes[:,:,:,:,2] += 10
prop_nodal_velocity[:,:,:,:,0] = -V_inf


# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()


mesh_list = [csdl.Variable(value=actuated_prop_meshes[i,:]) for i in range(num_blades)]
mesh_vel_list = [csdl.Variable(value=prop_nodal_velocity[i,:]) for i in range(num_blades)]
coll_vel_list = [csdl.Variable(value=collocation_velocity[i,:]) for i in range(num_blades)]

pitch = csdl.Variable(value=np.array([pitch_angle]))


input_dict = {
    'V_inf': V_inf,
    'alpha': pitch,
    # 'V_inf': mesh_vel_list,
    'collocation_velocity': coll_vel_list,
    'solver_mode': 'unsteady',
    'nt': nt,
    'dt': dt,

    'BC_PM': 'Neumann',

    'partition_size': 1,

    'free_wake': False,
    'meshes': mesh_list,
    'core_radius': chord*1e-3,
    'dissipation': True,
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

vlm_meshes_nt = []
for i in range(2):
    vlm_mesh_i = csdl.Variable(value=actuated_prop_meshes[i,:])
    vlm_meshes_nt.append(vlm_mesh_i)
    pfse.add_thin_surface(
        mesh=vlm_mesh_i,
        coll_vel=csdl.Variable(value=collocation_velocity[i,:])
    )

pfse_output_names = [
    'mu',
    'x_w',
    'mu_w',
    # 'mesh',
    'panel_forces',
    'surf_CL',
    'surf_CDi',
    'surf_L',
    'Cp',
    'Cp_static'
]

pfse.declare_outputs(pfse_output_names)

pfse_outputs = pfse.evaluate()

mu = pfse_outputs['mu']
x_w = pfse_outputs['x_w']
mu_w = pfse_outputs['mu_w']
# mesh = pfse_outputs['mesh']
panel_forces = pfse_outputs['panel_forces']
surf_CL = pfse_outputs['surf_CL']
surf_CDi = pfse_outputs['surf_CDi']
surf_L = pfse_outputs['surf_L']
Cp = pfse_outputs['Cp']
Cp_static = pfse_outputs['Cp_static']

inputs = [pitch]
outputs = [mu, x_w, mu_w, panel_forces]
outputs.append(panel_mesh_nt)
outputs.extend(vlm_meshes_nt)
outputs.extend([surf_CL, surf_CDi, surf_L])
outputs.append(Cp)
outputs.append(Cp_static)
# exit()
sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs,
    gpu=True
)

start = time.time()
sim.run()
stop = time.time()
print(f'compile + run time: {stop-start} seconds')

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

mu_val = sim[mu]
mu_w_val = sim[mu_w]
x_w_val = sim[x_w]
Cp_val = sim[Cp]
Cp_static_val = sim[Cp_static]
panel_forces_val = sim[panel_forces]
panel_mesh_nt_val = sim[panel_mesh_nt]
vlm_meshes_nt_val = [sim[val] for val in vlm_meshes_nt]

wake_form = 'lines'

# iso cam for testing
iso_cam = dict(
    pos=(-21.3394, -35.5395, 29.9108),
    focal_point=(3.87856, -0.619522, 7.28538),
    viewup=(0.265880, 0.381579, 0.885272),
    roll=56.7716,
    distance=48.6545,
    clipping_range=(35.4764, 64.9839),
)

# close-up view
iso_cam = dict(
    pos=(-5.87304, -14.2492, 8.71827),
    focal_point=(2.16218, -3.12258, 1.50911),
    viewup=(0.265880, 0.381579, 0.885271),
    roll=56.7716,
    distance=15.5028,
    clipping_range=(8.76841, 28.1232),
)

if True:
    pfse.plot_unsteady(
        panel_mesh_nt_val,
        vlm_meshes_nt_val,
        x_w_val,
        mu_val,
        mu_w_val,
        wake_form=wake_form,
        interactive=False,
        camera=iso_cam,
        name=f'sep_prop_wing_PFSE_iso_pitch_{pitch_angle}_nt_{nt}' + f'_{wake_form}',
        fps=10
    )

# front cam for testing

front_cam = dict(
    position=(-31.8757, 0.0928217, 0.197711),
    focal_point=(1.27338, -0.446571, 2.47714),
    viewup=(-0.0685583, 2.62192e-3, 0.997644),
    roll=89.8498,
    distance=33.2317,
    clipping_range=(28.6933, 38.9886),
)
if False:
    pfse.plot_unsteady(
        panel_mesh_nt_val,
        vlm_meshes_nt_val,
        x_w_val,
        mu_val,
        mu_w_val,
        wake_form=wake_form,
        interactive=False,
        camera=front_cam,
        name=f'sep_prop_wing_PFSE_front_pitch_{pitch_angle}_nt_{nt}' + f'_{wake_form}',
        fps=10
    )

wing_CL = sim[surf_CL][:,0]
rev_per_second = RPM/60.

nondim_rev_time = time_vec*rev_per_second
if True:
    plt.figure(figsize=(8,5))
    plt.plot(nondim_rev_time, wing_CL, label='prop-wing')
    plt.plot([nondim_rev_time[0], nondim_rev_time[-1]], [0.40109179]*2, label='wing only')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=15)
    plt.grid()
    plt.xlabel('Revolutions', fontsize=15)
    plt.ylabel(r'$C_L$', fontsize=15)
    plt.savefig(f'PFSE_prop_wing_CL_vs_rev_nt_{nt}_dt_{dt}.pdf')
    plt.show()

Cp_all_val = np.zeros(mu.shape)
num_PM_panels = Cp_val.shape[1]
Cp_all_val[:,:num_PM_panels] = Cp_val
# zeroing out the VLM pressures

if True:
    pfse.plot_unsteady(
        panel_mesh_nt_val,
        vlm_meshes_nt_val,
        x_w_val,
        Cp_all_val,
        mu_w_val*0.,
        wake_form=wake_form,
        color_wake=False,
        interactive=False,
        camera=iso_cam,
        name=f'sep_prop_wing_Cp_PFSE_iso_pitch_{pitch_angle}_nt_{nt}' + f'_{wake_form}',
        fps=10
    )

Cp_static_all_val = np.zeros(mu.shape)
num_PM_panels = Cp_val.shape[1]
Cp_static_all_val[:,:num_PM_panels] = Cp_static_val
# zeroing out the VLM pressures

if False:
    pfse.plot_unsteady(
        panel_mesh_nt_val,
        vlm_meshes_nt_val,
        x_w_val,
        Cp_static_all_val,
        mu_w_val*0.,
        wake_form=wake_form,
        color_wake=False,
        bounds=[-5,1],
        interactive=False,
        camera=iso_cam,
        name=f'sep_prop_wing_Cp_static_PFSE_iso_pitch_{pitch_angle}_nt_{nt}' + f'_{wake_form}',
        fps=10
    )