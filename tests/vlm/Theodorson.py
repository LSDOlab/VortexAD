import numpy as np
import csdl_alpha as csdl
import time
from scipy.special import j0, j1, y0, y1

from VortexAD import VortexLatticeMethod
from VortexAD.utils.meshing.gen_vlm_mesh import gen_vlm_mesh

import matplotlib.pyplot as plt

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
# pitch = csdl.Variable(value=np.array([5.]))
# pitch = csdl.Variable(value=np.array([3.06]))

rho = 1.225
V_inf = 10.
nt, dt = 100, 0.05
ns, nc = 14, 5
# ns, nc = 21, 7
AR = 10
c = 1
b = AR*c
# b, c = 10., 1.
ref_area = b*c
mesh_orig = gen_vlm_mesh(ns, nc, b, c)

k = 0.2
# k = omega*c/2/V_inf
omega = 2*V_inf*k/c

def theodorsen_func(k):
    j0_val = j0(k)
    j1_val = j1(k)
    y0_val = y0(k)
    y1_val = y1(k)

    # den = (j1_val+y0_val)**2 + (y1_val-j0_val)**2

    # F = (j1_val*(j1_val+y0_val) + y1_val*(y1_val-j0_val))/den
    # G = -(y1_val*y0_val + j1_val*j0_val)/den
    
    # C = F + G*1j

    H1 = j1_val-1j*y1_val
    H0 = j0_val - 1j*y0_val

    C = H1 / (H1 + 1j*H0)
    F = C.real
    G = C.imag
    return C, F, G

C_k, F_k, G_k = theodorsen_func(k)

t_vector = np.linspace(0, nt*dt, num=nt)
alpha_0 = 5.
alpha_amp = 5.
alpha_osc = alpha_0 + alpha_amp*np.sin(omega*t_vector) # deg
CL_qs = 2*np.pi*alpha_osc*np.pi/180.

RCL = CL_qs*F_k
ICL = CL_qs*G_k

phase = np.arctan(ICL/RCL)
# C_norm = (RCL**2 + ICL**2)**0.5
C_norm = np.abs(C_k)

CL_analytical = 2*np.pi*np.pi/180.*C_norm*(alpha_0 + alpha_amp*np.sin(omega*t_vector+phase))

V_vec = np.array([-V_inf, 0, 0])

alpha_osc_rad = alpha_osc*np.pi/180.
V_rot_mat = np.zeros((nt,3,3))
V_rot_mat[:,1,1] = 1.
V_rot_mat[:,0,0] = V_rot_mat[:,2,2] = np.cos(alpha_osc_rad)
V_rot_mat[:,2,0] = np.sin(alpha_osc_rad)
V_rot_mat[:,0,2] = -np.sin(alpha_osc_rad)

V_vec_rot = np.einsum('ijk,k->ij', V_rot_mat, V_vec)
point_velocities = np.einsum('ij,ab->iabj', V_vec_rot, np.ones((nc,ns)))


mesh_orig_var = csdl.Variable(value=mesh_orig)
mesh = csdl.expand(mesh_orig_var, (nt, nc, ns, 3), 'ijk->aijk')
point_velocities = csdl.Variable(value=point_velocities)


# input dict
input_dict = {
    'meshes': [mesh],
    'V_inf': point_velocities,
    'ref_area': ref_area, 
    # 'partition_size': 1,
    # 'partition_size': None,

    'solver_mode': 'unsteady',
    'free_wake': True,
    'dt': dt,
    'nt': nt,
    'core_radius': 1.e-3,
    # 'ROM': [dummy_ROM.transpose(), dummy_ROM], # [phi^T, phi]
}

vlm = VortexLatticeMethod(
    input_dict,
)

vlm_outputs = [
    'total_CL',
    'total_CDi',
    'gamma',
    'x_w',
    'gamma_w',
    # 'mesh',
]

vlm.declare_outputs(vlm_outputs)

outputs = vlm.evaluate()

# read outputs
CL = outputs['total_CL']
CDi = outputs['total_CDi']
gamma = outputs['gamma']
x_w = outputs['x_w']
gamma_w = outputs['gamma_w']
# mesh = outputs['mesh']
# total_force = outputs['F']

# csdl-jax stuff
inputs = [mesh_orig_var, point_velocities]
outputs = [CL, CDi, gamma, x_w, gamma_w, mesh]
# exit()
sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs,
    gpu=False
)
start = time.time()
sim.run()
stop = time.time()
print("=" * 25)
print(f'compile + run time: {stop-start} seconds')
print(f'number of timesteps: {nt}')
print(f'final simulation time: {nt*dt}')
print("=" * 25)

# num_runs = 5
# start_total = time.time()
# for i in range(num_runs):
#     start_run = time.time()
#     sim.run()
#     stop_run = time.time()
#     print(f'run time: {stop_run-start_run} seconds')
# stop_total = time.time()
# print(f'total run time for {num_runs} runs: {stop_total-start_total} seconds')
# print(f'average run time across {num_runs} runs: {(stop_total-start_total)/num_runs} seconds per run')

CL_val = sim[CL]
CDi_val = sim[CDi]
# CP_val = sim[CP]

print('CL:', CL_val)
print('CDi:', CDi_val)


mesh_val = sim[mesh]
x_w_val = sim[x_w]
gamma_val = sim[gamma]
gamma_w_val = sim[gamma_w]

wake_form = 'lines' # grid or lines
vid_name = f'Theodorson_k_{k}_' + wake_form
if True:
    # panel_method.plot(CP_val, bounds=[-3,1])
    cam = dict(
        pos=(-6.84211, -15.9857, 9.85074),
        focal_point=(3.15248, -2.72330, 0.754577),
        viewup=(0.135534, 0.488899, 0.861747),
        roll=46.7645,
        distance=18.9347,
        clipping_range=(2.82479, 39.1835),
    )

    cam = dict(
        pos=(-22.7709, -37.1225, 24.3477),
        focal_point=(3.15247, -2.72331, 0.754582),
        viewup=(0.135534, 0.488899, 0.861747),
        roll=46.7645,
        distance=49.1117,
        clipping_range=(41.4711, 61.6629),
    )

    vlm.plot_unsteady(
        [mesh_val], 
        x_w_val, 
        gamma_val, 
        gamma_w_val,
        camera=cam,
        wake_form=wake_form, # grid or lines
        interactive=False, 
        name=vid_name)
    
exit()

import matplotlib.pyplot as plt


# plt.figure()
# plt.plot(t_vector, h, label='position')
# plt.plot(t_vector, dhdt, label='velocity')
# plt.plot(t_vector, total_force_val[:,2].flatten()/AR, label='VortexAD')
# plt.plot(t_vector, Fz_analytical, label='Theodorson')
# plt.legend()

data_dict = {
    'alpha_osc': alpha_osc,
    'CL': CL_val.flatten(),
    'CL_analytical': CL_analytical
}

file_name = f'UVLM_Theodorsen_k_{k}.pkl'
import pickle
with open(file_name, 'wb') as file:
    pickle.dump(data_dict, file)


plt.figure()
plt.plot(t_vector, CL_val)

plt.figure()
plt.plot(alpha_osc, CL_val.flatten(), label=f'k={k}')
plt.plot(alpha_osc, CL_analytical, label='analytical')
plt.legend()
plt.grid()


plt.show()

