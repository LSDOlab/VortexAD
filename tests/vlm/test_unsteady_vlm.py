import numpy as np
import csdl_alpha as csdl
import time

from VortexAD import VortexLatticeMethod
from VortexAD.utils.meshing.gen_vlm_mesh import gen_vlm_mesh

import matplotlib.pyplot as plt
import pickle

V_inf = 10.
nt, dt = 15, 0.01
ns, nc = 14, 5
AR = 10
c = 1
b = AR*c
# b, c = 10., 1.
mesh_orig = gen_vlm_mesh(ns, nc, b, c)

mesh_dup = mesh_orig.copy()
mesh_dup[:,:,0] += 5
mesh_dup[:,:,2] += 0.5

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

pitch = csdl.Variable(value=np.array([5.]))

mesh = csdl.Variable(value=mesh_orig).expand((nt, nc, ns, 3), 'ijk->aijk')
mesh_list = [mesh]

# mesh_dup = csdl.Variable(value=mesh_dup).expand((nt, nc, ns, 3), 'ijk->aijk')
# mesh_list.append(mesh_dup)

input_dict = {
    'V_inf': 10.,
    'alpha': pitch,
    'solver_mode': 'unsteady',
    'nt': nt,
    'dt': dt,
    'ref_area': b*c, # NOTE: NEED TO FIX THE area VARIABLE WHEN COMPUTING COEFFICIENTS

    'free_wake': True,
    'meshes': mesh_list,
    'core_radius': 1.e-3
}

vlm = VortexLatticeMethod(
    input_dict
)
vlm_outputs = ['total_lift', 'total_CL', 'total_CDi', 'x_w', 'gamma', 'gamma_w', 'panel_force', 'net_gamma', 'wake_core_radius', 'dissipation_deriv']
# vlm_outputs = ['CL', 'CDi', 'x_w', 'surf_CL', 'surf_CDi', 'gamma', 'gamma_w']
vlm.declare_outputs(vlm_outputs)
output_dict = vlm.evaluate()

L = output_dict['total_lift']
CL = output_dict['total_CL']
CDi = output_dict['total_CDi']
x_w = output_dict['x_w']

# surf_CL = output_dict['surf_CL']
# surf_CDi = output_dict['surf_CDi']

# surf_0_CL = surf_CL[0]
# surf_1_CL = surf_CL[1]

# surf_0_CDi = surf_CDi[0]
# surf_1_CDi = surf_CDi[1]

gamma = output_dict['gamma']
gamma_w = output_dict['gamma_w']
net_gamma = output_dict['net_gamma']

panel_force =  output_dict['panel_force']

core_radius = output_dict['wake_core_radius']
diss_deriv = output_dict['dissipation_deriv']

inputs = [pitch]
outputs = [L, CL, CDi, x_w]
# outputs.extend([surf_0_CL, surf_1_CL, surf_0_CDi, surf_1_CDi])
outputs.extend([gamma, gamma_w, net_gamma])
outputs.extend(mesh_list)
outputs.extend([core_radius, diss_deriv])

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs,
    gpu=False
)
start = time.time()
sim.run()
end = time.time()

print(f'run + compile time: {end-start} seconds')
L_val = sim[L]
CL_val = sim[CL]
CDi_val = sim[CDi]
x_w_val = sim[x_w]
gamma_val = sim[gamma]
gamma_w_val = sim[gamma_w]
print(L_val)
print(CL_val)
print(CDi_val)

if True:
    mesh_val_list = [
        sim[mesh], 
        # sim[mesh_dup]
    ]

    vlm.plot_unsteady(
        mesh_val_list,
        x_w_val,
        gamma_val,
        gamma_w_val,
        interactive=False,
        name='uvlm_wing_sample_ani'
    )

t_vec = np.linspace(0,nt*dt,nt)

if True:
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(t_vec*V_inf/c, CL_val)
    axs[0].set_ylabel('CL', fontsize=12)
    axs[0].grid()
    axs[1].plot(t_vec*V_inf/c, CDi_val)
    axs[1].set_ylabel('CDi', fontsize=12)
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_xticks(np.arange(0,16,1))
    axs[1].grid()
    plt.show()

# data_dict = {
#     'time': t_vec,
#     'CL': CL_val,
#     'CDi': CDi_val
# }
# file_name = f'VLM_data_rect_wing_AR_{AR}.pkl'
# with open(file_name, 'wb') as file:
#     pickle.dump(data_dict, file)

# exit()

# verifying dissipation
bqs = 2.5
time_array = np.arange(0,nt*dt,dt)
dd_val = np.exp(-bqs*time_array)
gamma_w_col = np.zeros_like(dd_val)
for i in range(nt-1):
    gamma_w_col[i] = gamma_w_val[i+1].reshape(nt-1,ns-1)[:,0][i]

if True:
    plt.figure(figsize=(7,5))
    plt.plot(time_array[:-1], dd_val[:-1], '-', linewidth=3, label='Analytical dissipation')
    plt.plot(time_array[:-1], gamma_w_col[:-1]/gamma_w_col[0], '*', markersize=8, label='UVLM wake dissipation')
    plt.xlabel('Time (s)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Relative Wake Vortex Strength', fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    # plt.savefig('UVLM_dissipation_plot.pdf')
    plt.show()