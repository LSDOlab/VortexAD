import numpy as np
import csdl_alpha as csdl
import time

from VortexAD import VortexLatticeMethod
from VortexAD.utils.meshing.gen_vlm_mesh import gen_vlm_mesh

import matplotlib.pyplot as plt

V_inf = 10.
nt, dt = 40, 0.04
ns, nc = 21, 7
b, c = 10., 1.
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

    'free_wake': True,
    'meshes': mesh_list,
    'core_radius': 1.e-6
}

vlm = VortexLatticeMethod(
    input_dict
)
vlm_outputs = ['CL', 'CDi', 'x_w', 'gamma', 'gamma_w', 'panel_force', 'net_gamma']
# vlm_outputs = ['CL', 'CDi', 'x_w', 'surf_CL', 'surf_CDi', 'gamma', 'gamma_w']
vlm.declare_outputs(vlm_outputs)
output_dict = vlm.evaluate()

CL = output_dict['CL']
CDi = output_dict['CDi']
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

inputs = [pitch]
outputs = [CL, CDi, x_w]
# outputs.extend([surf_0_CL, surf_1_CL, surf_0_CDi, surf_1_CDi])
outputs.extend([gamma, gamma_w, net_gamma])
outputs.extend(mesh_list)

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

CL_val = sim[CL]
CDi_val = sim[CDi]
x_w_val = sim[x_w]
gamma_val = sim[gamma]
gamma_w_val = sim[gamma_w]

print(CL_val)
print(CDi_val)

mesh_val_list = [
    sim[mesh], 
    # sim[mesh_dup]
]

# vlm.plot_unsteady(
#     mesh_val_list,
#     x_w_val,
#     gamma_val,
#     gamma_w_val,
#     interactive=False
# )

40, 0.04
t_vec = np.linspace(0,40*.04,40)

if True:
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(t_vec, CL_val[:,0])
    axs[0].set_ylabel('CL', fontsize=12)
    axs[0].grid()
    axs[1].plot(t_vec, CDi_val[:,0])
    axs[1].set_ylabel('CDi', fontsize=12)
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].grid()

plt.show()