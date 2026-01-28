import numpy as np
import csdl_alpha as csdl

from VortexAD import VortexLatticeMethod
from VortexAD.utils.meshing.gen_vlm_mesh import gen_vlm_mesh

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

ns, nc = 21, 7
b, c = 10., 1.
mesh_orig = gen_vlm_mesh(ns, nc, b, c)
mesh = csdl.Variable(value=mesh_orig).expand((1, nc, ns, 3), 'ijk->aijk')
mesh_list = [mesh]

pitch = csdl.Variable(value=np.array([5.]))

input_dict = {
    'V_inf': 10.,
    'alpha': pitch,
    'meshes': mesh_list
}

vlm = VortexLatticeMethod(
    input_dict
)
vlm_outputs = ['surface_lift', 'surface_CL', 'surface_CDi', 'gamma', 'wake_vortex_mesh', 'net_gamma']
vlm.declare_outputs(vlm_outputs)

outputs = vlm.evaluate()
L = outputs['surface_lift'][0]
CL = outputs['surface_CL'][0]
CDi = outputs['surface_CDi'][0]
gamma = outputs['gamma']
wvm = outputs['wake_vortex_mesh']
net_gamma = outputs['net_gamma']

# csdl-jax stuff
inputs = [pitch]
outputs = [L, CL, CDi, gamma, wvm, net_gamma]

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs,
    gpu=False
)
sim.run()
L_val = sim[L]
CL_val = sim[CL]
CDi_val = sim[CDi]