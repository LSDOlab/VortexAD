import numpy as np
import csdl_alpha as csdl

from VortexAD import PanelMethod
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_mix.msh' # LE TE clustering + quads
pitch = csdl.Variable(value=np.array([5.]))
BC = 'Dirichlet'

batch_size = 1
num_cells = 808
basis_size = 10
basis_size = num_cells

# testing identity basis
# for a full identity basis, we should recover the FOM solution
U = np.eye(num_cells, basis_size)
ROM_basis = [U.T, U]

# input dict
input_dict = {
    'Mach': 0.25,
    'alpha': pitch,
    'Cp cutoff': -5.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': 10., 
    'BC': BC,
    'drag_type': 'Trefftz',
    'partition_size': batch_size, 
    'ROM': ROM_basis,
    # 'ROM': 'ROM-POD',
    # 'ROM_basis': [], # basis for ROM (POD or other offline-trained basis)
}

# instantiate PanelMethod class
panel_method = PanelMethod(
    input_dict
)
# declare outputs of interest
pm_outputs = [
    'CL',
    'CDi',
    'Cp',
    'mu',
    'L',
    'Di',

    'Di_Trefftz',
    'CDi_Trefftz',

    'AIC_mu',
    'RHS',
]
panel_method.declare_outputs(pm_outputs)

panel_method.setup_grid_properties(threshold_angle=125, plot=False) # optional for debugging

# run the panel method
outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
CP = outputs['Cp']
mu = outputs['mu']
L = outputs['L']
Di = outputs['Di']

Di_T = outputs['Di_Trefftz']
CDi_T = outputs['CDi_Trefftz']

AIC_mu = outputs['AIC_mu']
RHS = outputs['RHS']

# csdl-jax stuff
inputs = [pitch]
outputs = [CL, CDi, CP, mu]
outputs.extend([L, Di, Di_T, CDi_T])
outputs.extend([AIC_mu, RHS])

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs = outputs,
    gpu=False
)
sim.run()

CL_val = sim[CL]
CDi_val = sim[CDi]
CDi_T_val = sim[CDi_T]
CP_val = sim[CP]
mu_val = sim[mu]

print('CL:', CL_val)
print('CDi:', CDi_val)
print('CDi (Trefftz):', CDi_T_val)
exit()
panel_method.plot(CP_val, bounds=[-1.5,1])
panel_method.plot(mu_val)