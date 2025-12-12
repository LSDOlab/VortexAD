'''NACA0012 rectangular wing: 

Example of a rectangular wing with a NACA0012 airfoil. <br>
Mesh is made using OpenVSP.

'''

import numpy as np
import csdl_alpha as csdl

from VortexAD import PanelMethod
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster.stl' # LE TE clustering + tri
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_mix.msh' # LE TE clustering + tri
# mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch.stl' # same with tip bunch
# mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch_quad.msh' # quads?
pitch = csdl.Variable(value=np.array([5.]))
BC = 'Dirichlet'
# input dict
input_dict = {
    'Mach': 0.25,
    'alpha': pitch,
    'Cp cutoff': -5.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': 10., 
    'BC': BC,
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
    'mu'
]
panel_method.declare_outputs(pm_outputs)

panel_method.setup_grid_properties(threshold_angle=125, plot=True) # optional for debugging

# run the panel method
outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
CP = outputs['Cp']
mu = outputs['mu']

# csdl-jax stuff
inputs = [pitch]
outputs = [CL, CDi, CP, mu]

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs = outputs,
    gpu=False
)
sim.run()

CL_val = sim[CL]
CDi_val = sim[CDi]
CP_val = sim[CP]
mu_val = sim[mu]

print('CL:', CL_val)
print('CDi:', CDi_val)

panel_method.plot(CP_val, bounds=[-1.5,1])
panel_method.plot(mu_val)