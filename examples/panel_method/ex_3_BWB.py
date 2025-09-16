'''Blended wing body across many missions: 

Example of a BWB panel method analysis across many missions. <br>
Mesh is made using OpenVSP. <br>
We utilize additional solver inputs like compressibility corrections and reusing the AIC matrix.

<br>
<br>
Distribution Statement A: Approved for public release; distribution is unlimited. PA# AFRL-2025-3820.
'''

import numpy as np
import csdl_alpha as csdl

from VortexAD import PanelMethod
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/bwb.stl'
num_nodes = 6
pitch = csdl.Variable(value=np.arange(0,num_nodes))

# input dict
input_dict = {
    'Mach': 0.65,
    'alpha': pitch,
    'Cp cutoff': -5.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': 525., 
    'compressibility': True,
    'reuse_AIC': True
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
    'L',
    'Di'
]
panel_method.declare_outputs(pm_outputs)

panel_method.setup_grid_properties(threshold_angle=125, plot=True) # optional for debugging

# run the panel method
outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
CP = outputs['Cp']
L = outputs['L']
Di = outputs['Di']

# csdl-jax stuff
inputs = [pitch]
outputs = [CL, CDi, CP, L, Di]

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

L_val = sim[L]
Di_val = sim[Di]

print('CL:', CL_val)
print('CDi:', CDi_val)
print('L:', L_val)
print('Di:', Di_val)

panel_method.plot(CP_val[0,:], bounds=[-3,1])