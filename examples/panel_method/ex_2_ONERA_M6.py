'''ONERA M6 case: 

Example of the ONERA M6 case. <br>
Mesh is made using OpenVSP. <br>
We utilize additional solver inputs like compressibility corrections.

'''

import numpy as np
import csdl_alpha as csdl

from VortexAD import PanelMethod
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine.stl'
pitch = csdl.Variable(value=np.array([3.06]))

# input dict
input_dict = {
    'Mach': 0.7,
    'alpha': pitch,
    'Cp cutoff': -3.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': 1.51499, 
    'compressibility': True
}

# instantiate PanelMethod class
panel_method = PanelMethod(
    input_dict
)
# declare outputs of interest
pm_outputs = [
    'CL',
    'CDi',
    'Cp'
]
panel_method.declare_outputs(pm_outputs)

panel_method.setup_grid_properties(threshold_angle=90, plot=True) # optional for debugging

# run the panel method
outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
CP = outputs['Cp']

# csdl-jax stuff
inputs = [pitch]
outputs = [CL, CDi, CP]

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

print('CL:', CL_val)
print('CDi:', CDi_val)

panel_method.plot(CP_val, bounds=[-4,1])