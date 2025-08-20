import numpy as np
import csdl_alpha as csdl

from VortexAD import PanelMethod
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster.stl'
pitch = csdl.Variable(value=np.array([0.]))

# input dict
input_dict = {
    'Mach': 0.25,
    'alpha': pitch,
    'Cp cutoff': -5.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': 10., 
}

# instantiate PanelMethod class
panel_method = PanelMethod(
    input_dict
)
# declare outputs of interest
pm_outputs = [
    'CL',
    'CDi',
]
panel_method.declare_outputs(pm_outputs)

panel_method.setup_grid_properties(threshold_angle=125) # optional for debugging

# run the panel method
outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']

# csdl-jax stuff
inputs = [pitch]
outputs = [CL, CDi]

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[pitch],
    additional_outputs = [CL, CDi],
    gpu=False
)
sim.run()

CL_val = sim[CL]
CDi_val = sim[CDi]
