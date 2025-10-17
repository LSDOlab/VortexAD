import numpy as np
import csdl_alpha as csdl
import time

from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution

from VortexAD import PanelMethod, PanelMethodTri
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine_mixed.msh'# tri + quad
# mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine_quad.msh'# quads ONLY
# mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine.stl' # triangles

pitch = csdl.Variable(value=np.array([3.06]))
# pitch = csdl.Variable(value=np.array([0.]))

# input dict
input_dict = {
    'Mach': 0.7,
    'alpha': pitch,
    'Cp cutoff': -3.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': 1.51499, 
    'partition_size': 1,
    'compressibility': True,

    'solver_mode': 'unsteady',
    'free_wake': True,
    'dt': 0.1,
    'nt': 5,
}

panel_method = PanelMethod(
    input_dict,
)

pm_outputs = [
    'CL',
    'CDi',
    'Cp',
]

panel_method.declare_outputs(pm_outputs)
panel_method.setup_grid_properties(threshold_angle=125, plot=False) # optional for debugging

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
    additional_outputs=outputs,
    gpu=False
)
start = time.time()
sim.run()
stop = time.time()
print(f'compile + run time: {stop-start} seconds')

start_run = time.time()
sim.run()
stop_run = time.time()
print(f'run time: {stop_run-start_run} seconds')

CL_val = sim[CL]
CDi_val = sim[CDi]
CP_val = sim[CP]

print('CL:', CL_val)
print('CDi:', CDi_val)

panel_method.plot(CP_val, bounds=[-3,1])
# panel_method.plot_unsteady()