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
from VortexAD import find_cell_adjacency, TE_detection
import meshio


manual_mesh_upload = True
reuse_AIC = False

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
file_name = 'bwb.stl'
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/' + file_name

if manual_mesh_upload:
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/' + file_name 
    mesh = meshio.read(
        mesh_file_path,
    )
    
    points_orig = mesh.points
    cells = mesh.cells
    cells_dict = mesh.cells_dict
    
    cell_adjacency_data = find_cell_adjacency(points=points_orig, cells=cells_dict)
    
    points_orig = cell_adjacency_data[0] 
    cells_dict = cell_adjacency_data[1] 
    cell_adjacency = cell_adjacency_data[2] 
    edges2cells = cell_adjacency_data[3]
    points2cells = cell_adjacency_data[4]
    
    TE_properties = TE_detection(
        points=points_orig,
        cells=cells_dict,
        edges2cells=edges2cells,
        points2cells=points2cells,
        threshold_theta=125.
    )
    
    
    default_panel_mesh = csdl.Variable(value=points_orig)
    
    x_scaler = csdl.Variable(value=np.array([1.]))
    y_scaler = csdl.Variable(value=np.array([1.]))
    z_scaler = csdl.Variable(value=np.array([1.]))
    
    panel_mesh = csdl.Variable(value=points_orig)
    panel_mesh = panel_mesh.set(csdl.slice[:,0], default_panel_mesh[:,0]*x_scaler)
    panel_mesh = panel_mesh.set(csdl.slice[:,1], default_panel_mesh[:,1]*y_scaler)
    panel_mesh = panel_mesh.set(csdl.slice[:,2], default_panel_mesh[:,2]*z_scaler)


reuse_AIC = False
if reuse_AIC:
    num_nodes = 6
    pitch = csdl.Variable(value=np.arange(0,num_nodes))
    # pitch_val = np.linspace(-5, 5, 11)
    # pitch = csdl.Variable(value=pitch_val)
else:
    pitch = csdl.Variable(value=np.array([0.]))


# input dict
input_dict = {
    'Mach': 0.65,
    'alpha': pitch,
    'Cp cutoff': -5.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': 525., 
    'compressibility': True,
    'reuse_AIC': reuse_AIC,
    'drag_type': 'Trefftz',
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
    'Di',

    'Di_Trefftz',
    'CDi_Trefftz',
]
panel_method.declare_outputs(pm_outputs)

if manual_mesh_upload:
    panel_method.insert_grid_data(
        # mesh=panel_mesh[0,:],
        mesh=panel_mesh,
        cell_adjacency_data=cell_adjacency_data,
        TE_properties=TE_properties
    )
else:
    panel_method.setup_grid_properties(threshold_angle=125, plot=True) # optional for debugging

# run the panel method
outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
CP = outputs['Cp']
L = outputs['L']
Di = outputs['Di']

Di_T = outputs['Di_Trefftz']
CDi_T = outputs['CDi_Trefftz']

# csdl-jax stuff
inputs = [pitch, x_scaler, y_scaler, z_scaler]
# outputs = [CL, CDi, CP, L, Di]
outputs = [CL, CDi, L, Di]
outputs.extend([Di_T, CDi_T])

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs = outputs,
    gpu=False
)
sim.run()

CL_val = sim[CL]
CDi_val = sim[CDi]
L_val = sim[L]
Di_val = sim[Di]

# CP_val = sim[CP]
# panel_method.plot(CP_val[0,:], bounds=[-3,1])

print('CL:', CL_val)
print('CDi:', CDi_val)
print('L:', L_val)
print('Di:', Di_val)



x_scaler_vals = np.array([0.8, 0.9, 0.95, 1.0, 1.1, 1.2])
for val in x_scaler_vals:
    sim[x_scaler] = val
    asdf = sim.run()
    print('========')
    print(f'L: {sim[L]}')
    print(f'CL: {sim[CL]}')
    print(f'Di: {sim[Di]}')
    print(f'CDi: {sim[CDi]}')
    print(f'Di_T: {sim[Di_T]}')
    print(f'CDi_T: {sim[CDi_T]}')

exit()

if not reuse_AIC:

    pitch_array = np.linspace(-5, 5, 11)

    L_array = np.zeros_like(pitch_array)
    CL_array = np.zeros_like(pitch_array)
    Di_array = np.zeros_like(pitch_array)
    CDi_array = np.zeros_like(pitch_array)
    Di_T_array = np.zeros_like(pitch_array)
    CDi_T_array = np.zeros_like(pitch_array)

    for i, val in enumerate(pitch_array):
        sim[pitch] = val
        asdf = sim.run()

        L_array[i] = sim[L]
        CL_array[i] = sim[CL]
        Di_array[i] = sim[Di]
        CDi_array[i] = sim[CDi]
        Di_T_array[i] = sim[Di_T]
        CDi_T_array[i] = sim[CDi_T]

    if True:
        import matplotlib.pyplot as plt
        plt.plot(pitch_array, CDi_array, label='pressure integration')
        plt.plot(pitch_array, CDi_T_array, label='Trefftz plane')
        plt.grid()
        plt.legend()
        plt.show()