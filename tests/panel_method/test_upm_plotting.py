import numpy as np
import csdl_alpha as csdl
import time
import matplotlib.pyplot as plt
import pickle
import os
from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution

from VortexAD import PanelMethod
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
pitch = csdl.Variable(value=np.array([5.]))
# pitch = csdl.Variable(value=np.array([3.06]))

nt = 30

test_case = 'NACA'
if test_case == 'NACA':
    dt = .01
    V_inf = 10
    ref_area = 10.
    nt = 50
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster.stl' # triangles
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_quad.msh' # quad
    # mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch.stl' # triangles
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch_quad.msh' # quads?
    # mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch_fine_mix.msh' # quads?

elif test_case == 'BWB':
    # dt = csdl.Variable(value=0.025)
    dt = 0.025
    Mach = 0.7
    V_inf = Mach*340.3
    ref_area = 525.
    # mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/bwb.stl' # triangles
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/bwb_quad.msh' # triangles

elif test_case == 'ONERA':
    dt = csdl.Variable(value=0.05)
    V_inf = 10
    ref_area = 1.51499
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine_mixed.msh'# tri + quad
    # mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine_quad.msh'# quads ONLY
    # mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine.stl' # triangles

# dummy_ROM = np.einsum('i,jk->ijk', np.ones((nt,)), np.eye(1616, 15)) # time varying
# dummy_ROM = np.eye(1616, 1616) # static

# input dict
input_dict = {
    'V_inf': V_inf,
    'alpha': pitch,
    'Cp cutoff': -5.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': ref_area, 
    # 'partition_size': 1,
    'partition_size': None,
    'compressibility': False,

    'solver_mode': 'unsteady',
    'free_wake': True,
    'dt': dt,
    'nt': nt,
    'core_radius': 1.e-3,
    # 'ROM': [dummy_ROM.transpose(), dummy_ROM], # [phi^T, phi]
}

panel_method = PanelMethod(
    input_dict,
)

panel_method.setup_grid_properties(threshold_angle=125, plot=False) # optional for debugging

panel_method.setup_plotting_inputs()
with open('/home/luca/Packages/VortexAD/tests/panel_method' + '/' + 'PM_plotting_data.pkl', 'rb') as file:
    plotting_inputs = pickle.load(file)

mesh_val = plotting_inputs['mesh']
x_w_val = plotting_inputs['x_w']
mu_val = plotting_inputs['mu']
mu_w_val = plotting_inputs['mu_w']

wake_form = 'lines' # grid or lines
vid_name = test_case + '_' + wake_form
if True:
    # panel_method.plot(CP_val, bounds=[-3,1])
    cam = dict(
        pos=(-6.84211, -15.9857, 9.85074),
        focal_point=(3.15248, -2.72330, 0.754577),
        viewup=(0.135534, 0.488899, 0.861747),
        roll=46.7645,
        distance=18.9347,
        clipping_range=(2.82479, 39.1835),
    )
    panel_method.plot_unsteady(
        mesh_val, 
        x_w_val, 
        mu_val, 
        mu_w_val,
        # camera=cam,
        wake_form=wake_form, # grid or lines
        interactive=False, 
        name=vid_name)