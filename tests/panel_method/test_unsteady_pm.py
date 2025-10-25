import numpy as np
import csdl_alpha as csdl
import time

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
    dt = csdl.Variable(value=0.05)
    V_inf = 10
    ref_area = 10.
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster.stl' # triangles
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_quad.msh' # quad
    # mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch.stl' # triangles
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch.msh' # quads?

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
    'Cp cutoff': -3.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': ref_area, 
    # 'partition_size': 1,
    'partition_size': None,
    'compressibility': True,

    'solver_mode': 'unsteady',
    'free_wake': False,
    'dt': dt,
    'nt': nt,
    'core_radius': 1.e-3,
    # 'ROM': [dummy_ROM.transpose(), dummy_ROM], # [phi^T, phi]
}

panel_method = PanelMethod(
    input_dict,
)

pm_outputs = [
    'CL',
    'CDi',
    'Cp',
    'mu',
    # 'AIC_mu',
    # 'AIC_sigma',
    # 'AIC_mu_wake',
    'x_w',
    'mu_w',
    'mesh',
    # 'AIC_fw_sigma',
    # 'wake_vel',
]

panel_method.declare_outputs(pm_outputs)
panel_method.setup_grid_properties(threshold_angle=125, plot=False) # optional for debugging

outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
CP = outputs['Cp']
mu = outputs['mu']
# AIC_mu = outputs['AIC_mu']
# AIC_sigma = outputs['AIC_sigma']
# AIC_mu_wake = outputs['AIC_mu_wake']
x_w = outputs['x_w']
mu_w = outputs['mu_w']
mesh = outputs['mesh']
# AIC_fw_sigma = outputs['AIC_fw_sigma']
# wake_vel = outputs['wake_vel']

# csdl-jax stuff
inputs = [pitch]
# outputs = [CL, CDi, CP, mu, AIC_mu_wake, x_w]
# outputs = [CL, CDi, CP, mu, x_w, mu_w, mesh]
# outputs = [CL, CDi, CP, mu, x_w, mu_w, mesh, AIC_fw_sigma, wake_vel]
outputs = [CL, CDi, mu, x_w, mu_w, mesh]

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

# start_run = time.time()
# sim.run()
# stop_run = time.time()
# print(f'run time: {stop_run-start_run} seconds')

CL_val = sim[CL]
CDi_val = sim[CDi]
# CP_val = sim[CP]

print('CL:', CL_val)
print('CDi:', CDi_val)


mesh_val = sim[mesh]
x_w_val = sim[x_w]
mu_val = sim[mu]
mu_w_val = sim[mu_w]

wake_form = 'lines' # grid or lines
vid_name = test_case + '_' + wake_form
if False:
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