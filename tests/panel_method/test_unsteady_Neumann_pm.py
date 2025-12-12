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
    dt = csdl.Variable(value=0.1)
    V_inf = 10
    ref_area = 10.
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster.stl' # triangles
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_quad.msh' # quad
    mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch.stl' # triangles
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
    'Cp cutoff': -3.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': ref_area, 
    # 'partition_size': 1,
    'partition_size': None,
    'compressibility': False,
    # 'BC': 'Neumann',

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
panel_method.setup_grid_properties(threshold_angle=125, plot=True) # optional for debugging

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
recorder.active_graph.visualize()
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

num_runs = 5
start_total = time.time()
for i in range(num_runs):
    start_run = time.time()
    # sim.run()
    stop_run = time.time()
    print(f'run time: {stop_run-start_run} seconds')
stop_total = time.time()
print(f'total run time for {num_runs} runs: {stop_total-start_total} seconds')
print(f'average run time across {num_runs} runs: {(stop_total-start_total)/num_runs} seconds per run')

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
    


'''
NACA Data for 60 timesteps:
CL: [
 0.02536067, 0.03493555, 0.06171381, 0.09253669, 0.12014688, 0.14511074,
 0.16776847, 0.18834765, 0.20701238, 0.22402607, 0.23969752, 0.25376642,
 0.26650507, 0.27915003, 0.28945068, 0.29990465, 0.30861873, 0.31725834,
 0.32528997, 0.33211262, 0.3380989,  0.34411459, 0.35104886, 0.35575909,
 0.35981264, 0.36475638, 0.36843337, 0.37121042, 0.37499097, 0.37815458,
 0.38066648, 0.38382976, 0.38526945, 0.38795903, 0.389871,   0.3910531,
 0.39423671, 0.39521979, 0.397742,   0.39864417, 0.40118688, 0.40094152,
 0.40242101, 0.40378924, 0.40449431, 0.4075407,  0.40667181, 0.40808819,
 0.40707107, 0.40798877, 0.40929882, 0.41174788, 0.41203829, 0.41159881,
 0.41370783, 0.41212475, 0.4135777,  0.41168421, 0.41485791, 0.41479675]

 CDi: [
 -0.007451,   -0.0075075,  -0.0062224,  -0.00483262, -0.00355748, -0.00243766,
 -0.0014563,  -0.00059798,  0.0001436,   0.00078979,  0.00137892,  0.00188869,
  0.00230678,  0.00280969,  0.00310405,  0.00345371,  0.00363969,  0.00394069,
  0.00418914,  0.00438289,  0.00446701,  0.00448701,  0.00487922,  0.00505941,
  0.00509065,  0.00533852,  0.00539742,  0.00534015,  0.00543484,  0.00556244,
  0.00557434,  0.00576458,  0.00561559,  0.00567449,  0.0056796,   0.00558805,
  0.00567016,  0.00565917,  0.00584048,  0.00574715,  0.00599423,  0.00584219,
  0.00582521,  0.00583869,  0.0057905,   0.00614569,  0.00606201,  0.00620609,
  0.0058544,   0.00584088,  0.00579822,  0.00611835,  0.00618358,  0.00611876,
  0.00640031,  0.00609761,  0.00628823,  0.0059272,   0.00621403,  0.0062525 ]

'''

# if test_case == 'NACA':
CL_plot = np.array([
    0.02536067, 0.03493555, 0.06171381, 0.09253669, 0.12014688, 0.14511074,
    0.16776847, 0.18834765, 0.20701238, 0.22402607, 0.23969752, 0.25376642,
    0.26650507, 0.27915003, 0.28945068, 0.29990465, 0.30861873, 0.31725834,
    0.32528997, 0.33211262, 0.3380989,  0.34411459, 0.35104886, 0.35575909,
    0.35981264, 0.36475638, 0.36843337, 0.37121042, 0.37499097, 0.37815458,
    0.38066648, 0.38382976, 0.38526945, 0.38795903, 0.389871,   0.3910531,
    0.39423671, 0.39521979, 0.397742,   0.39864417, 0.40118688, 0.40094152,
    0.40242101, 0.40378924, 0.40449431, 0.4075407,  0.40667181, 0.40808819,
    0.40707107, 0.40798877, 0.40929882, 0.41174788, 0.41203829, 0.41159881,
    0.41370783, 0.41212475, 0.4135777,  0.41168421, 0.41485791, 0.41479675
])

CDi_plot = np.array([
    -0.007451,   -0.0075075,  -0.0062224,  -0.00483262, -0.00355748, -0.00243766,
    -0.0014563,  -0.00059798,  0.0001436,   0.00078979,  0.00137892,  0.00188869,
    0.00230678,  0.00280969,  0.00310405,  0.00345371,  0.00363969,  0.00394069,
    0.00418914,  0.00438289,  0.00446701,  0.00448701,  0.00487922,  0.00505941,
    0.00509065,  0.00533852,  0.00539742,  0.00534015,  0.00543484,  0.00556244,
    0.00557434,  0.00576458,  0.00561559,  0.00567449,  0.0056796,   0.00558805,
    0.00567016,  0.00565917,  0.00584048,  0.00574715,  0.00599423,  0.00584219,
    0.00582521,  0.00583869,  0.0057905,   0.00614569,  0.00606201,  0.00620609,
    0.0058544,   0.00584088,  0.00579822,  0.00611835,  0.00618358,  0.00611876,
    0.00640031,  0.00609761,  0.00628823,  0.0059272,   0.00621403,  0.0062525 
])
import matplotlib.pyplot as plt
time_vec = np.arange(nt)
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(9,5))
ax1.plot(time_vec, CL_plot)
ax1.grid()
ax1.set_ylabel('$C_L$', fontsize=15)
ax1.tick_params(axis='both', labelsize=12)
ax2.plot(time_vec, CDi_plot)
ax2.set_ylabel('$C_{Di}$', fontsize=15)
ax2.grid()
ax2.set_xlabel('Timesteps', fontsize=15)
ax2.tick_params(axis='both', labelsize=12)
plt.savefig('NACA0012_fw_convergence.pdf')
plt.show()
