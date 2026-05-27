import numpy as np
import csdl_alpha as csdl
import time
import matplotlib.pyplot as plt
import pickle
from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution

from VortexAD import PanelMethod
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
pitch = csdl.Variable(value=np.array([5.]))
# pitch = csdl.Variable(value=np.array([3.06]))

nt = 100

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
    'integration_method': 'ForwardEuler',
    # 'integration_method': 'RK2',
    # 'integration_method': 'RK3',
    # 'integration_method': 'RK4',
}

panel_method = PanelMethod(
    input_dict,
)

pm_outputs = [
    'CL',
    'CDi',
    'L',
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
# recorder.print_graph_structure()
# recorder.visualize_graph(filename='pm_graph', trim_loops=True, visualize_style='hierarchical')
# exit()
# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
L = outputs['L']
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
outputs.append(L)

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs,
    gpu=False
)
# start = time.time()
# sim.run()
# stop = time.time()
# print(f'compile + run time: {stop-start} seconds')

num_runs = 1
start_total = time.time()
for i in range(num_runs):
    start_run = time.time()
    sim.run()
    stop_run = time.time()
    print(f'run time: {stop_run-start_run} seconds')
stop_total = time.time()
print(f'total run time for {num_runs} runs: {stop_total-start_total} seconds')
print(f'average run time across {num_runs} runs: {(stop_total-start_total)/num_runs} seconds per run')

CL_val = sim[CL]
CDi_val = sim[CDi]
L_val = sim[L]
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
# exit()
t_vec = np.linspace(0,nt*dt,nt)
c=1
if True:
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(t_vec*V_inf/c, CL_val)
    axs[0].set_ylabel('CL', fontsize=12)
    axs[0].grid()
    axs[1].plot(t_vec*V_inf/c, CDi_val)
    axs[1].set_ylabel('CDi', fontsize=12)
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_xticks(np.arange(0,16,1))
    axs[1].grid()
    plt.show()

data_dict = {
    'time': t_vec,
    'CL': CL_val,
    'CDi': CDi_val,
    'L': L_val,
    'mu': mu_val,
    'mu_w': mu_w_val,
    'x_w': x_w_val,
}
file_name = 'PM_data_rect_wing.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(data_dict, file)

exit()

plotting_data_dict = {
    'mesh': mesh_val,
    'x_w': x_w_val,
    'mu': mu_val,
    'mu_w': mu_w_val,
}

file_name = 'PM_plotting_data.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(plotting_data_dict, file)


wake_conn = panel_method.wake_connectivity
wcs = wake_conn.shape
mu_w_val_grid = mu_w_val.reshape((nt,) + wcs[:-1])

bqs = 2.5
time_array = np.arange(0,nt*dt,dt)
dd_val = np.exp(-bqs*time_array)
mu_w_col = np.zeros_like(dd_val)
for i in range(nt-1):
    mu_w_col[i] = mu_w_val_grid[i+1].reshape(wcs[:-1])[:,0][i]

if True:
    plt.figure(figsize=(7,5))
    plt.plot(time_array[:-1], dd_val[:-1], '-', linewidth=3, label='Analytical dissipation')
    plt.plot(time_array[:-1], mu_w_col[:-1]/mu_w_col[0], '*', markersize=8, label='UPM wake dissipation')
    plt.xlabel('Time (s)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Relative Wake Vortex Strength', fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig('UPM_dissipation_plot.pdf')
    plt.show()