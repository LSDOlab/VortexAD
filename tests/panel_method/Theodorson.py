import numpy as np
import csdl_alpha as csdl
import time
from scipy.special import j0, j1, y0, y1

from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution

from VortexAD import PanelMethod
from VortexAD import SAMPLE_GEOMETRY_PATH
import meshio

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
pitch = csdl.Variable(value=np.array([5.]))
# pitch = csdl.Variable(value=np.array([3.06]))

nt = 150

dt = 0.25
V_inf = 1
AR = 10.
AR_0 = 10.
c = 1.
ref_area = AR*c
rho = 1.225
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster.stl' # triangles
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_quad.msh' # quad
# mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch.stl' # triangles
# mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch_quad.msh' # quads?
# mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_high_AR.msh' # quads?

mesh_details = meshio.read(
    mesh_file_path
)
num_panels = mesh_details.cells_dict['quad'].shape[0]
points_orig = mesh_details.points
points_new = np.copy(points_orig)
points_new[:,1] *= AR/AR_0

k = np.pi/10
# k = omega*c/2/V_inf
omega = 2*V_inf*k/c

t_vector = np.linspace(0, nt*dt, num=nt)
a = 0.05 # velocity amplitude
# h = a*np.cos(omega*t_vector)
dhdt = -a*np.sin(omega*t_vector)
d2hdt2 = -a*omega*np.cos(omega*t_vector)
h = a/omega*np.cos(omega*t_vector)

def theodorsen_func(k):
    j0_val, j1_val = j0(k), j1(k)
    y0_val, y1_val = y0(k), y1(k)

    den = (j1_val+y0_val)**2 + (y1_val-j0_val)**2

    F = (j1_val*(j1_val+y0_val) + y1_val*(y1_val-j0_val))/den
    G = -(y1_val*y0_val + j1_val*j0_val)/den
    
    C = F + G*1j
    return C, F, G

C_k, F_k, G_k = theodorsen_func(k)

# Fz_analytical = -np.pi*rho*V_inf*c*C_k.real*dhdt - np.pi*rho*c**2/4*d2hdt2
Fz_analytical = -np.pi*rho*V_inf*c*abs(C_k)*dhdt - np.pi*rho*c**2/4*d2hdt2


alpha_deg = 5.
alpha = alpha_deg*np.pi/180.
x_vel = V_inf * np.cos(alpha)
z_vel = V_inf * np.sin(alpha)
alpha_h = np.arctan(dhdt/V_inf)
alpha_eff = alpha - alpha_h
# alpha_eff = np.arctan((-dhdt+z_vel)/x_vel)

dhdt_grid = csdl.Variable(value=np.zeros((nt, num_panels, 3)))
# dhdt_grid = dhdt_grid.set(csdl.slice[:,:,2], csdl.expand(dhdt, (nt, num_panels), 'i->ia'))
# dhdt_grid = dhdt_grid.set(csdl.slice[:,:,0], -V_inf)
dhdt_grid = dhdt_grid.set(csdl.slice[:,:,2], -z_vel + np.cos(alpha)*csdl.expand(dhdt, (nt, num_panels), 'i->ia'))
dhdt_grid = dhdt_grid.set(csdl.slice[:,:,0], -x_vel - np.sin(alpha)*csdl.expand(dhdt, (nt, num_panels), 'i->ia'))


# input dict
input_dict = {
    # 'V_inf': V_inf,
    # 'collocation_velocity': dhdt_grid,
    'V_inf': dhdt_grid,
    'Cp cutoff': -3.,
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

pm_outputs = [
    'CL',
    'CDi',
    'Cp',
    'mu',
    'F',
    'x_w',
    'mu_w',
    'mesh',
]

panel_method.declare_outputs(pm_outputs)
panel_method.setup_grid_properties(threshold_angle=125, plot=False) # optional for debugging
panel_method.overwrite_mesh(points_new)

outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
CP = outputs['Cp']
mu = outputs['mu']
x_w = outputs['x_w']
mu_w = outputs['mu_w']
mesh = outputs['mesh']
total_force = outputs['F']

# csdl-jax stuff
inputs = [pitch]
outputs = [CL, CDi, mu, x_w, mu_w, mesh, total_force]
# exit()
sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs,
    gpu=False
)
start = time.time()
sim.run()
stop = time.time()
print("=" * 25)
print(f'compile + run time: {stop-start} seconds')
print(f'number of timesteps: {nt}')
print(f'final simulation time: {nt*dt}')
print("=" * 25)

# num_runs = 5
# start_total = time.time()
# for i in range(num_runs):
#     start_run = time.time()
#     sim.run()
#     stop_run = time.time()
#     print(f'run time: {stop_run-start_run} seconds')
# stop_total = time.time()
# print(f'total run time for {num_runs} runs: {stop_total-start_total} seconds')
# print(f'average run time across {num_runs} runs: {(stop_total-start_total)/num_runs} seconds per run')

CL_val = sim[CL]
CDi_val = sim[CDi]
# CP_val = sim[CP]

print('CL:', CL_val)
print('CDi:', CDi_val)


mesh_val = sim[mesh]
x_w_val = sim[x_w]
mu_val = sim[mu]
mu_w_val = sim[mu_w]
total_force_val = sim[total_force]

wake_form = 'lines' # grid or lines
vid_name = 'Theodorson' + '_' + wake_form
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
    
exit()

import matplotlib.pyplot as plt


# plt.figure()
# plt.plot(t_vector, h, label='position')
# plt.plot(t_vector, dhdt, label='velocity')
# plt.plot(t_vector, total_force_val[:,2].flatten()/AR, label='VortexAD')
# plt.plot(t_vector, Fz_analytical, label='Theodorson')
# plt.legend()

plt.figure()
plt.plot(t_vector, CL_val)

plt.figure()
plt.plot(alpha_eff*180/np.pi, CL_val.flatten(), label=f'k={k}')
plt.legend()
plt.grid()

plt.figure()
plt.plot(dhdt, CL_val.flatten(), label=f'k={k}')
plt.legend()
plt.grid()


plt.show()

'''
NOTE: need to change the sign of one of the inputs
- the slope of the last plot is negative, so somewhere the alpha is incorrect
'''