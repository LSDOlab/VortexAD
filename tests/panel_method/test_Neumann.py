'''NACA0012 rectangular wing: 

Example of a rectangular wing with a NACA0012 airfoil. <br>
Mesh is made using OpenVSP.

'''

import numpy as np
import scipy
import csdl_alpha as csdl

from VortexAD import PanelMethod
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster.stl' # LE TE clustering + tri
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_mix.msh' # LE TE clustering + tri
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch_fine_mix.msh' # LE TE clustering + tri
# mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch.stl' # same with tip bunch
# mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster_tip_bunch_quad.msh' # quads?
pitch = csdl.Variable(value=np.array([5.]))
BC = 'Dirichlet'
# input dict
input_dict = {
    # 'Mach': 0.25,
    'V_inf': 10.,
    'alpha': pitch,
    'Cp cutoff': -5.,
    'mesh_path': mesh_file_path, # can alternatively load mesh in with connectivity/TE data
    'ref_area': 10., 
    # 'BC': 'Neumann',
    'BC': BC,
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
    'mu',

    'sigma',
    'AIC_mu',
    'AIC_sigma',
    'RHS',
    'Di_panel',
    'panel_forces',
]
panel_method.declare_outputs(pm_outputs)

panel_method.setup_grid_properties(threshold_angle=125, plot=True) # optional for debugging

# run the panel method
outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
CP = outputs['Cp']
mu = outputs['mu']

sigma = outputs['sigma']
AIC_mu = outputs['AIC_mu']
AIC_sigma = outputs['AIC_sigma']
RHS = outputs['RHS']
Di_panel = outputs['Di_panel']
panel_forces = outputs['panel_forces']

# csdl-jax stuff
inputs = [pitch]
# outputs = [CL, CDi, CP, mu]
outputs = [CL, CDi, CP, mu, sigma, AIC_mu, AIC_sigma, RHS, Di_panel, panel_forces]

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
mu_val = sim[mu]

print('CL:', CL_val)
print('CDi:', CDi_val)

if BC == 'Dirichlet':
    import pickle
    file_data = {
        'Cp': CP_val,
        'mu': mu_val,
        'CL': CL_val,
        'CDi': CDi_val,
    }
    file_name = 'Dirichlet_data.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(file_data, file)

elif BC == 'Neumann':
    import pickle
    file_name = 'Dirichlet_data.pkl'
    with open(file_name, 'rb') as file:
        file_data = pickle.load(file)
    
    Cp_delta = CP_val - file_data['Cp']
    Cp_error = np.linalg.norm(Cp_delta)/np.linalg.norm(file_data['Cp'])

    CL_error = np.linalg.norm(CL_val-file_data['CL'])/np.linalg.norm(file_data['CL'])
    CDi_error = np.linalg.norm(CDi_val-file_data['CDi'])/np.linalg.norm(file_data['CDi'])

    print(f'Cp relative error norm (%): {Cp_error * 100}')
    print(f'CL relative error norm (%): {CL_error * 100}')
    print(f'CDi relative error norm (%): {CDi_error * 100}')


aaa = np.arange(AIC_mu.shape[1])
AIC_mu_diag = sim[AIC_mu][0,aaa,aaa]


# residual error
asdf = np.matmul(sim[AIC_mu][0,:], mu_val[0,:]) - sim[RHS][0,:]

# relative residual
rel_res = np.linalg.norm(asdf)/(np.linalg.norm(sim[AIC_mu][0,:])*np.linalg.norm(mu_val[0,:]))


AIC_svd = scipy.linalg.svd(sim[AIC_mu][0,:])


panel_method.plot(CP_val, bounds=[-3.,1])
panel_method.plot(mu_val)
if BC == 'Dirichlet':
    AIC_mu_diag_bounds = None
elif BC == 'Neumann':
    AIC_mu_diag_bounds = [0,15]
panel_method.plot(AIC_mu_diag, bounds=AIC_mu_diag_bounds)
panel_method.plot(asdf) # plot log scale for Neumann to show difference in order of magnitude
panel_method.plot(sim[Di_panel])

if BC == 'Neumann':
    panel_method.plot(Cp_delta)