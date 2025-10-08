'''NACA0012 rectangular wing: 

Example of a rectangular wing with a NACA0012 airfoil. <br>
Mesh is made using OpenVSP.

'''

import numpy as np
import csdl_alpha as csdl

from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution

from VortexAD import PanelMethod, PanelMethodMixed
from VortexAD import SAMPLE_GEOMETRY_PATH

# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

# set up input dictionary
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine_mixed.msh'# tri + quad
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine_quad.msh'# quads ONLY
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
    'compressibility': True
}

mixed = True

if not mixed:
    panel_method = PanelMethod(
        input_dict,
        # skip_geometry=True
    )

else:
    panel_method = PanelMethodMixed(
        input_dict,
    )


# declare outputs of interest
pm_outputs = [
    'CL',
    'CDi',
    'Cp',
    'sigma',
    'mu',
    'AIC_mu',
    'ql',
    'qm',
    'qn',
    'panel_area',
    'panel_center'
]

pm_outputs.append('delta_coll_point_quad')
# pm_outputs.append('delta_coll_point_triangle')

# if mixed:
#     # pm_outputs.append('delta_coll_point_triangle')
#     pm_outputs.append('delta_coll_point_quad')
# else:
#     pm_outputs.append('delta_coll_point')



panel_method.declare_outputs(pm_outputs)
# panel_method.mesh_filepath = mesh_file_path
# exit()

panel_method.setup_grid_properties(threshold_angle=125, plot=False) # optional for debugging

# run the panel method
outputs = panel_method.evaluate()

# read outputs
CL = outputs['CL']
CDi = outputs['CDi']
CP = outputs['Cp']
sigma = outputs['sigma']
mu = outputs['mu']
AIC_mu = outputs['AIC_mu']
ql = outputs['ql']
qm = outputs['qm']
qn = outputs['qn']
panel_area = outputs['panel_area']
panel_center = outputs['panel_center']
# delta_coll_point = outputs[pm_outputs[-1]]

dcp_quad = outputs['delta_coll_point_quad']
# dcp_tri = outputs['delta_coll_point_triangle']

# csdl-jax stuff
inputs = [pitch]
# outputs = [CL, CDi, CP, sigma, mu, AIC_mu, ql, qm, qn, panel_area, delta_coll_point]
outputs = [CL, CDi, CP, sigma, mu, AIC_mu, ql, qm, qn, panel_area, panel_center, dcp_quad]
# outputs.append(dcp_tri)

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
sigma_val = sim[sigma]
mu_val = sim[mu]
AIC_mu_val = sim[AIC_mu]
ql_val = sim[ql]
qm_val = sim[qm]
qn_val = sim[qn]
panel_area_val = sim[panel_area]
panel_center_val = sim[panel_center]
dcp_quad_val = sim[dcp_quad]
# dcp_tri_val = sim[dcp_tri]

print('CL:', CL_val)
print('CDi:', CDi_val)

data_dict = {
    'mu': mu_val,
    'Cp': CP_val,
    'ql': ql_val,
    'qm': qm_val,
    'qn': qn_val,
    'AIC_mu': AIC_mu_val,
    # 'delta_coll_point': dcp_val,
    'panel_area': panel_area_val
}

import pickle
file_name = f'data_mixed_{str(mixed)}.pkl'
# with open(file_name, 'wb') as file:
#     pickle.dump(data_dict, file)

panel_method.plot(CP_val, bounds=[-3,1])
panel_method.plot(mu_val)
# panel_method.plot(sigma_val)


upper_TE_cells = panel_method.upper_TE_cells
lower_TE_cells = panel_method.lower_TE_cells

points_orig = panel_method.points_orig
cells = panel_method.cells
cell_adjacency = panel_method.cell_adjacency
cell_types = cells.keys()
num_cells = np.sum([len(cells[cell_type]) for cell_type in cell_types])
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells[cell_type].tolist()

TE_coloring = np.zeros(shape=len(combined_cells))
# TE_coloring[upper_TE_cells] = 1
# TE_coloring[lower_TE_cells] = -1
TE_coloring[np.array(cell_adjacency['triangle']).flatten()] = -1
TE_coloring[:cells['triangle'].shape[0]] = 1

# TE_coloring[cells['quad']] = 2

plot_pressure_distribution(points_orig, TE_coloring, connectivity=combined_cells, interactive=True, top_view=False, cmap='rainbow')