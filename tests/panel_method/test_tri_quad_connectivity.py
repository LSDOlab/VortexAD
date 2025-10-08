'''NACA0012 rectangular wing: 

Example of a rectangular wing with a NACA0012 airfoil. <br>
Mesh is made using OpenVSP.

'''

import numpy as np
import csdl_alpha as csdl

import meshio

from VortexAD.utils.unstructured_grids.cell_adjacency import find_cell_adjacency
from VortexAD.utils.unstructured_grids.TE_detection import TE_detection

from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution

from VortexAD import SAMPLE_GEOMETRY_PATH

mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine_mixed.msh'# tri + quad
mesh_file_path = str(SAMPLE_GEOMETRY_PATH) + '/pm/onera_m6_fine.stl' # triangles

mesh = meshio.read(
    mesh_file_path
)

points_orig = mesh.points
# self.cells = mesh.cells
cells_dict_orig = mesh.cells_dict # keys are triangle, quad
cell_dict_keys = list(cells_dict_orig.keys())
combined_cells = []
if 'triangle' in cell_dict_keys:
    combined_cells.append(cells_dict_orig['triangle'].tolist())
if 'quad' in cell_dict_keys:
    combined_cells.append(cells_dict_orig['quad'].tolist())

cell_adjacency_data = find_cell_adjacency(
    points=points_orig, 
    cells=cells_dict_orig, 
)

points = cell_adjacency_data[0]
cells = cell_adjacency_data[1]
cell_adjacency = cell_adjacency_data[2]
edges2cells = cell_adjacency_data[3]
points2cells = cell_adjacency_data[4]


TE_properties = TE_detection(
    points=points,
    cells=cells,
    edges2cells=edges2cells,
    threshold_theta=125.
)
upper_TE_cells = TE_properties[0]
lower_TE_cells = TE_properties[1]
TE_edges = TE_properties[2]
TE_node_indices = TE_properties[3]


cell_types = cells.keys()
num_cells = np.sum([len(cells[cell_type]) for cell_type in cell_types])
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells[cell_type].tolist()


TE_coloring = np.zeros(shape=len(combined_cells))
# TE_coloring[upper_TE_cells] = 1
# TE_coloring[lower_TE_cells] = -1
TE_coloring[upper_TE_cells] = upper_TE_cells
TE_coloring[lower_TE_cells] = lower_TE_cells

plot_pressure_distribution(points_orig, TE_coloring, connectivity=combined_cells, interactive=True, top_view=False, cmap='rainbow')