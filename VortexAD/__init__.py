__version__ = '0.0.0'
from pathlib import Path

# solver classes
from VortexAD.core.pm_class_tri import PanelMethodTri
from VortexAD.core.pm_class import PanelMethod
from VortexAD.core.vlm_class import VLM

# panel method mesh/geometry utility functions
# from VortexAD.utils.unstructured_grids.cell_adjacency_old import find_cell_adjacency_old
# from VortexAD.utils.unstructured_grids.TE_detection_old import TE_detection_old
from VortexAD.utils.unstructured_grids.cell_adjacency import find_cell_adjacency
from VortexAD.utils.unstructured_grids.TE_detection import TE_detection

# plotting functions
from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution

# path to sample airfoils and geometries
ROOT = Path(__file__).parents[0]
SAMPLE_GEOMETRY_PATH = ROOT / 'core' / 'geometry' / 'sample_meshes'
AIRFOIL_PATH = ROOT / 'core' / 'geometry' / 'sample_airfoils'


