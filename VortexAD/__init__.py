__version__ = '0.0.0'
from pathlib import Path

# Solver classes.  Keep the VLM import independent of the panel method's
# optional mesh/JAX stack so lightweight VLM-only environments can use it.
from VortexAD.core.vlm_class import VortexLatticeMethod

try:
    from VortexAD.core.pm_class_tri import PanelMethodTri
    from VortexAD.core.pm_class import PanelMethod
except ModuleNotFoundError as error:
    if error.name not in {"jax", "jaxlib", "meshio"}:
        raise
    PanelMethodTri = None
    PanelMethod = None

# pfse (only unsteady)
try:
    from VortexAD.core.pfse_class import PFSE
except:
    pass

# panel method mesh/geometry utility functions
# from VortexAD.utils.unstructured_grids.cell_adjacency_old import find_cell_adjacency_old
# from VortexAD.utils.unstructured_grids.TE_detection_old import TE_detection_old
from VortexAD.utils.unstructured_grids.cell_adjacency import find_cell_adjacency
from VortexAD.utils.unstructured_grids.TE_detection import TE_detection

# Plotting is optional for solver-only installations.
try:
    from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution
except ModuleNotFoundError as error:
    if error.name != "vedo":
        raise
    plot_pressure_distribution = None

# path to sample airfoils and geometries
ROOT = Path(__file__).parents[0]
SAMPLE_GEOMETRY_PATH = ROOT / 'core' / 'geometry' / 'sample_meshes'
AIRFOIL_PATH = ROOT / 'core' / 'geometry' / 'sample_airfoils'
