__version__ = '0.0.0'
from pathlib import Path

from VortexAD.core.pm_class import PanelMethod
from VortexAD.core.vlm_class import VLM

# path to sample airfoils and geometries
ROOT = Path(__file__).parents[0]
SAMPLE_GEOMETRY_PATH = ROOT / 'core' / 'geometry' / 'sample_meshes'
AIRFOIL_PATH = ROOT / 'core' / 'geometry' / 'sample_airfoils'



