import numpy as np
from VortexAD import AIRFOIL_PATH

mesh_file_path = str(AIRFOIL_PATH) + '/onera_m6_airfoil_half.dat'

mesh_orig = np.genfromtxt(mesh_file_path)[::-1,:]

nc_half = mesh_orig.shape[0]

nc = int((nc_half-1)*2) + 1

mesh = np.zeros((nc,2))
mesh[:nc_half,:] = mesh_orig
mesh[nc_half:,0] = mesh_orig[:-1,0][::-1]
mesh[nc_half:,1] = -mesh_orig[:-1,1][::-1]

np.savetxt(str(AIRFOIL_PATH) + '/onera_m6_airfoil.dat', mesh)