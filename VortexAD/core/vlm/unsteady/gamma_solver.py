import numpy as np
import csdl_alpha as csdl

from VortexAD.core.vlm.unsteady.direct_solve import direct_solve
from VortexAD.core.vlm.unsteady.iterative_solve import iterative_solve

def gamma_solver(mesh_dict, vectorized_mesh_dict, solver_options_dict, x_w, gamma_w):
    '''
    Solves for body circulation strengths
    Two possibilities:
    - direct solve via assembling the AIC matrix
        - can use ROMs and partitioned vectorization
    - iterative solve (GMRES) via matvec products with batching

    '''
    iterative = solver_options_dict['iterative']
    if not iterative: # direct solve
        lin_solve_dict = direct_solve(mesh_dict, vectorized_mesh_dict, solver_options_dict, x_w, gamma_w)
    elif iterative: # mat-vec iterative solve (GMRES)
        gamma = iterative_solve(mesh_dict, vectorized_mesh_dict, solver_options_dict)
        lin_solve_dict = {
            'gamma': gamma
        }
    return lin_solve_dict