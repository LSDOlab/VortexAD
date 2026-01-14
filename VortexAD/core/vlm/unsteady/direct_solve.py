import numpy as np
import csdl_alpha as csdl

from VortexAD.core.vlm.unsteady.AIC_computation import compute_AIC

def direct_solve(mesh_dict, vectorized_mesh_dict, solver_options_dict, x_w, gamma_w):
    '''
    Docstring for direct_solve
    
    :param mesh_dict: Description
    :param solver_options_dict: Description

    Computing vortex ring strengths
    steps:
    - compute AIC for surface + wake (with Kutta condition)
        - use partitioned vectorization
    - matvec for linear system BC 
    '''
    ROM = solver_options_dict['ROM']
    batch_size = solver_options_dict['partition_size']

    AIC_matrices = compute_AIC(mesh_dict, vectorized_mesh_dict, solver_options_dict)
    AIC, AIC_w = AIC_matrices[0], AIC_matrices[1]

    panel_normal = vectorized_mesh_dict['panel_normal']
    coll_vel = vectorized_mesh_dict['collocation_velocity']

    BC = csdl.sum(coll_vel[0,:]*panel_normal[0,:], axes=(1,))
    wake_influence = csdl.matvec(AIC_w, gamma_w)

    # asdf = np.ones((AIC_w.shape[0], AIC_w.shape[1]))
    # wake_influence = csdl.matvec(asdf, gamma_w)

    RHS = -(BC + wake_influence)
    # RHS = -csdl.sum(coll_vel[0,:]*panel_normal[0,:], axes=(1,))

    gamma = csdl.solve_linear(AIC, RHS)

    lin_solve_dict = {
        'gamma': gamma,
        'AIC': AIC,
        'AIC_w': AIC_w,
        'RHS': RHS,
        'BC': BC,
        'wake_influence': wake_influence,
    }

    return lin_solve_dict