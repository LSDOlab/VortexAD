import numpy as np
import csdl_alpha as csdl

from VortexAD.core.elements.doublet import compute_doublet_influence_new
from VortexAD.core.elements.source import compute_source_influence_new
from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel

def AIC_computation(mesh_dict, wake_mesh_dict, mode='unstructured', batch_size=None, bc='Dirichlet', ROM=False, constant_geometry=False, compute_wake=True):
    num_nodes = 1

    AIC_list = []
    # wake AIC first
    coll_point_eval = mesh_dict['panel_center_mod']
    if not constant_geometry:
        cells = mesh_dict['cell_point_indices'] # keys are cell types, entries are points for each cell
        cell_types = list(cells.keys())
        cell_adjacency_types = mesh_dict['cell_adjacency'] # keys are cell types, entries are adjacent cell indices
        num_cells_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]
        num_tot_panels = sum(num_cells_per_type)

        upper_TE_cell_ind = mesh_dict['upper_TE_cells']
        lower_TE_cell_ind = mesh_dict['lower_TE_cells']
        num_wake_panels = wake_mesh_dict['num_panels']

        AIC_mu = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
        AIC_sigma = csdl.Variable(shape=AIC_mu.shape, value=0.)
        AIC_batch_func = csdl.experimental.batch_function(
            compute_aic_batched,
            batch_size=batch_size,
            batch_dims=[1]+[None]*8
        )

        start_j, stop_j = 0, 0
        for j, cell_type_j in enumerate(cell_types):
            num_cells_j = num_cells_per_type[j]
            stop_j += num_cells_j

            coll_point = mesh_dict['panel_center_' + cell_type_j] # (nn, num_tot_panels, 3)
            panel_corners = mesh_dict['panel_corners_' + cell_type_j] # (nn, num_tot_panels, 3, 3) 
            panel_x_dir = mesh_dict['panel_x_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
            panel_y_dir = mesh_dict['panel_y_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
            panel_normal = mesh_dict['panel_normal_' + cell_type_j] # (nn, num_tot_panels, 3)
            S = mesh_dict['S_' + cell_type_j]
            SL = mesh_dict['SL_' + cell_type_j]
            SM = mesh_dict['SM_' + cell_type_j]

            # insert batched assembly function here

            doublet_influence, source_influence = AIC_batch_func(
                coll_point_eval,
                coll_point,
                panel_corners,
                panel_x_dir,
                panel_y_dir,
                panel_normal,
                S,
                SL,
                SM,
                BC=bc,
                do_source=True,
            )

            doublet_influence = doublet_influence[:,0,:].reshape((num_tot_panels, num_cells_j))
            source_influence = source_influence[:,0,:].reshape((num_tot_panels, num_cells_j))

            AIC_mu = AIC_mu.set(csdl.slice[0, :, start_j:stop_j], doublet_influence)
            AIC_sigma = AIC_sigma.set(csdl.slice[0, :, start_j:stop_j], source_influence)

            start_j += num_cells_j


        # if bc == 'Dirichlet':
        #     AIC_mu, AIC_sigma = compute_AIC_Dirichlet(mesh_dict, eval_pt, do_source=True)
        # elif bc == 'Neumann':
        #     AIC_mu, AIC_sigma = compute_AIC_Neumann(mesh_dict, eval_pt, panel_normal, do_source=True)
        AIC_list.append(AIC_mu)
        AIC_list.append(AIC_sigma)

    if compute_wake:
        cells = mesh_dict['cell_point_indices'] # keys are cell types, entries are points for each cell
        cell_types = list(cells.keys())
        cell_adjacency_types = mesh_dict['cell_adjacency'] # keys are cell types, entries are adjacent cell indices
        num_cells_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]
        num_tot_panels = sum(num_cells_per_type)

        upper_TE_cell_ind = mesh_dict['upper_TE_cells']
        lower_TE_cell_ind = mesh_dict['lower_TE_cells']
        num_wake_panels = wake_mesh_dict['num_panels']

        AIC_wake = csdl.Variable(shape=(num_nodes, num_tot_panels, num_wake_panels), value=0.)

        AIC_batch_func = csdl.experimental.batch_function(
            compute_aic_batched,
            batch_size=batch_size,
            batch_dims=[1]+[None]*8
            # batch_dims=[None]+[1]*8
        )

        # compute the AIC wake matrix here (reduced shape of (num_panels,num_wake_panels))

        panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, np_w, 4, 3)
        coll_point_w = wake_mesh_dict['panel_center'] # (nn, np_w, 3)
        panel_x_dir_w = wake_mesh_dict['panel_x_dir'] # (nn, np_w, 3)
        panel_y_dir_w = wake_mesh_dict['panel_y_dir'] # (nn, np_w, 3)
        panel_normal_w = wake_mesh_dict['panel_normal'] # (nn, np_w, 3)
        S_w = wake_mesh_dict['S']
        SL_w = wake_mesh_dict['SL']
        SM_w = wake_mesh_dict['SM']

        wake_doublet_influence_vec = AIC_batch_func(
            coll_point_eval,
            coll_point_w,
            panel_corners_w,
            panel_x_dir_w,
            panel_y_dir_w,
            panel_normal_w,
            S_w,
            SL_w,
            SM_w,
            BC=bc,
            do_source=False,
        )
        wake_doublet_influence = wake_doublet_influence_vec.reshape((1,num_tot_panels,num_wake_panels))
        AIC_wake = wake_doublet_influence
        # AIC_wake = AIC_wake.set(csdl.slice[:,start_i:stop_i,:], wake_doublet_influence)

        AIC_list.append(AIC_wake)

    '''
    NOTE: We want to separate the AIC matrix generation of the doublets and wake for the ROMs here as well.
    For the unsteady solver, the wake AIC is treated as ITS OWN TERM and is not absorbed into the doublet AIC.
    Therefore, it's okay to separate them here (unlike the steady solver).
    '''
    return AIC_list

def compute_aic_batched(coll_point, panel_center, panel_corners, panel_x_dir, panel_y_dir,
                        panel_normal, S_j, SL_j, SM_j, BC='Dirichlet', do_source=True):
    if BC == 'Dirichlet':
        influence_mode = 'potential'
    elif BC == 'Neumann':
        influence_mode = 'velocity'
    num_nodes = coll_point.shape[0]
    num_eval_pts = coll_point.shape[1]
    num_induced_pts = panel_center.shape[1]
    num_interactions = num_eval_pts*num_induced_pts
    num_corners = panel_corners.shape[2]

    expanded_shape = (num_nodes, num_eval_pts, num_induced_pts, num_corners, 3)
    vectorized_shape = (num_nodes, num_interactions, num_corners, 3)

    # ============ expanding across columns ============
    coll_point_exp = csdl.expand(coll_point, expanded_shape, 'ijk->ijabk')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    # ============ expanding across rows ============
    coll_point_j_exp = csdl.expand(panel_center, expanded_shape, 'ijk->iajbk')
    coll_point_j_exp_vec = coll_point_j_exp.reshape(vectorized_shape)

    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'ijkl->iajkl')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    panel_x_dir_exp = csdl.expand(panel_x_dir, expanded_shape, 'ijk->iajbk')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)
    panel_y_dir_exp = csdl.expand(panel_y_dir, expanded_shape, 'ijk->iajbk')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)
    panel_normal_exp = csdl.expand(panel_normal, expanded_shape, 'ijk->iajbk')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    S_j_exp = csdl.expand(S_j, expanded_shape[:-1] , 'ijk->iajk')
    S_j_exp_vec = S_j_exp.reshape(vectorized_shape[:-1])

    SL_j_exp = csdl.expand(SL_j, expanded_shape[:-1], 'ijk->iajk')
    SL_j_exp_vec = SL_j_exp.reshape(vectorized_shape[:-1])

    SM_j_exp = csdl.expand(SM_j, expanded_shape[:-1], 'ijk->iajk')
    SM_j_exp_vec = SM_j_exp.reshape(vectorized_shape[:-1])

    a = coll_point_exp_vec - panel_corners_exp_vec # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_j_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_exp_vec, axes=(sum_ind,)) # normal projection of CP
    # print(A.shape)
    B = csdl.Variable(shape=A.shape, value=0.)
    B = B.set(csdl.slice[:,:,:-1], value=A[:,:,1:])
    B = B.set(csdl.slice[:,:,-1], value=A[:,:,0])

    BL = csdl.Variable(shape=AL.shape, value=0.)
    BL = BL.set(csdl.slice[:,:,:-1], value=BL[:,:,1:])
    BL = BL.set(csdl.slice[:,:,-1], value=BL[:,:,0])

    BM = csdl.Variable(shape=AM.shape, value=0.)
    BM = BM.set(csdl.slice[:,:,:-1], value=AM[:,:,1:])
    BM = BM.set(csdl.slice[:,:,-1], value=AM[:,:,0])

    A1 = AM*SL_j_exp_vec - AL*SM_j_exp_vec

    A_list = [A[:,:,ind] for ind in range(num_corners)]
    AM_list = [AM[:,:,ind] for ind in range(num_corners)]
    B_list = [B[:,:,ind] for ind in range(num_corners)]
    BM_list = [BM[:,:,ind] for ind in range(num_corners)]
    SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(num_corners)]
    SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(num_corners)]
    A1_list = [A1[:,:,ind] for ind in range(num_corners)]
    PN_list = [PN[:,:,ind] for ind in range(num_corners)]
    S_list = [S_j_exp_vec[:,:,ind] for ind in range(num_corners)]

    if BC == 'Dirichlet':
        AIC_mu_vec = compute_doublet_influence_new(
            A_list, 
            AM_list, 
            B_list, 
            BM_list, 
            SL_list, 
            SM_list, 
            A1_list, 
            PN_list, 
            mode='potential'
        )
    elif BC == 'Neumann':
        pass
    if do_source:
        AIC_sigma_vec = compute_source_influence_new(
            A_list, 
            AM_list, 
            B_list, 
            BM_list, 
            SL_list, 
            SM_list, 
            A1_list, 
            PN_list, 
            S_list, 
            mode=influence_mode
        )
    
    if BC == 'Neumann': # do normal vector projections here
        pass
    if do_source:
        return AIC_mu_vec, AIC_sigma_vec
    else:
        return AIC_mu_vec