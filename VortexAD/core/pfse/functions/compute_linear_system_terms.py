import numpy as np
import csdl_alpha as csdl

from VortexAD.core.elements.doublet import compute_doublet_influence_new
from VortexAD.core.elements.source import compute_source_influence_new
from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel

def compute_linear_system_terms(pm_mesh_dict, vlm_vectorized_dict, wake_mesh_dict, sigma, mu_w, num_tot_panels, batch_size, bc):

    num_nodes       = 1
    num_tot_panels  = num_tot_panels

    # panel method cell types
    cells = pm_mesh_dict['cell_point_indices'] # keys are cell types, entries are points for each cell
    cell_types = list(cells.keys())

    influence_methods = ['PM'] * len(cell_types) # NOTE: CHANGE NAME
    cell_types.append('VLM')
    influence_methods.append('VLM')
    bc_methods = ['PM', 'VLM']

    AIC_mu = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    RHS_sigma = csdl.Variable(shape=(num_nodes, num_tot_panels, len(influence_methods)), value=0.)
    start_i, stop_i = 0, 0

    for i, bc_method in enumerate(bc_methods): # looping over row sections
        if bc_method == 'PM': # applying either Dirichlet or Neumann BC for PM surface
            coll_point_eval = pm_mesh_dict['panel_center_mod']
            normal_vec_eval = pm_mesh_dict['panel_normal']
            batch_dims = [1]*2+[None]*8

        elif bc_method == 'VLM': # applying no-penetration BC for VLM surface
            coll_point_eval = vlm_vectorized_dict['panel_centers']
            normal_vec_eval = vlm_vectorized_dict['panel_normal']
            
        num_panels_iter_i = coll_point_eval.shape[1]
        stop_i += num_panels_iter_i

        batch_size_surf = batch_size
        if batch_size is None:
            batch_size_surf = num_tot_panels
        # AIC_batch_func = csdl.experimental.batch_function(
        #     AIC_func,
        #     # batch_size=batch_size,
        #     batch_size=batch_size_surf,
        #     batch_dims=batch_dims
        # )
        start_j, stop_j = 0, 0
        for j, cell_type_j in enumerate(cell_types): # looping over column sections
            influence_method = influence_methods[j] # method where singularity comes from
            
            
            AIC_func = PM_BC_AIC_batched
            if bc_method == 'PM':
                bc_batch = bc
                if influence_method == 'PM':
                    batch_dims = [1]*2+[None]*9
                elif influence_method == 'VLM':
                    batch_dims = [1]*2+[None]*8

            elif bc_method == 'VLM':
                bc_batch = 'Neumann'
                if influence_method == 'PM':
                    batch_dims = [1]*2+[None]*9
                elif influence_method == 'VLM':
                    batch_dims = [1]*2+[None]*1 # for bc_method == 'VLM'
                    AIC_func = VLM_BC_AIC_batched

            if influence_method == 'PM':
                mesh_dict = pm_mesh_dict
                panel_corners   = mesh_dict['panel_corners_' + cell_type_j] # (nn, num_tot_panels, 3, 3) 
                coll_point      = mesh_dict['panel_center_' + cell_type_j] # (nn, num_tot_panels, 3)
                panel_x_dir     = mesh_dict['panel_x_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
                panel_y_dir     = mesh_dict['panel_y_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
                panel_normal    = mesh_dict['panel_normal_' + cell_type_j] # (nn, num_tot_panels, 3)
                S               = mesh_dict['S_' + cell_type_j]
                SL              = mesh_dict['SL_' + cell_type_j]
                SM              = mesh_dict['SM_' + cell_type_j]
            elif influence_method == 'VLM':
                mesh_dict = vlm_vectorized_dict
                panel_corners   = mesh_dict['panel_corners'] # (nn, num_tot_panels, 3, 3)
                coll_point      = mesh_dict['panel_centers'] # (nn, num_tot_panels, 3)
                panel_x_dir     = mesh_dict['panel_x_dir'] # (nn, num_tot_panels, 3)
                panel_y_dir     = mesh_dict['panel_y_dir'] # (nn, num_tot_panels, 3)
                panel_normal    = mesh_dict['panel_normal'] # (nn, num_tot_panels, 3)
                S               = mesh_dict['S']
                SL              = mesh_dict['SL']
                SM              = mesh_dict['SM']
                
            # if influence_method == 'PM':
            #     AIC_func = PM_BC_AIC_batched
            #     mesh_dict = pm_mesh_dict
            #     panel_corners   = mesh_dict['panel_corners_' + cell_type_j] # (nn, num_tot_panels, 3, 3) 
            #     if bc_method == 'PM':
            #         coll_point      = mesh_dict['panel_center_' + cell_type_j] # (nn, num_tot_panels, 3)
            #         panel_x_dir     = mesh_dict['panel_x_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
            #         panel_y_dir     = mesh_dict['panel_y_dir_' + cell_type_j] # (nn, num_tot_panels, 3)
            #         panel_normal    = mesh_dict['panel_normal_' + cell_type_j] # (nn, num_tot_panels, 3)
            #         S               = mesh_dict['S_' + cell_type_j]
            #         SL              = mesh_dict['SL_' + cell_type_j]
            #         SM              = mesh_dict['SM_' + cell_type_j]

            #     batch_dims = [1]*2+[None]*9 # extra input at the end for the source terms

            # elif influence_method == 'VLM':
            #     batch_dims = [1]*2+[None]*1
            #     # AIC_func = VLM_BC_AIC_batched
            #     mesh_dict = vlm_vectorized_dict
            #     panel_corners   = mesh_dict['panel_corners'] # (nn, num_tot_panels, 3, 3) 
            #     if bc_method == 'PM':
            #         AIC_func = PM_BC_AIC_batched # need panel method interactions here
            #         coll_point      = mesh_dict['panel_centers'] # (nn, num_tot_panels, 3)
            #         panel_x_dir     = mesh_dict['panel_x_dir'] # (nn, num_tot_panels, 3)
            #         panel_y_dir     = mesh_dict['panel_y_dir'] # (nn, num_tot_panels, 3)
            #         panel_normal    = mesh_dict['panel_normal'] # (nn, num_tot_panels, 3)
            #         S               = mesh_dict['S']
            #         SL              = mesh_dict['SL']
            #         SM              = mesh_dict['SM']

            #         bc = 'Neumann' # hard coding bc we need normal velocity

            #         batch_dims = [1]*2+[None]*8 # no extra input because we only need the AIC mu matrix
            num_panels_iter_j = panel_corners.shape[1]
            stop_j += num_panels_iter_j
            AIC_batch_func = csdl.experimental.batch_function(
                AIC_func,
                # batch_size=batch_size,
                batch_size=batch_size_surf,
                batch_dims=batch_dims
            )
            if influence_method == 'VLM':
                if bc_method == 'PM':
                    AIC_mu_block = AIC_batch_func(
                        coll_point_eval, # where potential or velocity are induced
                        normal_vec_eval, # where potential or velocity are induced
                        coll_point, # panel that induces potential or velocity
                        panel_corners, # panel that induces potential or velocity
                        panel_x_dir, # panel that induces potential or velocity
                        panel_y_dir, # panel that induces potential or velocity
                        panel_normal, # panel that induces potential or velocity
                        S, # panel that induces potential or velocity
                        SL, # panel that induces potential or velocity
                        SM, # panel that induces potential or velocity
                        vec=None,
                        BC=bc,
                        do_source=False,
                        do_wake=False
                    )
                elif bc_method == 'VLM':
                    AIC_mu_block = AIC_batch_func(
                        coll_point_eval,
                        normal_vec_eval,
                        panel_corners,
                        # do_matvec=False
                    )
            elif influence_method == 'PM':
                # if bc_method == 'VLM':
                #     AIC_mu_block = AIC_batch_func(
                #         coll_point_eval, # where potential or velocity are induced
                #         normal_vec_eval, # where potential or velocity are induced
                #         coll_point, # panel that induces potential or velocity
                #         panel_corners, # panel that induces potential or velocity
                #         panel_x_dir, # panel that induces potential or velocity
                #         panel_y_dir, # panel that induces potential or velocity
                #         panel_normal, # panel that induces potential or velocity
                #         S, # panel that induces potential or velocity
                #         SL, # panel that induces potential or velocity
                #         SM, # panel that induces potential or velocity
                #         vec=None,
                #         BC=bc,
                #         do_source=False,
                #         do_wake=False
                #     )
                # if bc_method == 'PM':
                #     bc_batch = bc
                # elif bc_method == 'VLM':
                #     bc_batch='Neumann'
                
                outputs = AIC_batch_func(
                    coll_point_eval, # where potential or velocity are induced
                    normal_vec_eval, # where potential or velocity are induced
                    coll_point, # panel that induces potential or velocity
                    panel_corners, # panel that induces potential or velocity
                    panel_x_dir, # panel that induces potential or velocity
                    panel_y_dir, # panel that induces potential or velocity
                    panel_normal, # panel that induces potential or velocity
                    S, # panel that induces potential or velocity
                    SL, # panel that induces potential or velocity
                    SM, # panel that induces potential or velocity
                    sigma[:,start_j:stop_j],
                    BC=bc_batch,
                    do_source=True,
                    do_wake=False
                )
                AIC_mu_block = outputs[0]
                RHS_sigma_block = outputs[1]

            AIC_mu = AIC_mu.set(
                csdl.slice[:,start_i:stop_i, start_j:stop_j], 
                AIC_mu_block.reshape((1, num_panels_iter_i, num_panels_iter_j))
            )
            if influence_method == 'PM':
                RHS_sigma = RHS_sigma.set(
                    csdl.slice[:, start_i:stop_i, j], 
                    RHS_sigma_block.reshape(num_nodes, num_panels_iter_i)
                )

            start_j += num_panels_iter_j
        start_i += num_panels_iter_i

    RHS_sigma = csdl.sum(RHS_sigma, axes=(2,)) # building segments of the matvec

    RHS_w = csdl.Variable(shape=(num_nodes, num_tot_panels), value=0.)

    start_i, stop_i = 0, 0
    for i, bc_method in enumerate(bc_methods):
        if bc_method == 'PM': # applying either Dirichlet or Neumann BC for PM surface
            coll_point_eval = pm_mesh_dict['panel_center_mod']
            normal_vec_eval = pm_mesh_dict['panel_normal']
            batch_dims = [1]*2+[None]*9 # extra at the end bc of matvec
            AIC_func = PM_BC_AIC_batched

        elif bc_method == 'VLM': # applying no-penetration BC for VLM surface
            coll_point_eval = vlm_vectorized_dict['panel_centers']
            normal_vec_eval = vlm_vectorized_dict['panel_normal']
            batch_dims = [1]*2+[None]*2 # extra at the end bc of matvec
            AIC_func = VLM_BC_AIC_batched

        
        AIC_batch_func = csdl.experimental.batch_function(
            AIC_func,
            # batch_size=batch_size,
            batch_size=batch_size_surf, # NOTE: CHANGE TO BATCH SIZE OF WAKE?
            batch_dims=batch_dims
        )

        num_panels_iter = coll_point_eval.shape[1]
        stop_i += num_panels_iter

        panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, np_w, 4, 3)
        coll_point_w = wake_mesh_dict['panel_center'] # (nn, np_w, 3)
        panel_x_dir_w = wake_mesh_dict['panel_x_dir'] # (nn, np_w, 3)
        panel_y_dir_w = wake_mesh_dict['panel_y_dir'] # (nn, np_w, 3)
        panel_normal_w = wake_mesh_dict['panel_normal'] # (nn, np_w, 3)
        S_w = wake_mesh_dict['S']
        SL_w = wake_mesh_dict['SL']
        SM_w = wake_mesh_dict['SM']

        if bc_method == 'PM':
            AIC_mu_w_matvec_block = AIC_batch_func(
                coll_point_eval, # where potential or velocity are induced
                normal_vec_eval, # where potential or velocity are induced
                coll_point_w, # panel that induces potential or velocity
                panel_corners_w, # panel that induces potential or velocity
                panel_x_dir_w, # panel that induces potential or velocity
                panel_y_dir_w, # panel that induces potential or velocity
                panel_normal_w, # panel that induces potential or velocity
                S_w, # panel that induces potential or velocity
                SL_w, # panel that induces potential or velocity
                SM_w, # panel that induces potential or velocity
                mu_w,
                BC=bc,
                do_source=False,
                do_wake=True
            )
            # AIC_mu_w_matvec_block = csdl.Variable(value=np.ones(num_panels_iter,))
        elif bc_method == 'VLM':
            AIC_mu_w_matvec_block = AIC_batch_func(
                coll_point_eval,
                normal_vec_eval,
                panel_corners_w,
                mu_w,
                do_matvec=True
            )
            # AIC_mu_w_matvec_block = csdl.Variable(value=np.ones(num_panels_iter,))
        
        RHS_w = RHS_w.set(
            csdl.slice[:, start_i:stop_i], 
            AIC_mu_w_matvec_block.reshape(num_nodes, num_panels_iter)
        )
        start_i += num_panels_iter

    return AIC_mu, RHS_sigma, RHS_w

def PM_BC_AIC_batched(coll_point, normal_vec_eval, panel_center, panel_corners,  panel_x_dir, panel_y_dir,
                        panel_normal, S_j, SL_j, SM_j, vec=None, BC='Dirichlet', do_source=False, do_wake=False):
    '''
    Different options to support:
    - panel method boundary condition (Dirichlet or Neumann)
    - mode:
        - panel method
        - vortex ring
    
    '''
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

    if BC == 'Dirichlet':
        A_list = [A[:,:,ind] for ind in range(num_corners)]
        AM_list = [AM[:,:,ind] for ind in range(num_corners)]
        B_list = [B[:,:,ind] for ind in range(num_corners)]
        BM_list = [BM[:,:,ind] for ind in range(num_corners)]
        SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(num_corners)]
        SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(num_corners)]
        A1_list = [A1[:,:,ind] for ind in range(num_corners)]
        PN_list = [PN[:,:,ind] for ind in range(num_corners)]
        S_list = [S_j_exp_vec[:,:,ind] for ind in range(num_corners)]

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
                mode='potential'
            )
    elif BC == 'Neumann':
        num_edges = panel_corners.shape[2]
        AIC_vel_vec_list = []
        for i in range(num_edges-1):
            asdf = compute_vortex_line_ind_vel(
                panel_corners_exp_vec[:,:,i], 
                panel_corners_exp_vec[:,:,i+1], 
                coll_point_exp_vec[:,:,0], 
                mode='wake', 
                vc=None
            )
            AIC_vel_vec_list.append(asdf)
        asdf = compute_vortex_line_ind_vel(
            panel_corners_exp_vec[:,:,-1], 
            panel_corners_exp_vec[:,:,0], 
            coll_point_exp_vec[:,:,0], 
            mode='wake', 
            vc=None
        )
        AIC_vel_vec_list.append(asdf)
        AIC_mu_vel_vec = sum(AIC_vel_vec_list)

        expanded_shape_proj = (num_nodes, num_eval_pts, num_induced_pts, 3)
        vectorized_shape_proj = (num_nodes, num_interactions, 3)

        normal_vec_eval_exp = csdl.expand(normal_vec_eval, expanded_shape_proj, 'ijk->ijak')
        normal_vec_eval_exp_vec = normal_vec_eval_exp.reshape(vectorized_shape_proj)

        AIC_mu_vec = csdl.sum(normal_vec_eval_exp_vec*AIC_mu_vel_vec, axes=(2,))

        if do_source:
            # additional expansions for the (3,) dimension for velocity
            A = A.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
            AM = AM.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
            B = B.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
            BM = BM.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
            SL_j_exp_vec = SL_j_exp_vec.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
            SM_j_exp_vec = SM_j_exp_vec.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
            A1 = A1.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
            PN = PN.expand(panel_normal_exp_vec.shape, 'ijk->ijka')
            S_j_exp_vec = S_j_exp_vec.expand(panel_normal_exp_vec.shape, 'ijk->ijka')

            A_list = [A[:,:,ind] for ind in range(num_corners)]
            AM_list = [AM[:,:,ind] for ind in range(num_corners)]
            B_list = [B[:,:,ind] for ind in range(num_corners)]
            BM_list = [BM[:,:,ind] for ind in range(num_corners)]
            SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(num_corners)]
            SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(num_corners)]
            A1_list = [A1[:,:,ind] for ind in range(num_corners)]
            PN_list = [PN[:,:,ind] for ind in range(num_corners)]
            S_list = [S_j_exp_vec[:,:,ind] for ind in range(num_corners)]

            AIC_sigma_vel_vec = compute_source_influence_new(
                A_list, 
                AM_list, 
                B_list, 
                BM_list, 
                SL_list, 
                SM_list, 
                A1_list, 
                PN_list, 
                S_list, 
                panel_x_dir_exp_vec[:,:,0,:],
                panel_y_dir_exp_vec[:,:,0,:],
                panel_normal_exp_vec[:,:,0,:],
                mode='velocity'
            )

            AIC_sigma_vec = csdl.sum(normal_vec_eval_exp_vec*AIC_sigma_vel_vec, axes=(2,))

    AIC_mu = AIC_mu_vec.reshape((num_nodes, num_eval_pts, num_induced_pts))
    if do_source:
        AIC_sigma = AIC_sigma_vec.reshape((num_nodes, num_eval_pts, num_induced_pts))
    '''
    Added two options:
    if do_source:
        do matvec with the source AIC
    else:
        do matvec with the doublet AIC (this will be for the wakes)
    '''
    if do_source: # influence of surface doublets and surface sources
        AIC_sigma_matvec = csdl.einsum(AIC_sigma, vec, action='ijk,ik->ij')
        return AIC_mu, AIC_sigma_matvec
    elif do_wake: # influence of wake doublets
        AIC_mu_matvec = csdl.einsum(AIC_mu, vec, action='ijk,ik->ij')
        return AIC_mu_matvec
    else: # influence of VLM doublets (no need to compute sources for these)
        return AIC_mu # only the matrix is an output so no input vector is needed for matvec

def VLM_BC_AIC_batched(coll_point, normal_vec_eval, panel_corners, mu=None, vc=None, do_matvec=False):
    num_nodes = coll_point.shape[0]
    num_eval_pts = coll_point.shape[1]
    num_induced_pts = panel_corners.shape[1]
    num_interactions = num_eval_pts*num_induced_pts
    num_corners = panel_corners.shape[2]
    
    expanded_shape = (num_nodes, num_eval_pts, num_induced_pts, num_corners, 3)
    vectorized_shape = (num_nodes, num_interactions, num_corners, 3)

    # ============ expanding across columns ============
    coll_point_exp = csdl.expand(coll_point, expanded_shape, 'ijk->ijabk')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    normal_vec_eval_exp = csdl.expand(normal_vec_eval, expanded_shape, 'ijk->ijabk')
    normal_vec_eval_exp_vec = normal_vec_eval_exp.reshape(vectorized_shape)

    # ============ expanding across rows ============
    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'ijkl->iajkl')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    num_edges = num_corners

    vc_exp_vec = vc
    if isinstance(vc, csdl.Variable):
        vc_exp = csdl.expand(vc, (num_nodes, num_eval_pts, num_induced_pts, num_corners), 'ijk->iajk')
        vc_exp_vec = vc_exp.reshape((num_nodes, num_interactions, num_corners))
        vc_list = [vc_exp_vec[:,:,i] for i in range(num_edges)]
    else:
        vc_list = [vc]*num_edges


    AIC_vel_vec_list = []
    for  i in range(num_edges-1):
        asdf = compute_vortex_line_ind_vel(
            panel_corners_exp_vec[:,:,i], 
            panel_corners_exp_vec[:,:,i+1], 
            coll_point_exp_vec[:,:,0], 
            mode='wake', 
            vc=vc_list[i]
        )
        AIC_vel_vec_list.append(asdf)
    asdf = compute_vortex_line_ind_vel(
        panel_corners_exp_vec[:,:,-1], 
        panel_corners_exp_vec[:,:,0], 
        coll_point_exp_vec[:,:,0], 
        mode='wake', 
        vc=vc_list[-1]
    )
    AIC_vel_vec_list.append(asdf)
    AIC_vel_vec = sum(AIC_vel_vec_list)

    expanded_shape_proj = (num_nodes, num_eval_pts, num_induced_pts, 3)
    vectorized_shape_proj = (num_nodes, num_interactions, 3)

    normal_vec_eval_exp = csdl.expand(normal_vec_eval, expanded_shape_proj, 'ijk->ijak')
    normal_vec_eval_exp_vec = normal_vec_eval_exp.reshape(vectorized_shape_proj)

    AIC_vec = csdl.sum(normal_vec_eval_exp_vec*AIC_vel_vec, axes=(2,))
    AIC_grid = AIC_vec.reshape((num_nodes, num_eval_pts, num_induced_pts))

    if do_matvec:
        AIC_vec_matvec = csdl.einsum(AIC_grid, mu, action='ijk,ik->ij')
        return AIC_vec_matvec # (num_nodes, num_eval_pts)
    elif not do_matvec:
        return AIC_grid # (num_nodes, num_eval_pts, num_induced_pts)