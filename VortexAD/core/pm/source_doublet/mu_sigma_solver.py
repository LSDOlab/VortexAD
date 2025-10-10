import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.pm.source_doublet.fixed_wake_representation import fixed_wake_representation
from VortexAD.core.pm.source_doublet.compute_source_strength import compute_source_strength

from VortexAD.core.elements.source import compute_source_influence_new 
from VortexAD.core.elements.doublet import compute_doublet_influence_new
from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel


def mu_sigma_solver(num_nodes, mesh_dict, mode='structured', batch_size=None, bc='Dirichlet', ROM=False, constant_geometry=False):

    if mode == 'structured':
        surface_names = list(mesh_dict.keys())
        num_tot_panels = 0
        for i, surface in enumerate(surface_names):
            num_tot_panels += mesh_dict[surface]['num_panels']
            if i == 0:
                num_nodes = mesh_dict[surface]['nodal_velocity'].shape[0]
    
    elif mode == 'unstructured':
        num_tot_panels = len(mesh_dict['cell_adjacency'])
        num_nodes = mesh_dict['nodal_velocity'].shape[0]

    # wake_mesh_dict = fixed_wake_representation(mesh_dict, num_nodes, wake_propagation_dt=0.001)
    wake_mesh_dict = fixed_wake_representation(mesh_dict, num_nodes, wake_propagation_dt=100, mesh_mode=mode, constant_geometry=constant_geometry)

    sigma = compute_source_strength(mesh_dict, num_nodes, num_panels=num_tot_panels, mesh_mode=mode, constant_geometry=constant_geometry)
    if constant_geometry:
        num_nodes_geom = 1
    else:
        num_nodes_geom=num_nodes
    # graph = csdl.get_current_recorder().active_graph
    # graph.visualize('pre-subgraph')
    # static AIC matrices for linear system solve
    # return sigma
    if batch_size and ROM:
        sigma_nn = sigma[0,:]
        UT, U = ROM[0], ROM[1]
        RHS_red, AIC_red = compute_batched_aic_POD(mesh_dict, wake_mesh_dict, sigma_nn, batch_size, ROM)
        mu_red = csdl.solve_linear(AIC_red, RHS_red)
        mu = csdl.matvec(U, mu_red).reshape(sigma.shape)
        # print(mu.shape)
        # exit()

        return mu, sigma, wake_mesh_dict, None, None, None, None
    else:
        if mode == 'structured':
            if bc == 'Dirichlet':
                AIC_mu, AIC_sigma = AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, surface_names)
            elif bc == 'Neumann':
                AIC_mu, AIC_sigma = Neumann_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, surface_names)
        elif mode == 'unstructured':
            if batch_size is None:
                print('no batching')
                # AIC_mu, AIC_sigma, AIC_mu_orig = unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels)
                AIC_mu, AIC_sigma, AIC_mu_orig = unstructured_AIC_computation_UW(mesh_dict, wake_mesh_dict, num_nodes_geom, num_tot_panels)
            else:
                print(f'yes batching; batch size = {batch_size}')
                # AIC_mu, AIC_sigma, AIC_mu_orig = unstructured_AIC_computation_UW_looping(mesh_dict, wake_mesh_dict, num_nodes_geom, num_tot_panels, batch_size)
                AIC_mu, AIC_sigma, AIC_mu_orig = unstructured_AIC_computation_UW_looping_mixed(mesh_dict, wake_mesh_dict, num_nodes_geom, num_tot_panels, batch_size)
            # for i in csdl.frange(1):
            #     AIC_mu, AIC_sigma = unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels)
            #     graph = csdl.get_current_recorder().active_graph
            # graph.visualize('subgraph')
            # csdl.get_current_recorder().visualize_graph('entire_graph', visualize_style='hierarchical')
            # exit()

        else:
            raise ValueError('Mode must be structured or unstructured')

        # return AIC_mu
        # AIC_mu = AIC_mu.set(csdl.slice[0,asdf,asdf], value=0.5)
        # print(AIC_mu[0,asdf,asdf].value)
        # print(AIC_mu[0,0,:].value)
        # exit()
        if constant_geometry: # AIC_sigma num_nodes is 1
            loop_vals = np.arange(num_nodes).tolist()
            with csdl.experimental.enter_loop(vals=[loop_vals]) as loop_builder:
                n = loop_builder.get_loop_indices()
                sigma_BC_influence = csdl.matvec(AIC_sigma[0,:,:], sigma[n,:])
            sigma_BC_influence = loop_builder.add_stack(sigma_BC_influence)
            loop_builder.finalize()
        else:
            sigma_BC_influence = csdl.einsum(AIC_sigma, sigma, action='ijk,ik->ij')

    if bc == 'Dirichlet':
        RHS = -sigma_BC_influence
    elif bc == 'Neumann':
        surf_normal_vel = sigma # sigma = -V_inf \cdot normal_vec
        RHS = -sigma_BC_influence + surf_normal_vel
        # RHS =  surf_normal_vel

    mu = csdl.Variable(value=np.zeros(sigma.shape))
    if ROM:
        UT, U = ROM[0], ROM[1]

    # direct linear solve
    loop_vals = np.arange(num_nodes).tolist()
    with csdl.experimental.enter_loop(vals=[loop_vals]) as loop_builder:
        nn = loop_builder.get_loop_indices()
        RHS_nn = RHS[nn,:]
        if ROM:
            '''
            reduce AIC with U^T AIC U
            reduce RHS with U^T b
            solve reduced system
            reconstruct larger mu with U mu
            '''
            AIC_red = csdl.matmat(csdl.matmat(UT, AIC_mu[nn,:]), U)
            RHS_red_nn = csdl.matvec(UT, RHS_nn)
            mu_red_nn = csdl.solve_linear(AIC_red, RHS_red_nn)
            mu_nn = csdl.matvec(U,mu_red_nn)
            mu = mu.set(csdl.slice[nn,:], value=mu_nn)
        else:
            if constant_geometry:
                mu = csdl.solve_linear(AIC_mu[0,:,:], RHS_nn)
                # mu = csdl.matvec(AIC_mu[0,:,:], RHS_nn)
            else:
                mu = csdl.solve_linear(AIC_mu[nn,:,:], RHS_nn)
    mu = loop_builder.add_stack(mu)
    loop_builder.finalize()

    
    # iterative nonlinear solver
    # for nn in csdl.frange(num_nodes):
    #     RHS_nn = RHS[nn,:]
    #     AIC_mu_nn = AIC_mu[nn,:,:]
    #     mu_nn = csdl.ImplicitVariable(value=np.zeros(RHS_nn.shape))

    #     residual = csdl.matvec(AIC_mu_nn, mu_nn) - RHS_nn
    #     solver = csdl.nonlinear_solvers.GaussSeidel(
    #         'nl_solver_mu',
    #         tolerance=1e-10,
    #         max_iter=100
    #     )
    #     # solver.add_state(mu_nn, residual)
    #     solver.add_state(mu_nn, residual, state_update=mu_nn+2.*residual)
    #     solver.run()

    #     mu = mu.set(csdl.slice[nn,:], value=mu_nn)

    # return mu, sigma, wake_mesh_dict, AIC_mu, AIC_sigma, AIC_mu_orig
    return mu, sigma, wake_mesh_dict, AIC_mu, AIC_sigma, RHS, AIC_mu_orig

def AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, surface_names):
    AIC_sigma = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    AIC_mu = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    num_surfaces = len(surface_names)
    start_i, stop_i = 0, 0
    for i in range(num_surfaces):
        surf_i_name = surface_names[i]

        coll_point_i = mesh_dict[surf_i_name]['panel_center_mod'] # evaluation point
        nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
        num_panels_i = mesh_dict[surf_i_name]['num_panels']
        stop_i += num_panels_i

        start_j, stop_j = 0, 0
        start_w_j, stop_w_j = 0, 0
        for j in range(num_surfaces):
            surf_j_name = surface_names[j]
            nc_j, ns_j = mesh_dict[surf_j_name]['nc'], mesh_dict[surf_j_name]['ns']
            num_panels_j = mesh_dict[surf_j_name]['num_panels']
            stop_j += num_panels_j

            panel_corners_j = mesh_dict[surf_j_name]['panel_corners']
            coll_point_j = mesh_dict[surf_j_name]['panel_center']
            panel_x_dir_j = mesh_dict[surf_j_name]['panel_x_dir']
            panel_y_dir_j = mesh_dict[surf_j_name]['panel_y_dir']
            panel_normal_j = mesh_dict[surf_j_name]['panel_normal']

            S_j = mesh_dict[surf_j_name]['S']
            SL_j = mesh_dict[surf_j_name]['SL']
            SM_j = mesh_dict[surf_j_name]['SM']

            num_interactions = num_panels_i*num_panels_j

            coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'jklm->jklabm')
            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions, 4, 3))

            panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklmn->jaklmn')
            panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions, 4, 3))

            coll_point_j_exp = csdl.expand(coll_point_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            coll_point_j_exp_vec = coll_point_j_exp.reshape((num_nodes, num_interactions, 4, 3))

            panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
            panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
            panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, num_interactions, 4, 3))

            S_j_exp = csdl.expand(S_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            S_j_exp_vec = S_j_exp.reshape((num_nodes, num_interactions, 4))

            SL_j_exp = csdl.expand(SL_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            SL_j_exp_vec = SL_j_exp.reshape((num_nodes, num_interactions, 4))

            SM_j_exp = csdl.expand(SM_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            SM_j_exp_vec = SM_j_exp.reshape((num_nodes, num_interactions, 4))
            
            a = coll_point_i_exp_vec - panel_corners_j_exp_vec # Rc - Ri
            P_JK = coll_point_i_exp_vec - coll_point_j_exp_vec # RcJ - RcK
            sum_ind = len(a.shape) - 1

            A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
            AL = csdl.sum(a*panel_x_dir_j_exp_vec, axes=(sum_ind,))
            AM = csdl.sum(a*panel_y_dir_j_exp_vec, axes=(sum_ind,)) # m-direction projection 
            PN = csdl.sum(P_JK*panel_normal_j_exp_vec, axes=(sum_ind,)) # normal projection of CP

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

            A_list = [A[:,:,ind] for ind in range(4)]
            AM_list = [AM[:,:,ind] for ind in range(4)]
            B_list = [B[:,:,ind] for ind in range(4)]
            BM_list = [BM[:,:,ind] for ind in range(4)]
            SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(4)]
            SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(4)]
            A1_list = [A1[:,:,ind] for ind in range(4)]
            PN_list = [PN[:,:,ind] for ind in range(4)]
            S_list = [S_j_exp_vec[:,:,ind] for ind in range(4)]

            doublet_influence_vec = compute_doublet_influence_new(
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
            doublet_influence = doublet_influence_vec.reshape((num_nodes, num_panels_i, num_panels_j))
            AIC_mu = AIC_mu.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=doublet_influence)

            source_influence_vec = compute_source_influence_new(
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
            source_influence = source_influence_vec.reshape((num_nodes, num_panels_i, num_panels_j))
            AIC_sigma = AIC_sigma.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=source_influence)
            start_j += num_panels_j

            # ================ wake influence here ================

            nc_w_j, ns_w_j = wake_mesh_dict[surf_j_name]['nc'], wake_mesh_dict[surf_j_name]['ns']
            num_panels_w_j = wake_mesh_dict[surf_j_name]['num_panels']
            stop_w_j += num_panels_w_j

            panel_corners_j = wake_mesh_dict[surf_j_name]['panel_corners']
            coll_point_j = wake_mesh_dict[surf_j_name]['panel_center']
            panel_x_dir_j = wake_mesh_dict[surf_j_name]['panel_x_dir']
            panel_y_dir_j = wake_mesh_dict[surf_j_name]['panel_y_dir']
            panel_normal_j = wake_mesh_dict[surf_j_name]['panel_normal']

            # S_j = wake_mesh_dict[surf_j_name]['S']
            SL_j = wake_mesh_dict[surf_j_name]['SL']
            SM_j = wake_mesh_dict[surf_j_name]['SM']

            num_interactions_w = num_panels_i*num_panels_w_j

            coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_w_j, 4, 3), 'jklm->jklabm')
            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklmn->jaklmn')
            panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            coll_point_j_exp = csdl.expand(coll_point_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklm->jaklbm')
            coll_point_j_exp_vec = coll_point_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklm->jaklbm')
            panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))
            panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklm->jaklbm')
            panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))
            panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklm->jaklbm')
            panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            # S_j_exp = csdl.expand(S_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            # S_j_exp_vec = S_j_exp.reshape((num_nodes, num_interactions, 4))

            SL_j_exp = csdl.expand(SL_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4), 'jklm->jaklm')
            SL_j_exp_vec = SL_j_exp.reshape((num_nodes, num_interactions_w, 4))

            SM_j_exp = csdl.expand(SM_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4), 'jklm->jaklm')
            SM_j_exp_vec = SM_j_exp.reshape((num_nodes, num_interactions_w, 4))
            
            a = coll_point_i_exp_vec - panel_corners_j_exp_vec # Rc - Ri
            P_JK = coll_point_i_exp_vec - coll_point_j_exp_vec # RcJ - RcK
            sum_ind = len(a.shape) - 1

            A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
            AL = csdl.sum(a*panel_x_dir_j_exp_vec, axes=(sum_ind,))
            AM = csdl.sum(a*panel_y_dir_j_exp_vec, axes=(sum_ind,)) # m-direction projection 
            PN = csdl.sum(P_JK*panel_normal_j_exp_vec, axes=(sum_ind,)) # normal projection of CP

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

            A_list = [A[:,:,ind] for ind in range(4)]
            AM_list = [AM[:,:,ind] for ind in range(4)]
            B_list = [B[:,:,ind] for ind in range(4)]
            BM_list = [BM[:,:,ind] for ind in range(4)]
            SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(4)]
            SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(4)]
            A1_list = [A1[:,:,ind] for ind in range(4)]
            PN_list = [PN[:,:,ind] for ind in range(4)]
            # S_list = [S_j_exp_vec[:,:,ind] for ind in range(4)]

            doublet_influence_w_vec = compute_doublet_influence_new(
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
            doublet_influence_w = doublet_influence_w_vec.reshape((num_nodes, num_panels_i, num_panels_w_j))

            # doublet_influence_KC = csdl.Variable(value=np.zeros(shape=doublet_influence.shape))
            # doublet_influence_KC = doublet_influence_KC.set(
            #     csdl.slice[:,:,:],
            #     value=doublet_influence
            # )
            # doublet_influence_KC = doublet_influence_KC.set(
            #     csdl.slice[:,:,:(ns_j-1)],
            #     value=doublet_influence[:,:,:(ns_j-1)]-doublet_influence_w
            # )
            # doublet_influence_KC = doublet_influence_KC.set(
            #     csdl.slice[:,:,-(ns_j-1):],
            #     value=doublet_influence[:,:,-(ns_j-1):]+doublet_influence_w
            # )

            asdf = stop_j-num_panels_j
            AIC_mu = AIC_mu.set(
                csdl.slice[:,start_i:stop_i, asdf:(asdf+(ns_w_j-1))],
                value=AIC_mu[:,start_i:stop_i, asdf:(asdf+(ns_w_j-1))] - doublet_influence_w
            )

            AIC_mu = AIC_mu.set(
                csdl.slice[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j],
                value=AIC_mu[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j] + doublet_influence_w
            )

            # start_j += num_panels_j
            start_w_j += num_panels_w_j
        start_i += num_panels_i

    return AIC_mu, AIC_sigma

def Neumann_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, surface_names):
    #I M LUCA, I M SMART, I M THE BEST - NICHOLAS ORNDORFF

    AIC_sigma = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    AIC_mu = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    num_surfaces = len(surface_names)
    start_i, stop_i = 0, 0
    for i in range(num_surfaces):
        surf_i_name = surface_names[i]

        coll_point_i = mesh_dict[surf_i_name]['panel_center_mod'] # evaluation point
        panel_normal_i = mesh_dict[surf_i_name]['panel_normal']
        nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
        num_panels_i = mesh_dict[surf_i_name]['num_panels']
        stop_i += num_panels_i

        start_j, stop_j = 0, 0
        start_w_j, stop_w_j = 0, 0
        for j in range(num_surfaces):
            surf_j_name = surface_names[j]
            nc_j, ns_j = mesh_dict[surf_j_name]['nc'], mesh_dict[surf_j_name]['ns']
            num_panels_j = mesh_dict[surf_j_name]['num_panels']
            stop_j += num_panels_j

            panel_corners_j = mesh_dict[surf_j_name]['panel_corners']
            coll_point_j = mesh_dict[surf_j_name]['panel_center']
            panel_x_dir_j = mesh_dict[surf_j_name]['panel_x_dir']
            panel_y_dir_j = mesh_dict[surf_j_name]['panel_y_dir']
            panel_normal_j = mesh_dict[surf_j_name]['panel_normal']

            S_j = mesh_dict[surf_j_name]['S']
            SL_j = mesh_dict[surf_j_name]['SL']
            SM_j = mesh_dict[surf_j_name]['SM']

            num_interactions = num_panels_i*num_panels_j

            coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'jklm->jklabm')
            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions, 4, 3))

            panel_normal_i_exp = csdl.expand(panel_normal_i, (num_nodes, nc_i-1, ns_i-1, num_panels_j, 3), 'jklm->jklam')
            panel_normal_i_grid = panel_normal_i_exp.reshape((num_nodes, num_panels_i, num_panels_j, 3)) # for normal projection

            # panel_normal_i_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 3), 'jklm->jaklm')
            # panel_normal_i_grid = panel_normal_i_exp.reshape((num_nodes, num_panels_i, num_panels_j, 3)) # for normal projection

            panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklmn->jaklmn')
            panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions, 4, 3))

            coll_point_j_exp = csdl.expand(coll_point_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            coll_point_j_exp_vec = coll_point_j_exp.reshape((num_nodes, num_interactions, 4, 3))

            panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
            panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
            panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, num_interactions, 4, 3))

            S_j_exp = csdl.expand(S_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            S_j_exp_vec = S_j_exp.reshape((num_nodes, num_interactions, 4))

            SL_j_exp = csdl.expand(SL_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            SL_j_exp_vec = SL_j_exp.reshape((num_nodes, num_interactions, 4))

            SM_j_exp = csdl.expand(SM_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            SM_j_exp_vec = SM_j_exp.reshape((num_nodes, num_interactions, 4))
            
            a = coll_point_i_exp_vec - panel_corners_j_exp_vec # Rc - Ri
            P_JK = coll_point_i_exp_vec - coll_point_j_exp_vec # RcJ - RcK
            sum_ind = len(a.shape) - 1

            A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
            AL = csdl.sum(a*panel_x_dir_j_exp_vec, axes=(sum_ind,))
            AM = csdl.sum(a*panel_y_dir_j_exp_vec, axes=(sum_ind,)) # m-direction projection 
            PN = csdl.sum(P_JK*panel_normal_j_exp_vec, axes=(sum_ind,)) # normal projection of CP

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

            A = A.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
            AM = AM.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
            B = B.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
            BM = BM.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
            SL_j_exp_vec = SL_j_exp_vec.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
            SM_j_exp_vec = SM_j_exp_vec.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
            A1 = A1.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
            PN = PN.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')
            S_j_exp_vec = S_j_exp_vec.expand(panel_normal_j_exp_vec.shape, 'ijk->ijka')

            A_list = [A[:,:,ind] for ind in range(4)]
            AM_list = [AM[:,:,ind] for ind in range(4)]
            B_list = [B[:,:,ind] for ind in range(4)]
            BM_list = [BM[:,:,ind] for ind in range(4)]
            SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(4)]
            SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(4)]
            A1_list = [A1[:,:,ind] for ind in range(4)]
            PN_list = [PN[:,:,ind] for ind in range(4)]
            S_list = [S_j_exp_vec[:,:,ind] for ind in range(4)]

            # doublet_influence_vec = compute_doublet_influence_new(
            #     A_list, 
            #     AM_list, 
            #     B_list, 
            #     BM_list, 
            #     SL_list, 
            #     SM_list, 
            #     A1_list, 
            #     PN_list, 
            #     mode='potential'
            # )
            # doublet_influence = doublet_influence_vec.reshape((num_nodes, num_panels_i, num_panels_j))
            # AIC_mu = AIC_mu.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=doublet_influence)

            source_vel_vec = compute_source_influence_new(
                A_list, 
                AM_list, 
                B_list, 
                BM_list, 
                SL_list, 
                SM_list, 
                A1_list, 
                PN_list, 
                S_list, 
                panel_x_dir_j_exp_vec[:,:,0,:],
                panel_y_dir_j_exp_vec[:,:,0,:],
                panel_normal_j_exp_vec[:,:,0,:],
                mode='velocity'
            )
            source_vel = source_vel_vec.reshape((num_nodes, num_panels_i, num_panels_j, 3))
            source_vel_normal_proj = csdl.sum(source_vel*panel_normal_i_grid, axes=(3,))

            AIC_sigma = AIC_sigma.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=source_vel_normal_proj)

            ind_vel_s_12 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,0,:],panel_corners_j_exp_vec[:,:,1,:],p_eval=coll_point_i_exp_vec[:,:,0,:], mode='wake', vc=1.e-7)
            ind_vel_s_23 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,1,:],panel_corners_j_exp_vec[:,:,2,:],p_eval=coll_point_i_exp_vec[:,:,1,:], mode='wake', vc=1.e-7)
            ind_vel_s_34 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,2,:],panel_corners_j_exp_vec[:,:,3,:],p_eval=coll_point_i_exp_vec[:,:,2,:], mode='wake', vc=1.e-7)
            ind_vel_s_41 = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,3,:],panel_corners_j_exp_vec[:,:,0,:],p_eval=coll_point_i_exp_vec[:,:,3,:], mode='wake', vc=1.e-7)

            ind_vel_vr = ind_vel_s_12+ind_vel_s_23+ind_vel_s_34+ind_vel_s_41
            ind_vel_vr_mat = ind_vel_vr.reshape((num_nodes, num_panels_i, num_panels_j, 3))
            ind_vel_vr_normal_proj = csdl.sum(ind_vel_vr_mat*panel_normal_i_grid, axes=(3,))
            AIC_mu = AIC_mu.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=ind_vel_vr_normal_proj)

            start_j += num_panels_j

            # ================ wake influence here ================

            nc_w_j, ns_w_j = wake_mesh_dict[surf_j_name]['nc'], wake_mesh_dict[surf_j_name]['ns']
            num_panels_w_j = wake_mesh_dict[surf_j_name]['num_panels']
            stop_w_j += num_panels_w_j

            panel_corners_j = wake_mesh_dict[surf_j_name]['panel_corners']
            panel_normal_j = wake_mesh_dict[surf_j_name]['panel_normal']

            num_interactions_w = num_panels_i*num_panels_w_j

            coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_w_j, 4, 3), 'jklm->jklabm')
            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            panel_normal_i_exp = csdl.expand(panel_normal_i, (num_nodes, nc_i-1, ns_i-1, num_panels_w_j, 3), 'jklm->jklam')
            panel_normal_i_grid = panel_normal_i_exp.reshape((num_nodes, num_panels_i, num_panels_w_j, 3)) # for normal projection

            # panel_normal_i_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 3), 'jklm->jaklm')
            # panel_normal_i_grid = panel_normal_i_exp.reshape((num_nodes, num_panels_i, num_panels_w_j, 3)) # for normal projection

            panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_w_j-1, ns_w_j-1, 4, 3), 'jklmn->jaklmn')
            panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions_w, 4, 3))

            # doublet_influence_w_vec = compute_doublet_influence_new(
            #     A_list, 
            #     AM_list, 
            #     B_list, 
            #     BM_list, 
            #     SL_list, 
            #     SM_list, 
            #     A1_list, 
            #     PN_list, 
            #     mode='potential'
            # )
            # doublet_influence_w = doublet_influence_w_vec.reshape((num_nodes, num_panels_i, num_panels_w_j))

            ind_vel_s_12_w = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,0,:],panel_corners_j_exp_vec[:,:,1,:],p_eval=coll_point_i_exp_vec[:,:,0,:], mode='wake', vc=1.e-4)
            ind_vel_s_23_w = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,1,:],panel_corners_j_exp_vec[:,:,2,:],p_eval=coll_point_i_exp_vec[:,:,0,:], mode='wake', vc=1.e-4)
            ind_vel_s_34_w = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,2,:],panel_corners_j_exp_vec[:,:,3,:],p_eval=coll_point_i_exp_vec[:,:,0,:], mode='wake', vc=1.e-4)
            ind_vel_s_41_w = compute_vortex_line_ind_vel(panel_corners_j_exp_vec[:,:,3,:],panel_corners_j_exp_vec[:,:,0,:],p_eval=coll_point_i_exp_vec[:,:,0,:], mode='wake', vc=1.e-4)

            ind_vel_vr_w = ind_vel_s_12_w+ind_vel_s_23_w+ind_vel_s_34_w+ind_vel_s_41_w
            ind_vel_vr_mat = ind_vel_vr_w.reshape((num_nodes, num_panels_i, num_panels_w_j, 3))
            ind_vel_vr_normal_proj = csdl.sum(ind_vel_vr_mat*panel_normal_i_grid, axes=(3,))

            # doublet_influence_KC = csdl.Variable(value=np.zeros(shape=doublet_influence.shape))
            # doublet_influence_KC = doublet_influence_KC.set(
            #     csdl.slice[:,:,:],
            #     value=doublet_influence
            # )
            # doublet_influence_KC = doublet_influence_KC.set(
            #     csdl.slice[:,:,:(ns_j-1)],
            #     value=doublet_influence[:,:,:(ns_j-1)]-doublet_influence_w
            # )
            # doublet_influence_KC = doublet_influence_KC.set(
            #     csdl.slice[:,:,-(ns_j-1):],
            #     value=doublet_influence[:,:,-(ns_j-1):]+doublet_influence_w
            # )

            asdf = stop_j-num_panels_j
            AIC_mu = AIC_mu.set(
                csdl.slice[:,start_i:stop_i, asdf:(asdf+(ns_w_j-1))],
                value=AIC_mu[:,start_i:stop_i, asdf:(asdf+(ns_w_j-1))] - ind_vel_vr_normal_proj
            )

            AIC_mu = AIC_mu.set(
                csdl.slice[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j],
                value=AIC_mu[:,start_i:stop_i, (stop_j-(ns_w_j-1)):stop_j] + ind_vel_vr_normal_proj
            )

            # start_j += num_panels_j
            start_w_j += num_panels_w_j
        start_i += num_panels_i
    
    return AIC_mu, AIC_sigma
    
def unstructured_AIC_computation(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels):
    
    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']
    num_first_wake_panels = len(upper_TE_cell_ind)
    num_wake_panels = wake_mesh_dict['num_panels']

    coll_point_eval = mesh_dict['panel_center_mod']

    coll_point = mesh_dict['panel_center'] # (nn, nt, num_tot_panels, 3)
    panel_corners = mesh_dict['panel_corners'] # (nn, nt, num_tot_panels, 3, 3) 
    panel_x_dir = mesh_dict['panel_x_dir'] # (nn, nt, num_tot_panels, 3)
    panel_y_dir = mesh_dict['panel_y_dir'] # (nn, nt, num_tot_panels, 3)
    panel_normal = mesh_dict['panel_normal'] # (nn, nt, num_tot_panels, 3)
    S_j = mesh_dict['S']
    SL_j = mesh_dict['SL']
    SM_j = mesh_dict['SM']

    num_interactions = num_tot_panels**2
    expanded_shape = (num_nodes, num_tot_panels, num_tot_panels, 3, 3)
    vectorized_shape = (num_nodes, num_interactions, 3, 3)

    # expanding collocation points (where boundary condition is applied, the "i-th" expansion-vectorization)
    coll_point_exp = csdl.expand(coll_point_eval, expanded_shape, 'jkl->jkabl')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    # expanding the panel terms used to compute influences AT the collocation points
    # -> the "j-th" expansion-vectorization
    coll_point_j_exp = csdl.expand(coll_point, expanded_shape, 'jkl->jakbl')
    coll_point_j_exp_vec = coll_point_j_exp.reshape(vectorized_shape)

    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'jklm->jaklm')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    panel_x_dir_exp = csdl.expand(panel_x_dir, expanded_shape, 'jkl->jakbl')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)
    panel_y_dir_exp = csdl.expand(panel_y_dir, expanded_shape, 'jkl->jakbl')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)
    panel_normal_exp = csdl.expand(panel_normal, expanded_shape, 'jkl->jakbl')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    # dpij_exp = csdl.expand(S_j, expanded_shape[:-1] + (2,), 'ijklm->ijaklm')
    # dpij_exp_vec = dpij_exp.reshape(vectorized_shape[:-1] + (2,))
    # dij_exp = csdl.expand(dij, expanded_shape[:-1], 'ijkl->ijakl')
    # dij_exp_vec = dij_exp.reshape(vectorized_shape[:-1])

    S_j_exp = csdl.expand(S_j, expanded_shape[:-1] , 'jkl->jakl')
    S_j_exp_vec = S_j_exp.reshape(vectorized_shape[:-1])

    SL_j_exp = csdl.expand(SL_j, expanded_shape[:-1], 'jkl->jakl')
    SL_j_exp_vec = SL_j_exp.reshape(vectorized_shape[:-1])

    SM_j_exp = csdl.expand(SM_j, expanded_shape[:-1], 'jkl->jakl')
    SM_j_exp_vec = SM_j_exp.reshape(vectorized_shape[:-1])

    a = coll_point_exp_vec - panel_corners_exp_vec # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_j_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    # for i in csdl.frange(1):

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_exp_vec, axes=(sum_ind,)) # normal projection of CP

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

    A_list = [A[:,:,ind] for ind in range(3)]
    AM_list = [AM[:,:,ind] for ind in range(3)]
    B_list = [B[:,:,ind] for ind in range(3)]
    BM_list = [BM[:,:,ind] for ind in range(3)]
    SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(3)]
    SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(3)]
    A1_list = [A1[:,:,ind] for ind in range(3)]
    PN_list = [PN[:,:,ind] for ind in range(3)]
    S_list = [S_j_exp_vec[:,:,ind] for ind in range(3)]

    doublet_influence_vec = compute_doublet_influence_new(
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
    doublet_influence = doublet_influence_vec.reshape((num_nodes, num_tot_panels, num_tot_panels))
#     graph = csdl.get_current_recorder().active_graph
    # graph.visualize('subgraph_lo')
    # print(len(graph.node_table))
    # exit()
    AIC_mu_orig = doublet_influence

    source_influence_vec = compute_source_influence_new(
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
    source_influence = source_influence_vec.reshape((num_nodes, num_tot_panels, num_tot_panels))
    AIC_sigma = source_influence

    # wake AIC influence
    # location where wake influence is computed
    coll_point = mesh_dict['panel_center_mod'][:,:,:]

    # wake values
    panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, nc_w, ns_w, 4, 3)
    coll_point_w = wake_mesh_dict['panel_center'] # (nn, nc_w, ns_w, 4, 3)
    panel_x_dir_w = wake_mesh_dict['panel_x_dir'] # (nn, nc_w, ns_w, 3)
    panel_y_dir_w = wake_mesh_dict['panel_y_dir'] # (nn, nc_w, ns_w, 3)
    panel_normal_w = wake_mesh_dict['panel_normal'] # (nn, nc_w, ns_w, 3)
    SL_w = wake_mesh_dict['SL']
    SM_w = wake_mesh_dict['SM']
    
    nc_w, ns_w = panel_corners_w.shape[1], panel_corners_w.shape[2]

    # target expansion and vectorization shapes
    # TODO: add support for multisurface
    num_wake_interactions = num_tot_panels*num_wake_panels
    expanded_shape = (num_nodes, num_tot_panels, nc_w, ns_w, 4, 3)
    vectorized_shape = (num_nodes, num_wake_interactions, 4, 3)

    # expanding and vectorizing terms
    coll_point_exp = csdl.expand(coll_point, (num_nodes, num_tot_panels, num_wake_panels, 4, 3), 'ijk->ijabk')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    coll_point_w_exp = csdl.expand(coll_point_w, expanded_shape, 'ijkl->iajkbl')
    coll_point_w_exp_vec = coll_point_w_exp.reshape(vectorized_shape)

    panel_corners_w_exp = csdl.expand(panel_corners_w, expanded_shape, 'ijklm->iajklm')
    panel_corners_w_exp_vec = panel_corners_w_exp.reshape(vectorized_shape)

    panel_x_dir_w_exp = csdl.expand(panel_x_dir_w, expanded_shape, 'ijkl->iajkbl')
    panel_x_dir_w_exp_vec = panel_x_dir_w_exp.reshape(vectorized_shape)
    panel_y_dir_w_exp = csdl.expand(panel_y_dir_w, expanded_shape, 'ijkl->iajkbl')
    panel_y_dir_w_exp_vec = panel_y_dir_w_exp.reshape(vectorized_shape)
    panel_normal_w_exp = csdl.expand(panel_normal_w, expanded_shape, 'ijkl->iajkbl')
    panel_normal_w_exp_vec = panel_normal_w_exp.reshape(vectorized_shape)

    SL_w_exp = csdl.expand(SL_w, expanded_shape[:-1], 'jklm->jaklm')
    SL_w_exp_vec = SL_w_exp.reshape(vectorized_shape[:-1])

    SM_w_exp = csdl.expand(SM_w, expanded_shape[:-1], 'jklm->jaklm')
    SM_w_exp_vec = SM_w_exp.reshape(vectorized_shape[:-1])

    a = coll_point_exp_vec - panel_corners_w_exp_vec # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_w_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_w_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_w_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_w_exp_vec, axes=(sum_ind,)) # normal projection of CP

    B = csdl.Variable(shape=A.shape, value=0.)
    B = B.set(csdl.slice[:,:,:-1], value=A[:,:,1:])
    B = B.set(csdl.slice[:,:,-1], value=A[:,:,0])

    BL = csdl.Variable(shape=AL.shape, value=0.)
    BL = BL.set(csdl.slice[:,:,:-1], value=BL[:,:,1:])
    BL = BL.set(csdl.slice[:,:,-1], value=BL[:,:,0])

    BM = csdl.Variable(shape=AM.shape, value=0.)
    BM = BM.set(csdl.slice[:,:,:-1], value=AM[:,:,1:])
    BM = BM.set(csdl.slice[:,:,-1], value=AM[:,:,0])

    A1 = AM*SL_w_exp_vec - AL*SM_w_exp_vec

    # print(A.shape)

    A_list = [A[:,:,ind] for ind in range(4)]
    AM_list = [AM[:,:,ind] for ind in range(4)]
    B_list = [B[:,:,ind] for ind in range(4)]
    BM_list = [BM[:,:,ind] for ind in range(4)]
    SL_list = [SL_w_exp_vec[:,:,ind] for ind in range(4)]
    SM_list = [SM_w_exp_vec[:,:,ind] for ind in range(4)]
    A1_list = [A1[:,:,ind] for ind in range(4)]
    PN_list = [PN[:,:,ind] for ind in range(4)]

    wake_doublet_influence_vec = compute_doublet_influence_new(
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
    wake_doublet_influence = wake_doublet_influence_vec.reshape((num_nodes, num_tot_panels, num_wake_panels))
    AIC_mu_adjustment = csdl.Variable(value=np.zeros(AIC_mu_orig.shape))
    # for te_ind in range(len(list(lower_TE_cell_ind))):
    te_ind_list = list(np.arange(len(list(lower_TE_cell_ind)), dtype=int))
    for te_ind, lower_ind, upper_ind in csdl.frange(vals=(te_ind_list, lower_TE_cell_ind, upper_TE_cell_ind)):
        # lower_ind = lower_TE_cell_ind[te_ind]
        # upper_ind = upper_TE_cell_ind[te_ind]
        AIC_mu_adjustment = AIC_mu_adjustment.set(
            csdl.slice[:,:,lower_ind],
            value=-wake_doublet_influence[:,:,te_ind]
        )
        AIC_mu_adjustment = AIC_mu_adjustment.set(
            csdl.slice[:,:,upper_ind],
            value=wake_doublet_influence[:,:,te_ind]
        )

        # AIC_mu = AIC_mu.set(
        #     csdl.slice[:,:,lower_ind],
        #     value=AIC_mu[:,:,lower_ind]-wake_doublet_influence[:,:,te_ind]
        # )

    AIC_mu = AIC_mu_orig + AIC_mu_adjustment

    return AIC_mu, AIC_sigma, AIC_mu_orig


def unstructured_AIC_computation_UW(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels):
    
    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']
    num_first_wake_panels = len(upper_TE_cell_ind)

    coll_point_eval = mesh_dict['panel_center_mod']

    coll_point = mesh_dict['panel_center'] # (nn, nt, num_tot_panels, 3)
    panel_corners = mesh_dict['panel_corners'] # (nn, nt, num_tot_panels, 3, 3) 
    panel_x_dir = mesh_dict['panel_x_dir'] # (nn, nt, num_tot_panels, 3)
    panel_y_dir = mesh_dict['panel_y_dir'] # (nn, nt, num_tot_panels, 3)
    panel_normal = mesh_dict['panel_normal'] # (nn, nt, num_tot_panels, 3)
    S_j = mesh_dict['S']
    SL_j = mesh_dict['SL']
    SM_j = mesh_dict['SM']

    num_interactions = num_tot_panels**2
    expanded_shape = (num_nodes, num_tot_panels, num_tot_panels, 3, 3)
    vectorized_shape = (num_nodes, num_interactions, 3, 3)

    # expanding collocation points (where boundary condition is applied, the "i-th" expansion-vectorization)
    coll_point_exp = csdl.expand(coll_point_eval, expanded_shape, 'jkl->jkabl')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    # expanding the panel terms used to compute influences AT the collocation points
    # -> the "j-th" expansion-vectorization
    coll_point_j_exp = csdl.expand(coll_point, expanded_shape, 'jkl->jakbl')
    coll_point_j_exp_vec = coll_point_j_exp.reshape(vectorized_shape)

    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'jklm->jaklm')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    panel_x_dir_exp = csdl.expand(panel_x_dir, expanded_shape, 'jkl->jakbl')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)
    panel_y_dir_exp = csdl.expand(panel_y_dir, expanded_shape, 'jkl->jakbl')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)
    panel_normal_exp = csdl.expand(panel_normal, expanded_shape, 'jkl->jakbl')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    # dpij_exp = csdl.expand(S_j, expanded_shape[:-1] + (2,), 'ijklm->ijaklm')
    # dpij_exp_vec = dpij_exp.reshape(vectorized_shape[:-1] + (2,))
    # dij_exp = csdl.expand(dij, expanded_shape[:-1], 'ijkl->ijakl')
    # dij_exp_vec = dij_exp.reshape(vectorized_shape[:-1])

    S_j_exp = csdl.expand(S_j, expanded_shape[:-1] , 'jkl->jakl')
    S_j_exp_vec = S_j_exp.reshape(vectorized_shape[:-1])

    SL_j_exp = csdl.expand(SL_j, expanded_shape[:-1], 'jkl->jakl')
    SL_j_exp_vec = SL_j_exp.reshape(vectorized_shape[:-1])

    SM_j_exp = csdl.expand(SM_j, expanded_shape[:-1], 'jkl->jakl')
    SM_j_exp_vec = SM_j_exp.reshape(vectorized_shape[:-1])

    a = coll_point_exp_vec - panel_corners_exp_vec # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_j_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    # for i in csdl.frange(1):

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_exp_vec, axes=(sum_ind,)) # normal projection of CP

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

    A_list = [A[:,:,ind] for ind in range(3)]
    AM_list = [AM[:,:,ind] for ind in range(3)]
    B_list = [B[:,:,ind] for ind in range(3)]
    BM_list = [BM[:,:,ind] for ind in range(3)]
    SL_list = [SL_j_exp_vec[:,:,ind] for ind in range(3)]
    SM_list = [SM_j_exp_vec[:,:,ind] for ind in range(3)]
    A1_list = [A1[:,:,ind] for ind in range(3)]
    PN_list = [PN[:,:,ind] for ind in range(3)]
    S_list = [S_j_exp_vec[:,:,ind] for ind in range(3)]

    doublet_influence_vec = compute_doublet_influence_new(
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
    doublet_influence = doublet_influence_vec.reshape((num_nodes, num_tot_panels, num_tot_panels))
#     graph = csdl.get_current_recorder().active_graph
    # graph.visualize('subgraph_lo')
    # print(len(graph.node_table))
    # exit()
    AIC_mu_orig = doublet_influence

    source_influence_vec = compute_source_influence_new(
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
    source_influence = source_influence_vec.reshape((num_nodes, num_tot_panels, num_tot_panels))
    AIC_sigma = source_influence

    # wake AIC influence
    # location where wake influence is computed
    coll_point = mesh_dict['panel_center_mod'][:,:,:]

    # wake values
    panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, np_w, 4, 3)
    coll_point_w = wake_mesh_dict['panel_center'] # (nn, np_w, 3)
    panel_x_dir_w = wake_mesh_dict['panel_x_dir'] # (nn, np_w, 3)
    panel_y_dir_w = wake_mesh_dict['panel_y_dir'] # (nn, np_w, 3)
    panel_normal_w = wake_mesh_dict['panel_normal'] # (nn, np_w, 3)
    SL_w = wake_mesh_dict['SL']
    SM_w = wake_mesh_dict['SM']
    
    nc_w, ns_w = panel_corners_w.shape[1], panel_corners_w.shape[2]
    num_wake_panels = wake_mesh_dict['num_panels']
    
    # target expansion and vectorization shapes
    # TODO: add support for multisurface
    num_wake_interactions = num_tot_panels*num_wake_panels
    expanded_shape = (num_nodes, num_tot_panels, num_wake_panels, 4, 3)
    vectorized_shape = (num_nodes, num_wake_interactions, 4, 3)

    # expanding and vectorizing terms
    coll_point_exp = csdl.expand(coll_point, (num_nodes, num_tot_panels, num_wake_panels, 4, 3), 'ijk->ijabk')
    coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape)

    coll_point_w_exp = csdl.expand(coll_point_w, expanded_shape, 'ijk->iajbk') #FIX EXPANSION SHAPE
    coll_point_w_exp_vec = coll_point_w_exp.reshape(vectorized_shape)

    panel_corners_w_exp = csdl.expand(panel_corners_w, expanded_shape, 'ijkl->iajkl')
    panel_corners_w_exp_vec = panel_corners_w_exp.reshape(vectorized_shape)

    panel_x_dir_w_exp = csdl.expand(panel_x_dir_w, expanded_shape, 'ijk->iajbk')
    panel_x_dir_w_exp_vec = panel_x_dir_w_exp.reshape(vectorized_shape)
    panel_y_dir_w_exp = csdl.expand(panel_y_dir_w, expanded_shape, 'ijk->iajbk')
    panel_y_dir_w_exp_vec = panel_y_dir_w_exp.reshape(vectorized_shape)
    panel_normal_w_exp = csdl.expand(panel_normal_w, expanded_shape, 'ijk->iajbk')
    panel_normal_w_exp_vec = panel_normal_w_exp.reshape(vectorized_shape)

    SL_w_exp = csdl.expand(SL_w, expanded_shape[:-1], 'ijk->iajk')
    SL_w_exp_vec = SL_w_exp.reshape(vectorized_shape[:-1])

    SM_w_exp = csdl.expand(SM_w, expanded_shape[:-1], 'ijk->iajk')
    SM_w_exp_vec = SM_w_exp.reshape(vectorized_shape[:-1])

    a = coll_point_exp_vec - panel_corners_w_exp_vec # Rc - Ri
    P_JK = coll_point_exp_vec - coll_point_w_exp_vec # RcJ - RcK
    sum_ind = len(a.shape) - 1

    A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
    AL = csdl.sum(a*panel_x_dir_w_exp_vec, axes=(sum_ind,))
    AM = csdl.sum(a*panel_y_dir_w_exp_vec, axes=(sum_ind,)) # m-direction projection 
    PN = csdl.sum(P_JK*panel_normal_w_exp_vec, axes=(sum_ind,)) # normal projection of CP

    B = csdl.Variable(shape=A.shape, value=0.)
    B = B.set(csdl.slice[:,:,:-1], value=A[:,:,1:])
    B = B.set(csdl.slice[:,:,-1], value=A[:,:,0])

    BL = csdl.Variable(shape=AL.shape, value=0.)
    BL = BL.set(csdl.slice[:,:,:-1], value=BL[:,:,1:])
    BL = BL.set(csdl.slice[:,:,-1], value=BL[:,:,0])

    BM = csdl.Variable(shape=AM.shape, value=0.)
    BM = BM.set(csdl.slice[:,:,:-1], value=AM[:,:,1:])
    BM = BM.set(csdl.slice[:,:,-1], value=AM[:,:,0])

    A1 = AM*SL_w_exp_vec - AL*SM_w_exp_vec

    # print(A.shape)

    A_list = [A[:,:,ind] for ind in range(4)]
    AM_list = [AM[:,:,ind] for ind in range(4)]
    B_list = [B[:,:,ind] for ind in range(4)]
    BM_list = [BM[:,:,ind] for ind in range(4)]
    SL_list = [SL_w_exp_vec[:,:,ind] for ind in range(4)]
    SM_list = [SM_w_exp_vec[:,:,ind] for ind in range(4)]
    A1_list = [A1[:,:,ind] for ind in range(4)]
    PN_list = [PN[:,:,ind] for ind in range(4)]

    wake_doublet_influence_vec = compute_doublet_influence_new(
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
    wake_doublet_influence = wake_doublet_influence_vec.reshape((num_nodes, num_tot_panels, num_wake_panels))
    
    te_ind_list = np.arange(len(list(lower_TE_cell_ind)), dtype=int).tolist()
    # ==== USING CSDL FRANGE ====
    # AIC_mu_adjustment = csdl.Variable(value=np.zeros(AIC_mu_orig.shape))
    # for te_ind, lower_ind, upper_ind in csdl.frange(vals=(te_ind_list, lower_TE_cell_ind, upper_TE_cell_ind)):
    #     # lower_ind = lower_TE_cell_ind[te_ind]
    #     # upper_ind = upper_TE_cell_ind[te_ind]
    #     AIC_mu_adjustment = AIC_mu_adjustment.set(
    #         csdl.slice[:,:,lower_ind],
    #         value=-wake_doublet_influence[:,:,te_ind]
    #     )
    #     AIC_mu_adjustment = AIC_mu_adjustment.set(
    #         csdl.slice[:,:,upper_ind],
    #         value=wake_doublet_influence[:,:,te_ind]
    #     )

    # AIC_mu = AIC_mu_orig + AIC_mu_adjustment

    # ==== USING STACK VIA LOOP BUILDER ====
    loop_vals = [te_ind_list]
    with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
        i = loop_builder.get_loop_indices()
        AIC_KC_lower = -wake_doublet_influence[:,:,i]
        AIC_KC_upper = wake_doublet_influence[:,:,i]
    AIC_KC_lower = loop_builder.add_stack(AIC_KC_lower)
    AIC_KC_upper = loop_builder.add_stack(AIC_KC_upper)
    loop_builder.finalize()
    num_TE_panels = len(te_ind_list)
    AIC_KC_lower = AIC_KC_lower.reshape((num_TE_panels, num_tot_panels)).T().reshape((num_nodes, num_tot_panels, num_TE_panels))
    AIC_KC_upper = AIC_KC_upper.reshape((num_TE_panels, num_tot_panels)).T().reshape((num_nodes, num_tot_panels, num_TE_panels))
    
    AIC_mu = AIC_mu_orig
    AIC_mu = AIC_mu.set(
        csdl.slice[:,:,lower_TE_cell_ind],
        AIC_mu_orig[:,:,lower_TE_cell_ind] + AIC_KC_lower
    )

    AIC_mu = AIC_mu.set(
        csdl.slice[:,:,upper_TE_cell_ind],
        AIC_mu_orig[:,:,upper_TE_cell_ind] + AIC_KC_upper
    )

    return AIC_mu, AIC_sigma, AIC_mu_orig

def unstructured_AIC_computation_UW_looping(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, batch_size):
    
    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']

    coll_point_eval = mesh_dict['panel_center_mod']

    coll_point = mesh_dict['panel_center'] # (nn, num_tot_panels, 3)
    panel_corners = mesh_dict['panel_corners'] # (nn, num_tot_panels, 3, 3) 
    panel_x_dir = mesh_dict['panel_x_dir'] # (nn, num_tot_panels, 3)
    panel_y_dir = mesh_dict['panel_y_dir'] # (nn, num_tot_panels, 3)
    panel_normal = mesh_dict['panel_normal'] # (nn, num_tot_panels, 3)
    S_j = mesh_dict['S']
    SL_j = mesh_dict['SL']
    SM_j = mesh_dict['SM']

    # batch_size = num_tot_panels // 3984 # must divide num_panels into an integer
    num_batches = num_tot_panels // batch_size
    loop_vals = [np.arange(0, num_batches, dtype=int).tolist()]
    num_interactions_in_loop = batch_size*num_tot_panels
    expanded_shape = (num_nodes, batch_size, num_tot_panels, 3, 3)
    vectorized_shape = (num_nodes, num_interactions_in_loop, 3, 3)

    # expanding the panel terms used to compute influences AT the collocation points
    # -> the "j-th" expansion-vectorization
    coll_point_j_exp = csdl.expand(coll_point, expanded_shape, 'jkl->jakbl')
    coll_point_j_exp_vec = coll_point_j_exp.reshape(vectorized_shape)

    panel_corners_exp = csdl.expand(panel_corners, expanded_shape, 'jklm->jaklm')
    panel_corners_exp_vec = panel_corners_exp.reshape(vectorized_shape)

    panel_x_dir_exp = csdl.expand(panel_x_dir, expanded_shape, 'jkl->jakbl')
    panel_x_dir_exp_vec = panel_x_dir_exp.reshape(vectorized_shape)
    panel_y_dir_exp = csdl.expand(panel_y_dir, expanded_shape, 'jkl->jakbl')
    panel_y_dir_exp_vec = panel_y_dir_exp.reshape(vectorized_shape)
    panel_normal_exp = csdl.expand(panel_normal, expanded_shape, 'jkl->jakbl')
    panel_normal_exp_vec = panel_normal_exp.reshape(vectorized_shape)

    S_j_exp = csdl.expand(S_j, expanded_shape[:-1] , 'jkl->jakl')
    S_j_exp_vec = S_j_exp.reshape(vectorized_shape[:-1])
    SL_j_exp = csdl.expand(SL_j, expanded_shape[:-1], 'jkl->jakl')
    SL_j_exp_vec = SL_j_exp.reshape(vectorized_shape[:-1])
    SM_j_exp = csdl.expand(SM_j, expanded_shape[:-1], 'jkl->jakl')
    SM_j_exp_vec = SM_j_exp.reshape(vectorized_shape[:-1])

    coll_point_eval = coll_point_eval.reshape((num_nodes, num_batches, batch_size, 3))

    nn_loop_vals = [np.arange(0, num_nodes, dtype=int).tolist()]
    with csdl.experimental.enter_loop(vals=nn_loop_vals) as nn_loop_builder:
        n = nn_loop_builder.get_loop_indices()

        with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
            i = loop_builder.get_loop_indices()

            # coll_point_eval_batch = coll_point_eval[0,batch_size*i:batch_size*(i+1),:]
            coll_point_eval_batch = coll_point_eval[n,i,:,:]

            # expanding collocation points (where boundary condition is applied, the "i-th" expansion-vectorization)
            coll_point_exp = csdl.expand(coll_point_eval_batch, expanded_shape[1:], 'kl->kabl')
            coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape[1:])

            a = coll_point_exp_vec - panel_corners_exp_vec[n,:] # Rc - Ri
            P_JK = coll_point_exp_vec - coll_point_j_exp_vec[n,:] # RcJ - RcK
            sum_ind = len(a.shape) - 1

            A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
            AL = csdl.sum(a*panel_x_dir_exp_vec[n,:], axes=(sum_ind,))
            AM = csdl.sum(a*panel_y_dir_exp_vec[n,:], axes=(sum_ind,)) # m-direction projection 
            PN = csdl.sum(P_JK*panel_normal_exp_vec[n,:], axes=(sum_ind,)) # normal projection of CP

            B = csdl.Variable(shape=A.shape, value=0.)
            B = B.set(csdl.slice[:,:-1], value=A[:,1:])
            B = B.set(csdl.slice[:,-1], value=A[:,0])

            BL = csdl.Variable(shape=AL.shape, value=0.)
            BL = BL.set(csdl.slice[:,:-1], value=BL[:,1:])
            BL = BL.set(csdl.slice[:,-1], value=BL[:,0])

            BM = csdl.Variable(shape=AM.shape, value=0.)
            BM = BM.set(csdl.slice[:,:-1], value=AM[:,1:])
            BM = BM.set(csdl.slice[:,-1], value=AM[:,0])

            A1 = AM*SL_j_exp_vec[n,:] - AL*SM_j_exp_vec[n,:]

            A_list = [A[:,ind] for ind in range(3)]
            AM_list = [AM[:,ind] for ind in range(3)]
            B_list = [B[:,ind] for ind in range(3)]
            BM_list = [BM[:,ind] for ind in range(3)]
            SL_list = [SL_j_exp_vec[n,:,ind] for ind in range(3)]
            SM_list = [SM_j_exp_vec[n,:,ind] for ind in range(3)]
            A1_list = [A1[:,ind] for ind in range(3)]
            PN_list = [PN[:,ind] for ind in range(3)]
            S_list = [S_j_exp_vec[n,:,ind] for ind in range(3)]

            doublet_influence_vec_nn = compute_doublet_influence_new(
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

            source_influence_vec_nn = compute_source_influence_new(
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
            # print('====')
            # print(doublet_influence_vec.shape)

            # doublet_influence_grid = doublet_influence_vec.reshape((batch_size, num_tot_panels))
            # source_influence_grid = source_influence_vec.reshape((batch_size, num_tot_panels))
            # print('====')
            # print(doublet_influence_grid.shape)

        doublet_influence_vec = loop_builder.add_stack(doublet_influence_vec_nn)
        source_influence_vec = loop_builder.add_stack(source_influence_vec_nn)
        loop_builder.finalize()
    
    doublet_influence_vec = nn_loop_builder.add_stack(doublet_influence_vec)
    source_influence_vec = nn_loop_builder.add_stack(source_influence_vec)
    nn_loop_builder.finalize()
    # doublet_influence_vec = csdl.Variable(value=0, shape=(num_nodes, num_interactions))
    # source_influence_vec = csdl.Variable(value=0, shape=(num_nodes, num_interactions))
    # print(doublet_influence_vec.shape)
    # doublet_influence_batch_grid = doublet_influence_vec.reshape((num_batches, batch_size, num_tot_panels))
    # source_influence_batch_grid = source_influence_vec.reshape((num_batches, batch_size, num_tot_panels))

    # AIC_mu_orig = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    # AIC_sigma = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    # for i in csdl.frange(num_batches):
    #     AIC_mu_orig = AIC_mu_orig.set(
    #         csdl.slice[0,batch_size*i:batch_size*(i+1),:],
    #         doublet_influence_batch_grid[i,:,:]
    #     )

    #     AIC_sigma = AIC_sigma.set(
    #         csdl.slice[0,batch_size*i:batch_size*(i+1),:],
    #         source_influence_batch_grid[i,:,:]
    #     )

    # print(AIC_mu_orig.shape)
    # print(AIC_sigma.shape)
    # exit()
    AIC_mu_orig = doublet_influence_vec.reshape((num_nodes, num_tot_panels, num_tot_panels))
    AIC_sigma = source_influence_vec.reshape((num_nodes, num_tot_panels, num_tot_panels))

    # doublet_influence_grid_batch = loop_builder.add_stack(doublet_influence_grid)
    # source_influence_grid_batch = loop_builder.add_stack(source_influence_grid)
    # loop_builder.finalize()
    # print(doublet_influence_grid.shape)
    # exit()
    # AIC_mu_orig = doublet_influence_grid.reshape((num_nodes, num_tot_panels, num_tot_panels))
    # AIC_sigma = source_influence_grid.reshape((num_nodes, num_tot_panels, num_tot_panels))

    # wake AIC influence
    # location where wake influence is computed
    coll_point = mesh_dict['panel_center_mod']

    # wake values
    panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, np_w, 4, 3)
    coll_point_w = wake_mesh_dict['panel_center'] # (nn, np_w, 3)
    panel_x_dir_w = wake_mesh_dict['panel_x_dir'] # (nn, np_w, 3)
    panel_y_dir_w = wake_mesh_dict['panel_y_dir'] # (nn, np_w, 3)
    panel_normal_w = wake_mesh_dict['panel_normal'] # (nn, np_w, 3)
    SL_w = wake_mesh_dict['SL']
    SM_w = wake_mesh_dict['SM']
    
    num_wake_panels = wake_mesh_dict['num_panels']
    batch_size = batch_size # must divide num_panels into an integer
    num_batches = num_batches
    loop_vals = [np.arange(0, num_batches, dtype=int).tolist()]
    num_interactions_in_loop = batch_size*num_wake_panels
    expanded_shape = (num_nodes, batch_size, num_wake_panels, 4, 3)
    vectorized_shape = (num_nodes, num_interactions_in_loop, 4, 3)

    coll_point = coll_point.reshape((num_nodes, num_batches, batch_size, 3))

    # expanding and vectorizing terms
    coll_point_w_exp = csdl.expand(coll_point_w, expanded_shape, 'ijk->iajbk') #FIX EXPANSION SHAPE
    coll_point_w_exp_vec = coll_point_w_exp.reshape(vectorized_shape)

    panel_corners_w_exp = csdl.expand(panel_corners_w, expanded_shape, 'ijkl->iajkl')
    panel_corners_w_exp_vec = panel_corners_w_exp.reshape(vectorized_shape)

    panel_x_dir_w_exp = csdl.expand(panel_x_dir_w, expanded_shape, 'ijk->iajbk')
    panel_x_dir_w_exp_vec = panel_x_dir_w_exp.reshape(vectorized_shape)
    panel_y_dir_w_exp = csdl.expand(panel_y_dir_w, expanded_shape, 'ijk->iajbk')
    panel_y_dir_w_exp_vec = panel_y_dir_w_exp.reshape(vectorized_shape)
    panel_normal_w_exp = csdl.expand(panel_normal_w, expanded_shape, 'ijk->iajbk')
    panel_normal_w_exp_vec = panel_normal_w_exp.reshape(vectorized_shape)

    SL_w_exp = csdl.expand(SL_w, expanded_shape[:-1], 'ijk->iajk')
    SL_w_exp_vec = SL_w_exp.reshape(vectorized_shape[:-1])
    SM_w_exp = csdl.expand(SM_w, expanded_shape[:-1], 'ijk->iajk')
    SM_w_exp_vec = SM_w_exp.reshape(vectorized_shape[:-1])

    with csdl.experimental.enter_loop(vals=nn_loop_vals) as nn_loop_builder:
        n = nn_loop_builder.get_loop_indices()

        with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
            i = loop_builder.get_loop_indices()

            coll_point_exp = csdl.expand(coll_point[n,i,:,:], expanded_shape[1:], 'jk->jabk')
            coll_point_exp_vec = coll_point_exp.reshape(vectorized_shape[1:])

            a = coll_point_exp_vec - panel_corners_w_exp_vec[n,:] # Rc - Ri
            P_JK = coll_point_exp_vec - coll_point_w_exp_vec[n,:] # RcJ - RcK
            sum_ind = len(a.shape) - 1
            
            A = csdl.norm(a, axes=(sum_ind,)) # norm of distance from CP of i to corners of j
            AL = csdl.sum(a*panel_x_dir_w_exp_vec[n,:], axes=(sum_ind,))
            AM = csdl.sum(a*panel_y_dir_w_exp_vec[n,:], axes=(sum_ind,)) # m-direction projection 
            PN = csdl.sum(P_JK*panel_normal_w_exp_vec[n,:], axes=(sum_ind,)) # normal projection of CP

            B = csdl.Variable(shape=A.shape, value=0.)
            B = B.set(csdl.slice[:,:-1], value=A[:,1:])
            B = B.set(csdl.slice[:,-1], value=A[:,0])

            BL = csdl.Variable(shape=AL.shape, value=0.)
            BL = BL.set(csdl.slice[:,:-1], value=BL[:,1:])
            BL = BL.set(csdl.slice[:,-1], value=BL[:,0])

            BM = csdl.Variable(shape=AM.shape, value=0.)
            BM = BM.set(csdl.slice[:,:-1], value=AM[:,1:])
            BM = BM.set(csdl.slice[:,-1], value=AM[:,0])

            A1 = AM*SL_w_exp_vec[n,:] - AL*SM_w_exp_vec[n,:]

            # print(A.shape)

            A_list = [A[:,ind] for ind in range(4)]
            AM_list = [AM[:,ind] for ind in range(4)]
            B_list = [B[:,ind] for ind in range(4)]
            BM_list = [BM[:,ind] for ind in range(4)]
            SL_list = [SL_w_exp_vec[n,:,ind] for ind in range(4)]
            SM_list = [SM_w_exp_vec[n,:,ind] for ind in range(4)]
            A1_list = [A1[:,ind] for ind in range(4)]
            PN_list = [PN[:,ind] for ind in range(4)]

            wake_doublet_influence_vec_nn = compute_doublet_influence_new(
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
        
        wake_doublet_influence_vec = loop_builder.add_stack(wake_doublet_influence_vec_nn)
        loop_builder.finalize()
    wake_doublet_influence_vec = nn_loop_builder.add_stack(wake_doublet_influence_vec)
    nn_loop_builder.finalize()
    # wake_doublet_influence_vec = csdl.Variable(value=0, shape=((num_nodes, num_tot_panels*num_wake_panels)))
    wake_doublet_influence = wake_doublet_influence_vec.reshape((num_nodes, num_tot_panels, num_wake_panels))
    # NOTE: FIX THIS LINE ABOVE
    # wake_doublet_influence = wake_doublet_influence_vec
    # print(wake_doublet_influence.shape)
    # exit()
    
    te_ind_list = np.arange(len(list(lower_TE_cell_ind)), dtype=int).tolist()
    # ==== USING CSDL FRANGE ====
    # AIC_mu_adjustment = csdl.Variable(value=np.zeros(AIC_mu_orig.shape))
    # for te_ind, lower_ind, upper_ind in csdl.frange(vals=(te_ind_list, lower_TE_cell_ind, upper_TE_cell_ind)):
    #     # lower_ind = lower_TE_cell_ind[te_ind]
    #     # upper_ind = upper_TE_cell_ind[te_ind]
    #     AIC_mu_adjustment = AIC_mu_adjustment.set(
    #         csdl.slice[:,:,lower_ind],
    #         value=-wake_doublet_influence[:,:,te_ind]
    #     )
    #     AIC_mu_adjustment = AIC_mu_adjustment.set(
    #         csdl.slice[:,:,upper_ind],
    #         value=wake_doublet_influence[:,:,te_ind]
    #     )

    # AIC_mu = AIC_mu_orig + AIC_mu_adjustment

    # ==== USING STACK VIA LOOP BUILDER ====
    loop_vals = [te_ind_list]
    with csdl.experimental.enter_loop(vals=nn_loop_vals) as nn_loop_builder:
        n = nn_loop_builder.get_loop_indices()
        with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
            i = loop_builder.get_loop_indices()
            AIC_KC_lower_nn = -wake_doublet_influence[n,:,i]
            AIC_KC_upper_nn = wake_doublet_influence[n,:,i]
            # print(AIC_KC_lower_nn.shape)
        AIC_KC_lower = loop_builder.add_stack(AIC_KC_lower_nn)
        AIC_KC_upper = loop_builder.add_stack(AIC_KC_upper_nn)
        loop_builder.finalize()
        # print(AIC_KC_lower.shape)
        AIC_KC_lower = AIC_KC_lower.T()
        AIC_KC_upper = AIC_KC_upper.T()
        # print(AIC_KC_lower.shape)
    
    AIC_KC_lower = nn_loop_builder.add_stack(AIC_KC_lower)
    AIC_KC_upper = nn_loop_builder.add_stack(AIC_KC_upper)
    nn_loop_builder.finalize()
    # print(AIC_KC_lower.shape)
    # exit()
    num_TE_panels = len(te_ind_list)
    # AIC_KC_lower = AIC_KC_lower.reshape((num_nodes, num_TE_panels, num_tot_panels)).T().reshape((num_nodes, num_tot_panels, num_TE_panels))
    # AIC_KC_upper = AIC_KC_upper.reshape((num_nodes, num_TE_panels, num_tot_panels)).T().reshape((num_nodes, num_tot_panels, num_TE_panels))
    # AIC_KC_lower = csdl.einsum(AIC_KC_lower, action='ijk->ikj')
    # AIC_KC_upper = csdl.einsum(AIC_KC_upper, action='ijk->ikj')
    
    AIC_mu = AIC_mu_orig
    AIC_mu = AIC_mu.set(
        csdl.slice[:,:,lower_TE_cell_ind],
        AIC_mu_orig[:,:,lower_TE_cell_ind] + AIC_KC_lower
    )

    AIC_mu = AIC_mu.set(
        csdl.slice[:,:,upper_TE_cell_ind],
        AIC_mu_orig[:,:,upper_TE_cell_ind] + AIC_KC_upper
    )

    return AIC_mu, AIC_sigma, AIC_mu_orig

def unstructured_AIC_computation_UW_looping_mixed(mesh_dict, wake_mesh_dict, num_nodes, num_tot_panels, batch_size):
    cells = mesh_dict['cell_point_indices'] # keys are cell types, entries are points for each cell
    cell_types = list(cells.keys())
    cell_adjacency_types = mesh_dict['cell_adjacency'] # keys are cell types, entries are adjacent cell indices
    num_cells_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]
    num_tot_panels = sum(num_cells_per_type)

    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']
    num_wake_panels = wake_mesh_dict['num_panels']

    AIC_mu_orig = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    AIC_sigma = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)
    AIC_mu_wake = csdl.Variable(shape=(num_nodes, num_tot_panels, num_wake_panels), value=0.)

    AIC_batch_func = csdl.experimental.batch_function(
        compute_aic_batched,
        batch_size=batch_size,
        batch_dims=[1]+[None]*8
        # batch_dims=[None]+[1]*8
    )

    AIC_batch_func_mu = csdl.experimental.batch_function(
        compute_aic_batched,
        batch_size=batch_size,
        batch_dims=[1]+[None]*8
        # batch_dims=[None]+[1]*8
    )

    AIC_batch_func_sigma = csdl.experimental.batch_function(
        compute_aic_batched,
        batch_size=batch_size,
        batch_dims=[1]+[None]*8
        # batch_dims=[None]+[1]*8
    )

    start_i, stop_i = 0, 0
    for i, cell_type_i in enumerate(cell_types):
        num_cells_i = num_cells_per_type[i]
        stop_i += num_cells_i

        coll_point_eval = mesh_dict['panel_center_mod_' + cell_type_i]

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

            doublet_influence = AIC_batch_func_mu(
                coll_point_eval,
                coll_point,
                panel_corners,
                panel_x_dir,
                panel_y_dir,
                panel_normal,
                S,
                SL,
                SM,
                mode='doublet'
            )

            source_influence = AIC_batch_func_sigma(
                coll_point_eval,
                coll_point,
                panel_corners,
                panel_x_dir,
                panel_y_dir,
                panel_normal,
                S,
                SL,
                SM,
                mode='source'
            )

            doublet_influence = doublet_influence[:,0,:].reshape((num_cells_i, num_cells_j))
            source_influence = source_influence[:,0,:].reshape((num_cells_i, num_cells_j))

            AIC_mu_orig = AIC_mu_orig.set(csdl.slice[0, start_i:stop_i, start_j:stop_j], doublet_influence)
            AIC_sigma = AIC_sigma.set(csdl.slice[0, start_i:stop_i, start_j:stop_j], source_influence)

            start_j += num_cells_j

        # wake influence outside of the inner loop
        # compute the AIC wake matrix here (reduced shape of (num_panels,num_wake_panels))

        panel_corners_w = wake_mesh_dict['panel_corners'] # (nn, np_w, 4, 3)
        coll_point_w = wake_mesh_dict['panel_center'] # (nn, np_w, 3)
        panel_x_dir_w = wake_mesh_dict['panel_x_dir'] # (nn, np_w, 3)
        panel_y_dir_w = wake_mesh_dict['panel_y_dir'] # (nn, np_w, 3)
        panel_normal_w = wake_mesh_dict['panel_normal'] # (nn, np_w, 3)
        S_w = wake_mesh_dict['S']
        SL_w = wake_mesh_dict['SL']
        SM_w = wake_mesh_dict['SM']

        wake_doublet_influence = AIC_batch_func(
            coll_point_eval,
            coll_point_w,
            panel_corners_w,
            panel_x_dir_w,
            panel_y_dir_w,
            panel_normal_w,
            S_w,
            SL_w,
            SM_w,
            mode='wake'
        )
        wake_doublet_influence = wake_doublet_influence.reshape((1,num_cells_i,num_wake_panels))
        AIC_mu_wake = AIC_mu_wake.set(csdl.slice[:,start_i:stop_i,:], wake_doublet_influence)

        start_i += num_cells_i

    # ==== USING STACK VIA LOOP BUILDER ====
    nn_loop_vals = [np.arange(0, num_nodes, dtype=int).tolist()]
    te_ind_list = np.arange(len(list(lower_TE_cell_ind)), dtype=int).tolist()
    loop_vals = [te_ind_list]
    with csdl.experimental.enter_loop(vals=nn_loop_vals) as nn_loop_builder:
        n = nn_loop_builder.get_loop_indices()
        with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
            i = loop_builder.get_loop_indices()
            AIC_KC_lower_nn = -AIC_mu_wake[n,:,i]
            AIC_KC_upper_nn = AIC_mu_wake[n,:,i]
            # print(AIC_KC_lower_nn.shape)
        AIC_KC_lower = loop_builder.add_stack(AIC_KC_lower_nn)
        AIC_KC_upper = loop_builder.add_stack(AIC_KC_upper_nn)
        loop_builder.finalize()

        AIC_KC_lower = AIC_KC_lower.T()
        AIC_KC_upper = AIC_KC_upper.T()
    
    AIC_KC_lower = nn_loop_builder.add_stack(AIC_KC_lower)
    AIC_KC_upper = nn_loop_builder.add_stack(AIC_KC_upper)
    nn_loop_builder.finalize()
    
    AIC_mu = AIC_mu_orig
    AIC_mu = AIC_mu.set(
        csdl.slice[:,:,lower_TE_cell_ind],
        AIC_mu_orig[:,:,lower_TE_cell_ind] + AIC_KC_lower
    )

    AIC_mu = AIC_mu.set(
        csdl.slice[:,:,upper_TE_cell_ind],
        AIC_mu_orig[:,:,upper_TE_cell_ind] + AIC_KC_upper
    )

    return AIC_mu, AIC_sigma, AIC_mu_orig


def compute_aic_batched(coll_point, panel_center, panel_corners, panel_x_dir, panel_y_dir,
                        panel_normal, S_j, SL_j, SM_j, mode='doublet'):
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

    if mode == 'doublet' or mode == 'wake':
        AIC_vec = compute_doublet_influence_new(
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
    elif mode == 'source':
        AIC_vec = compute_source_influence_new(
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

    # A_out = AIC_vec.reshape((num_nodes, num_eval_pts, num_induced_pts))

    return AIC_vec

def compute_batched_aic_POD(mesh_dict, wake_mesh_dict, sigma, batch_size, ROM):
    num_tot_panels = len(mesh_dict['cell_adjacency'])
    UT, U = ROM[0], ROM[1]

    coll_point_eval = mesh_dict['panel_center_mod'] # (nn, num_tot_panels, 3)

    coll_point_mesh = mesh_dict['panel_center'] # (nn, num_tot_panels, 3)
    panel_corners = mesh_dict['panel_corners'] # (nn, num_tot_panels, 3, 3) 
    panel_x_dir = mesh_dict['panel_x_dir'] # (nn, num_tot_panels, 3)
    panel_y_dir = mesh_dict['panel_y_dir'] # (nn, num_tot_panels, 3)
    panel_normal = mesh_dict['panel_normal'] # (nn, num_tot_panels, 3)
    S_j = mesh_dict['S']
    SL_j = mesh_dict['SL']
    SM_j = mesh_dict['SM']

    coll_point_wake = wake_mesh_dict['panel_center'] # (nn, num_wake_panels, 3)
    panel_corners_wake = wake_mesh_dict['panel_corners'] # (nn, num_wake_panels, 3, 3) 
    panel_x_dir_wake = wake_mesh_dict['panel_x_dir'] # (nn, num_wake_panels, 3)
    panel_y_dir_wake = wake_mesh_dict['panel_y_dir'] # (nn, num_wake_panels, 3)
    panel_normal_wake = wake_mesh_dict['panel_normal'] # (nn, num_wake_panels, 3)
    S_j_wake = wake_mesh_dict['S']
    SL_j_wake = wake_mesh_dict['SL']
    SM_j_wake = wake_mesh_dict['SM']

    RHS_batched_func = csdl.experimental.batch_function(compute_aic_mat_vec, batch_size=batch_size, batch_dims=[1]+[None]*10)

    RHS = RHS_batched_func(
        coll_point_eval,
        coll_point_mesh,
        panel_corners,
        panel_x_dir,
        panel_y_dir,
        panel_normal,
        S_j,
        SL_j,
        SM_j,
        sigma,
        'source'
    ) * -1.
    RHS = RHS.reshape((num_tot_panels,1))
    RHS_red = csdl.matvec(UT, RHS)

    phi_T_AIC_batched_func = csdl.experimental.batch_function(compute_phi_T_aic_mat_mat, batch_size=batch_size, batch_dims=[None] + [1]*8 + [None, None])
    phi_T_AIC = phi_T_AIC_batched_func(
        coll_point_eval,
        coll_point_mesh,
        panel_corners,
        panel_x_dir,
        panel_y_dir,
        panel_normal,
        S_j,
        SL_j,
        SM_j,
        UT,
        'doublet'
    )
    # print(phi_T_AIC.shape)
    # exit()
    loop_vals = [np.arange(U.shape[1], dtype=int).tolist()]
    with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
        i = loop_builder.get_loop_indices()
        phi_T_AIC_ind = phi_T_AIC[:,i,:]
    
    phi_T_AIC_ind = loop_builder.add_stack(phi_T_AIC_ind)
    loop_builder.finalize()
    phi_T_AIC = phi_T_AIC_ind.reshape(UT.shape)

    # phi_T_AIC = csdl.einsum(phi_T_AIC, action='ijk->jik').reshape(UT.shape)
    AIC_red_surf = csdl.matmat(phi_T_AIC, U)

    upper_TE_cell_ind = mesh_dict['upper_TE_cells']
    lower_TE_cell_ind = mesh_dict['lower_TE_cells']
    nw = len(upper_TE_cell_ind)
    phi_T_AIC_wake_batched_func = csdl.experimental.batch_function(compute_phi_T_aic_mat_mat, batch_size=2, batch_dims=[None] + [1]*8 + [None, None])
    phi_T_AIC_wake = phi_T_AIC_wake_batched_func(
        coll_point_eval,
        coll_point_wake,
        panel_corners_wake,
        panel_x_dir_wake,
        panel_y_dir_wake,
        panel_normal_wake,
        S_j_wake,
        SL_j_wake,
        SM_j_wake,
        UT,
        'wake'
    )

    loop_vals = [np.arange(U.shape[1], dtype=int).tolist()]
    with csdl.experimental.enter_loop(vals=loop_vals) as loop_builder:
        i = loop_builder.get_loop_indices()
        phi_T_AIC_ind_wake = phi_T_AIC_wake[:,i,:]
    
    phi_T_AIC_ind_wake = loop_builder.add_stack(phi_T_AIC_ind_wake)
    loop_builder.finalize()
    phi_T_AIC_wake = phi_T_AIC_ind_wake.reshape(UT.shape[0], nw)

    # phi_T_AIC_wake = csdl.einsum(phi_T_AIC_wake, action='ijk->jik').reshape((UT.shape[0], nw))
    U_upper_TE_cells = U[upper_TE_cell_ind,:]
    U_lower_TE_cells = U[lower_TE_cell_ind,:]
    U_TE_diff = U_upper_TE_cells-U_lower_TE_cells

    AIC_red_wake = csdl.matmat(phi_T_AIC_wake, U_TE_diff)

    AIC_red = AIC_red_surf + AIC_red_wake

    return RHS_red, AIC_red


def compute_aic_mat_vec(coll_point, panel_center, panel_corners, panel_x_dir, panel_y_dir,
                        panel_normal, S_j, SL_j, SM_j, v, mode='doublet'):
    '''
    This function computes the matrix vector product Av, where A is the 
    AIC matrix for the doublets, and v represents the vector that converges
    to the doublet strengths

    The three modes are doublet, source, and wake (corresponding to which AIC matrix to compute)
    '''
    num_nodes = coll_point.shape[0]
    num_eval_pts = coll_point.shape[1]
    num_induced_pts = panel_center.shape[1]
    
    num_interactions = num_eval_pts*num_induced_pts
    num_corners = 3
    if mode == 'wake':
        num_corners = 4
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

    if mode == 'doublet' or mode == 'wake':
        AIC_vec = compute_doublet_influence_new(
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
    elif mode == 'source':
        AIC_vec = compute_source_influence_new(
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

    # A = AIC_vec.reshape((num_nodes, num_eval_pts, num_induced_pts))
    # Av = csdl.einsum(A,v,action='ijk,ik->ij')
    A_out = AIC_vec.reshape((num_eval_pts, num_induced_pts))
    # print(v.shape)
    # print(A_out.shape)

    # Av = csdl.einsum(A_out,v,action='jk,k->j')
    Av = csdl.matvec(A_out,v)

    return Av

def compute_phi_T_aic_mat_mat(coll_point, panel_center, panel_corners, panel_x_dir, panel_y_dir,
                        panel_normal, S_j, SL_j, SM_j, UT, mode='doublet'):
    '''
    This function computes the matrix vector product Av, where A is the 
    AIC matrix for the doublets, and v represents the vector that converges
    to the doublet strengths

    The three modes are doublet, source, and wake (corresponding to which AIC matrix to compute)
    '''
    num_nodes = coll_point.shape[0]
    num_eval_pts = coll_point.shape[1]
    num_induced_pts = panel_center.shape[1]
    
    num_interactions = num_eval_pts*num_induced_pts
    num_corners = 3
    if mode == 'wake':
        num_corners = 4
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

    if mode == 'doublet' or mode == 'wake':
        AIC_vec = compute_doublet_influence_new(
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
    elif mode == 'source':
        AIC_vec = compute_source_influence_new(
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

    # A = AIC_vec.reshape((num_nodes, num_eval_pts, num_induced_pts))
    # Av = csdl.einsum(A,v,action='ijk,ik->ij')
    A_out = AIC_vec.reshape((num_eval_pts, num_induced_pts))
    # print(UT.shape)
    # print(A_out.shape)
    # exit()
    # UT_A = csdl.einsum(UT, A_out, action='jk,ki->ji')
    UT_A = csdl.matmat(UT, A_out)
    # exit()
    # Av_no_num_nodes = Av[0,:]
    return UT_A