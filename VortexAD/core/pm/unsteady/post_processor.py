import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.pm.source_doublet.least_squares_velocity import least_squares_velocity, unstructured_least_squares_velocity

def steady_pressure_computation(mesh_dict, mu, sigma, num_nodes, constant_geometry=False, Cp_cutoff=-100.):
    
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])
    output_dict = {}

    qn = sigma
    
    # looping over cell types to compute induced velocities
    cells = mesh_dict['cell_point_indices']
    cell_types = list(cells.keys())
    cell_adjacency_types = mesh_dict['cell_adjacency']
    num_cells_per_type = [len(cell_adjacency_types[cell_type]) for cell_type in cell_types]
    num_cells = sum(num_cells_per_type)

    ql = csdl.Variable(value=np.zeros(qn.shape))
    qm = csdl.Variable(value=np.zeros(qn.shape))
    start, stop = 0, 0
    for i, cell_type in enumerate(cell_types):
        stop += num_cells_per_type[i]
        
        delta_coll_point = mesh_dict['delta_coll_point_' + cell_type]
        cell_adjacency = np.array(cell_adjacency_types[cell_type])
        ql_iter, qm_iter = unstructured_least_squares_velocity(mu, delta_coll_point, cell_adjacency, start, constant_geometry)
        # NOTE: we add "start" to this to signify the shift in panel indices with different types
        # this is only needed with mixed grids
        ql = ql.set(csdl.slice[:,start:stop], ql_iter)
        qm = qm.set(csdl.slice[:,start:stop], qm_iter)

        start += num_cells_per_type[i]

    panel_x_dir = mesh_dict['panel_x_dir']
    panel_y_dir = mesh_dict['panel_y_dir']
    panel_normal = mesh_dict['panel_normal']
    panel_area = mesh_dict['panel_area']
    panel_center = mesh_dict['panel_center']
    coll_vel = mesh_dict['coll_point_velocity']

    if constant_geometry:
        panel_x_dir = csdl.expand(panel_x_dir.reshape(panel_x_dir.shape[1:]), (num_nodes,) + panel_x_dir.shape[1:], 'ij->aij')
        panel_y_dir = csdl.expand(panel_y_dir.reshape(panel_y_dir.shape[1:]), (num_nodes,) + panel_y_dir.shape[1:], 'ij->aij')
        panel_normal = csdl.expand(panel_normal.reshape(panel_normal.shape[1:]), (num_nodes,) + panel_normal.shape[1:], 'ij->aij')
        panel_area = csdl.expand(panel_area.reshape(panel_area.shape[1:]), (num_nodes,) + panel_area.shape[1:], 'i->ai')
        panel_center = csdl.expand(panel_center.reshape(panel_center.shape[1:]), (num_nodes,) + panel_center.shape[1:], 'ij->aij')

    # free_stream_l = csdl.einsum(coll_vel, panel_x_dir, action='jkl,jkl->jk')
    # free_stream_m = csdl.einsum(coll_vel, panel_y_dir, action='jkl,jkl->jk')
    # free_stream_n = csdl.einsum(coll_vel, panel_normal, action='jkl,jkl->jk')

    free_stream_l = csdl.sum(coll_vel*panel_x_dir, axes=(2,))
    free_stream_m = csdl.sum(coll_vel*panel_y_dir, axes=(2,))
    free_stream_n = csdl.sum(coll_vel*panel_normal, axes=(2,))

    Ql = free_stream_l + ql
    Qm = free_stream_m + qm
    Qn = free_stream_n + qn
    Q_inf_norm = csdl.norm(coll_vel, axes=(2,))
    
    perturbed_vel_mag = (Ql**2 + Qm**2 + Qn**2)**0.5
    Cp_static = 1 - perturbed_vel_mag**2/Q_inf_norm**2
    # Cp_dynamic = -dmu_dt*2./Q_inf_norm**2
    Cp = Cp_static

    Cp_cutoff_exp = csdl.expand(Cp_cutoff, Cp.shape)
    Cp = csdl.maximum(Cp, Cp_cutoff_exp, rho=100)

    output_dict = {
        'Cp_static': Cp_static,
        'Q_inf_norm': Q_inf_norm,
        'ql': ql,
        'qm': qm,
        'qn': qn,
    }

    return output_dict

def unsteady_post_processor(mesh_dict, output_dict, mu, num_nodes, dt, nt, 
                            compressibility=False, rho=1.225, Cp_cutoff=-100., 
                            constant_geometry=False, ref_point=np.zeros(3), 
                            ref_area=10., ref_chord=1., sos=340.3):
    '''
    Compute unsteady pressure and loads
    '''
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])

    # panel_x_dir = mesh_dict['panel_x_dir']
    # panel_y_dir = mesh_dict['panel_y_dir']
    panel_normal = mesh_dict['panel_normal']
    panel_area = mesh_dict['panel_area']
    panel_center = mesh_dict['panel_center']
    coll_vel = mesh_dict['coll_point_velocity']

    ql = output_dict['ql']
    qm = output_dict['qm']
    qn = output_dict['qn']

    # free_stream_l = csdl.sum(coll_vel*panel_x_dir, axes=(2,))
    # free_stream_m = csdl.sum(coll_vel*panel_y_dir, axes=(2,))
    # free_stream_n = csdl.sum(coll_vel*panel_normal, axes=(2,))

    # Ql = free_stream_l + ql
    # Qm = free_stream_m + qm
    # Qn = free_stream_n + qn
    Q_inf_norm = csdl.norm(coll_vel, axes=(2,))
    # perturbed_vel_mag = (Ql**2 + Qm**2 + Qn**2)**0.5

    num_panels = mu.shape[-1]
    dmu_dt = csdl.Variable(value=np.zeros((nt, num_panels)))
    dmu_dt = dmu_dt.set(csdl.slice[1:,:], (mu[1:,:]-mu[:-1,:])/dt)

    Cp_dynamic = -dmu_dt*2./Q_inf_norm**2
    Cp_static = output_dict['Cp_static']
    Cp = Cp_static + Cp_dynamic
    Q_inf = csdl.average(Q_inf_norm, axes=(1,))
    print(Q_inf.shape)
    if compressibility:
        
        M_inf = Q_inf/sos
        M_inf = 0.7
        beta = (1-M_inf**2)**0.5
        if constant_geometry:
            beta  = csdl.expand(beta, (num_nodes, Cp.shape[1]),'i->ia')
        # sos = 340.3
        # M_inf = perturbed_vel_mag/sos
        # von Karman and Tsien correction --> better at M=0.7-0.8
        # denom = beta + (M_inf**2/(1+beta))*Cp/2
        # denom = beta + 1/2*Cp*(1-beta)
        # Cp = Cp/denom

        # PG compressibility correction
        Cp = Cp/beta

    if rho.shape[0] == num_nodes:
        rho_exp = csdl.expand(rho, panel_area.shape, 'i->ia')
    else:
        rho_exp = rho

    dF_no_normal = -0.5*rho_exp*Q_inf_norm**2*panel_area*Cp
    dF = csdl.expand(dF_no_normal, panel_normal.shape, 'jk->jka')*panel_normal
    total_force = csdl.sum(dF, axes=(1,))
    Fz_panel = csdl.tensordot(dF, z_dir_global, axes=([2],[0]))
    Fx_panel = csdl.tensordot(dF, x_dir_global, axes=([2],[0]))

    aoa = csdl.arctan(coll_vel[:,:,2]/coll_vel[:,:,0])
    cosa, sina = csdl.cos(aoa), csdl.sin(aoa)

    panel_L = Fz_panel*cosa - Fx_panel*sina
    panel_Di = Fz_panel*sina + Fx_panel*cosa

    L = csdl.sum(panel_L, axes=(1,))
    Di = csdl.sum(panel_Di, axes=(1,))

    CL = L/(0.5*rho*ref_area*Q_inf**2)
    CDi = Di/(0.5*rho*ref_area*Q_inf**2)

    # ref_point = csdl.Variable(value=np.array([0., 0., 0.,]))
    ref_pt_exp = csdl.expand(ref_point, dF.shape, 'i->abi')
    
    panel_moment_arm = panel_center-ref_pt_exp
    panel_moment = csdl.cross(panel_moment_arm, dF, axis=2)
    moment = csdl.sum(panel_moment, axes=(1,))
    Q_inf_exp = csdl.expand(Q_inf, moment.shape, 'i->ia')
    # if rho.shape[0] == num_nodes:
    #     rho_exp_CM = csdl.expand(rho, (num_nodes, 3), 'i->ia')
    # else:
    #     rho_exp_CM = rho
    rho_exp_CM = rho
    CM = moment/(0.5*rho_exp_CM*ref_area*Q_inf_exp**2*ref_chord)

    # scalar coefficients
    output_dict['CL'] = CL
    output_dict['CDi'] = CDi
    output_dict['CM'] = CM

    # scalar/vector forces
    output_dict['L'] = L
    output_dict['Di'] = Di
    output_dict['F'] = total_force
    output_dict['M'] = moment

    # force + pressure distributions
    output_dict['Cp'] = Cp
    output_dict['panel_forces'] = dF
    output_dict['L_panel'] = panel_L
    output_dict['Di_panel'] = panel_Di

    # flow field
    # output_dict['Qn'] = Qn
    # output_dict['Ql'] = Ql
    # output_dict['V_mag'] = perturbed_vel_mag
    output_dict['ql'] = ql
    output_dict['qm'] = qm
    output_dict['qn'] = qn

    return output_dict