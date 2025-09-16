import numpy as np 
import csdl_alpha as csdl

import time

from VortexAD.core.pm.source_doublet.least_squares_velocity import least_squares_velocity, unstructured_least_squares_velocity

def post_processor(mesh_dict, mu, sigma, num_nodes, rho=1.225, Cp_cutoff=-100.):
    surface_names = list(mesh_dict.keys())
    start, stop = 0, 0
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])

    output_dict = {}
    for i in range(len(surface_names)):
        surface_name = surface_names[i]
        surf_dict = {}

        num_panels = mesh_dict[surface_name]['num_panels']
        nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        stop += num_panels

        mu_grid = mu[:,start:stop].reshape((num_nodes, nc-1, ns-1))

        # perturbation velocities
        qn = sigma[:,start:stop].reshape((num_nodes, nc-1, ns-1)) # num_nodes, nt, num_panels for surface

        # region least squares method for perturbation velocities (derivatives)
        delta_coll_point = mesh_dict[surface_name]['delta_coll_point']
        ql, qm = least_squares_velocity(mu_grid, delta_coll_point)
        # ql, qm = least_squares_velocity_old(mu_grid, delta_coll_point)
        # endregion

        panel_x_dir = mesh_dict[surface_name]['panel_x_dir']
        panel_y_dir = mesh_dict[surface_name]['panel_y_dir']
        panel_normal = mesh_dict[surface_name]['panel_normal']
        nodal_cp_velocity = mesh_dict[surface_name]['nodal_cp_velocity']
        coll_vel = mesh_dict[surface_name]['coll_point_velocity']
        if coll_vel:
            total_vel = nodal_cp_velocity+coll_vel
        else:
            total_vel = nodal_cp_velocity

        free_stream_l = csdl.einsum(total_vel, panel_x_dir, action='jklm,jklm->jkl')
        free_stream_m = csdl.einsum(total_vel, panel_y_dir, action='jklm,jklm->jkl')
        free_stream_n = csdl.einsum(total_vel, panel_normal, action='jklm,jklm->jkl')
        # print(mu_grid[0,0,:,:].value)
        # print(ql[0,0,:,:].value)
        # exit()
        Ql = free_stream_l + ql
        Qm = free_stream_m + qm
        Qn = free_stream_n + qn
        Q_inf_norm = csdl.norm(total_vel, axes=(3,))

        body_vel = csdl.Variable(shape=Ql.shape + (3,), value=0.)
        body_vel = body_vel.set(csdl.slice[:,:,:,0], value=Ql)
        body_vel = body_vel.set(csdl.slice[:,:,:,1], value=Qm)
        body_vel = body_vel.set(csdl.slice[:,:,:,2], value=Qn)
        body_vel_norm = csdl.norm(body_vel, axes=(3,))

        rot_mat = mesh_dict[surface_name]['rot_mat']
        body_vel_global = csdl.einsum(body_vel, rot_mat, action='jklm,jklmn->jkln')


        perturbed_vel_mag = (Ql**2 + Qm**2 + Qn**2)**0.5 
        Cp_static = 1 - (Ql**2 + Qm**2 + Qn**2)/Q_inf_norm**2
        Cp = Cp_static
        
        panel_area = mesh_dict[surface_name]['panel_area']

        dP = -0.5*rho*Q_inf_norm**2*Cp
        dF_no_normal = dP*panel_area
        dF = csdl.expand(dF_no_normal, panel_normal.shape, 'jkl->jkla') * panel_normal

        Fz_panel = csdl.tensordot(dF, z_dir_global, axes=([3],[0]))
        Fx_panel = csdl.tensordot(dF, x_dir_global, axes=([3],[0]))

        nc_panels = int(num_panels/(ns-1))

        LE_velocity = (total_vel[:,int((nc_panels/2)-1),:,:] + total_vel[:,int(nc_panels/2),:,:])/2.
        aoa = csdl.arctan(LE_velocity[:,:,2]/LE_velocity[:,:,0])

        aoa_exp = csdl.expand(aoa, Fz_panel.shape, 'jk->jak')

        cosa, sina = csdl.cos(aoa_exp), csdl.sin(aoa_exp)

        panel_L = Fz_panel*cosa - Fx_panel*sina
        panel_Di = Fz_panel*sina + Fx_panel*cosa

        L = csdl.sum(panel_L, axes=(1,2))
        Di = csdl.sum(panel_Di, axes=(1,2))

        Q_inf = csdl.norm(csdl.average(LE_velocity, axes=(1,)), axes=(1,))

        planform_area = mesh_dict[surface_name]['planform_area']
        CL = L/(0.5*rho*planform_area*Q_inf**2)
        CDi = Di/(0.5*rho*planform_area*Q_inf**2)

        surf_dict['Cp'] = Cp
        surf_dict['CL'] = CL
        surf_dict['CDi'] = CDi
        surf_dict['Fx_panel'] = Fx_panel
        surf_dict['Fz_panel'] = Fz_panel
        surf_dict['panel_forces'] = dF
        surf_dict['L'] = L
        
        surf_dict['body_vel'] = body_vel_norm

        surf_dict['panel_pressure'] = dP
        surf_dict['surface_vel'] = body_vel_global

        start += num_panels
        
        output_dict[surface_name] = surf_dict

    return output_dict


def unstructured_post_processor(mesh_dict, mu, sigma, num_nodes, compressibility=False, 
                                rho=1.225, Cp_cutoff=-100., constant_geometry=False, 
                                ref_point=np.zeros(3), ref_area=10., ref_chord=1.,
                                sos=340.3):
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])
    output_dict = {}

    qn = sigma
    delta_coll_point = mesh_dict['delta_coll_point']
    cell_adjacency = mesh_dict['cell_adjacency']

    ql, qm = unstructured_least_squares_velocity(mu, delta_coll_point, cell_adjacency, constant_geometry)

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


    Cp_cutoff_exp = csdl.expand(Cp_cutoff, Cp.shape)
    Cp = csdl.maximum(Cp, Cp_cutoff_exp, rho=100)

    if rho.shape[0] == num_nodes:
        rho_exp = csdl.expand(rho, panel_area.shape, 'i->ia')
    else:
        rho_exp = rho

    dF_no_normal = -0.5*rho_exp*Q_inf_norm**2*panel_area*Cp
    dF = csdl.expand(dF_no_normal, panel_normal.shape, 'jk->jka')*panel_normal
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
    if rho.shape[0] == num_nodes:
        rho_exp_CM = csdl.expand(rho, (num_nodes, 3), 'i->ia')
    else:
        rho_exp_CM = rho
    CM = moment/(0.5*rho_exp_CM*ref_area*Q_inf_exp**2*ref_chord)

    output_dict['CL'] = CL
    output_dict['CDi'] = CDi
    output_dict['CM'] = CM
    output_dict['Cp'] = Cp
    output_dict['panel_forces'] = dF
    output_dict['M'] = moment
    output_dict['Qn'] = Qn
    # output_dict['Ql'] = Ql
    output_dict['L'] = L
    output_dict['Di'] = Di
    output_dict['V_mag'] = perturbed_vel_mag
    output_dict['L_panel'] = panel_L
    output_dict['Di_panel'] = panel_Di

    return output_dict