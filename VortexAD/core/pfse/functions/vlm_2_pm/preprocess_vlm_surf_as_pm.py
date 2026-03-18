import numpy as np
import csdl_alpha as csdl

def preprocess_vlm_surf_as_pm(mesh_dict):
    surface_names = list(mesh_dict.keys())
    for i, surf_name in enumerate(surface_names):

        surf_mesh = mesh_dict[surf_name]['mesh'] # MESH OF SURFACE
        mesh_shape = surf_mesh.shape # (nn, nc, ns, 3)
        nc, ns = mesh_shape[1], mesh_shape[2]

        mesh_dict[surf_name]['num_panels'] = (nc-1)*(ns-1)
        mesh_dict[surf_name]['nc'] = nc
        mesh_dict[surf_name]['ns'] = ns

        # bound vortex --> for simplicity we will refer to this as mesh
        mesh = csdl.Variable(shape=mesh.shape, value=0.)
        mesh = mesh.set(csdl.slice[:,:-1,:,:], value=(3*surf_mesh[:,:-1,:,:] + surf_mesh[:,1:,:,:])/4)
        mesh = mesh.set(csdl.slice[:,-1,:,:], value=surf_mesh[:,-1,:,:] + (surf_mesh[:,-1,:,:] - surf_mesh[:,-2,:,:])/4)

        mesh_dict[surf_name]['bound_vortex_mesh'] = mesh

        R1 = mesh[:,:-1,:-1,:]
        R2 = mesh[:,1:,:-1,:]
        R3 = mesh[:,1:,1:,:]
        R4 = mesh[:,:-1,1:,:]
    
        S1 = (R1+R2)/2.
        S2 = (R2+R3)/2.
        S3 = (R3+R4)/2.
        S4 = (R4+R1)/2.

        Rc = (R1+R2+R3+R4)/4.
        mesh_dict[surf_name]['panel_center'] = Rc

        panel_corners = csdl.Variable(shape=(Rc.shape[:-1] + (4,3)), value=0.)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,0,:], value=R1)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,1,:], value=R2)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,2,:], value=R3)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,3,:], value=R4)
        mesh_dict[surf_name]['panel_corners'] = panel_corners

        D1 = R3-R1
        D2 = R4-R2

        D1D2_cross = csdl.cross(D1, D2, axis=3)
        D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(3,))
        panel_area = D1D2_cross_norm/2.
        mesh_dict[surf_name]['panel_area'] = panel_area

        normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'jkl->jkla')
        mesh_dict[surf_name]['panel_normal'] = normal_vec

        m_dir = S3 - Rc
        m_norm = csdl.norm(m_dir, axes=(3,))
        m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'jkl->jkla')
        l_vec = csdl.cross(m_vec, normal_vec, axis=3)
        # this also tells us that normal_vec = cross(l_vec, m_vec)

        panel_center_mod = Rc 

        mesh_dict[surf_name]['panel_center_mod'] = panel_center_mod

        mesh_dict[surf_name]['panel_x_dir'] = l_vec
        mesh_dict[surf_name]['panel_y_dir'] = m_vec

        rot_mat = csdl.Variable(value=np.zeros(normal_vec.shape + (3,))) # taken from dissertation of Pranav Prashant Ladkat, Pg. 26 eq. 4.5 
        rot_mat = rot_mat.set(csdl.slice[:,:,:,:,0], value=l_vec)
        rot_mat = rot_mat.set(csdl.slice[:,:,:,:,1], value=m_vec)
        rot_mat = rot_mat.set(csdl.slice[:,:,:,:,2], value=normal_vec)
        mesh_dict[surf_name]['rot_mat'] = rot_mat # rotation matrix transforms panel coordinates to global coordinates

        SMP = csdl.norm((S2)/2 - Rc, axes=(3,))
        SMQ = csdl.norm((S3)/2 - Rc, axes=(3,)) # same as m_norm

        mesh_dict[surf_name]['SMP'] = SMP
        mesh_dict[surf_name]['SMQ'] = SMQ

        s = csdl.Variable(shape=panel_corners.shape, value=0.)
        s = s.set(csdl.slice[:,:,:,:-1,:], value=panel_corners[:,:,:,1:,:] - panel_corners[:,:,:,:-1,:])
        s = s.set(csdl.slice[:,:,:,-1,:], value=panel_corners[:,:,:,0,:] - panel_corners[:,:,:,-1,:])

        l_exp = csdl.expand(l_vec, panel_corners.shape, 'jklm->jklam')
        m_exp = csdl.expand(m_vec, panel_corners.shape, 'jklm->jklam')
        
        S = csdl.norm(s, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0 --> added to the equations instead
        # S = csdl.norm(s, axes=(5,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
        SL = csdl.sum(s*l_exp, axes=(4,))
        SM = csdl.sum(s*m_exp, axes=(4,))

        mesh_dict[surf_name]['S'] = S
        mesh_dict[surf_name]['SL'] = SL
        mesh_dict[surf_name]['SM'] = SM

        delta_coll_point = csdl.Variable(Rc.shape[:-1] + (4,2), value=0.)
        delta_coll_point = delta_coll_point.set(csdl.slice[:,1:,:,0,0], value=csdl.sum((Rc[:,:-1,:,:]-Rc[:,1:,:,:])*l_vec[:,1:,:,:], axes=(3,)))
        delta_coll_point = delta_coll_point.set(csdl.slice[:,1:,:,0,1], value=csdl.sum((Rc[:,:-1,:,:]-Rc[:,1:,:,:])*m_vec[:,1:,:,:], axes=(3,)))
        delta_coll_point = delta_coll_point.set(csdl.slice[:,:-1,:,1,0], value=csdl.sum((Rc[:,1:,:,:]-Rc[:,:-1,:,:])*l_vec[:,:-1,:,:], axes=(3,)))
        delta_coll_point = delta_coll_point.set(csdl.slice[:,:-1,:,1,1], value=csdl.sum((Rc[:,1:,:,:]-Rc[:,:-1,:,:])*m_vec[:,:-1,:,:], axes=(3,)))
        delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,2,0], value=csdl.sum((Rc[:,:,:-1,:]-Rc[:,:,1:,:])*l_vec[:,:,1:,:], axes=(3,)))
        delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,2,1], value=csdl.sum((Rc[:,:,:-1,:]-Rc[:,:,1:,:])*m_vec[:,:,1:,:], axes=(3,)))
        delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,3,0], value=csdl.sum((Rc[:,:,1:,:]-Rc[:,:,:-1,:])*l_vec[:,:,:-1,:], axes=(3,)))
        delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,3,1], value=csdl.sum((Rc[:,:,1:,:]-Rc[:,:,:-1,:])*m_vec[:,:,:-1,:], axes=(3,)))

        # # setting deltas for panels wrapping around TE to zero
        # delta_coll_point = delta_coll_point.set(csdl.slice[:,0,:,:,:], value=0.)

        mesh_dict[surf_name]['delta_coll_point'] = delta_coll_point

        nodal_vel = mesh_dict[surf_name]['nodal_velocity']
        # mesh_dict[surf_name]['nodal_cp_velocity'] = (
        #     nodal_vel[:,:-1,:-1,:]+nodal_vel[:,:-1,1:,:]+\
        #     nodal_vel[:,1:,1:,:]+nodal_vel[:,1:,:-1,:]) / 4.
        coll_point_velocity = (
            nodal_vel[:,:-1,:-1,:]+nodal_vel[:,:-1,1:,:]+\
            nodal_vel[:,1:,1:,:]+nodal_vel[:,1:,:-1,:]) / 4.
        collocation_velocity = mesh_dict['collocation_velocity'] # prescribed velocity @ collocation
        
        if collocation_velocity is None:
            mesh_dict[surf_name]['coll_point_velocity'] = coll_point_velocity
        else:
            mesh_dict[surf_name]['coll_point_velocity'] = coll_point_velocity + collocation_velocity

        # computing planform area
        panel_width_spanwise = csdl.norm((mesh[:,:,1:,:] - mesh[:,:,:-1,:]), axes=(3,))
        avg_panel_width_spanwise = csdl.average(panel_width_spanwise, axes=(1,)) # num_nodes, ns - 1
        surface_TE = (mesh[:,-1,:-1,:] + mesh[:,0,:-1,:] + mesh[:,-1,1:,:] + mesh[:,0,1:,:])/4
        surface_LE = (mesh[:,int((nc-1)/2),:-1,:] + mesh[:,int((nc-1)/2),1:,:])/2 # num_nodes, ns - 1, 3

        chord_spanwise = csdl.norm(surface_TE - surface_LE, axes=(2,)) # num_nodes,  ns - 1

        planform_area = csdl.sum(chord_spanwise*avg_panel_width_spanwise, axes=(1,))
        mesh_dict[surf_name]['planform_area'] = planform_area