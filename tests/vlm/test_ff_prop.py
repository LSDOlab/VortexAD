import numpy as np
import csdl_alpha as csdl
import time

from VortexAD import VortexLatticeMethod
from VortexAD.utils.meshing.gen_prop_mesh import gen_prop_mesh

import matplotlib.pyplot as plt

V_inf = 10.
V_inf = 185.
nt, dt = 100, 0.001
RPM = 850.
RPM2omega = (2*np.pi) / 60.
omega = RPM * RPM2omega

radius = 2.
chord = 0.2
twist = 45.
num_blades = 2
nr = 5

prop_meshes = gen_prop_mesh(
    radius, 
    chord, 
    twist, 
    num_blades, 
    num_radial=nr, 
    direction='forward',
    plot=False
)
# exit()
pms = prop_meshes.shape[1:]

actuated_prop_meshes = np.zeros((num_blades, nt) + pms)
prop_nodal_velocity = np.zeros((num_blades, nt) + pms)
collocation_velocity = np.zeros((num_blades, nt, pms[0]-1, pms[1]-1, 3))
time_vec = np.linspace(0, nt*dt, nt)
omega_vector = -omega*np.array([1., 0., 0.])
# omega_vector = omega*np.array([0., 0., 1.])
for i in range(nt):
    dtheta = time_vec[i] * omega

    # rotated meshes
    rot_mat = np.zeros((3,3))
    rot_mat[0,0] = 1
    rot_mat[1,1] = rot_mat[2,2] = np.cos(dtheta)
    rot_mat[1,2] = np.sin(dtheta)
    rot_mat[2,1] = -np.sin(dtheta)

    # rot_mat = np.zeros((3,3))
    # rot_mat[2,2] = 1
    # rot_mat[0,0] = rot_mat[1,1] = np.cos(dtheta)
    # rot_mat[0,1] = -np.sin(dtheta)
    # rot_mat[1,0] = np.sin(dtheta)

    asdf = np.einsum('ij,abcj->abci', rot_mat, prop_meshes)

    actuated_prop_meshes[:,i,:] = asdf

    collocation_points = (asdf[:,:-1,:-1,:]+asdf[:,1:,:-1,:]+asdf[:,1:,1:,:]+asdf[:,:-1,1:,:])/4

    ref_point = np.array([0., 0., 0.])
    vel_arm_collocation = collocation_points - ref_point

    coll_vel_t = np.cross(omega_vector, vel_arm_collocation)
    
    collocation_velocity[:,i,:] = coll_vel_t

    vel_arm = asdf - ref_point
    nodal_vel_t = np.cross(omega_vector, vel_arm)
    # prop_nodal_velocity[:,i,:] = nodal_vel_t


prop_nodal_velocity[:,:,:,:,0] = -V_inf


# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

mesh_list = [csdl.Variable(value=actuated_prop_meshes[i,:]) for i in range(num_blades)]
mesh_vel_list = [csdl.Variable(value=prop_nodal_velocity[i,:]) for i in range(num_blades)]
coll_vel_list = [csdl.Variable(value=collocation_velocity[i,:]) for i in range(num_blades)]

pitch = csdl.Variable(value=np.array([0.]))

input_dict = {
    # 'V_inf': 10.,
    # 'alpha': pitch,
    'V_inf': mesh_vel_list,
    'collocation_velocity': coll_vel_list,
    'solver_mode': 'unsteady',
    'nt': nt,
    'dt': dt,

    'partition_size': 1,

    'free_wake': True,
    'meshes': mesh_list,
    'core_radius': 1.e-6,
}


vlm = VortexLatticeMethod(
    input_dict
)

vlm_outputs = ['x_w', 'gamma', 'gamma_w']
# vlm_outputs.append('dxw_dt')
vlm_outputs.append('panel_force')
vlm_outputs.extend(['AIC', 'AIC_w', 'RHS', 'BC', 'wake_influence'])
vlm_outputs.extend(['panel_centers', 'panel_normal', 'wake_corners'])
vlm.declare_outputs(vlm_outputs)
output_dict = vlm.evaluate()

x_w = output_dict['x_w']
gamma = output_dict['gamma']
gamma_w = output_dict['gamma_w']
# dxw_dt = output_dict['dxw_dt']
panel_force = output_dict['panel_force']

panel_center = output_dict['panel_centers']
panel_normal = output_dict['panel_normal']
wake_corners = output_dict['wake_corners']

AIC = output_dict['AIC']
AIC_w = output_dict['AIC_w']
RHS = output_dict['RHS']
BC = output_dict['BC']
wake_influence = output_dict['wake_influence']

inputs = [pitch]
inputs.extend(mesh_list)
outputs = [x_w, gamma, gamma_w]
# outputs.append(dxw_dt)
outputs.append(panel_force)
outputs.extend([AIC, AIC_w, RHS, BC, wake_influence])
outputs.extend([panel_normal, panel_center, wake_corners])

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs,
    gpu=False
)
start = time.time()
sim.run()
end = time.time()

print(f'run + compile time: {end-start} seconds')

x_w_val = sim[x_w]
gamma_val = sim[gamma]
gamma_w_val = sim[gamma_w]

mesh_val_list = [sim[val] for val in mesh_list]

wake_form  = 'lines'

# iso
iso_cam = dict(
    position=(-27.2696, -16.3214, 8.61178),
    focal_point=(1.27338, -0.446573, 2.47714),
    viewup=(0.0807063, 0.229700, 0.969909),
    roll=74.8432,
    distance=33.2317,
    clipping_range=(21.2845, 48.3509),
)

vlm.plot_unsteady(
    mesh_val_list,
    x_w_val,
    gamma_val,
    gamma_w_val,
    wake_form=wake_form,
    interactive=False,
    camera=iso_cam,
    name='prop_ani_iso' + f'_{wake_form}'
)

# front
front_cam = dict(
    position=(-31.8757, 0.0928217, 0.197711),
    focal_point=(1.27338, -0.446571, 2.47714),
    viewup=(-0.0685583, 2.62192e-3, 0.997644),
    roll=89.8498,
    distance=33.2317,
    clipping_range=(28.6933, 38.9886),
)

vlm.plot_unsteady(
    mesh_val_list,
    x_w_val,
    gamma_val,
    gamma_w_val,
    wake_form=wake_form,
    interactive=False,
    camera=front_cam,
    name='prop_ani_front' + f'_{wake_form}'
)


exit()

def AIC_wake_function(panel_center_val, panel_normal_val, wake_corners_val, vc=1.e-6):
    
    from VortexAD.core.elements.vortex_ring import compute_vortex_line_ind_vel
    rec = csdl.Recorder(inline=True)
    rec.start()

    pcv = csdl.Variable(value=panel_center_val)
    pnv = csdl.Variable(value=panel_normal_val)
    wcv = csdl.Variable(value=wake_corners_val)

    num_eval = pcv.shape[0]
    num_induced = wcv.shape[0]
    num_interactions = num_eval*num_induced

    expanded_shape = (1, num_eval, num_induced, 4, 3)
    vectorized_shape = (1, num_interactions, 4, 3)

    pcve = pcv.expand(expanded_shape, 'ij->aibcj')
    pcvev = pcve.reshape(vectorized_shape)

    wcve = wcv.expand(expanded_shape, 'ijk->abijk')
    wcvev = wcve.reshape(vectorized_shape)

    num_edges = 4

    AIC_vel_vec_list = []
    for  i in range(num_edges-1):
        asdf = compute_vortex_line_ind_vel(
            wcvev[:,:,i], 
            wcvev[:,:,i+1], 
            pcvev[:,:,0], 
            mode='wake', 
            vc=vc
        )
        AIC_vel_vec_list.append(asdf)
    asdf = compute_vortex_line_ind_vel(
        wcvev[:,:,-1], 
        wcvev[:,:,0], 
        pcvev[:,:,0], 
        mode='wake', 
        vc=vc
    )
    AIC_vel_vec_list.append(asdf)
    AIC_vel_vec = sum(AIC_vel_vec_list)

    expanded_shape_proj = (1, num_eval, num_induced, 3)
    vectorized_shape_proj = (1, num_interactions, 3)

    pnve = pnv.expand(expanded_shape_proj, 'ij->aibj')
    pnvev = pnve.reshape(vectorized_shape_proj)

    AIC_vec = csdl.sum(pnvev*AIC_vel_vec, axes=(2,))
    
    return AIC_vec.reshape((num_eval, num_induced))

time_ind = 1

wake_mesh = x_w_val[time_ind,:,:].reshape((2*nt, nr, 3))
panel_center_val = sim[panel_center][time_ind,:]
panel_normal_val = sim[panel_normal][time_ind,:]
wake_corners_val = sim[wake_corners][time_ind,:]

asdf = AIC_wake_function(panel_center_val[0,:].reshape((1,3)), panel_normal_val[0,:].reshape((1,3)), wake_corners_val)