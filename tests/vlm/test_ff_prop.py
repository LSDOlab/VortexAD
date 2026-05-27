import numpy as np
import csdl_alpha as csdl
import time

from VortexAD import VortexLatticeMethod
from VortexAD.utils.meshing.gen_prop_mesh import gen_prop_mesh

import matplotlib.pyplot as plt

inch_2_m = 1/12*0.3048 # divide by 12 to convert to feet, then x 0.3048 to convert to meters

rotor_type = 'Q400'

if rotor_type == 'Ingraham':
    # parameters from Ingraham et al.
    nb = 3 # number of blades
    D_inch = 24
    c_inch = 1.5
    pitch_inch = 16

    RPM = 7199.
    J = 0.523 # advance ratio; can vary between 0.4 and 0.6

    RPM = 4644.368
    J = 0.3411
    # # CT of 0.0778

    # RPM = 7516.7586
    # J = 0.21096
    # CT of 0.11267

    RPS = RPM / 60.
    omega = RPS*2*np.pi
    V_inf = J*RPS*D_inch*inch_2_m

    # twist dist function from Ingraham et al.
    def twist_dist_func(pitch, D, nondim_r):
        twist_dist = np.arctan(pitch/(np.pi*D*nondim_r))*180/np.pi
        return twist_dist

    nt, dt = 50, 0.0002

    revs = 4
    nt_per_rev = 25
    nt = nt_per_rev*revs
    total_t = revs/RPS
    dt = total_t/nt
    
    nr = 20
    nc = 6
    nondim_r = np.linspace(0.2,1,nr)
    twist_dist = twist_dist_func(pitch_inch, D_inch, nondim_r)
    # twist_dist = 60.

    # convert to metric SI
    radius = D_inch/2*inch_2_m
    chord = c_inch*inch_2_m

    ref_chord = chord

elif rotor_type == 'Q400':
    nb = 6
    D = 4.1
    radius = D/2
    J_scaler = 1
    V_inf = 140/J_scaler # 140 to 170
    RPM = 850.
    RPS = RPM / 60.
    omega = RPS*2*np.pi
    J = V_inf/(RPS*D)

    collective = 30.

    def twist_dist_func(V_inf, omega, radius, nondim_r):
        r = radius*nondim_r
        a = 0.25
        a_prime = 0.04
        alpha_design = 4.
        theta_offset = -18.
        twist_dist = np.arctan(V_inf*(1+a)/(omega*r*(1-a_prime)))*180/np.pi + \
                        alpha_design + theta_offset

        theta_0 = 21
        twist_dist = theta_0 + 180/np.pi*np.arctan(V_inf/(omega*r)) - \
                        180/np.pi*np.arctan(V_inf/(omega*radius*0.75))

        return twist_dist
    
    def chord_dist_func(radius, nondim_r):
        chord_dist = radius*(0.17-0.12*nondim_r+0.03*nondim_r**2)
        
        A = 0.42
        B = 3.2
        C = 0.035
        chord_dist = radius*(A*nondim_r*np.exp(-B*nondim_r)+C)
        return chord_dist
    
    t_per_rev = 1/RPS

    nt, dt = 50, 0.001
    nr = 11
    nc = 4
    nondim_r = np.linspace(0.2,1,nr)
    twist_dist = twist_dist_func(V_inf*J_scaler, omega, radius, nondim_r)
    # twist_dist = twist_dist_func(V_inf*J_scaler, omega, radius, nondim_r) - 21
    # twist_dist = 90.
    chord = chord_dist_func(radius, nondim_r)
    twist_dist = twist_dist + collective

    ref_chord = np.max(chord)

print(f'Advance ratio: {J}')
print(f'Twist distribution: {twist_dist}')
prop_meshes = gen_prop_mesh(
    radius=radius, 
    chord=chord, 
    # twist=twist, 
    twist=twist_dist, 
    num_blades=nb, 
    num_radial=nr, 
    nc=nc,
    direction='forward',
    plot=True
)
# exit()
pms = prop_meshes.shape[1:]

actuated_prop_meshes = np.zeros((nb, nt) + pms)
prop_nodal_velocity = np.zeros((nb, nt) + pms)
collocation_velocity = np.zeros((nb, nt, pms[0]-1, pms[1]-1, 3))


actuated_prop_bv_meshes = np.zeros_like(actuated_prop_meshes)
bvm_coll_vel = np.zeros_like(collocation_velocity)
bound_vec_coll_vel = np.zeros_like(collocation_velocity)


time_vec = np.linspace(0, nt*dt, nt)
omega_vector = -omega*np.array([1., 0., 0.])
# omega_vector = omega*np.array([1., 0., 0.])
for i in range(nt):
    dtheta = time_vec[i] * omega

    # rotated meshes
    rot_mat = np.zeros((3,3))
    rot_mat[0,0] = 1
    rot_mat[1,1] = rot_mat[2,2] = np.cos(dtheta)
    rot_mat[1,2] = np.sin(dtheta)
    rot_mat[2,1] = -np.sin(dtheta)
    # rot_mat[1,2] = -np.sin(dtheta)
    # rot_mat[2,1] = np.sin(dtheta)

    asdf = np.einsum('ij,abcj->abci', rot_mat, prop_meshes)

    actuated_prop_meshes[:,i,:] = asdf # asdf has shape (nb, nc, ns, 3)

    bvm = np.zeros_like(asdf)
    bvm[:,:-1,:,:] = 0.75*asdf[:,:-1,:,:] + 0.25*asdf[:,1:,:,:]
    bvm[:,-1,:,:] = -0.25*asdf[:,-2,:,:] + 1.25*asdf[:,-1,:,:]
    actuated_prop_bv_meshes[:,i,:] = bvm

    collocation_points = (asdf[:,:-1,:-1,:]+asdf[:,1:,:-1,:]+asdf[:,1:,1:,:]+asdf[:,:-1,1:,:])/4
    bvm_collocation_points = (bvm[:,:-1,:-1,:]+bvm[:,1:,:-1,:]+bvm[:,1:,1:,:]+bvm[:,:-1,1:,:])/4
    bound_vec_center = (bvm[:,:-1,:-1,:] + bvm[:,:-1,1:])/2


    ref_point = np.array([0., 0., 0.])
    vel_arm_collocation = collocation_points - ref_point
    vel_arm_bvm_collocation = bvm_collocation_points - ref_point
    vel_arm_bound_vec_center = bound_vec_center - ref_point

    coll_vel_t = np.cross(omega_vector, vel_arm_collocation)
    bvm_coll_vel_t = np.cross(omega_vector, vel_arm_bvm_collocation)
    bound_vec_coll_vel_t = np.cross(omega_vector, vel_arm_bound_vec_center)
    
    
    collocation_velocity[:,i,:] = coll_vel_t
    bvm_coll_vel[:,i,:] = bvm_coll_vel_t
    bound_vec_coll_vel[:,i,:] = bound_vec_coll_vel_t

    vel_arm = asdf - ref_point
    nodal_vel_t = np.cross(omega_vector, vel_arm)
    # prop_nodal_velocity[:,i,:] = nodal_vel_t


prop_nodal_velocity[:,:,:,:,0] = -V_inf

# exit()
# instantiate recorder to assemble the graph
recorder = csdl.Recorder(inline=False)
recorder.start()

mesh_list = [csdl.Variable(value=actuated_prop_meshes[i,:]) for i in range(nb)]
mesh_vel_list = [csdl.Variable(value=prop_nodal_velocity[i,:]) for i in range(nb)]
# coll_vel_list = [csdl.Variable(value=collocation_velocity[i,:]) for i in range(nb)]
coll_vel_list = [csdl.Variable(value=bvm_coll_vel[i,:]) for i in range(nb)]

pitch = csdl.Variable(value=np.array([0.]))
rho = 1.17573
input_dict = {
    # 'V_inf': csdl.Variable(value=V_inf),
    # 'alpha': pitch,
    'V_inf': mesh_vel_list,
    'collocation_velocity': coll_vel_list,
    'solver_mode': 'unsteady',
    'nt': nt,
    'dt': dt,
    'rho': rho,
    'sos': 344.58,
    'compressibility': False,

    'partition_size': 1,

    'free_wake': True,
    'meshes': mesh_list,
    'core_radius': ref_chord*1e-6,
    'dissipation': True,
    # 'core_radius': 1.e-6,
}


vlm = VortexLatticeMethod(
    input_dict
)

vlm_outputs = ['x_w', 'gamma', 'gamma_w']
# vlm_outputs.append('dxw_dt')
vlm_outputs.append('panel_force')
vlm_outputs.extend(['AIC', 'AIC_w', 'RHS', 'BC', 'wake_influence'])
vlm_outputs.extend(['panel_centers', 'panel_normal', 'wake_corners'])
# vlm_outputs.append(['panel_force'])
vlm_outputs.append('total_CL')
vlm_outputs.append('steady_panel_force')
vlm_outputs.extend([
    'bound_vec_velocity',
    'body_ind_vel',
    'wake_ind_vel',
    'self_ind_vel',
])
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

CL = output_dict['total_CL']
steady_panel_force = output_dict['steady_panel_force']

bound_vec_velocity = output_dict['bound_vec_velocity']
body_ind_vel = output_dict['body_ind_vel']
wake_ind_vel = output_dict['wake_ind_vel']
self_ind_vel = output_dict['self_ind_vel']

inputs = [pitch]
inputs.extend(mesh_list)
outputs = [x_w, gamma, gamma_w]
# outputs.append(dxw_dt)
outputs.append(panel_force)
outputs.extend([AIC, AIC_w, RHS, BC, wake_influence])
outputs.extend([panel_normal, panel_center, wake_corners])
outputs.append(CL)
outputs.append(steady_panel_force)
outputs.extend([
    bound_vec_velocity,
    body_ind_vel,
    wake_ind_vel,
    self_ind_vel
])

sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs,
    gpu=True
)
# exit()
start = time.time()
sim.run()
end = time.time()

print(f'run + compile time: {end-start} seconds')

# start = time.time()
# sim.run()
# end = time.time()

# print(f'run time: {end-start} seconds')

x_w_val = sim[x_w]
gamma_val = sim[gamma]
gamma_w_val = sim[gamma_w]

mesh_val_list = [sim[val] for val in mesh_list]

panel_forces = sim[panel_force]
steady_panel_forces = sim[steady_panel_force]
thrust = np.sum(panel_forces[:,:,0], axis=1)*-1.
rev_per_second = RPM/60.
CT = thrust/(rho*(radius*2)**4*rev_per_second**2)

nondim_rev_time = time_vec*rev_per_second
plt.figure(figsize=(7,5))
plt.plot(nondim_rev_time, CT)
plt.grid()
plt.xlabel('Revolutions', fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel(r'$C_T$', fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f'{rotor_type}_prop_CT_vs_rev.pdf')
plt.show()
# exit()
wake_form  = 'grid'
# bounds for nr=5
# bounds = [-72.65183418851787, -47.69002929478038]

# bounds for nr=11
bounds = [-108.98585845736565, -53.929196894121624]


# iso
if rotor_type == 'Q400':
    iso_cam = dict(
    position=(-27.2696, -16.3214, 8.61178),
    focal_point=(1.27338, -0.446573, 2.47714),
    viewup=(0.0807063, 0.229700, 0.969909),
    roll=74.8432,
    distance=33.2317,
    clipping_range=(21.2845, 48.3509),
)
elif rotor_type == 'Ingraham':
    iso_cam = dict(
    pos=(-3.77961, -2.20647, 1.17118),
    focal_point=(0.463121, 0.153225, 0.259305),
    viewup=(0.0807065, 0.229700, 0.969910),
    roll=74.8432,
    distance=4.93968,
    clipping_range=(2.87531, 5.75842),
)

vlm.plot_unsteady(
    mesh_val_list,
    x_w_val,
    gamma_val,
    gamma_w_val,
    # bounds=bounds, 
    wake_form=wake_form,
    interactive=False,
    camera=iso_cam,
    name=f'{rotor_type}_prop_ani_iso_nr_{nr}_nt_{nt}' + f'_{wake_form}',
    fps=10
)

# front
if rotor_type == 'Q400':
    front_cam = dict(
        position=(-31.8757, 0.0928217, 0.197711),
        focal_point=(1.27338, -0.446571, 2.47714),
        viewup=(-0.0685583, 2.62192e-3, 0.997644),
        roll=89.8498,
        distance=33.2317,
        clipping_range=(28.6933, 38.9886),
    )
elif rotor_type == 'Ingraham':
    front_cam = dict(
        pos=(-2.63560, 5.74572e-3, -0.0860976),
        focal_point=(1.43663, -0.0605164, 0.193921),
        viewup=(-0.0685583, 2.62114e-3, 0.997644),
        roll=89.8498,
        distance=4.08238,
        clipping_range=(2.47157, 2.85754),
    )

# vlm.plot_unsteady(
#     mesh_val_list,
#     x_w_val,
#     gamma_val,
#     gamma_w_val,
#     # bounds=bounds,
#     wake_form=wake_form,
#     interactive=False,
#     camera=front_cam,
#     name=f'{rotor_type}_prop_ani_front_nr_{nr}_nt_{nt}' + f'_{wake_form}',
#     fps=10
# )

# spanwise thrust distribution
panel_thrust_b1 = -1.*panel_forces[:,:(nc-1)*(nr-1),0].reshape((nt, nc-1, nr-1))

plt.figure()
plt.plot(panel_thrust_b1[-1].T, label=[f'chord {ind}' for ind in range(nc-1)])
plt.legend()
plt.grid()
plt.show()

steady_panel_thrust_b1 = -1.*steady_panel_forces[:,:(nc-1)*(nr-1),0].reshape((nt, nc-1, nr-1))

# bound vec induced velocity
bound_vec_velocity_val = sim[bound_vec_velocity][:,:(nc-1)*(nr-1)].reshape((nt,nc-1,nr-1,3))
body_ind_vel_val = sim[body_ind_vel][:,:(nc-1)*(nr-1)].reshape((nt,nc-1,nr-1,3))
wake_ind_vel_val = sim[wake_ind_vel][:,:(nc-1)*(nr-1)].reshape((nt,nc-1,nr-1,3))
self_ind_vel_val = sim[self_ind_vel][:,:(nc-1)*(nr-1)].reshape((nt,nc-1,nr-1,3))


# local angle of attack

tangential_vel = np.linalg.norm(collocation_velocity[:,:,:,:,1:], axis=4)

tangential_vel = radius*nondim_r*omega
inflow_angle = np.arctan(V_inf/tangential_vel)*180/np.pi

alpha_eff = twist_dist - inflow_angle

# verifying dissipation
bqs = 2.5
time_array = np.arange(0,nt*dt,dt)
dd_val = np.exp(-bqs*time_array)
gamma_w_col = np.zeros_like(dd_val)
for i in range(nt-1):
    gamma_w_col[i] = gamma_w_val[i+1,:(nt-1)*(nr-1)].reshape(nt-1,nr-1)[:,0][-1]

# gamma_w_col = gamma_w_val[-1].reshape(nt-1,ns-1)[:,0]

gamma_w_col_rel = gamma_w_col/gamma_w_col[0]

rel_gamma_diff = dd_val-gamma_w_col_rel
# rel_gamma_error = rel_gamma_diff/dd_val

if False:
    plt.figure(figsize=(7,5))
    plt.plot(time_array[:-1], dd_val[:-1], '-', linewidth=3, label='Analytical dissipation')
    plt.plot(time_array[:-1], gamma_w_col_rel[:-1], '*', markersize=8, label='UVLM wake dissipation')
    plt.xlabel('Time (s)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Relative Wake Vortex Strength', fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    # plt.savefig('UVLM_dissipation_plot.pdf')
    plt.show()



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