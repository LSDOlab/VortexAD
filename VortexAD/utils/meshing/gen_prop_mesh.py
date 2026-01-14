import numpy as np
from VortexAD import AIRFOIL_PATH

def gen_prop_mesh(radius, chord, twist, num_blades, direction='up', r0=0.2, num_radial=5, nc=2, plot=False):
    '''
    Docstring for gen_prop_mesh
    
    :param radius: radius
    :param chord: Chord length or distribution
    :param twist: Twist angle or distribution
    :param num_blades: Number of blades
    :param direction: Prop disk normal direction
    :param r0: inner radius start (normalized between 0 and 1)
    :param num_radial: Number of radial/spanwise stations
    :param nc: Number of chordwise nodes
    '''
    if direction == 'forward':
        normal_vec = np.array([1., 0., 0.])
    elif direction == 'up':
        normal_vec = np.array([0., 0., 1.])
    
    # chord distribution vs. cst chord.
    if isinstance(chord, float):
        chord_dist = np.ones((num_radial,)) * chord
    elif isinstance(chord, np.array):
        num_radial_from_chord = np.prod(chord.shape)
        if np.prod(chord.shape) == 1:
            chord_dist = np.ones((num_radial,)) * chord
        else:
            num_radial = num_radial_from_chord
            chord_dist = chord

    # generating twist distribution
    if isinstance(twist, float):
        twist_dist = np.ones(num_radial) * twist
    elif isinstance(twist, np.array):
        if np.prod(twist.shape) > 1:
            twist_dist = twist
        else:
            twist_dist = np.ones(num_radial) * twist

    if np.prod(twist_dist.shape) != np.prod(chord_dist.shape):
        ValueError('Mismatched twist and chord shapes.')
            

    mesh_shape = (nc, num_radial, 3)
    mesh_blade = np.zeros(mesh_shape)
    radial_stations = np.linspace(r0, 1., num_radial) * radius
    ndsl = 0.25 # nondimensional spar location --> twist rotation point
    for i in range(num_radial):
        sec_twist = twist_dist[i]*np.pi/180.
        sec_chord = chord_dist[i]
        radial_station = radial_stations[i]

        mesh_section = np.zeros((nc, 3))
        mesh_section[:,1] = radial_station
        if direction == 'up':
            mesh_section[:,0] = -sec_chord*ndsl + np.linspace(0,sec_chord,nc)
        elif direction == 'forward':
            mesh_section[:,2] = -sec_chord*ndsl + np.linspace(0,sec_chord,nc)

        rotation_point = np.array([0., radial_station, 0.])

        # radial_dir = np.array([0., 1., 0.])
        # rot_vec = sec_twist*radial_dir
        # rotation_arm = mesh_section-rotation_point
        # mesh_section_twisted = np.cross(rot_vec, rotation_arm)

        # mesh_blade[i,:] = mesh_section_twisted

        sec_rot_mat = np.zeros((3,3))
        sec_rot_mat[1,1] = 1.
        sec_rot_mat[0,0] = sec_rot_mat[2,2] = np.cos(sec_twist)
        sec_rot_mat[0,2] = np.sin(sec_twist)
        sec_rot_mat[2,0] = -np.sin(sec_twist)

        mesh_section_twisted = np.einsum('ij,aj->ai', sec_rot_mat, mesh_section-rotation_point) + rotation_point

        mesh_blade[:,i,:] = mesh_section_twisted

    mesh_array = np.zeros((num_blades,) + mesh_shape)
    rotation_angles = np.linspace(0, 2*np.pi, num_blades+1)[:-1]
    for i in range(num_blades):
        # copy and rotate mesh a certain number of degrees in the plane
        theta = rotation_angles[i]
        if theta == 0.: # no need to apply rotation if zero
            mesh_array[i,:] = mesh_blade
            continue
        
        rot_mat = np.zeros((3,3))
        if direction == 'up':
            rot_mat[2,2] = 1.
            rot_mat[0,0] = rot_mat[1,1] = np.cos(theta)
            rot_mat[1,0] = np.sin(theta)
            rot_mat[0,1] = -np.sin(theta)
            
        elif direction == 'forward':
            rot_mat[0,0] = 1.
            rot_mat[1,1] = rot_mat[2,2] = np.cos(theta)
            rot_mat[1,2] = np.sin(theta)
            rot_mat[2,1] = -np.sin(theta)

        mesh_blade_rotated = np.einsum('ij,abj->abi', rot_mat, mesh_blade)
        mesh_array[i,:] = mesh_blade_rotated

    # if direction == 'forward':
    #     # apply 270 deg rotation from pointing up (0,0,1) to forward (1,0,0)
    #     rot_angle = 3*np.pi/2
    #     rot_mat = np.zeros((3,3))
    #     rot_mat[1,1] = 1.
    #     rot_mat[0,0] = rot_mat[2,2] = np.cos(theta)
    #     rot_mat[0,2] = np.sin(theta)
    #     rot_mat[2,0] = -np.sin(theta)
    #     final_mesh = np.einsum('ij,abcj->abci', rot_mat, mesh_array)
    # else:
    #     final_mesh = mesh_array
    final_mesh = mesh_array

    if plot == True:
        plot_prop_mesh(final_mesh)
        
    
    return final_mesh
    # meshes = [final_mesh[i,:] for i in range(num_blades)]
    # return meshes


def plot_prop_mesh(mesh):
    import vedo
    from vedo import Axes, Plotter, Mesh, show
    vedo.settings.default_backend = 'vtk'
    axs = Axes(
        xrange=(0,3),
        yrange=(-7.5, 7.5),
        zrange=(0, 5),
    )
    vp = Plotter(
        bg='white',
        # bg2='white',
        # axes=0,
        #  pos=(0, 0),
        offscreen=False,
        interactive=1,
        size=(2500,2500)
    )
    ms = mesh.shape
    nb, nc, ns = ms[0], ms[1], ms[2]

    connectivity = np.array([[[
        j + i*ns,
        j + (i+1)*ns,
        j+1 + (i+1)*ns,
        j+1 + i*ns,
    ] for j in range(ns-1)] for i in range(nc-1)]).reshape((-1,4))

    for i in range(nb):
        mesh_points = mesh[i,:]
        vps = Mesh(
            [np.reshape(mesh_points, (-1,3)), connectivity],
            c='gray',
            alpha=1.
        ).linecolor('black')
        vp += vps

    show([vps, axs], elevation=-45, azimuth=-45, roll=45,
                    axes=False, interactive=True)  # render the scene
