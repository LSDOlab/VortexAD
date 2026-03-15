import numpy as np
import csdl_alpha as csdl

from VortexAD.core.pfse.pfse_solver import pfse_solver

default_input_dict = {
    # flow properties
    'V_inf': None, # m/s
    'Mach': None,
    'sos': 340.3, # m/s, 
    'alpha': None, # user can provide grid of velocities as well
    'rho': 1.225, # kg/m^3
    'nu': 1.46e-5,
    'compressibility': False, # PG correction
    'Cp cutoff': -5., # minimum Cp (numerical reasons)

    # mesh
    'meshes': None, # NOTE: set up default mesh here 
    'mesh_names': None,

    # collocation velocity
    'collocation_velocity': False,

    # partition size for linear system assembly
    'partition_size': 1, # for full vectorization, set to None

    # GMRES linear system solve
    'iterative': False,
    
    # ROM options
    'ROM': False, # 'ROM-POD or ROM-Krylov

    # others
    'ref_area': 10., # reference area (l^2, l being the input length unit)
    'ref_chord': 1.,
    'moment_reference': np.zeros(3), 

    # unsteady solver
    'dt': 0.1, # time step (s)
    'nt': 10, # number of time steps
    'store_state_history': True, # flag to store state history
    'core_radius': 1.e-3, # vortex core radius
    'vc_parameters': [1.25643, 0, 2.5], # alpha, a1, bqs from core model
    'free_wake': False,

    # ML airfoil model
    'alpha_ML': False

}

mesh_dict = {
    'type': 
}

class PFSE(object):
    def __init__(self, solver_input_dict):
        options_dict = default_input_dict
        for key in solver_input_dict.keys():
            options_dict[key] = solver_input_dict[key]
        self.options_dict = options_dict

        # instantiating dictionary
        self.meshes = []
        self.mesh_names = []
        self.surf_counter = 0
        self.connectivity_list = []
        self.coll_vel_list = []
        self.coll_vel_flag_list = []

    def add_thick_surface(self, mesh, name=None, connectivity=None, coll_vel=None):
        if name is None:
            name = f'surface_{self.surf_counter}'

        if coll_vel is None:
            coll_vel_flag = False
        else:
            coll_vel = -coll_vel # velocity relative to body --> sign change
            coll_vel_flag = True
        
        self.meshes.append(mesh)
        self.mesh_names.append(name)
        self.connectivity_list.append(connectivity)
        self.coll_vel_list.append(coll_vel)
        self.coll_vel_flag_list.append(coll_vel_flag)

        self.surf_counter += 1

    def add_thin_surface(self, mesh, name=None, connectivity=None, coll_vel=None):
        if name is None:
            name = f'surface_{self.surf_counter}'
        coll_vel_flag = True
        if coll_vel is None:
            coll_vel_flag = False
        else:
            coll_vel = -coll_vel # velocity relative to body --> sign change
            coll_vel_flag = True

        self.meshes.append(mesh)
        self.mesh_names.append(name)
        self.connectivity_list.append(None)
        self.coll_vel_list.append(coll_vel)
        self.coll_vel_flag_list.append(coll_vel_flag)

        self.surf_counter += 1

    def setup_flow_properties(self):
        V_inf   = self.options_dict['V_inf']
        mach    = self.options_dict['Mach']
        sos     = self.options_dict['sos']
        alpha   = self.options_dict['alpha']
        
        # checking if V_inf is defined vs. mach #
        if V_inf is None:
            if mach is None:
                raise ValueError('Need to define a speed or Mach number')
            else:
                V_inf = mach*sos
        # checking num_nodes
        def check_num_nodes(val):
            if isinstance(val, float) or isinstance(val, int):
                nn_val = 1
            elif isinstance(val, csdl.Variable):
                nn_val = val.shape[0]
            elif val is None:
                nn_val = 0
            else:
                nn_val = len(val) # list, set or np.array()
            return nn_val

        if alpha is not None:
            nn_V_inf = check_num_nodes(V_inf)
            nn_V_alpha = check_num_nodes(alpha)

            if nn_V_inf != nn_V_alpha:
                if nn_V_inf != 1 and nn_V_alpha != 1:
                    raise ValueError('Error in defining shape of velocity and inflow angle.')
                
            num_nodes = np.max([nn_V_alpha, nn_V_inf])
        else:
            num_nodes = check_num_nodes(V_inf)
            nn_V_inf = num_nodes

        if self.solver_mode == 'unsteady':
            nt = self.options_dict['nt']
            nn_V_inf = nt
            num_nodes = nt

        # converting flow velocity into a grid

        # case where V_inf is a scalar and not a csdl variable:
        if isinstance(V_inf, float) or isinstance(V_inf, int):
        # if nn_V_inf == 1:
            V_vec = csdl.Variable(value=0., shape=(num_nodes,3))
            V_vec = V_vec.set(csdl.slice[:,0], value=-V_inf)
            if alpha is None:
                V_vec_nn = V_vec
                # grid_velocity = csdl.expand(V_vec, (num_nodes,) + grid_shape, 'ij->iaj')
            else:
                pitch_rad = alpha * np.pi/180.
                V_rot_mat = csdl.Variable(value=0., shape=(num_nodes, 3,3))
                V_rot_mat = V_rot_mat.set(csdl.slice[:,1,1], value=1.)
                V_rot_mat = V_rot_mat.set(csdl.slice[:,0,0], value=csdl.cos(pitch_rad))
                V_rot_mat = V_rot_mat.set(csdl.slice[:,2,2], value=csdl.cos(pitch_rad))
                V_rot_mat = V_rot_mat.set(csdl.slice[:,2,0], value=csdl.sin(pitch_rad))
                V_rot_mat = V_rot_mat.set(csdl.slice[:,0,2], value=-csdl.sin(pitch_rad))

                V_vec_rot = csdl.einsum(V_rot_mat, V_vec, action='ijk,ik->ij')
                V_vec_nn = V_vec_rot

                # grid_velocity = csdl.expand(V_vec_rot, (num_nodes,) + grid_shape, 'ij->iaj')
        elif isinstance(V_inf, list):
            V_vec_nn = np.array([0.])
        else:
            num_nodes = V_inf.shape[0] # FIRST DIMENSION IS ALWAYS NUM NODES
            if not isinstance(V_inf, csdl.Variable):
                V_inf = csdl.Variable(value=-V_inf)
            # shape of (3,) means 3 flow instances with a x-velocity
            # shape of (1,3) implies 1 case with 3 velocity components

            if len(V_inf.shape) == 1:
                V_vec = csdl.Variable(value=0., shape=(num_nodes,3))
                V_vec = V_vec.set(csdl.slice[:,0], value=-V_inf)
                if alpha is None:
                    V_vec_nn = V_vec_rot
                    # grid_velocity = csdl.expand(V_inf, grid_shape)
                else:
                    pitch_rad = alpha * np.pi/180.
                    V_rot_mat = csdl.Variable(value=0., shape=(num_nodes, 3,3))
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,1,1], value=1.)
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,0,0], value=csdl.cos(pitch_rad))
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,2,2], value=csdl.cos(pitch_rad))
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,2,0], value=csdl.sin(pitch_rad))
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,0,2], value=-csdl.sin(pitch_rad))

                    V_vec_rot = csdl.einsum(V_rot_mat, V_vec, action='ijk,ik->ij')
                    V_vec_nn = V_vec_rot
                    # print(grid_shape)
                    # grid_velocity = csdl.expand(V_vec_rot, (num_nodes,) + grid_shape, 'ij->iaj')
            

            elif V_inf.shape == (num_nodes, 3): # velocity 
                V_vec_rot = V_inf
                # V_vec_rot = csdl.expand(V_inf, grid_shape, 'ij->iaj')
            
            # case where velocity is a tensor of shape (nn, n_points, 3)
            elif len(V_inf.shape) == 4:
                V_vec_nn = V_inf

        self.mesh_velocities = []
        for i, mesh in enumerate(self.meshes):
            # nc, ns = mesh.shape[1], mesh.shape[2]
            #  
            # flipping sign due to coordinate systems
            if len(mesh.shape) == 3: # mesh is steady
                mesh_velocity = csdl.expand(-V_vec_nn, (num_nodes,) + mesh.shape, 'ij->iabj')
            elif len(mesh.shape) == 4: # mesh is unsteady
                # mesh_velocity = csdl.expand(-V_vec_nn, mesh.shape, 'ij->iabj')
                if len(V_vec_nn.shape) == 4:
                    mesh_velocity = -V_vec_nn
                else:
                    mesh_velocity = csdl.expand(-V_vec_nn, mesh.shape, 'ij->iabj')
            self.mesh_velocities.append(mesh_velocity)

        if isinstance(V_inf, list):
            self.mesh_velocities = [-val for val in V_inf]

        self.num_nodes = num_nodes
        self.options_dict['num_nodes'] = num_nodes

        # collocation velocities are handled in the functions that load meshes
        # for i in range(self.num_surfaces):
        #     mvs = self.mesh_velocities[i].shape
        #     expected_shape = (num_nodes, mvs[1]-1, mvs[2]-1, 3) # collocation points
        #     if input_coll_vel[i].shape != expected_shape:
        #         raise ValueError(f'collocation velocity shape does not match nodal velocity shape: {expected_shape}')
            
        #     self.coll_velocity[i] = -input_coll_vel[i] # velocity relative to body --> sign change
        #     self.coll_vel_flag[i] = True

    def generate_wake_connectivity(self):
        pass

    def evaluate(self):
        self.options_dict['meshes'] = self.meshes
        self.options_dict['connectivity'] = self.connectivity_list
        self.options_dict['collocation_velocity'] = self.coll_vel_list

        self.generate_wake_connectivity()

        self.setup_flow_properties()

        self.__assemble_input_dict__()


        solver_output_dict = pfse_solver(
            self.orig_mesh_dict,
            self.options_dict
        )
        
        output_dict = {}
        for output_name in self.output_name_list:
            output_dict[output_name] = solver_output_dict[output_name]

        return output_dict
    
    def __assemble_input_dict__(self):
        self.orig_mesh_dict = {}

        num_surfaces = len(self.meshes)
        for i in range(num_surfaces):
            surf_name = self.mesh_names[i]

            sub_dict = {
                'mesh': self.meshes[i],
                'nodal_velocity': self.mesh_velocities[i],
                'coll_vel': self.coll_vel_list[i],
                'coll_vel_flag': self.coll_vel_flag_list[i]
            }

            self.orig_mesh_dict[surf_name] = sub_dict