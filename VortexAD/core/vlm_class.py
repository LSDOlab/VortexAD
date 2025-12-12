import numpy as np
import csdl_alpha as csdl

default_input_dict = {
    # flow properties
    'V_inf': None, # m/s
    'Mach': None,
    'sos': 340.3, # m/s, 
    'alpha': None, # user can provide grid of velocities as well
    'rho': 1.225, # kg/m^3
    'compressibility': False, # PG correction

    # mesh
    'meshes': None, # NOTE: set up default mesh here 

    # collocation velocity
    'collocation_velocity': False,

    # solver and wake mode; steady is fixed, unsteady is prescribed or free
    'solver_mode': 'steady',
    'wake_mode': 'fixed',

    # partition size for linear system assembly
    'partition_size': 1, # for full vectorization, set to None

    # GMRES linear system solve
    'iterative': False,
    
    # ROM options
    'ROM': False, # 'ROM-POD or ROM-Krylov

    # reusing AIC (no alpha dependence on wake) --> this only applies to fixed wake
    'reuse_AIC': False,

    # others
    'ref_area': 10., # reference area (l^2, l being the input length unit)
    'ref_chord': 1.,
    'moment_reference': np.zeros(3), 

    # unsteady solver
    'dt': 0.1, # time step (s)
    'nt': 10, # number of time steps
    'store_state_history': True, # flag to store state history
    'core_radius': 1.e-6, # vortex core radius
    'free_wake': False,
}



class VLM(object):
    def __init__(self, solver_input_dict):
        options_dict = default_input_dict
        for key in solver_input_dict.keys():
            options_dict[key] = solver_input_dict[key]
        self.options_dict = options_dict

        if self.options_dict['meshes'] is None:
            print('No mesh input. Generating default mesh.')

        self.reuse_AIC = self.options_dict['reuse_AIC']
        self.solver_mode = self.options_dict['solver_mode']

        self.setup_flow_properties()

    def setup_flow_properties(self):
        '''
        Input velocity options
        - mach + sos to get V_inf
        - V_inf as a:
            - scalar
            - vector
            - grid
        '''
        grid_shape = self.points.shape
        nn_grid, num_pts = grid_shape[0], grid_shape[1]

        V_inf   = self.options_dict['V_inf']
        mach    = self.options_dict['Mach']
        sos     = self.options_dict['sos']
        alpha   = self.options_dict['alpha']
        mesh_mode = self.options_dict['mesh_mode'] # structured or unstructured
        # if alpha is None:
        #     alpha = np.zeros(V_inf.shape)

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
                if nn_V_inf != 1 and nn_V_inf != 1:
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
                grid_velocity = csdl.expand(V_vec, (num_nodes,) + grid_shape, 'ij->iaj')
            else:
                pitch_rad = alpha * np.pi/180.
                V_rot_mat = csdl.Variable(value=0., shape=(num_nodes, 3,3))
                V_rot_mat = V_rot_mat.set(csdl.slice[:,1,1], value=1.)
                V_rot_mat = V_rot_mat.set(csdl.slice[:,0,0], value=csdl.cos(pitch_rad))
                V_rot_mat = V_rot_mat.set(csdl.slice[:,2,2], value=csdl.cos(pitch_rad))
                V_rot_mat = V_rot_mat.set(csdl.slice[:,2,0], value=csdl.sin(pitch_rad))
                V_rot_mat = V_rot_mat.set(csdl.slice[:,0,2], value=-csdl.sin(pitch_rad))

                V_vec_rot = csdl.einsum(V_rot_mat, V_vec, action='ijk,ik->ij')

                grid_velocity = csdl.expand(V_vec_rot, (num_nodes,) + grid_shape, 'ij->iaj')

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
                    grid_velocity = csdl.expand(V_inf, grid_shape)
                else:
                    pitch_rad = alpha * np.pi/180.
                    V_rot_mat = csdl.Variable(value=0., shape=(num_nodes, 3,3))
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,1,1], value=1.)
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,0,0], value=csdl.cos(pitch_rad))
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,2,2], value=csdl.cos(pitch_rad))
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,2,0], value=csdl.sin(pitch_rad))
                    V_rot_mat = V_rot_mat.set(csdl.slice[:,0,2], value=-csdl.sin(pitch_rad))

                    V_vec_rot = csdl.einsum(V_rot_mat, V_vec, action='ijk,ik->ij')
                    print(grid_shape)
                    grid_velocity = csdl.expand(V_vec_rot, (num_nodes,) + grid_shape, 'ij->iaj')

            elif V_inf.shape == (num_nodes, 3): # velocity 
                grid_velocity = csdl.expand(V_inf, grid_shape, 'ij->iaj')
            
            # case where velocity is a tensor of shape (nn, n_points, 3)
            elif len(V_inf.shape) == 3:
                grid_velocity = V_inf

        # flipping sign due to coordinate systems
        self.grid_velocity = -grid_velocity

        self.coll_velocity = None
        self.coll_vel_flag = False
        input_coll_vel = self.options_dict['collocation_velocity']
        if input_coll_vel:
            self.coll_velocity = -input_coll_vel # velocity relative to body --> sign change
            self.coll_vel_flag = True

        self.num_nodes = num_nodes

        # self.flow_dict = {
        #     'nodal_velocity': self.grid_velocity,
        #     'coll_point_velocity': None # NOTE: change in the future
        # }
        self.flow_properties_flag = True

    def evaluate(self):
        pass