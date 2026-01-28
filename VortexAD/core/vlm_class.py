import numpy as np
import csdl_alpha as csdl

from VortexAD.core.vlm.steady_vlm_solver import steady_vlm_solver
from VortexAD.core.vlm.unsteady_vlm_solver import unsteady_vlm_solver

from VortexAD.utils.plotting.plot_vlm import plot_wireframe

default_input_dict = {
    # flow properties
    'V_inf': None, # m/s
    'Mach': None,
    'sos': 340.3, # m/s, 
    'alpha': None, # user can provide grid of velocities as well
    'rho': 1.225, # kg/m^3
    'nu': 1.46e-5,
    'compressibility': False, # PG correction

    # mesh
    'meshes': None, # NOTE: set up default mesh here 
    'mesh_names': None,

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
    'core_radius': 1.e-3, # vortex core radius
    'vc_parameters': [1.25643, 0, 2.5], # alpha, a1, bqs from core model
    'free_wake': False,

    # ML airfoil model
    'alpha_ML': False
}



class VortexLatticeMethod(object):
    def __init__(self, solver_input_dict):
        options_dict = default_input_dict
        for key in solver_input_dict.keys():
            options_dict[key] = solver_input_dict[key]
        self.options_dict = options_dict

        if self.options_dict['meshes'] is None:
            from VortexAD.utils.meshing.gen_vlm_mesh import gen_vlm_mesh
            print('No mesh input. Generating default mesh.')
            ns, nc = 11, 3
            b, c = 10., 1.
            mesh = gen_vlm_mesh(ns, nc, b, c)
            self.options_dict['meshes'] = [mesh]

        self.meshes = self.options_dict['meshes']
        self.num_surfaces = len(self.options_dict['meshes'])
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

        Ways to input velocity:
        - Mach*sos and alpha as a scalar
        - Mach*sos and alpha across num_nodes
        - V_inf and alpha as a scalar
        - V_inf and alpha across num_nodes
        - nodal velocity across num_nodes, grid points and 3

        Steps to propagate velocities:
        - take one of the inputs from above list
        - convert to 3-components (x,y,z) 
            - this is a vector representing the flow field
        - loop over mesh list and make velocity across the nodes
            - V_vec --> nodal velocities (nn, nc, ns, 3) using einsum
        '''
        # grid_shape = self.points.shape

        V_inf   = self.options_dict['V_inf']
        mach    = self.options_dict['Mach']
        sos     = self.options_dict['sos']
        alpha   = self.options_dict['alpha']
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
                if V_vec_nn.shape == 4:
                    mesh_velocity = -V_vec_nn
                else:
                    mesh_velocity = csdl.expand(-V_vec_nn, mesh.shape, 'ij->iabj')
            self.mesh_velocities.append(mesh_velocity)
        
        if isinstance(V_inf, list):
            self.mesh_velocities = [-val for val in V_inf]

        self.coll_velocity = self.num_surfaces*[None]
        self.coll_vel_flag = self.num_surfaces*[False]
        input_coll_vel = self.options_dict['collocation_velocity']

        for i in range(self.num_surfaces):
            if input_coll_vel:
                mvs = self.mesh_velocities[i].shape
                expected_shape = (num_nodes, mvs[1]-1, mvs[2]-1, 3) # collocation points
                if input_coll_vel[i].shape != expected_shape:
                    raise ValueError(f'collocation velocity shape does not match nodal velocity shape: {expected_shape}')
                
                self.coll_velocity[i] = -input_coll_vel[i] # velocity relative to body --> sign change
                self.coll_vel_flag[i] = True

        self.num_nodes = num_nodes
        self.options_dict['num_nodes'] = num_nodes

        # self.flow_dict = {
        #     'nodal_velocity': self.grid_velocity,
        #     'coll_point_velocity': None # NOTE: change in the future
        # }
        self.flow_properties_flag = True

    def declare_outputs(self, outputs):
        '''
        Declare outputs to be saved
        '''
        self.output_name_list = outputs

    def __assemble_input_dict__(self):

        nn_geom = self.num_nodes
        if self.reuse_AIC:
            nn_geom = 1
        print(self.num_nodes)
        # exit()
        if len(self.meshes[0].shape) == 4: # (nn, nc, ns, 3)
            self.meshes = self.meshes
        else:
            # self.meshes = csdl.expand(
            #     self.meshes,
            #     (nn_geom,) + self.points.shape,
            #     'ij->aij'
            # )
            meshes = [
                csdl.expand(
                mesh,
                (nn_geom,) + mesh.shape,
                'ijk->aijk'
            ) for mesh in self.meshes
            ]
            self.meshes = meshes
        num_surfaces = len(self.meshes)
        if self.options_dict['mesh_names'] is None:
            mesh_names = [f'surface_{i}' for i in range(num_surfaces)]
        if len(mesh_names) != num_surfaces: # only for custom mesh names
            raise ValueError('List of mesh names must match the number of surfaces.')
        self.orig_mesh_dict = {}
        for i in range(num_surfaces):
            surf_name = mesh_names[i]
            self.orig_mesh_dict[surf_name] = {
                'mesh': self.meshes[i],
                'nodal_velocity': self.mesh_velocities[i],
                'coll_vel_flag': self.coll_vel_flag[i],
                'coll_vel': self.coll_velocity[i]
            }

        # self.orig_mesh_dict = {
        #     'points': self.meshes,
        #     'nodal_velocity': self.grid_velocity,
        #     'collocation_velocity': self.coll_velocity,
        #     'coll_vel_flag': self.coll_vel_flag,
        #     'wake_connectivity': self.wake_connectivity
        # }

    def evaluate(self):
        if not self.flow_properties_flag:
            self.setup_flow_properties()

        self.__assemble_input_dict__()

        if self.solver_mode == 'steady':
            vlm_output_dict = steady_vlm_solver(
                self.orig_mesh_dict,
                self.options_dict,
            )
        elif self.solver_mode == 'unsteady':
            vlm_output_dict = unsteady_vlm_solver(
                self.orig_mesh_dict,
                self.options_dict,
            )

        output_dict = {}
        for output_name in self.output_name_list:
            output_dict[output_name] = vlm_output_dict[output_name]

        return output_dict

    def plot_unsteady(self, meshes, wake_mesh, surface_data, wake_data, wake_form='grid', bounds=None, cmap='jet', interactive=False, camera=False, screenshot=False, name='sample_vlm_ani'):
        num_meshes = len(meshes)
        mesh_connectivity = []
        wake_connectivity = []
        for i in range(num_meshes):
            ms = meshes[i].shape
            nt, nc, ns = ms[0], ms[1], ms[2] # num points
            nt_p, nc_p, ns_p = nt-1, nc-1, ns-1 # num panels
            surf_mesh_con = np.array([[[
                j + i*ns,
                j + (i+1)*ns,
                j+1 + (i+1)*ns,
                j+1 + i*ns,
            ] for j in range(ns-1)] for i in range(nc-1)])
            mesh_connectivity.append(surf_mesh_con)
            wake_mesh_con = np.array([[[
                j + i*ns,
                j + (i+1)*ns,
                j+1 + (i+1)*ns,
                j+1 + i*ns,
            ] for j in range(ns-1)] for i in range(nt-1)])
            wake_connectivity.append(wake_mesh_con)
        


        plot_wireframe(meshes, mesh_connectivity, wake_mesh, wake_connectivity, surface_data, wake_data, 
                       wake_form=wake_form, interactive=interactive, camera=camera, name=name)

            