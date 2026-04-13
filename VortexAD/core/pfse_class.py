import numpy as np
import csdl_alpha as csdl

from VortexAD.core.pfse.pfse_solver import pfse_solver

from VortexAD.utils.unstructured_grids.cell_adjacency import find_cell_adjacency, find_wake_cell_adjacency
from VortexAD.utils.unstructured_grids.TE_detection import TE_detection

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

    # partition size for linear system assembly
    'partition_size': 1, # for full vectorization, set to None

    # GMRES linear system solve
    'iterative': False,
    
    # ROM options
    'ROM': False, # 'ROM-POD or ROM-Krylov

    # reference values
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

    # panel method options
    'BC_PM': 'Dirichlet',
    'Cp cutoff': -5., # minimum Cp (numerical reasons)
    'reuse_AIC': False, # not used I think


    # ML airfoil model
    'alpha_ML': False

}

# mesh_dict = {
#     'type': 
# }

class PFSE(object):
    def __init__(self, solver_input_dict):
        options_dict = default_input_dict
        for key in solver_input_dict.keys():
            options_dict[key] = solver_input_dict[key]
        self.options_dict = options_dict

        # instantiating dictionary
        self.meshes = []
        self.mesh_types = [] # either thick or thin
        self.mesh_names = []
        self.surf_counter = 0
        self.coll_vel_list = []
        self.coll_vel_flag_list = []

        # VLM surface data
        self.vlm_meshes = []
        self.vlm_coll_vel_list = []
        self.vlm_coll_vel_flag_list = []

    def add_thick_surface(self, mesh, connectivity, TE_properties, name=None, coll_vel=None):
        if name is None:
            name = f'surface_{self.surf_counter}'

        if coll_vel is None:
            coll_vel_flag = False
        else:
            coll_vel = -coll_vel # velocity relative to body --> sign change
            coll_vel_flag = True
        
        # self.meshes.append(mesh)
        self.mesh_types.append('thick')
        self.mesh_names.append(name)
        # self.coll_vel_list.append(coll_vel)
        # self.coll_vel_flag_list.append(coll_vel_flag)

        self.pm_coll_vel = coll_vel
        self.pm_coll_vel_flag = coll_vel_flag

        self.surf_counter += 1

        # separating connectivity data
        # self.points = connectivity[0]
        self.points = mesh
        self.cells = connectivity[1]
        self.cell_adjacency = connectivity[2]
        self.edges2cells = connectivity[3]
        self.points2cells = connectivity[4]

        # separating TE data

        self.upper_TE_cells = TE_properties[0]
        self.lower_TE_cells = TE_properties[1]
        self.TE_edges = TE_properties[2]
        self.TE_node_indices = TE_properties[3]

    def add_thin_surface(self, mesh, name=None, coll_vel=None):
        if name is None:
            name = f'surface_{self.surf_counter}'
        coll_vel_flag = True
        if coll_vel is None:
            coll_vel_flag = False
        else:
            coll_vel = -coll_vel # velocity relative to body --> sign change
            coll_vel_flag = True

        # self.meshes.append(mesh)
        self.mesh_types.append('thin')
        self.mesh_names.append(name)
        # self.coll_vel_list.append(coll_vel)
        # self.coll_vel_flag_list.append(coll_vel_flag)

        self.vlm_meshes.append(mesh)
        self.vlm_coll_vel_list.append(coll_vel)
        self.vlm_coll_vel_flag_list.append(coll_vel_flag)

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

        # if self.solver_mode == 'unsteady':
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

        # setting up PM velocities
        pm_mesh = self.points
        if V_vec_nn.shape == (num_nodes, 3):
            pm_velocity = -V_vec_nn.expand(pm_mesh.shape, 'ij->iaj')
            self.pm_grid_velocity = pm_velocity

        else:
            ValueError('panel method velocity input error')

        # setting up VLM velocities
        self.vlm_mesh_velocities = []
        for i, mesh in enumerate(self.vlm_meshes):
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
            self.vlm_mesh_velocities.append(mesh_velocity)

        # if isinstance(V_inf, list):
        #     self.mesh_velocities = [-val for val in V_inf]

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

        # PM
        ns = len(self.TE_node_indices)
        num_TE_edges = len(self.TE_edges)
        TE_edges_zeroed = []
        TE_nodes_zeroed_dup = []
        for i in range(num_TE_edges):
            edge = self.TE_edges[i]
            new_edge = []
            for j in range(2):
                ind = np.where(self.TE_node_indices == edge[j])[0][0]
                new_edge.append(ind)
            TE_edges_zeroed.append(tuple(new_edge))
            TE_nodes_zeroed_dup.extend(new_edge)
        self.TE_nodes_zeroed = list(set(TE_nodes_zeroed_dup))


        nt = self.options_dict['nt']
        self.pm_wake_connectivity = np.array([[[
            edge[0] + i*ns,
            edge[0] + (i+1)*ns,
            edge[1] + (i+1)*ns,
            edge[1] + i*ns,
        ] for edge in TE_edges_zeroed] for i in range(nt-1)])
    
        wake_cell_adjacency = find_wake_cell_adjacency(self.pm_wake_connectivity)
        self.edges2cells_w = wake_cell_adjacency[0]

        # VLM



    def declare_outputs(self, outputs):
        '''
        Declare outputs to be saved
        '''
        self.output_name_list = outputs

    def evaluate(self):
        # self.options_dict['meshes'] = self.meshes
        # self.options_dict['connectivity'] = self.connectivity_list
        # self.options_dict['collocation_velocity'] = self.coll_vel_list

        self.generate_wake_connectivity()

        self.setup_flow_properties()

        self.__assemble_input_dict__()

        solver_output_dict = pfse_solver(
            self.pm_orig_mesh_dict,
            self.vlm_orig_mesh_dict,
            self.options_dict
        )
        
        output_dict = {}
        for output_name in self.output_name_list:
            output_dict[output_name] = solver_output_dict[output_name]

        return output_dict
    
    def __assemble_input_dict__(self):
        # PM
        self.pm_orig_mesh_dict = {
            'points': self.points,
            'nodal_velocity': self.pm_grid_velocity,
            'collocation_velocity': self.pm_coll_vel,
            'coll_vel_flag': self.pm_coll_vel_flag,
            'cell_point_indices': self.cells,
            'cell_adjacency': self.cell_adjacency,
            'points2cells': self.points2cells, # used for higher-order methods
            'TE_node_indices': self.TE_node_indices,
            'TE_edges': self.TE_edges,
            'upper_TE_cells': self.upper_TE_cells,
            'lower_TE_cells': self.lower_TE_cells,
            'wake_connectivity': self.pm_wake_connectivity
        }
        # VLM
        self.vlm_orig_mesh_dict = {}

        num_surfaces = len(self.vlm_meshes)
        for i in range(num_surfaces):
            surf_name = self.mesh_names[i]

            sub_dict = {
                'mesh': self.vlm_meshes[i],
                'nodal_velocity': self.vlm_mesh_velocities[i],
                'coll_vel': self.vlm_coll_vel_list[i],
                'coll_vel_flag': self.vlm_coll_vel_flag_list[i]
            }

            self.vlm_orig_mesh_dict[surf_name] = sub_dict
    

    def plot_unsteady(self, pm_mesh, vlm_meshes, x_w, surface_data, wake_data, 
                      wake_form='grid', bounds=None, cmap='jet', interactive=False, camera=False, screenshot=False, name='sample_vlm_ani', fps=5):
        from VortexAD.utils.plotting.plot_pfse import plot_wireframe

        # PM plotting setup
        cell_types = self.cells.keys()
        num_cells = np.sum([len(self.cells[cell_type]) for cell_type in cell_types])
        combined_cells = []
        for cell_type in cell_types:
            combined_cells += self.cells[cell_type].tolist()

        pm_conn_params = [
            self.TE_node_indices, 
            self.TE_nodes_zeroed, 
            self.edges2cells_w
        ]

        # VLM plotting setup
        num_meshes = len(vlm_meshes)
        vlm_mesh_connectivity = []
        vlm_wake_connectivity = []
        for i in range(num_meshes):
            ms = vlm_meshes[i].shape
            nt, nc, ns = ms[0], ms[1], ms[2] # num points
            nt_p, nc_p, ns_p = nt-1, nc-1, ns-1 # num panels
            surf_mesh_con = np.array([[[
                j + i*ns,
                j + (i+1)*ns,
                j+1 + (i+1)*ns,
                j+1 + i*ns,
            ] for j in range(ns-1)] for i in range(nc-1)])
            vlm_mesh_connectivity.append(surf_mesh_con)
            wake_mesh_con = np.array([[[
                j + i*ns,
                j + (i+1)*ns,
                j+1 + (i+1)*ns,
                j+1 + i*ns,
            ] for j in range(ns-1)] for i in range(nt-1)])
            vlm_wake_connectivity.append(wake_mesh_con)

        wake_connectivities = [self.pm_wake_connectivity] + vlm_wake_connectivity

        plot_wireframe(pm_mesh, combined_cells, pm_conn_params, vlm_meshes, vlm_mesh_connectivity, x_w, wake_connectivities, surface_data, wake_data, 
                       bounds=bounds, wake_form=wake_form, interactive=interactive, camera=camera, name=name, fps=fps)