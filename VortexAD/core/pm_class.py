import numpy as np
import csdl_alpha as csdl
import meshio

from VortexAD.utils.unstructured_grids.cell_adjacency import find_cell_adjacency, find_wake_cell_adjacency
from VortexAD.utils.unstructured_grids.TE_detection import TE_detection

from VortexAD.core.pm.steady_panel_solver import steady_panel_solver
from VortexAD.core.pm.unsteady_panel_solver import unsteady_panel_solver
import os

current_directory = os.getcwd()
default_mesh_path = current_directory + '/geometry/sample_meshes/naca0012_LE_TE_cluster.stl'

default_input_dict = {
    # flow properties
    'V_inf': 10., # m/s
    'Mach': None,
    'sos': 340.3, # m/s, 
    'alpha': None, # user can provide grid of velocities as well
    'rho': 1.225, # kg/m^3
    'compressibility': False, # PG correction
    'Cp cutoff': -5., # minimum Cp (numerical reasons)

    # mesh
    'mesh_path': default_mesh_path,
    'mesh_mode': 'unstructured',

    # collocation velocity
    'collocation_velocity': False,

    # panel method conditions
    'BC': 'Dirichlet',
    'higher_order': False,

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
    'drag_type': 'pressure', # pressure or Trefftz (not implemented yet)

    # unsteady solver
    'dt': 0.1, # time step (s)
    'nt': 10, # number of time steps
    'store_state_history': True, # flag to store state history
    'core_radius': 1.e-6, # vortex core radius
    'free_wake': False,
}

output_options_dict = {
    # forces and coefficients:
    'CL': ['lift coefficient (unitless)', '(num_nodes,)'],
    'CDi': ['induced drag coefficient (unitless)', '(num_nodes,)'],
    'CM': ['moment coefficient (unitless)', '(num_nodes, 3)'],
    'L': ['lift force (N)', '(num_nodes,)'],
    'Di': ['induced drag force (N)', '(num_nodes,)'],
    'M': ['moment (Nm)', '(num_nodes, 3)'],

    # force and pressure DISTRIBUTIONS
    'Cp': ['pressure coefficient distribution (unitless)', '(num_nodes, num_panels) or (num_nodes, nc, ns)'],
    'panel_forces': ['force on each panel (N)', '(num_nodes, num_panels, 3) or (num_nodes, nc, ns, 3)'],
    'L_panel': ['lift force on each panel (N)', '(num_nodes, num_panels) or (num_nodes, nc, ns)'],
    'Di_panel': ['induced drag force on each panel (N)', '(num_nodes, num_panels) or (num_nodes, nc, ns)'],

    # flow field information
    'V_mag': ['velocity magnitude at collocation points (m/s)', '(num_nodes, num_panels) or (num_nodes, nc, ns)'],

    # geometry/mesh parameters
    'panel center': ['physical coordinate of panel centers (m)', '(num_nodes, num_panels, 3) or (num_nodes, nc, ns, 3)'],
    'panel area': ['area of each panel (m^2)', '(num_nodes, num_panels) or (num_nodes, nc, ns)'],
    'panel_x_dir': ['local coordinate system in-plane vector 1 (m)', '(num_nodes, num_panels, 3) or (num_nodes, nc, ns, 3)'],
    'panel_y_dir': ['local coordinate system in-plane vector 2 (m)', '(num_nodes, num_panels, 3) or (num_nodes, nc, ns, 3)'],
    'panel_normal': ['panel normal vector (m)', '(num_nodes, num_panels, 3) or (num_nodes, nc, ns, 3)'],

    # others (usually for debugging)
}

class PanelMethod(object):
    def __init__(self, solver_input_dict, threshold_angle=125., skip_geometry=False):

        # load to existing default dictionary
        options_dict = default_input_dict
        for key in solver_input_dict.keys():
            options_dict[key] = solver_input_dict[key]
        self.options_dict = options_dict

        self.reuse_AIC = self.options_dict['reuse_AIC']
        self.solver_mode = self.options_dict['solver_mode']

        self.cell_adjacency_flag = False
        self.TE_properties_flag = False
        self.flow_properties_flag = False
        
        # self.solver_mode = solver_input_dict['solver_mode'] # steady or unsteady
        if not skip_geometry:
            self.mesh_filepath = solver_input_dict['mesh_path']
            self.setup_grid_properties(threshold_angle=125.)
            self.setup_flow_properties()

        # self.threshold_theta_default = 125


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
                grid_velocity = csdl.expand(V_vec, grid_shape)
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
                V_inf = csdl.Variable(value=V_inf)
            # shape of (3,) means 3 flow instances with a x-velocity
            # shape of (1,3) implies 1 case with 3 velocity components

            if len(V_inf.shape) == 1:
                V_vec = csdl.Variable(value=0., shape=(num_nodes,3))
                V_vec = V_vec.set(csdl.slice[:,0], value=V_inf)
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
                    grid_velocity = csdl.expand(V_vec_rot, grid_shape, 'ij->iaj')

            elif V_inf.shape == (num_nodes, 3): # velocity 
                grid_velocity = csdl.expand(V_inf, grid_shape, 'ij->iaj')
            
            # case where velocity is a tensor of shape (nn, n_points, 3)
            elif len(V_inf.shape) == 3:
                grid_velocity = V_inf

        # flipping sign due to coordinate systems
        self.grid_velocity = -grid_velocity
        self.coll_velocity = None

        self.num_nodes = num_nodes

        # self.flow_dict = {
        #     'nodal_velocity': self.grid_velocity,
        #     'coll_point_velocity': None # NOTE: change in the future
        # }
        self.flow_properties_flag = True

    def setup_grid_properties(self, threshold_angle=125, plot=False):
        '''
        Sets up the mesh, cell adjacency, and trailing edge properties.

        The trailing edge can be tweaked using the threshold angle input.
        '''
        self.import_mesh()

        self.compute_cell_adjacency()

        self.compute_TE_properties(threshold_angle=threshold_angle, plot=plot)

        self.grid_properties = True

    def declare_outputs(self, outputs):
        '''
        Declare outputs to be saved
        '''
        self.output_name_list = outputs

    def __assemble_input_dict__(self):

        nn_geom = self.num_nodes
        if self.reuse_AIC:
            nn_geom = 1

        self.points = csdl.expand(
            self.points,
            (nn_geom,) + self.points.shape,
            'ij->aij'
        )

        self.orig_mesh_dict = {
            'points': self.points,
            'nodal_velocity': self.grid_velocity,
            'coll_point_velocity': self.coll_velocity,
            'cell_point_indices': self.cells,
            'cell_adjacency': self.cell_adjacency,
            'points2cells': self.points2cells, # used for higher-order methods
            'TE_node_indices': self.TE_node_indices,
            'TE_edges': self.TE_edges,
            'upper_TE_cells': self.upper_TE_cells,
            'lower_TE_cells': self.lower_TE_cells,
            'wake_connectivity': self.wake_connectivity
        }

    def evaluate(self):
        '''
        Function call to set up and run the panel method.

        The output is a dictionary containing the string names set
        in self.declare_outputs()
        '''
        if not self.cell_adjacency_flag:
            raise ValueError('No cell adjacency data. Please check mesh path or use self.insert_grid_data.')
        if not self.TE_properties_flag:
            raise ValueError('No TE properties data. Please check mesh path or use self.insert_grid_data.')

        if not self.flow_properties_flag:
            self.setup_flow_properties()
        
        self.generate_wake_connectivity()

        self.__assemble_input_dict__()

        if self.solver_mode == 'steady':
            pm_output_dict = steady_panel_solver(
                self.orig_mesh_dict,
                self.options_dict,
            )
        elif self.solver_mode == 'unsteady':
            pm_output_dict = unsteady_panel_solver(
                self.orig_mesh_dict,
                self.options_dict,
            )

        output_dict = {}
        for output_name in self.output_name_list:
            output_dict[output_name] = pm_output_dict[output_name]

        return output_dict

    def print_output_options(self, output_names=False):
        '''
        Prints the output options with a description + shape
        If no input is provided, the entire dictionary is printed.

        Inputs:
        - output_names: list
        
        Returns the output options dictionary 
        '''
        if not output_names:
            print(output_options_dict)
        else:
            for key in output_names:
                if key in output_options_dict.keys():
                    print(output_options_dict[key])
        return output_options_dict

    def import_mesh(self):
        '''
        Importing the mesh using meshio.
        '''
        mesh = meshio.read(
            self.mesh_filepath
        )

        self.points_orig = mesh.points
        # self.cells = mesh.cells
        self.cells_dict_orig = mesh.cells_dict
        # self.cells_dict_orig = mesh.cells_dict['triangle']
    
    def overwrite_mesh(self, mesh):
        self.points = mesh

    def compute_cell_adjacency(self, radius=1.e-10):
        '''
        Computing cell adjacency informatiom.
        '''
        cell_adjacency_data = find_cell_adjacency(
            points=self.points_orig, 
            cells=self.cells_dict_orig, 
            radius=radius
        )

        self.points = cell_adjacency_data[0]
        self.cells = cell_adjacency_data[1]
        self.cell_adjacency = cell_adjacency_data[2]
        self.edges2cells = cell_adjacency_data[3]
        self.points2cells = cell_adjacency_data[4]

        self.cell_adjacency_flag = True


    def compute_TE_properties(self, threshold_angle=125, plot=False):
        '''
        Computing trailing edge properties to set the Kutta condition
        '''
        if not self.cell_adjacency_flag:
            self.compute_cell_adjacency()

        TE_properties = TE_detection(
            points=self.points,
            cells=self.cells,
            edges2cells=self.edges2cells,
            threshold_theta=threshold_angle
        )
        self.upper_TE_cells = TE_properties[0]
        self.lower_TE_cells = TE_properties[1]
        self.TE_edges = TE_properties[2]
        self.TE_node_indices = TE_properties[3]

        self.TE_properties_flag = True


        self.TE_data = ...

        if plot:
            cell_types = self.cells.keys()
            num_cells = np.sum([len(self.cells[cell_type]) for cell_type in cell_types])
            combined_cells = []
            for cell_type in cell_types:
                combined_cells += self.cells[cell_type].tolist()


            TE_coloring = np.zeros(shape=len(combined_cells))
            # TE_coloring = np.zeros(shape=self.cells.shape[0])
            TE_coloring[self.upper_TE_cells] = 1
            TE_coloring[self.lower_TE_cells] = -1

            from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution

            plot_pressure_distribution(self.points_orig, TE_coloring, connectivity=combined_cells, interactive=True, top_view=False, cmap='rainbow')

        return self.TE_data
    
    def generate_wake_connectivity(self):
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

        if self.solver_mode == 'steady':
            self.wake_connectivity = np.array([[
                edge[0],
                edge[0]+ns,
                edge[1]+ns,
                edge[1]
            ] for edge in TE_edges_zeroed])

        elif self.solver_mode == 'unsteady':
            nt = self.options_dict['nt']
            self.wake_connectivity = np.array([[[
                edge[0] + i*ns,
                edge[0] + (i+1)*ns,
                edge[1] + (i+1)*ns,
                edge[1] + i*ns,
            ] for edge in TE_edges_zeroed] for i in range(nt-1)])
        
        wake_cell_adjacency = find_wake_cell_adjacency(self.wake_connectivity)
        self.edges2cells_w = wake_cell_adjacency[0]

    # these functions are when we want to use the functions externally
    # this helps when doing optimization or using FFD to move a mesh
    def insert_grid_data(self, mesh, cell_adjacency_data, TE_properties):
        self.insert_mesh(mesh)
        self.insert_cell_adjacency(cell_adjacency_data)
        self.insert_TE_properties(TE_properties)

    def insert_mesh(self, mesh):
        self.points = mesh
        # print(self.points.shape)
        # exit()

    def insert_cell_adjacency(self, cell_adjacency_data):
        # self.points = cell_adjacency_data[0]
        self.cells = cell_adjacency_data[1]
        self.cell_adjacency = cell_adjacency_data[2]
        self.edges2cells = cell_adjacency_data[3]
        self.points2cells = cell_adjacency_data[4]

        self.cell_adjacency_flag = True

    def insert_TE_properties(self, TE_properties):
        self.upper_TE_cells = TE_properties[0]
        self.lower_TE_cells = TE_properties[1]
        self.TE_edges = TE_properties[2]
        self.TE_node_indices = TE_properties[3]

        self.TE_properties_flag = True

    
    def plot(self, data_to_plot, bounds=None, cmap='jet', camera=False, screenshot=False):
        '''
        Plotting function for scalar field variables.
        '''
        from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution
        # plot_pressure_distribution(self.points_orig, data_to_plot, connectivity=self.cells, bounds=bounds, interactive=True, top_view=False, cmap=cmap)
        cell_types = self.cells.keys()
        num_cells = np.sum([len(self.cells[cell_type]) for cell_type in cell_types])
        combined_cells = []
        for cell_type in cell_types:
            combined_cells += self.cells[cell_type].tolist()
        plot_pressure_distribution(self.points_orig, data_to_plot, connectivity=combined_cells, bounds=bounds, interactive=True, top_view=False, cmap=cmap, camera=camera, screenshot=screenshot)

    def plot_unsteady(self, mesh, wake_mesh, surface_data, wake_data, wake_form='grid', bounds=None, cmap='jet', interactive=False, camera=False, screenshot=False, name='panel_method'):
        from VortexAD.utils.plotting.plot_unstructured import plot_wireframe
        cell_types = self.cells.keys()
        num_cells = np.sum([len(self.cells[cell_type]) for cell_type in cell_types])
        combined_cells = []
        for cell_type in cell_types:
            combined_cells += self.cells[cell_type].tolist()

        plot_wireframe(mesh, wake_mesh, surface_data, wake_data, combined_cells, self.wake_connectivity, wake_form, self.TE_nodes_zeroed, self.edges2cells_w, interactive=interactive, camera=camera, name=name)
    
    # def conduct_off_body_analysis(self, eval_pts):
    #     velocity = off_body_analysis(eval_pts)

    #     return velocity


# class PMOutputs(csdl.VariableGroup):