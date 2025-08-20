import numpy as np
import csdl_alpha as csdl
import meshio

from VortexAD.utils.unstructured_grids.cell_adjacency import find_cell_adjacency
from VortexAD.utils.unstructured_grids.TE_detection import TE_detection


from VortexAD.core.pm.steady_panel_solver import steady_panel_solver
import os

current_directory = os.getcwd()
default_mesh_path = current_directory + '/geometry/sample_meshes/naca0012_LE_TE_cluster.stl'

default_input_dict = {
    # flow properties
    'V_inf': None,
    'Mach': None,
    'sos': 340.3, # m/s, 
    'alpha': np.array([0.]), # user can provide grid of velocities as well
    'rho': 1.225, # kg/m^3
    'compressibility': False, # PG correction
    'Cp cutoff': -5., # minimum Cp (numerical reasons)

    # mesh
    'mesh_path': default_mesh_path,

    # collocation velocity
    'collocation_velocity': False,

    # panel method conditions
    'BC': 'Dirichlet',
    'higher_order': False,

    # grid type
    'mesh_mode': 'unstructured',

    # solver mode
    'solver_mode': 'steady',

    # wake mode
    # steady is fixed
    # unsteady is prescribed or free
    'wake_mode': 'fixed',

    # partition size for linear system assembly
    'partition_size': 1,

    # GMRES linear system solve
    'iterative': False,
    
    # ROM options
    'ROM': False,
    # 'ROM-POD': False,
    # 'ROM-Krylov': False,

    # reusing AIC (no alpha dependence on wake)
    # this only applies to fixed wake
    'reuse_AIC': False,

    # others
    'ref_area': 10., # reference area (l^2, l being the input length unit)
    'ref_chord': 1.,
    'moment_reference': np.zeros(3), 
    'drag_type': 'pressure' # pressure or Trefftz (not implemented yet)
}

output_options_dict = {
    'CL': ['lift coefficient (unitless)', '(num_nodes,)'],
    'L': ['lift force (N)', '(num_nodes,)'],

    'CDi': ['induced drag coefficient (unitless)', '(num_nodes,)'],
    'Di': ['induced drag force (N)', '(num_nodes,)'],

    'CM': ['moment coefficient (unitless)', '(num_nodes, 3)'],
    'M': ['moment (Nm)', '(num_nodes, 3)'],

    'Cp': ['pressure coefficient distribution (unitless)', '(num_nodes, num_panels) or (num_nodes, nc, ns)'],

    'panel forces': ['force on each panel (N)', '(num_nodes, num_panels, 3) or (num_nodes, nc, ns, 3)'],
}

class PanelMethod(object):
    def __init__(self, solver_input_dict, threshold_angle=125.):

        # load to existing default dictionary
        options_dict = default_input_dict
        for key in solver_input_dict.keys():
            options_dict[key] = solver_input_dict[key]
        self.options_dict = options_dict

        self.mesh_filepath = solver_input_dict['mesh_path']
        self.reuse_AIC = self.options_dict['reuse_AIC']
        
        # self.solver_mode = solver_input_dict['solver_mode'] # steady or unsteady

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
            else:
                nn_val = len(val) # list, set or np.array()
            return nn_val

        nn_V_inf = check_num_nodes(V_inf)
        nn_V_alpha = check_num_nodes(alpha)

        if nn_V_inf != nn_V_alpha:
            if nn_V_inf != 1 and nn_V_inf != 1:
                raise ValueError('Error in defining shape of velocity and inflow angle.')
            
        num_nodes = np.max([nn_V_alpha, nn_V_inf])

        # converting flow velocity into a grid

        # case where V_inf is a scalar and not a csdl variable:
        # if isinstance(V_inf, float) or isinstance(V_inf, int):
        if nn_V_inf == 1:
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

    def setup_grid_properties(self, threshold_angle=125):
        '''
        Sets up the mesh, cell adjacency, and trailing edge properties.

        The trailing edge can be tweaked using the threshold angle input.
        '''
        self.import_mesh()

        self.compute_cell_adjacency()

        self.compute_TE_properties()

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
            'points2cells': self.points2cells,
            'TE_node_indices': self.TE_node_indices,
            'TE_edges': self.TE_edges,
            'upper_TE_cells': self.upper_TE_cells,
            'lower_TE_cells': self.lower_TE_cells,
        }

    def evaluate(self):
        '''
        Function call to set up and run the panel method.

        The output is a dictionary containing the string names set
        in self.declare_outputs()
        '''

        self.__assemble_input_dict__()

        pm_output_dict, mesh_dict = steady_panel_solver(
            self.orig_mesh_dict,
            self.options_dict,
        )


        output_dict = {}
        for output_name in self.output_name_list:
            output_dict[output_name] = pm_output_dict[output_name]

        return output_dict

    def import_mesh(self):
        '''
        Importing the mesh using meshio.
        '''
        mesh = meshio.read(
            self.mesh_filepath
        )

        self.points_orig = mesh.points
        # self.cells = mesh.cells
        cells_dict = mesh.cells_dict
        self.cells_orig = cells_dict['triangle']
        # _orig bc we update these in the cell_adjacency function

    def compute_cell_adjacency(self, radius=1.e-10):
        '''
        Computing cell adjacency informatiom.
        '''
        cell_adjacency_data = find_cell_adjacency(
            points=self.points_orig, 
            cells=self.cells_orig, 
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
            TE_coloring = np.zeros(shape=self.cells.shape[0])
            TE_coloring[self.upper_TE_cells] = 1
            TE_coloring[self.lower_TE_cells] = -1

            from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution

            plot_pressure_distribution(self.points_orig, TE_coloring, connectivity=self.cells, interactive=True, top_view=False, cmap='rainbow')

        return self.TE_data
    
    def plot(self, data_to_plot, bounds=None, cmap='jet'):
        '''
        Plotting function for scalar field variables.
        '''
        from VortexAD.utils.plotting.plot_unstructured import plot_pressure_distribution
        plot_pressure_distribution(self.points_orig, data_to_plot, connectivity=self.cells, bounds=bounds, interactive=True, top_view=False, cmap=cmap)
    
    # def conduct_off_body_analysis(self, eval_pts):
    #     velocity = off_body_analysis(eval_pts)

    #     return velocity


# class PMOutputs(csdl.VariableGroup):