# region Imports and Setup

import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

from modopt import CSDLAlphaProblem, PySLSQP
import lsdo_geo
import VortexAD
import meshio
import pickle

import resource
import time
from datetime import datetime
import pyvista as pv

lfs.num_workers=1

recorder = csdl.Recorder(inline=True)
recorder.start()

imported_function_set = lfs.import_file_patched(file_name="../rectangular_wing_naca0012_10ar.stp", parallelize=False)
geometry = lsdo_geo.Geometry(functions=imported_function_set.functions, 
                                      function_names=imported_function_set.function_names,
                                      name='imported_geometry',
                                      space=imported_function_set.space)

# geometry.plot()
# geometry.functions[0].plot()
# geometry.functions[1].plot()
# geometry.functions[2].plot()
# geometry.functions[3].plot()

exit()
# endregion Imports

# region panel mesh import
mesh = meshio.read("../rectangular_wing_naca0012_10ar.msh")

points_orig = mesh.points
cells = mesh.cells
cells_dict = mesh.cells_dict
cell_adjacency_data = VortexAD.find_cell_adjacency(points=points_orig, cells=cells_dict)

points_orig = cell_adjacency_data[0] 
cells_dict = cell_adjacency_data[1] 
cell_adjacency = cell_adjacency_data[2] 
edges2cells = cell_adjacency_data[3]
points2cells = cell_adjacency_data[4]

TE_properties = VortexAD.TE_detection(points=points_orig,
                             cells=cells_dict,
                             edges2cells=edges2cells,
                             points2cells=points2cells,
                             threshold_theta=125.
                             )

upper_TE_cells = TE_properties[0] 
lower_TE_cells = TE_properties[1] 
TE_edges = TE_properties[2] 
TE_node_indices = TE_properties[3]

cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells_dict[cell_type].tolist()

projected_panel_mesh = geometry.project(points_orig, 
                                        grid_search_density_parameter=1, 
                                        newton_tolerance=1.e-10, 
                                        grid_search_density_cutoff=30,
                                        projection_tolerance=1.e-3,
                                        force_reprojection=False, 
                                        plot=False
                                        )

# project panel centers
cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types:
    combined_cells += cells_dict[cell_type].tolist()

panel_centers_orig = np.zeros((len(combined_cells), 3))
for i, cell in enumerate(combined_cells):
    panel_centers_orig[i] = np.mean(points_orig[cell], axis=0)

projected_panel_centers = geometry.project(panel_centers_orig, 
                            grid_search_density_parameter=1,
                            newton_tolerance=1.e-10,
                            grid_search_density_cutoff=30,
                            projection_tolerance=1.e-2,
                            force_reprojection=False, 
                            plot=False,
                            )
# endregion

# region Key locations

# The following points are used to define the key locations of the geometry 
# that can be used to define meshes and/or design parameters. The inputs are numpy arrays
# with the initial locations in physical space. The output of the projection is the parametric 
# location of the point on the geometry. It is important to have the coordinates in parametric space
# because the parametric coordinates will not change as the geometry is deformed.

leading_edge_left = geometry.project(np.array([0.0, -5.0, 0.0]))
leading_edge_right = geometry.project(np.array([0.0, 5.0, 0.0]))
trailing_edge_left = geometry.project(np.array([1.0, -5.0, 0.0]))
trailing_edge_right = geometry.project(np.array([1.0, 5.0, 0.0]))
leading_edge_center = geometry.project(np.array([0.0, 0.0, 0.0]))
trailing_edge_center = geometry.project(np.array([1.0, 0.0, 0.0]))
quarter_chord_left = geometry.project(np.array([0.25, -5.0, 0.0]))
quarter_chord_right = geometry.project(np.array([0.25, 5.0, 0.0]))
quarter_chord_center = geometry.project(np.array([0.25, 0.0, 0.0]))
# endregion

# region Create Parameterization Objects

root_twist_dv = csdl.Variable(value=0.)
tip_twist_dv = csdl.Variable(value=0.)
twist_coeff = csdl.Variable(value=np.zeros((3,)))
twist_coeff = twist_coeff.set(csdl.slice[1], root_twist_dv * np.pi/180)
twist_coeff = twist_coeff.set(csdl.slice[0::2], tip_twist_dv * np.pi/180)
# twist_coeff_dv = csdl.Variable(value=np.array([0., 10., 20])*np.pi/180)

# Construct a Free Form Deformation (FFD) block around the geometry
num_ffd_coefficients_chordwise = 8
num_ffd_sections = 3
# Note: This FFD block construction is one of a few helper functions that can be used to create a FFD block.
#       The "manual" method is to use construct_ffd_block_from_corners, which allows for defining the coefficients directly.
ffd_block = construct_ffd_block_around_entities(entities=geometry, 
                                                num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2), degree=(3,1,1))
# ffd_block.plot()

# Define an axial sectional parameterization for the FFD volume. 
# This views the FFD volume as a series of 2D sections (as defined by the control points) 
# that can be allowed to stretch, translate, and rotate independently.
# The sectional parameterization is chosen to have the spanwise direction as the principal 
# parametric dimension (0,1,2 corresponds to u,v,w of the FFD block, which in this case corresponds to x,y,z).
ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=1,
)
# ffd_sectional_parameterization.plot()

# Although unnecessary for this example, this section defines B-spline functions that can be used to independently
# parameterize the sectional parameters (this method is commonly used, so it's included here for completeness).
# The coefficients will be used as the states of the parameterization solver, which will be manipulated to solve
# for the desired geometry (satisfies the design parameters and constraints). The initial values are mainly for
# debugging to see what the deformation modes do to the geometry since the solver will solve for the actual values.
space_of_linear_3_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
space_of_linear_2_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

chord_stretching_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                         coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])), name='chord_stretching_b_spline_coefficients')

wingspan_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                             coefficients=csdl.Variable(shape=(2,), value=np.array([-0., 0.])), name='wingspan_stretching_b_spline_coefficients')

sweep_translation_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])), name='sweep_translation_b_spline_coefficients')
# sweep_translation_b_spline.plot()

# twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
#                                 coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])*np.pi/180), name='twist_b_spline_coefficients')

twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
                                coefficients=twist_coeff, name='twist_b_spline_coefficients')


# endregion Create Parameterization Objects

# region Evaluate Inner Parameterization Map To Define Forward Model For Parameterization Solver
# Evaluate the B-splines to get the sectional parameters
parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(parametric_b_spline_inputs)
wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(parametric_b_spline_inputs)
sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(parametric_b_spline_inputs)
twist_sectional_parameters = twist_b_spline.evaluate(parametric_b_spline_inputs)

# Evaluate the sectional parameterization to get the FFD coefficients
sectional_parameters = VolumeSectionalParameterizationInputs()
sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translation_sectional_parameters)
sectional_parameters.add_sectional_rotation(axis=1, rotation=twist_sectional_parameters)
ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

# Evaluate the FFD and set the coefficients of the geometry
geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) # type: ignore
# geometry.plot()

# Define the design parameters as a function of the geometry (which is now a function of the parameterization states)
wingspan = csdl.norm(geometry.evaluate(leading_edge_right) - geometry.evaluate(leading_edge_left)) # type: ignore
root_chord = csdl.norm(geometry.evaluate(trailing_edge_center) - geometry.evaluate(leading_edge_center)) # type: ignore
tip_chord_left = csdl.norm(geometry.evaluate(trailing_edge_left) - geometry.evaluate(leading_edge_left)) # type: ignore
tip_chord_right = csdl.norm(geometry.evaluate(trailing_edge_right) - geometry.evaluate(leading_edge_right)) # type: ignore

spanwise_direction_left = geometry.evaluate(quarter_chord_left) - geometry.evaluate(quarter_chord_center)
spanwise_direction_right = geometry.evaluate(quarter_chord_right) - geometry.evaluate(quarter_chord_center)
sweep_angle_left = csdl.arctan(-spanwise_direction_left[0] / spanwise_direction_left[1]) # type: ignore
sweep_angle_right = csdl.arctan(spanwise_direction_right[0] / spanwise_direction_right[1]) # type: ignore
# endregion Evaluate Parameterization To Define Parameterization Forward Model For Parameterization Solver

# region Set Up and Evaluate Geometry Parameterization Solver
# Define design variables for the optimizer (for the solver, these are desired values)
wingspan_outer_dv = csdl.Variable(shape=(1,), value=np.array([10.0]))
root_chord_outer_dv = csdl.Variable(shape=(1,), value=np.array([1.0]))
taper_ratio_dv = csdl.Variable(shape=(1,), value=np.array([1.0]))
sweep_angle_outer_dv = csdl.Variable(shape=(1,), value=np.array([0.*np.pi/180]))
tip_chord_outer_dv = root_chord_outer_dv*taper_ratio_dv

geometry_solver = ParameterizationSolver()

# Define the states for the parameterization solver (solver will manipulate these to achieve the variables)
geometry_solver.add_state(chord_stretching_b_spline.coefficients)
geometry_solver.add_state(wingspan_stretching_b_spline.coefficients)
geometry_solver.add_state(sweep_translation_b_spline.coefficients)

# Define the geometric variables/constraints that the solver will enforce.
geometric_variables = GeometricVariables()
geometric_variables.add_variable(wingspan, wingspan_outer_dv, penalty_value=None)
geometric_variables.add_variable(root_chord, root_chord_outer_dv, penalty_value=None)
geometric_variables.add_variable(tip_chord_left, tip_chord_outer_dv, penalty_value=None)
geometric_variables.add_variable(tip_chord_right, tip_chord_outer_dv, penalty_value=None)
# geometric_variables.add_variable(sweep_angle_left, sweep_angle_outer_dv, penalty_value=None)
# geometric_variables.add_variable(sweep_angle_right, sweep_angle_outer_dv, penalty_value=None)
geometric_variables.add_variable(-spanwise_direction_left[0], csdl.tan(sweep_angle_outer_dv) * spanwise_direction_left[1], penalty_value=None)
geometric_variables.add_variable(spanwise_direction_right[0], csdl.tan(sweep_angle_outer_dv) * spanwise_direction_right[1], penalty_value=None)

print("Wingspan: ", wingspan.value) # type: ignore
print("Root Chord: ", root_chord.value) # type: ignore
print("Tip Chord Left: ", tip_chord_left.value) # type: ignore
print("Tip Chord Right: ", tip_chord_right.value) # type: ignore
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi) # type: ignore
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi) # type: ignore

# geometry.plot()
geometry_solver.evaluate(geometric_variables)
# geometry.plot()

print()
print("Wingspan: ", wingspan.value) # type: ignore
print("Root Chord: ", root_chord.value) # type: ignore
print("Tip Chord Left: ", tip_chord_left.value) # type: ignore
print("Tip Chord Right: ", tip_chord_right.value) # type: ignore
print("Sweep Angle Left: ", sweep_angle_left.value*180/np.pi) # type: ignore
print("Sweep Angle Right: ", sweep_angle_right.value*180/np.pi) # type: ignore
print("Chord Stretching: ", chord_stretching_b_spline.coefficients.value) # type: ignore
print("Wingspan Stretching: ", wingspan_stretching_b_spline.coefficients.value) # type: ignore
print("Sweep Translation: ", sweep_translation_b_spline.coefficients.value) # type: ignore
# endregion Setup and Evaluate Geometry Parameterization Solver

geometry_coefficients = geometry.stack_coefficients()
geometry_coefficients.add_name('geometry_coefficients')
# geometry_coefficients.save()

planform_area = (root_chord_outer_dv + tip_chord_outer_dv) / 2 * wingspan_outer_dv
planform_area.add_name('planform_area')
planform_area_0 = planform_area.value

asdf = geometry.plot(show=False)

def _flatten_plot_items(plot_output):
    if isinstance(plot_output, list):
        items = []
        for item in plot_output:
            items.extend(_flatten_plot_items(item))
        return items
    return [plot_output]

def _to_pyvista_mesh_items(plot_output):
    """Convert geometry.plot() outputs to (mesh, kwargs) items for plotter.add_mesh()."""
    plot_items = _flatten_plot_items(plot_output)
    mesh_items = []

    def _with_normals(dataset):
        if hasattr(dataset, 'compute_normals'):
            try:
                return dataset.compute_normals(
                    cell_normals=False,
                    point_normals=True,
                    split_vertices=False,
                    inplace=False,
                )
            except Exception:
                return dataset
        return dataset

    for item in plot_items:
        mesh = None
        kwargs = {}

        if isinstance(item, dict) and 'mesh' in item:
            mesh = item['mesh']
            kwargs = dict(item.get('kwargs', {}))
        elif isinstance(item, tuple) and len(item) == 2:
            mesh, raw_kwargs = item
            kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        else:
            mesh = item.dataset if hasattr(item, 'dataset') else item

        if mesh is None:
            continue

        try:
            wrapped = pv.wrap(mesh)
        except Exception:
            continue

        if isinstance(wrapped, pv.MultiBlock):
            for block in wrapped:
                if block is not None and getattr(block, 'n_points', 0) > 0:
                    mesh_items.append((_with_normals(block), dict(kwargs)))
        elif getattr(wrapped, 'n_points', 0) > 0:
            mesh_items.append((_with_normals(wrapped), kwargs))

    return mesh_items


# importing geometry

# fname = 'beam_stability_no_airfoil_shape_stab_tight_opt_SLSQP_1_missions_2'
# final_ind = 131

# fname = 'beam_stability_no_airfoil_shape_stab_tight_opt_SLSQP_1_missions_1_MS'
# final_ind = 60

fname = 'NACA0012_opt_demo_2'
final_ind = 9

full_fname = fname

# initial geometry
vars = csdl.inline_import('../' + full_fname + '.hdf5', f'iteration_{0}')
geometry_coefficients = vars['geometry_coefficients'].value
geometry.unstack_coefficients(geometry_coefficients)

plot_init = geometry.plot(opacity=0.25, color='yellow', show=False, screenshot='aaa.png')

plotter = pv.Plotter(off_screen=True)
plot_init_geometry = _to_pyvista_mesh_items(plot_init)
for mesh, mesh_kwargs in plot_init_geometry:
    render_kwargs = dict(mesh_kwargs)
    render_kwargs.setdefault('show_edges', False)
    render_kwargs.setdefault('smooth_shading', True)
    plotter.add_mesh(mesh, **render_kwargs)

camera_obj = plotter.camera

viewup = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}.get("z")

camera_obj.SetViewUp(*viewup)

camera_obj.roll = 0.
plotter.camera = camera_obj

camera = {
    'position': (-30, -25, 50),
    'focal_point': (15, -5, 5),
    'viewup': (0, 0, 1),
    'distance': 20,
}

camera = {
    'position': (-14.524996400871494, -6.587320545830874, 17.063483701110826),
    'focal_point': (4.721129598625581, 1.9665132317233698, -2.182642298386252),
    'viewup': (0, 0, 1),
    'distance': 20,
}

plotter.camera_position = [camera['position'], camera['focal_point'], camera['viewup']]
# plotter.camera = camera

plotter.show(title='test', screenshot='test.png')

print("Camera position:", plotter.camera.position)
print("Focal point:    ", plotter.camera.focal_point)
print("Azimuth:        ", plotter.camera.azimuth)
print("Elevation:      ", plotter.camera.elevation)
print("Roll:           ", plotter.camera.roll)

# If you specifically want the view angle (degrees):
print("View angle:     ", plotter.camera.GetViewAngle())

# exit()

plotter = pv.Plotter(off_screen=True)
plotter.set_background('white')

camera = {
    'position': (-30, -25, 50),
    'focal_point': (15, -5, 5),
    'viewup': (0, 0, 1),
    'distance': 20,
}

camera = {
    'position': (-14.524996400871494, -6.587320545830874, 17.063483701110826),
    'focal_point': (4.721129598625581, 1.9665132317233698, -2.182642298386252),
    'viewup': (0, 0, 1),
    'distance': 20,
}

video_writer = None
fps = 2

plotter.open_movie(f'{fname}_ani.mp4', quality=8, framerate=2)
plotter.show(auto_close=False)

for i in range(final_ind+1):
    print(f'working on frame {i} of {final_ind}')
    vars = csdl.inline_import('../' + full_fname + '.hdf5', f'iteration_{i}')
    geometry_coefficients = vars['geometry_coefficients'].value
    geometry.unstack_coefficients(geometry_coefficients)

    plot_iter = geometry.plot(opacity=0.25, color='blue', show=False)
    geometry_mesh_items = _to_pyvista_mesh_items(plot_iter+plot_init_geometry)

    plotter.clear()
    for mesh, mesh_kwargs in geometry_mesh_items:
        render_kwargs = dict(mesh_kwargs)
        render_kwargs.setdefault('show_edges', False)
        render_kwargs.setdefault('smooth_shading', True)
        plotter.add_mesh(mesh, **render_kwargs)
    plotter.add_text(
        f"iteration {i}",
        position='lower_left',
        font_size=16,
        color='black',
        shadow=False,
    )
    plotter.camera_position = [camera['position'], camera['focal_point'], camera['viewup']]
    plotter.render()
    plotter.write_frame()

    frame_rgb = plotter.screenshot(return_img=True)

#     if video_writer is None:
#         frame_height, frame_width = frame_rgb.shape[:2]
#         video_writer = cv2.VideoWriter(
#             str(video_path),
#             cv2.VideoWriter_fourcc(*'mp4v'),
#             fps,
#             (frame_width, frame_height),
#         )
#         if not video_writer.isOpened():
#             raise RuntimeError(f"Failed to create OpenCV VideoWriter for {video_path}")

#     frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
#     video_writer.write(frame_bgr)

# video_writer.release()
plotter.close()