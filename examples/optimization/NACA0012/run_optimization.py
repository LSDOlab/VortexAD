import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from geometry_function import setup_geometry_parameterization

from modopt import CSDLAlphaProblem, PySLSQP
import lsdo_geo
import VortexAD
import meshio
import pickle

import resource
import time
from datetime import datetime

lfs.num_workers=1

recorder = csdl.Recorder(inline=True)
recorder.start()

imported_function_set = lfs.import_file_patched(file_name="rectangular_wing_naca0012_10ar.stp", parallelize=False)
geometry = lsdo_geo.Geometry(functions=imported_function_set.functions, 
                                      function_names=imported_function_set.function_names,
                                      name='imported_geometry',
                                      space=imported_function_set.space)

# geometry.plot() # plots geometry using pyvista

# region panel mesh import
mesh = meshio.read("rectangular_wing_naca0012_10ar.msh")

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

geometry, DV_dict = setup_geometry_parameterization(geometry)

root_twist_dv = DV_dict['root_twist_dv']
tip_twist_dv = DV_dict['tip_twist_dv']
wingspan_outer_dv = DV_dict['wingspan_outer_dv']
taper_ratio_dv = DV_dict['taper_ratio_dv']

root_chord_outer_dv = DV_dict['root_chord_outer_dv']
tip_chord_outer_dv = DV_dict['tip_chord_outer_dv']

geometry_coefficients = geometry.stack_coefficients()
geometry_coefficients.add_name('geometry_coefficients')
# geometry_coefficients.save()

planform_area = (root_chord_outer_dv + tip_chord_outer_dv) / 2 * wingspan_outer_dv
planform_area.add_name('planform_area')
planform_area_0 = planform_area.value

# region Aerodynamics

recorder.inline = False
# geometry needs inline to be true

cruise_speed = csdl.Variable(value=50.)
velocity = csdl.concatenate([cruise_speed, csdl.Variable(value=0.), csdl.Variable(value=0.)])

num_nodes = 1

panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')
panel_mesh.add_name('panel_mesh')

panel_centers = geometry.evaluate(projected_panel_centers, plot=False)
panel_centers.add_name('panel_centers')

point_velocities = csdl.expand(-velocity, (num_nodes,) + panel_mesh.shape[1:], 'j->iaj')
rho_array = csdl.Variable(shape=(num_nodes,), value=np.array([1.225]))
sos_array = csdl.Variable(shape=(num_nodes,), value=np.array([343.0]))

pm_solver_inputs = {
    'V_inf': point_velocities,
    'rho': rho_array,
    'sos': sos_array,
    'compressibility': True,
    'Cp cutoff': -5.,
    'partition_size': 1,
    # 'reuse_AIC': True,
    # 'mesh_path': file_path+file_name, # already done externally
    'ref_area': planform_area # does not matter bc we don't use the coefficients,
}
# we leave out the mesh path because we need FFD to move the mesh

panel_method = VortexAD.PanelMethod(
    solver_input_dict=pm_solver_inputs,
    skip_geometry=True # not running geometry
)
# inserting grid data from above
panel_method.insert_grid_data(
    mesh=panel_mesh[0,:],
    cell_adjacency_data=cell_adjacency_data,
    TE_properties=TE_properties
)

panel_method.declare_outputs([
    'Cp',
    'L',
    'Di',
    'M',
    'panel_forces',
    'CL',
    # 'CDi',
    'CDi_Trefftz',
    'CM',
])
outputs = panel_method.evaluate()

CL = outputs['CL']
CDi = outputs['CDi_Trefftz']

Cp = outputs['Cp']
Cp.add_name('Cp')

# endregion



# region optimization parameters
root_twist_dv.set_as_design_variable(lower=0., upper=5.)
root_twist_dv.add_name('root_twist')

tip_twist_dv.set_as_design_variable(lower=-5., upper=5.)
tip_twist_dv.add_name('tip_twist')

# added for attempt 2
wingspan_outer_dv.set_as_design_variable(lower=5., upper=15.)
wingspan_outer_dv.add_name('wingspan')

taper_ratio_dv.set_as_design_variable(lower=0.1, upper=1.5)
taper_ratio_dv.add_name('taper_ratio')


CL.set_as_constraint(equals=0.2, scaler=1e1)
CL.add_name('CL')

# added for attempt 2
planform_area.set_as_constraint(equals=planform_area_0, scaler=1e-1)

CDi.set_as_objective(scaler=1e3)
CDi.add_name('CDi')

# endregion

# region setting up optimization and testing structure
csdl.save_optimization_variables()

fname = f'NACA0012_opt_demo'

testing = False
check_derivs = False

# if testing is True, then code exits before optimization setup

if not check_derivs:
    planform_area.save()
    panel_mesh.save()
    panel_centers.save()
    geometry_coefficients.save()
    Cp.save()

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    # additional_inputs=[wingspan_outer_dv, root_chord_outer_dv, tip_chord_outer_dv, sweep_angle_outer_dv, tip_twist_dv],
    # additional_outputs=[panel_mesh],
    gpu=False,
    save_on_update=True, 
    filename=fname, 
    output_saved=True
)

def get_peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On Mac, ru_maxrss is in bytes
    return usage.ru_maxrss / (1024 ** 2)  # convert to MB

start_time = time.time()
jax_sim.run()
end_time = time.time()
print("Simulator compile + run time: ", end_time - start_time)

if testing:
    start_time = time.time()
    jax_sim.run()
    end_time = time.time()
    print("Simulator run time: ", end_time - start_time)

    print(f"Peak memory so far (after sim.run()): {get_peak_memory_mb():.2f} MB")

    if check_derivs:
        print('starting compile of check totals')
        start_time = time.time()
        asdf = jax_sim.check_totals(step_size=1.e-6)
        end_time = time.time()
        print(f'check totals run+compile time: {end_time-start_time} seconds') 
        print(f"Peak memory so far (after sim.check_totals()): {get_peak_memory_mb():.2f} MB")

    exit()

print('setting up Problem')

now = datetime.now()
prob_start_date = now.date()
prob_start_time = now.time()
print('================')
print(f'prob start date: {prob_start_date}')
print(f'prob start time: {prob_start_time}')
print('================')

print(f'file name: {fname}')

start_problem_time = time.time()
prob = CSDLAlphaProblem(problem_name=fname, simulator=jax_sim)
end_problem_time = time.time()
print(f'problem compile+run time: {end_problem_time-start_problem_time} seconds')

now = datetime.now()
prob_end_date = now.date()
prob_end_time = now.time()
print('================')
print(f'prob end date: {prob_end_date}')
print(f'prob end time: {prob_end_time}')
print('================')

print(f"Peak memory so far (after problem compile): {get_peak_memory_mb():.2f} MB")

print('setting up optimizer')
start_opt_setup = time.time()
# optimizer = PySLSQP(prob, solver_options={'maxiter':300, 'acc':1e-5})
save_vars_list = ['x', 'objective', 'optimality', 'feasibility', 
                    'step', 'mode', 'iter', 'majiter', 'ismajor', 
                    'constraints', 'gradient', 'multipliers', 'jacobian']
SLSQP_solver_options={
    'maxiter':100, 
    'acc':1e-6,
    # 'save_itr': 'all',
    # 'save_vars': save_vars_list,
    # 'save_filename': fname + '_optimizer_data.hdf5',
}

optimizer = PySLSQP(
    prob, 
    solver_options=SLSQP_solver_options
)

end_opt_setup = time.time()
print(f'optimizer setup time: {end_opt_setup-start_opt_setup} seconds')

now = datetime.now()
opt_start_date = now.date()
opt_start_time = now.time()
print('================')
print(f'optimizer start date: {opt_start_date}')
print(f'optimizer start time: {opt_start_time}')
print('================')

print('running optimizer')
opt_start = time.time()
optimizer.solve()
opt_end = time.time()

now = datetime.now()
opt_end_date = now.date()
opt_end_time = now.time()
print('================')
print(f'optimizer end date: {opt_end_date}')
print(f'optimizer end time: {opt_end_time}')
print('================')

success = optimizer.results['success']
print('\tTime:', opt_end - opt_start)
print('\tSuccess:', success)
# print('\tOptimized vars:', optimizer.results['x'])
print('\tOptimized obj:', optimizer.results['objective'])