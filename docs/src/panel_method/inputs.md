# Inputs

VortexAD uses an input dictionary to pass information to the main solver. The solver inputs build onto a default input dictionary; it is up to the user to provide any updated inputs in a python dictionary format. The input dictionary is stored above the `PanelMethod` class [here](https://github.com/LSDOlab/VortexAD/blob/main/VortexAD/core/pm_class.py#L15-L57), and is also shown below. 

```python
default_input_dict = {
    # flow properties
    'V_inf': None,                          # m/s
    'Mach': None,                           # Unitless 
    'sos': 340.3,                           # m/s, 
    'alpha': None,                          # user can provide grid of velocities as well
    'rho': 1.225,                           # density (kg/m^3)
    'nu': 1.46e-5,                          # kinematic viscosity (m^2/s) 
    'compressibility': False,               # PG correction
    'Cp cutoff': -5.,                       # minimum Cp (numerical reasons)
    'nwpcl': 30,                            # Number of Wake Propagation (reference) Chord Lengths 

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
    'partition_size': 1,                    # for full vectorization, set to None

    # GMRES linear system solve
    'iterative': False,
    
    # ROM options
    'ROM': False,                           # 'ROM-POD or ROM-Krylov
    'ROM_basis': None,                      # basis for ROM (POD or other offline-trained basis)

    # reusing AIC (no alpha dependence on wake) --> this only applies to fixed wake
    'reuse_AIC': False,

    # others
    'ref_area': 10.,                        # reference area (l^2, l being the input length unit)
    'ref_chord': 1.,                        # reference chord
    'moment_reference': np.zeros(3),        # reference point for moment computation
    'drag_type': 'Trefftz',                 # pressure or Trefftz
    'Trefftz_integration_pts': 5,           # number of Trefftz plane evaluation points per wake element column

    # steady solver wake relaxation parameters
    'wake_relaxation': False,               # wake relaxation flag
    'K': 0.5,                               # relaxation parameter, between 0.5 and 5
    'num_wake_planes': 5,                   # number of wake planes 
    'num_wake_iterations': 3,               # number of wake iterations 


    # unsteady solver
    'dt': 0.1,                              # time step (s)
    'nt': 10,                               # number of time steps
    'store_state_history': True,            # flag to store state history
    'core_radius': 1.e-3,                   # vortex core radius
    'vc_parameters': [1.25643, 0, 2.5],     # alpha, a1, bqs from core model (referred to in paper)
    'free_wake': False,
    'dissipation': False,
    'integration_method': 'ForwardEuler'
}
```

The table below summarizes each input. This work is still in development; any inputs noted with (WIP) signify that they are a work in progress.


| Key | Meaning | Default | Units | Shape/Type | Options |
|:---|:---|:---|:---|:---|:---|
| V_inf | Freestream velocity | 10 | m/s | scalar/vector/tensor | N/A |
| Mach | Mach number | None | None | scalar/vector/tensor | N/A |
| sos | Speed of sound | 340.3 | m/s | scalar/vector/tensor | N/A |
| alpha | Inflow angle | None | degrees | scalar/vector/tensor | N/A |
| rho | Density | 1.225 | kg/m^3 | scalar/vector/tensor | N/A |
| nu | Kinematic viscosity | 1.46e-5 | m^2/s | scalar/vector/tensor | N/A |
| compressibility | Compressibility <br/> correction | False | N/A | string | (False, True) |
| Cp cutoff | Pressure coefficient <br/> minimum cutoff | -5. | None | scalar | N/A |
| nwpcl | Wake propagation <br/> reference chord lengths | 30 | None | int | N/A |
| mesh_path | Path to mesh file | [path to this file](https://github.com/LSDOlab/VortexAD/blob/main/VortexAD/core/geometry/sample_meshes/pm/naca0012_LE_TE_cluster.stl) | N/A | string | N/A |
| mesh_mode | Mesh type | unstructured | N/A | string | (structured, unstructured) |
| collocation_velocity | Collocation velocity | False | m/s | vector/tensor | N/A |
| BC | Boundary condition | Dirichlet | N/A | string | (Dirichlet, Neumann) |
| higher_order  (WIP)| Higher-order methods | False | N/A | bool | (False, True) |
| solver_mode | Steady or unsteady | 'steady' | N/A | string | ('steady', 'unsteady') |
| wake_mode | Mode for wake | 'fixed' | N/A | string | ('fixed', 'prescribed', 'free') |
| partition_size | AIC assembly <br/> partition size | 1 | N/A | int | [1, total panels] |
| iterative (WIP)| Flag to toggle <br/> GMRES iterative solver | False | N/A | bool | (False, True) |
| ROM | Flag to toggle ROMs | False | N/A | (string, bool) | (False, 'ROM-POD', 'ROM-Krylov' (WIP)) |
| ROM_basis | offline-trained <br/> ROM basis | None | N/A | list | N/A |
| reuse_AIC | Flag to reuse AIC <br/> across evaluations | False | N/A | bool | (False, True) |
| ref_area | Reference area <br/> for coefficients | 10. | m^2 | scalar | N/A |
| ref_chord | Reference chord <br/> for coefficients | 1. | m | scalar | N/A |
| moment_reference | Moment reference point | np.array([0., 0., 0.]) | m | (3,) | N/A |
| drag_type | Method to compute drag | 'pressure' | N/A | string | ('pressure', 'Trefftz') |
| Trefftz_integration_pts | Trefftz plane <br/> integration points <br/> per wake element  | 5 | N/A | int | N/A |
| wake_relaxation | wake relaxation flag  | False | N/A | bool | (False, True (WIP)) |
| K | relaxation parameter  | 0.5 | N/A | float | N/A |
| num_wake_planes | number of wake planes  | 5 | N/A | int | N/A |
| num_wake_iterations | wake iterations  | 3 | N/A | int | N/A |
| dt | time step  | 0.1 | s | float | N/A |
| nt | number of time steps | 10 | N/A | int | N/A |
| store_state_history | storing Ozone <br/> state history  | True | N/A | bool | (True, False) |
| core_radius | vortex core radius  | 1.e-3 | m | float | N/A |
| vc_parameters | vortex core parameters <br/> $ \alpha, \text{a1, bqs} $ (see [here](https://vortexad.readthedocs.io/en/latest/src/background/pm_background/unsteady_wake.html)) | [1.25643, 0, 2.5] | N/A | list | N/A |
| free_wake | toggle free wake  | False | N/A | bool | (True, False) |
| dissipation | toggle wake dissipation | False | N/A | bool | (True, False) |
| integration_method | time integration method  | 'ForwardEuler' | N/A | str | ('ForwardEuler') |
