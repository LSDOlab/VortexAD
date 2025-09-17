# Inputs

VortexAD uses an input dictionary to pass information to the main solver. The solver inputs build onto a default input dictionary; it is up to the user to provide any updated inputs in a python dictionary format. The input dictionary is stored above the `PanelMethod` class [here](https://github.com/LSDOlab/VortexAD/blob/main/VortexAD/core/pm_class.py#L15-L57), and is also shown below. 

```python
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
    'partition_size': 1,

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
    'drag_type': 'pressure' # pressure or Trefftz (not implemented yet)
}
```

The table below summarizes each input. This work is still in development; any inputs noted with (WIP) signify that they are a work in progress.


| Key | Meaning | Default | Units | Shape/Type | Options |
|:---|:---|:---|:---|:---|:---|
| V_inf | Freestream velocity | 10 | m/s | scalar/vector/tensor | N/A |
| Mach | Mach number | None | None | scalar/vector/tensor | N/A |
| sos | Speed of sound | 340.3 | m/s | scalar/vector/tensor | N/A |
| alpha | Inflow angle | None | degrees | scalar/vector/tensor | N/A |
| rho | Fluid density | 1.225 | kg/m^3 | scalar/vector/tensor | N/A |
| compressibility | Compressibility <br/> correction | False | N/A | string | (False, True) |
| Cp cutoff | Pressure coefficient <br/> minimum cutoff | -5. | None | scalar | N/A |
| mesh_path | Path to mesh file | [path to this file](https://github.com/LSDOlab/VortexAD/blob/main/VortexAD/core/geometry/sample_meshes/pm/naca0012_LE_TE_cluster.stl) | N/A | string | N/A |
| mesh_mode | Mesh type | unstructured | N/A | string | (structured, unstructured) |
| BC | Boundary condition | Dirichlet | N/A | string | (Dirichlet, Neumann (WIP)) |
| higher_order  (WIP)| Higher-order methods | False | N/A | bool | (False, True) |
| solver_mode | Steady or unsteady | 'steady' | N/A | string | ('steady', 'unsteady' (WIP)) |
| wake_mode | Mode for wake | 'fixed' | N/A | string | ('fixed', 'prescribed', 'free') |
| partition_size | AIC assembly <br/> partition size | 1 | N/A | int | [1, total panels] |
| iterative (WIP)| Flag to toggle <br/> GMRES iterative solver | False | N/A | bool | (False, True) |
| ROM | Flag to toggle ROMs | False | N/A | (string, bool) | (False, 'ROM-POD', 'ROM-Krylov' (WIP)) |
| reuse_AIC | Flag to reuse AIC <br/> across evaluations | False | N/A | bool | (False, True) |
| ref_area | Reference area <br/> for coefficients | 10. | m^2 | scalar | N/A |
| ref_chord | Reference chord <br/> for coefficients | 1. | m | scalar | N/A |
| moment_reference | Moment reference point | np.array([0., 0., 0.]) | m | (3,) | N/A |
| drag_type | Method to compute drag | 'pressure' | N/A | string | ('pressure', 'Trefftz' (WIP)) |
<!-- | Mach | Flow properties <br/> asdf | 10. m/s | scaler or tensor | | -->