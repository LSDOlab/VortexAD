# Outputs

The panel method in VortexAD offers a set of outputs to select from. Lie the inputs, the solver output dictionary can be found above the `PanelMethod` class [here](https://github.com/LSDOlab/VortexAD/blob/main/VortexAD/core/pm_class.py#L15-L57). It is also shown below. The values corresponding to each output key represent the output shapes for the (unstructured, structured) grids. These outputs can be printed to the terminal window by calling `PanelMethod.print_output_options()`.

```python
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
}
```
The table below summarizes each outout. The shape of each output is dictated by the number of parallel/vectorized analyses (`num_nodes`) and the number of mesh panels (`num_panels`).

| Key | Meaning | Units | Shape |
|:---|:---|:---|:---|
| `CL` | Lift coefficient | None | `(num_nodes,)` |
| `CDi` | Induced drag <br/> coefficient | None | `(num_nodes,)` |
| `CM` | Moment coefficient | None | `(num_nodes,3)` |
| `L` | Lift force | `N` | `(num_nodes,)` |
| `Di` | Induced drag <br/> force | `N` | `(num_nodes,)` |
| `M` | Moment | `Nm` | `(num_nodes,3)` |
| `Cp` | Pressure coefficient <br/> distribution | None | `(num_nodes, num_panels)` |
| `panel_forces` | Panel forces | `N` | `(num_nodes, num_panels, 3)` |
| `L_panel` | Panel lift forces | `N` | `(num_nodes,num_panels)` |
| `Di_panel` | Panel induced <br/> drag forces | `N` | `(num_nodes, num_panels)` |
| `V_mag` | Surface velocity <br/> magnitude | `m/s` | `(num_nodes, num_panels)` |