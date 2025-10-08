---
title: In Development
---

This page highlights areas of current development,
separated into relevant categories.

### Flow physics

Unsteady free-wake flow solver

Neumann-BC solver
- VortexAD only supports a Dirichlet boundary condition formulation. The Neumann no-penetration BC is better suited for unsteady flows.

VLM solver
- we have written a VLM code and are working on including it in this repository. 

### Geometry and meshing
Support for unstructured quad meshes and quad+tri meshes
- VortexAD currently only supports structured grids with quads and unstructured tri meshes. 