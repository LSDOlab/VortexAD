---
title: Potential flow
---

<!-- Flow fields can generally be described by the Navier-Stokes equations, a set of complex nonlinear 
partial differential equations that relate different flow properties such as density, velocity, 
pressure and temperature.

*INCLUDE CONTINUITY AND MOMENTUM EQUATIONS*

Attempting to directly solve some form of these equations numerically is difficult and computationally expensive.
This is essentially what CFD is (which encompasses Euler, RANS, URANS, LES and DNS). CFD is necessary for many
complex flows, such as separated flows, viscous flows, etc ...

Through dimensional analysis, we can define an important quantity of the NS equations called the Reynolds number.
The Reynolds number defines the ratio of the relative magnitudes of the inertial and viscous terms: 

$$Re = \frac{\rho VL}{\mu}$$

where $\rho$ is the flow density, $V$ is the characteristic flow speed, $L$ is the characteristic length, 
and $\mu$ is the dynamic viscosity. In many aircraft-related applications, we are solving high-Reynolds number 
flows, which indicates that the overall influence of vorticity and viscosity are minimal. These two assumptions are 
key to simplify the Navier-Stokes equations. Potential flow utilizes these two approximations.


Potential flow simplifies these equations based on a set of key assumptions that are valid in the high-Reynolds number regime:
the flow field is assumed to be **irrotational** and **inviscid**.



To include one final reduction, we defined the Mach number, which represents the ratio of the flow speed to the sound speed in the medium. In low-speed flows at low Mach, the density changes caused by pressure changes in the flow become negligible, meaning the flow can be treated as incompressible (constant density).

*include equations for these and show how this simplifies the continuity equation*


Irrotational flows mean that there is no local rotation of fluid particles, and can be described mathematically as 

$$\nabla \times \vec{V} = 0 $$

This tells us that there exists a scalar $\phi$ such that $V = \nabla\phi$. Introducing this into the continuity equation, we see that the governing equation for potential flow becomes 

$$\nabla^2\phi = 0$$ -->

<!-- ## Referencing using bib files

You can add references in the `references.bib` file and cite them 
in the page like this {cite:p}`perez2011python`. 
You can also include a list of references cited at the end as shown below.

## Bibliography

```{bibliography} references.bib
``` -->