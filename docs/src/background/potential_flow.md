---
title: Potential flow
---

Flow fields can generally be described by the Navier-Stokes equations, a set of complex nonlinear 
partial differential equations that relate different flow properties. Shown below are the continuity and momentum equations.

<!-- such as density, velocity and pressure.   -->

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \textbf{u}) = 0
$$

$$
\rho \frac{D\textbf{u}}{Dt} = \rho \left( \frac{\partial \textbf{u}}{\partial t} + \textbf{u} \cdot \nabla \textbf{u} \right) = -\nabla p + \mu \nabla^2\textbf{u} + \frac{\mu}{3}\nabla (\nabla \cdot \textbf{u}) + \rho f
$$

Directly solving some form of these equations numerically is difficult and computationally expensive, and exact solutions exist only in few specific cases. Computational fluid dynamics (CFD) (which encompasses Euler, RANS, URANS, LES and DNS) tackles this problem directly to generally solve complex flows that include separation, viscous effects, thermo coupling and changes to flow properties (compressibility, viscosity).

Flow fields around aircraft exhibit many of the above complexities; however, due to the inherent properties of these flow regimes, most of the physics can still be captured with a set of simplifications. 
We can take advantage of incompressible **potential flow**, which model irrotational and inviscid flows, to simplify flow field models with reasonable accuracy and computational cost. We want to take advantage of these simplicities to maintain reasonable accuracy with low computational cost, especially in aircraft design and optimization. 

<!-- We will see shortly that under the assumptions of incompressible, irrotational and inviscid flows, we can solve for flow field behavior We can utilize certain assumptions like ... to take advantage of **potential flow** and model significant fidelity of the flow physics with reasonable accuracy and computational cost -->

Through dimensional analysis, we can define an important quantity of the flow field called the Reynolds number. The Reynolds number defines the ratio of the relative magnitudes of the inertial and viscous terms: 

$$Re = \frac{\rho VL}{\mu}$$

where $\rho$ is the flow density, $V$ is the characteristic flow speed, $L$ is the characteristic length, and $\mu$ is the dynamic viscosity. The effects of viscosity refer to the friction present from flow particles and objects in the flow field. In many aircraft-related applications, we are exploring high-Reynolds number flows, which means that most of the flow field contains minimal influence from viscosity. In these types of flows, the effects of viscosity are reserved to small regions near bodies called boundary layers, where the flow goes to zero velocity at the boundary. Within the boundary layer is also where **vorticity** is usually confined; vorticity is defined mathematically as 

$$
\omega = \nabla \times \textbf{u}
$$

and refers to the *local rotation of fluid or fluid particles*. Some sources use a different convention and define vorticity as $\zeta$ and the angular velocity as $\zeta = 2\omega$. Outside of the boundary layer, both viscosity and vorticity have minimal influence on the overall flow field. As a result, a large area in the flow field can be treated as **inviscid** and **irrotational**, two key simplifications that model potential flows. 

<!-- Irrotational flows mean that there is no local rotation of fluid particles, and can be described mathematically as 

$$\omega = \nabla \times \textbf{u} = 0 $$

This tells us that there exists a scalar $\phi$ such that $\textbf{u} = \nabla\phi$. -->

To include one final simplification, we identify the Mach number, which represents the ratio of the flow speed to the sound speed in the medium:

$$M = \frac{||\textbf{u}_\infty||}{a}$$

where $a$ is the speed of sound and $\textbf{u}_\infty$ is the characteristic flow speed. At low Mach (typically coinciding with low-speed flows), the effects of pressure changes have a minimal influence on how the density of fluid molecules change due to compression. We can then treat the flow field as **incompressible** by assuming a constant density. 

```{note}
In CFD, incompressibility also decouples the dependence of the pressure field on temperature, which subsequently decouples the energy equation in Navier-Stokes as well (not shown above).
```

We have defined three simplifications to the Navier-Stokes equations that are reasonable for most aircraft-related flows that will simplify the flow analysis. 
These three simplifications are key aspects of solving for incompressible potential flow. 
First, we can simplify the continuity equation to

$$
\nabla \cdot \textbf{u} = 0
$$

under the assumption of incompressibility because density is constant. Second, irrotational flows specify that there is no vorticity in the flow:

$$ \omega = \nabla \times \textbf{u} = 0 $$

which allows us to define a scalar quantity $\phi$ such that $\textbf{u} = \nabla \phi$. This quantity $\phi$ is referred to as the **velocity potential**.

```{note}
This relationship is deduced from calculus identities, noting that the **curl of a gradient** is always zero.
```
We can introduce this relationship into the incompressible continuity equation to formulate our governing equation:

<!-- We can formulate our governing equation by inserting this into the incompressible continuity equation: -->

$$
\nabla \cdot \textbf{u} = \nabla \cdot \nabla \phi = \nabla^2 \phi = 0
$$

where we return the *Laplace equation* as our governing equation in the flow field. For potential flow, this equation is satisfied everywhere in space

Finally, the inviscid flow condition is used to simplify the momentum equation by ignoring the viscous terms. At this stage, we will also assume a steady flow with no body or gravitational forces. Our inviscid momentum equation then becomes  

$$
\textbf{u} \cdot \nabla \textbf{u} = -\frac{\nabla p}{\rho}
$$

We can rearrange the left side using product rule:

$$
\frac{1}{2} \nabla||\textbf{u}||^2= \frac{1}{2} \nabla (\textbf{u} \cdot \textbf{u}) = (\nabla \cdot \textbf{u}) \textbf{u} + \textbf{u} \cdot \nabla \textbf{u}
$$
and setting the first term on the right side to zero from continuity. Introducing this formula into our reduced momentum equation, we see that

$$
\nabla \left( \frac{1}{2} ||\textbf{u}||^2 + \frac{p}{\rho} \right) = 0
$$

which tells us that the sum of overall pressure contributions must be constant. We arrive at Bernoulli's equation:

$$
\frac{1}{2} \rho ||\textbf{u}||^2 + p = C
$$

where $C$ represents a constant pressure head along a given streamline. The density term $\rho$ has been moved around to represent each term in Bernoulli's equation as a form of *pressure*. The velocity term represents the **dynamic** pressure, and $p$ represents the **static** pressure. Common forms of Bernoulli's equation include a gravitational contribution $\rho gh$ to represent fluid pressure changes due to elevation. This term is often neglected for aircraft analysis because the pressure changes from vertical displacements of fluids around bodies is small compared to the static and dynamic pressure terms. 

<!-- ## Referencing using bib files

You can add references in the `references.bib` file and cite them 
in the page like this {cite:p}`perez2011python`. 
You can also include a list of references cited at the end as shown below.

## Bibliography

```{bibliography} references.bib
``` -->