# Unsteady ODE formulation

In this subsection, we describe our ODE formulation for the wake-shedding problem.
The two ODE states are the wake nodes $x_w$ and the wake doublet strengths $\mu_w$.
Given the equivalence of the doublet and vortex ring induced velocities, the ODE formulation also applies to the wake vortex rings $\Gamma_w$.
For the sake of simplicity, we denote the wake vortex strengths using $\mu_w$.

% With our graph-based framework, we must preallocate the states across all time steps.
Graph-based modeling is well-suited for efficient ODE time integration.
For each iteration, the sequence of operations to compute state derivatives is the same. 
Thus, the computational graph can be reused.
A consequence of reusing the computational graph is that the states and dynamic parameters must be preallocated across all time steps. 
As a result, we store inactive data for the wake states that have yet to shed into the flow.
To access the active wake data at each time step, we define activation arrays that function as a continuous switch to the time derivatives.
These activation arrays use zero-based indexing, like Python.

We solve the ODE using a Lagrangian perspective, where states are tracked along the wake particles and panels.
The equivalent Eulerian perspective would be to shift the ODE states along an array as the wake sheds downstream.
However, in developing this framework, our numerical experiments showed that the accumulation of finite difference error led to significant wake instability.

Before outlining our formulation, we present basic nomenclature.
The total number of time steps is denoted $nt$, while the time step index is denoted $n$. 
For each timestep, the wake node derivatives are defined as:
<!-- \begin{equation}
\frac{dx_w}{dt}[n,:] = \dot{x}_w[n,:] = V_{w}[n,:]A_{x}[n,:],
\end{equation} -->
$$
\frac{dx_w}{dt}[n,:] = \dot{x}_w[n,:] = V_{w}[n,:]A_{x}[n,:],
$$
where $V_{w} = V_{\infty} + V_i$ represents the total wake velocity vector as a sum of the freestream and induced free-wake velocities. 
We define a velocity activation array:
<!-- \begin{equation}
\label{eq:vel_activation}
A_x[n,i,:] = \begin{cases}
% 1 & [-(1+n):]\\
% 1 & nt-n < i < nt\\
1 & i \in [nt-n-1, nt-1]\\
0 & \text{else} \\
\end{cases},
\end{equation} -->
$$
A_x[n,i,:] = \begin{cases}
% 1 & [-(1+n):]\\
% 1 & nt-n < i < nt\\
1 & i \in [nt-n-1, nt-1]\\
0 & \text{else} \\
\end{cases},
$$
to identify which row of wake panels has been shed, based on the time step index $n$.
We apply a similar logic to the wake doublet strengths using activation arrays.
At each timestep, the wake doublet derivatives are defined as
<!-- \begin{equation}
\label{eq:wake_deriv}
\frac{d\mu_w}{dt}[n,:] = \dot{\mu}_w[n,:] = \frac{\mu_{TE}[n,:]}{\Delta t}A_{\mu}[n,:] + \beta[n,:],
\end{equation} -->
$$
\frac{d\mu_w}{dt}[n,:] = \dot{\mu}_w[n,:] = \frac{\mu_{TE}[n,:]}{\Delta t}A_{\mu}[n,:] + \beta[n,:],
$$
where changes to the wake doublet field depend on the Kutta condition and vortex dissipation.
The Kutta condition derivative is responsible for generating lifting flow, while the vortex dissipation models energy losses in the wake.
First, we discuss the Kutta condition derivative.
We formulate a finite-difference approximation to enforce the Kutta condition, where $\mu_{TE}$ is the expected wake doublet strength defined by the Kutta condition.
This term applies only to the newest row of wake panels.
Thus, we define a sparse activation array for the Kutta condition derivative as
<!-- \begin{equation}
\label{eq:TE_activation}
A_{\mu}[n,i] = \begin{cases}
1 & i = nt-n-2\\
0 & \text{else}\\
\end{cases},
\end{equation} -->
$$
A_{\mu}[n,i] = \begin{cases}
1 & i = nt-n-2\\
0 & \text{else}\\
\end{cases},
$$
where the only non-zero index corresponds to the index of the newest row of wake panels.
In the absence of dissipation, the wake doublet strengths remain constant once they are shed.
The dissipation derivatives $\beta$ are applied to the remaining shed wake panels and are modeled as
<!-- \begin{equation}
\label{eq:diss_term}
\beta[n,:] = \begin{cases}
\dot{\mu}_{diss}[n,:]B_{\mu}[n,:] & \text{yes dissipation}\\
0 & \text{no dissipation} \\
\end{cases},
\end{equation} -->
$$
\beta[n,:] = \begin{cases}
\dot{\mu}_{diss}[n,:]B_{\mu}[n,:] & \text{yes dissipation}\\
0 & \text{no dissipation} \\
\end{cases},
$$
where
<!-- \begin{equation}
\label{eq:diss_deriv}
% \dot{\mu}_{diss}[n,:] = \mu_w[n,:] \frac{\text{exp} \left(- \frac{bq}{s}\Delta t \right) - 1}{\Delta t}
\dot{\mu}_{diss}[n,:] = -\frac{bq}{s}\mu_w[n,:]
\end{equation} -->
$$
\dot{\mu}_{diss}[n,:] = -\frac{bq}{s}\mu_w[n,:]
$$
is the time derivative of our empirical vortex dissipation formulation, mentioned in the previous section.
We define a final activation array for the dissipation derivative as
<!-- \begin{equation}
\label{eq:diss_activation}
B_{\mu}[n,i] = \begin{cases}
1 & i \in [nt-n-2, nt-2] \\
0 & \text{else} \\
\end{cases},
\end{equation} -->
$$
B_{\mu}[n,i] = \begin{cases}
1 & i \in [nt-n-2, nt-2] \\
0 & \text{else} \\
\end{cases},
$$
where the indices containing 1 correspond to wake rows that have already been shed.
As a result, the dissipation derivative is not applied to wake elements that are pre-allocated but have yet to shed from the surface.

We use [Ozone](https://github.com/LSDOlab/ozone), an open-source Python library for solving ordinary differential equations, to solve our ODE {cite:p}`sperry2025ozone`. Ozone is developed using CSDL, so adjoint-based sensitivities are automatically propagated through the ODE solver.
Ozone offers various integration techniques, such as time-marching and collocation methods.
In addition, Ozone offers checkpointing for efficient memory management.
We use a time-marching procedure for the wake-shedding problem.
Besides the Kutta condition derivative for the wake doublet strengths, the ODE formulation uses analytical derivatives to update the ODE states.

## Bibliography

```{bibliography} ../../references.bib
```