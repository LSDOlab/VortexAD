# Unsteady wake
For unsteady flows, we propagate the wake in time using a free-wake method.
Free-wake methods incorporate surface-and-wake-induced velocities into the wake-shedding process to capture realistic flow behavior, such as wing-tip vortices.
The induced velocities are computed using the source, doublet, and wake doublet strengths. 
The total velocity of a wake element can be written as 
<!-- \begin{equation}
    \vec{V}_{w} = \vec{V}_{\infty} + \vec{v}_{wi} 
    \label{eq:wake_velocity}
\end{equation} -->
$$
\vec{V}_{w} = \vec{V}_{\infty} + \vec{v}_{wi},
$$
where the induced velocity of the wake elements is
<!-- \begin{equation}
    \label{eq:wake_vel_ind}
    \vec{v}_{wi} = A\mu + B\sigma + C\mu_w,
\end{equation} -->
$$
\vec{v}_{wi} = A\mu + B\sigma + C\mu_w,
$$
and the tensors $A, B, C$ correspond to the unit-strength induced velocity of the surface doublets, surface sources, and wake doublets, respectively. <!-- We recycle the matrix/tensor notation from Equation \ref{eq:lin_sys} to represent induced velocities. -->
The induced velocities for each source term in the global reference frame can be found in {cite:p}`maskew1987program`.
However, the induced velocities of doublets and vortex rings are identical.
We compute the induced velocities of doublets and vortex rings using the Biot-Savart law:
<!-- \begin{equation}
\vec{q}_{ab} = 
    \frac{\Gamma}{4 \pi} 
    \frac{\boldsymbol{r_a} \times \boldsymbol{r_b}}
            {\|\boldsymbol{r_a} \times \boldsymbol{r_b}\|^2} 
    \boldsymbol{r_{o}} \cdot 
    \left( \frac{\boldsymbol{r_a}}{\|\boldsymbol{r_b}\|}- \frac{\boldsymbol{r_b}}{\|\boldsymbol{r_b}\|} \right),
    \label{eq:bs_law}
\end{equation} -->
$$
\vec{q}_{ab} = 
    \frac{\Gamma}{4 \pi} 
    \frac{\boldsymbol{r_a} \times \boldsymbol{r_b}}
            {\|\boldsymbol{r_a} \times \boldsymbol{r_b}\|^2} 
    \boldsymbol{r_{o}} \cdot 
    \left( \frac{\boldsymbol{r_a}}{\|\boldsymbol{r_b}\|}- \frac{\boldsymbol{r_b}}{\|\boldsymbol{r_b}\|} \right),
$$
where each element is decomposed into vortex line elements.
This computation is repeated for each edge of the panel. <!-- The total wake velocity in Equation \ref{eq:wake_velocity} is then used to update the wake locations for the next time step. -->
The total wake velocity is then used to update the wake locations for the next time step.

The wake-shedding incorporates vortex diffusion and dissipation to represent viscosity and energy losses in the flow. 
These phenomena are not naturally captured by potential flow, so we model these effects using empirical methods previously incorporated in potential flow models {cite:p}`de2021experimental,ingraham2023low`.
Vortex diffusion captures the effects of viscosity in the flow using a vortex-core model.
We introduce a time dependence to the finite-core radius $r_c$:  
<!-- \begin{equation}
    \label{eq:vortex_core}
    r_c = \sqrt{r_{c0}^2 + 4 \alpha \delta \nu t},
\end{equation} -->
$$
r_c = \sqrt{r_{c0}^2 + 4 \alpha \delta \nu t},
$$
where $r_{c0}$ is the initial finite core size when the wake first sheds and $t$ is the age of the wake element. The $\delta \nu$ is the average effective turbulent viscosity defined as 
<!-- \begin{equation}
    \label{eq:eddy_viscosity}
    \delta \nu = \nu + a_1 \Gamma,
\end{equation} -->
$$
    \delta \nu = \nu + a_1 \Gamma,
$$
where $\nu$ is the kinematic viscosity and $\Gamma$ is the wake element circulation strength.
Note that $\Gamma$ is a time-dependent term.
The empirical parameters $\alpha$ and $a_1$ in the above equations represent the Oseen coefficient and Squire's coefficient, respectively. 
Vortex dissipation models the decay in wake circulation strength across time.
In this work, we model the vortex dissipation as
<!-- \begin{equation}
    \label{eq:vortex_dissipation}
    \Gamma(t) = \Gamma_0\text{exp} \left( -\frac{bq}{s}t\right),
\end{equation} -->
$$
    \Gamma(t) = \Gamma_0\text{exp} \left( -\frac{bq}{s}t\right),
$$
where $\Gamma_0$ is the initial wake element circulation strength when it is first shed from a surface. 
The term $\frac{bq}{s}$ is another empirical parameter representing the wake decay as a function of the ambient turbulence level.
In this work, we set $\alpha = 1.25643$ and $\frac{bq}{s} = 2.5$, based on experimental observations from De Gregorio et al {cite:p}`de2021experimental`.
We also ignore the effect of circulation strength on the vortex core size by setting $a_1 = 0$.

## Bibliography

```{bibliography} ../../references.bib
```