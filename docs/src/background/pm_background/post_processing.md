# Post-processing

Under the assumptions of potential flow and incompressibility, the Navier-Stokes momentum equation can be formulated as Bernoulli's equation:

$$\frac{1}{2}\rho ||V||^2 + P + \rho gh = C$$

where $\rho$ is the density, $V$ is the velocity vector, $P$ is the static pressure and C represents the constant pressure **head** along the streamline. The $\rho g h$ term represents the gravitational effects on the total pressure; this is commonly ignored in flow analysis of aircraft due to the relative minimal vertical displacement of the flow. Bernoulli's equation can be used to relate the velocity of the flow to the pressure of the flow *at any point in space*. Ignoring the gravitational term, we can relate the pressure at any point on the surface to the freestream properties as

$$\frac{1}{2}\rho ||V||^2 + P = \frac{1}{2}\rho ||V_{\infty}||^2 + P_{\infty}$$

which relates different static and dynamic pressure terms between the freestream and the surface. We define a nondimensional pressure coefficient $C_p$ by rearranging the above equation as

$$C_p = \frac{P - P_{\infty}}{\frac{1}{2}\rho ||V_{\infty}||^2} = 1 - \frac{||V||^2}{||V_{\infty}||^2}$$

which only depends on the freestream velocity and the velocity along the streamline defining the body. From this equation, we see that variations in pressure on the surface are due to local acceleration of the flow.

The velocity on the body contains two contributions: the free-stream velocity $V_{\infty}$ and the induced velocity from the sources and doublets $V_{ind}$:

$$V = V_{\infty} + V_{ind}$$

The induced velocities represent the changes to the flow field velocity due to the presence of the body, which leads to local areas of acceleration and pressure changes. To compute the induced velocity, we utilize the fundamental definitions of doublets and sources. Recall that doublets and sources represent the 
incremental **potential and velocity** jump across the surface of the body, respectively. We can compute the induced velocity using 

<!-- Mathematically, these can be represented as 

$$\mu = \phi_o - \phi_i$$
$$\sigma = (V_o-V_i) \cdot \vec{n}$$ -->

 
$$ V_{ind} = \nabla \phi_{ind}$$
from the definition of potential flow. This relationship holds true regardless of the coordinate frame, although the exact velocity components will differ. For simplicity, we consider this gradient term in the local coordinate frame of each panel; with this approach, the coordinate frame aligns better with the surface and allows us to use the fundamental definitions of the singularities. With the local coordinate frame, we can represent the induced velocities as 

$$
V_{ind} = \nabla \phi_{ind} = \begin{pmatrix} \nabla \mu \\ \sigma \end{pmatrix} = \begin{pmatrix} \frac{\partial \mu}{\partial l} \\ \frac{\partial \mu}{\partial m} \\ \sigma \end{pmatrix}
$$ 

<!-- # NOTE: maybe need to change the nabla terms for phi and mu  -->

where the doublet gradient term $\nabla \mu$ represents a surface gradient. We are interested in computing this term for each **panel**.<!-- We know the sources $\sigma$ represent the normal component of the velocity jump across the boundary; therefore, we can deduce that the doublet gradient contains only terms in the tangent plane.  -->
We use numerical differentiation to estimate $\nabla \mu$ at panel $i$. Using a first-order Taylor series expansion, we can approximate the neighboring panel doublet strength $\mu_j$ using $\mu_i$ as

$$
\mu_{j} = \mu_i + (\delta p)^T \nabla \mu_i
$$

where

$$
\delta p = p_{j}-p_{i} = \begin{pmatrix} (p_j - p_i)_l \\ (p_j-p_i)_m \end{pmatrix} = \begin{pmatrix} \delta l_j \\ \delta m_j \end{pmatrix}
$$

represents the vector between the centroids of panel $i$ and $j$. We can assemble a series of these approximations for $j \in [1,n]$, where $n$ represents the number of panels adjacent to panel $i$. This is used to formulate an overdetermined linear system:

$$
\begin{split}
\begin{pmatrix} | & | \\ \delta l & \delta m \\ | & |\end{pmatrix} \begin{pmatrix} \frac{\partial \mu}{\partial l} \\ \frac{\partial \mu}{\partial m} \end{pmatrix} = \begin{pmatrix}| \\ \delta \mu\\ | \end{pmatrix}
\end{split}
$$

which can be written in shorthand as $Av = b$. The rows in the linear system represent the number of neighboring panels; recall that the VortexAD panel code supports only closed, triangulated grids, so the linear system for each panel will have three rows. The delta vectors for positions and $\mu$ are defined as

$$
\delta l = \begin{pmatrix} \delta l_1 \\ \delta l_2 \\ \delta l_3 \end{pmatrix}, \hspace{2mm} \delta m = \begin{pmatrix} \delta m_1 \\ \delta m_2 \\ \delta m_3 \end{pmatrix}, \hspace{2mm} \delta \mu = \begin{pmatrix} \delta \mu_1 \\ \delta \mu_2 \\ \delta \mu_3 \end{pmatrix}
$$

where $l$ and $m$ represent the two in-plane vectors of the local coordinate system.

<!-- We pose this as a least-squares regression problem:
$$
min_v |Av - b|
$$ -->

We solve this system of equations using the normal equations:

$$
A^TAv = A^Tb \hspace{2mm} \text{or} \hspace{2mm} Cv = d
$$

where our new system of equations is represented by a 2x2 linear system. Rather than assembling the entire matrix $A$ and doing a series of expensive matrix-matrix and matrix=vector products, we can assemble the new linear system directly by computing elements of $C$ and $d$; each term in the linear system is shown below based on the delta vectors of $l, m, \mu$.

$$
% \begin{pmatrix} \delta l^2 & \delta l\delta m \\ \delta l\delta m &  \delta m^2 \end{pmatrix}
\begin{split}
\begin{pmatrix} \delta l \cdot \delta l &  \delta l \cdot \delta m \\ \delta l \cdot \delta m &  \delta m \cdot \delta m \end{pmatrix} \begin{pmatrix} \frac{\partial \mu}{\partial l} \\ \frac{\partial \mu}{\partial m} \end{pmatrix} = \begin{pmatrix} \delta \mu \cdot \delta l \\ \delta \mu \cdot \delta m \end{pmatrix}
\end{split}
$$

This linear system needs to be solved for each panel in the mesh; for a linear system of this size, we analytically compute the elements of $\nabla \mu$ rather than using a linear solve via LU decomposition sequentially across panels. With the analytical solution, we can take advantage of vectorization for speed.