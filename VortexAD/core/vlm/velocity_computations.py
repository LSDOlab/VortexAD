import numpy as np 
import csdl_alpha as csdl

def compute_normal_velocity(velocity, normal_vectors):
    # velocity_expanded = csdl.expand(velocity, normal_vectors.shape, 'ij->iabj')
    # proj = csdl.sum(normal_vectors*velocity_expanded, axes=(3,))
    
    proj = csdl.sum(normal_vectors*velocity, axes=(3,))

    return proj

def compute_induced_velocity_old(p1, p2, p_eval, gamma=1.):
    # r1 = p1-p_eval
    # r2 = p2-p_eval

    r1 = p_eval-p1
    r2 = p_eval-p2
    input_shape = r1.shape
    xyz_dim = len(input_shape) - 1

    # find cross product and norm
    r1r2_cross = csdl.cross(r1, r2, axis=xyz_dim)
    r1r2_cross_norm = csdl.norm(r1r2_cross+1.e-12, axes=(xyz_dim,))
    r1r2_cross_norm_exp = csdl.expand(r1r2_cross_norm, r1.shape, 'ij->ija')

    # compute norms of r1 and r2
    r1_norm = csdl.norm(r1+1.e-12, axes=(xyz_dim, ))
    r1_norm_exp = csdl.expand(r1_norm, r1.shape, 'ij->ija')
    r2_norm = csdl.norm(r2+1.e-12, axes=(xyz_dim, ))
    r2_norm_exp = csdl.expand(r2_norm, r2.shape, 'ij->ija')

    # compute dot products 
    r0 = r1-r2
    r0r1_dot = csdl.sum(r0*r1, axes=(xyz_dim,))
    r0r1_dot_exp = csdl.expand(r0r1_dot, r1.shape, 'ij->ija')
    r0r2_dot = csdl.sum(r0*r2, axes=(xyz_dim,))
    r0r2_dot_exp = csdl.expand(r0r2_dot, r2.shape, 'ij->ija')

    induced_vel = gamma/(4*np.pi)*r1r2_cross/(r1r2_cross_norm_exp + 1.e-8)**2 * \
                (r0r1_dot_exp/r1_norm_exp - r0r2_dot_exp/r2_norm_exp)

    # term_pre_dot = r0*(r1/r1_norm_exp + r2/r2_norm_exp)

    # dot_prod_term = csdl.sum(term_pre_dot, axes=(xyz_dim,))
    # dot_prod_term_exp = csdl.expand(dot_prod_term, r0.shape, 'ij->ija')

    # induced_vel = gamma/(4*np.pi)*r1r2_cross/(r1r2_cross_norm_exp + 1.e-8)**2 * dot_prod_term_exp

    return induced_vel

def compute_induced_velocity(p1, p2, p_eval, gamma=1., vc=None):
    # r1 = p1-p_eval
    # r2 = p2-p_eval

    r1 = p_eval-p1
    r2 = p_eval-p2
    input_shape = r1.shape
    xyz_dim = len(input_shape) - 1

    # compute norms of r1 and r2
    r1_norm = csdl.norm(r1+1.e-12, axes=(xyz_dim, ))
    r1_norm_exp = csdl.expand(r1_norm, r1.shape, 'ij->ija')
    r2_norm = csdl.norm(r2+1.e-12, axes=(xyz_dim, ))
    r2_norm_exp = csdl.expand(r2_norm, r2.shape, 'ij->ija')

    # find cross product and norm
    r1r2_cross = csdl.cross(r1, r2, axis=xyz_dim)
    r1r2_cross_norm = csdl.norm(r1r2_cross+1.e-12, axes=(xyz_dim,))
    r1r2_cross_norm_exp = csdl.expand(r1r2_cross_norm, r1.shape, 'ij->ija')
    
    if vc is None:
        # compute dot products 
        r0 = r1-r2
        r0r1_dot = csdl.sum(r0*r1, axes=(xyz_dim,))
        r0r1_dot_exp = csdl.expand(r0r1_dot, r1.shape, 'ij->ija')
        r0r2_dot = csdl.sum(r0*r2, axes=(xyz_dim,))
        r0r2_dot_exp = csdl.expand(r0r2_dot, r2.shape, 'ij->ija')

        induced_vel = gamma/(4*np.pi)*r1r2_cross/(r1r2_cross_norm_exp + 1.e-8)**2 * \
                    (r0r1_dot_exp/r1_norm_exp - r0r2_dot_exp/r2_norm_exp)
        
    else:
        # finite core from BYU VortexLattice approach:
        # if finite_core
        #     rdot = dot(r1, r2)
        #     r1s, r2s, εs = nr1^2, nr2^2, core_size^2
        #     f1 = cross(r1, r2)/(r1s*r2s - rdot^2 + εs*(r1s + r2s - 2*nr1*nr2))
        #     f2 = (r1s - rdot)/sqrt(r1s + εs) + (r2s - rdot)/sqrt(r2s + εs)

        # Vhat = (f1*f2)/(4*pi)
        rdot = csdl.sum(r1*r2, axes=(xyz_dim,))
        rdot_exp = csdl.expand(rdot, r1.shape, 'ij->ija')

        r1s = r1_norm_exp**2
        r2s = r2_norm_exp**2
        eps_s = vc**2

        f1 = r1r2_cross/(r1s*r2s - rdot_exp**2 + vc*(r1s+r2s-2*r1_norm_exp*r2_norm_exp) + 1.e-12)
        f2 = (r1s - rdot_exp)/(r1s+eps_s)**0.5 + (r2s - rdot_exp)/(r2s+eps_s)**0.5
        induced_vel = f1*f2/4./np.pi

    return induced_vel