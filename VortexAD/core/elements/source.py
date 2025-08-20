import numpy as np
import csdl_alpha as csdl 

import jax
jax.config.update("jax_enable_x64", True)

def compute_source_strengths(mesh_dict, num_nodes, nt, num_panels, mesh_mode='structured'):
    
    if mesh_mode == 'structured':
        surface_names = list(mesh_dict.keys())

        sigma = csdl.Variable(shape=(num_nodes, nt, num_panels), value=0.)
        start, stop = 0, 0
        for surface in surface_names:
            num_surf_panels = mesh_dict[surface]['num_panels']
            stop += num_surf_panels

            center_pt_velocity = mesh_dict[surface]['nodal_cp_velocity']
            coll_point_vel = mesh_dict[surface]['coll_point_velocity']
            panel_normal = mesh_dict[surface]['panel_normal']

            if coll_point_vel:
                total_vel = center_pt_velocity+coll_point_vel
            else:
                total_vel = center_pt_velocity

            # vel_projection = csdl.einsum(coll_point_velocity, panel_normal, action='ijklm,ijklm->ijkl')
            vel_projection = csdl.sum(-total_vel*panel_normal, axes=(4,))

            sigma = sigma.set(csdl.slice[:,:,start:stop], value=csdl.reshape(vel_projection, shape=(num_nodes, nt, num_surf_panels)))
            start += num_surf_panels

    elif mesh_mode == 'unstructured':
        coll_point_velocity = mesh_dict['coll_point_velocity']
        panel_normal = mesh_dict['panel_normal']

        vel_projection = -csdl.sum(coll_point_velocity*panel_normal, axes=(3,)) # summing along 4th dimension
        sigma = vel_projection

    return sigma # VECTORIZED in shape=(num_nodes, nt, num_surf_panels)

def compute_source_influence(dij, mij, dpij, dx, dy, dz, rk, ek, hk, sigma=1., mode='potential'):
    '''
    UPDATE TO EXPLAIN EACH INPUT
    each input is a list of length 4, holding the values at the corners
    ex: mij = [mij_i for i in range(4)] where each index i corresponds to a corner of the panel
    NOTE: dpij uses the same notation, but contains a sub-index to differentiate between x and y values
    ex: dpij[0][0] refers to the x-component of dpij for point 1
    '''
    num_edges = len(dx) # tells us the number of vertices per panel (hence the edges)
    if mode == 'potential':

        term_1 = []
        term_2 = []

        for i in range(num_edges):
            n = i+1
            if i == num_edges-1: # end
                n = 0

            source_only_term = ((dx[i]*dpij[i][1] - dy[i]*dpij[i][0])/(dij[i]+1.e-12)*csdl.log((rk[i] + rk[n] + dij[i]+1.e-12)/(rk[i] + rk[n] - dij[i] + 1.e-12)))
            term_1.append(source_only_term)

            atan2_mode = 'atan2'

            if atan2_mode == 'K_P': # original arctan term from K&P
                atan2_term = csdl.arctan((mij[i]*ek[i]-hk[i])/(dz[i]*rk[i]+1.e-12)) - csdl.arctan((mij[i]*ek[n]-hk[n])/(dz[i]*rk[n]+1.e-12))
                
            elif atan2_mode == 'atan2': # arctan terms combined using atan2
                t_y = dz[i]*dpij[i][0] * ((ek[i]*dpij[i][1]-hk[i]*dpij[i][0])*rk[n] - (ek[n]*dpij[i][1]-hk[n]*dpij[i][0])*rk[i])
                t_x = dz[i]**2*rk[i]*rk[n]*dpij[i][0]**2 + (ek[i]*dpij[i][1]-hk[i]*dpij[i][0])*(ek[n]*dpij[i][1]-hk[n]*dpij[i][0])
                # atan2_term = 2*csdl.arctan(t_y / ((t_x**2 + t_y**2 + 1.e-12)**0.5 + t_x))
                atan2_term = 2*csdl.arctan(((t_x**2 + t_y**2)**0.5 - t_x) / (t_y + 1.e-12))

            term_2.append(atan2_term)

        source_influence = -sigma/(4*np.pi) * (sum(term_1) - dz[0] * sum(term_2))

        # t1_y = dz[0]*dpij[0][0] * ((ek[0]*dpij[0][1]-hk[0]*dpij[0][0])*rk[1] - (ek[1]*dpij[0][1]-hk[1]*dpij[0][0])*rk[0])
        # t2_y = dz[1]*dpij[1][0] * ((ek[1]*dpij[1][1]-hk[1]*dpij[1][0])*rk[2] - (ek[2]*dpij[1][1]-hk[2]*dpij[1][0])*rk[1])
        # t3_y = dz[2]*dpij[2][0] * ((ek[2]*dpij[2][1]-hk[2]*dpij[2][0])*rk[3] - (ek[3]*dpij[2][1]-hk[3]*dpij[2][0])*rk[2])
        # t4_y = dz[3]*dpij[3][0] * ((ek[3]*dpij[3][1]-hk[3]*dpij[3][0])*rk[0] - (ek[0]*dpij[3][1]-hk[0]*dpij[3][0])*rk[3])

        # t1_x = dz[0]**2*rk[0]*rk[1]*dpij[0][0]**2 + (ek[0]*dpij[0][1]-hk[0]*dpij[0][0])*(ek[1]*dpij[0][1]-hk[1]*dpij[0][0])
        # t2_x = dz[1]**2*rk[1]*rk[2]*dpij[1][0]**2 + (ek[1]*dpij[1][1]-hk[1]*dpij[1][0])*(ek[2]*dpij[1][1]-hk[2]*dpij[1][0])
        # t3_x = dz[2]**2*rk[2]*rk[3]*dpij[2][0]**2 + (ek[2]*dpij[2][1]-hk[2]*dpij[2][0])*(ek[3]*dpij[2][1]-hk[3]*dpij[2][0])
        # t4_x = dz[3]**2*rk[3]*rk[0]*dpij[3][0]**2 + (ek[3]*dpij[3][1]-hk[3]*dpij[3][0])*(ek[0]*dpij[3][1]-hk[0]*dpij[3][0])

        # # atan2_1 = 2*csdl.arctan(t1_y / ((t1_x**2 + t1_y**2 + 1.e-12)**0.5 + t1_x))
        # # atan2_2 = 2*csdl.arctan(t2_y / ((t2_x**2 + t2_y**2 + 1.e-12)**0.5 + t2_x))
        # # atan2_3 = 2*csdl.arctan(t3_y / ((t3_x**2 + t3_y**2 + 1.e-12)**0.5 + t3_x))
        # # atan2_4 = 2*csdl.arctan(t4_y / ((t4_x**2 + t4_y**2 + 1.e-12)**0.5 + t4_x))

        # atan2_1 = 2*csdl.arctan(((t1_x**2 + t1_y**2)**0.5 - t1_x) / (t1_y + 1.e-12))
        # atan2_2 = 2*csdl.arctan(((t2_x**2 + t2_y**2)**0.5 - t2_x) / (t2_y + 1.e-12))
        # atan2_3 = 2*csdl.arctan(((t3_x**2 + t3_y**2)**0.5 - t3_x) / (t3_y + 1.e-12))
        # atan2_4 = 2*csdl.arctan(((t4_x**2 + t4_y**2)**0.5 - t4_x) / (t4_y + 1.e-12))

        # source_influence = -sigma/4/np.pi*(
        #     ( # CHANGE TO USE dx, dy, dz
        #     ((dx[0]*dpij[0][1] - dy[0]*dpij[0][0])/(dij[0]+1.e-12)*csdl.log((rk[0] + rk[1] + dij[0]+1.e-12)/(rk[0] + rk[1] - dij[0] + 1.e-12))) + 
        #     ((dx[1]*dpij[1][1] - dy[1]*dpij[1][0])/(dij[1]+1.e-12)*csdl.log((rk[1] + rk[2] + dij[1]+1.e-12)/(rk[1] + rk[2] - dij[1] + 1.e-12))) + 
        #     ((dx[2]*dpij[2][1] - dy[2]*dpij[2][0])/(dij[2]+1.e-12)*csdl.log((rk[2] + rk[3] + dij[2]+1.e-12)/(rk[2] + rk[3] - dij[2] + 1.e-12))) + 
        #     ((dx[3]*dpij[3][1] - dy[3]*dpij[3][0])/(dij[3]+1.e-12)*csdl.log((rk[3] + rk[0] + dij[3]+1.e-12)/(rk[3] + rk[0] - dij[3] + 1.e-12)))
        # )
        # - dz[0] * ( 
        #     # atan2_1 + atan2_2 + atan2_3 + atan2_4
        #     csdl.arctan((mij[0]*ek[0]-hk[0])/(dz[0]*rk[0]+1.e-12)) - csdl.arctan((mij[0]*ek[1]-hk[1])/(dz[0]*rk[1]+1.e-12)) + 
        #     csdl.arctan((mij[1]*ek[1]-hk[1])/(dz[1]*rk[1]+1.e-12)) - csdl.arctan((mij[1]*ek[2]-hk[2])/(dz[1]*rk[2]+1.e-12)) + 
        #     csdl.arctan((mij[2]*ek[2]-hk[2])/(dz[2]*rk[2]+1.e-12)) - csdl.arctan((mij[2]*ek[3]-hk[3])/(dz[2]*rk[3]+1.e-12)) + 
        #     csdl.arctan((mij[3]*ek[3]-hk[3])/(dz[3]*rk[3]+1.e-12)) - csdl.arctan((mij[3]*ek[0]-hk[0])/(dz[3]*rk[0]+1.e-12))
        # )) # note that dk[:,i,2] is the same for all i
        return source_influence

    elif mode == 'velocity':
        # vel = csdl.Variable(shape=dz.shape[],value=0.)
        u = sigma/(4*np.pi) * (
            dpij[0][1]/(dij[0] + 1.e-12)*csdl.log((rk[0] + rk[1] - dij[0])/(rk[0] + rk[1] + dij[0] + 1.e-12)) + 
            dpij[1][1]/(dij[1] + 1.e-12)*csdl.log((rk[1] + rk[2] - dij[1])/(rk[1] + rk[2] + dij[1] + 1.e-12)) + 
            dpij[2][1]/(dij[2] + 1.e-12)*csdl.log((rk[2] + rk[3] - dij[2])/(rk[2] + rk[3] + dij[2] + 1.e-12)) + 
            dpij[3][1]/(dij[3] + 1.e-12)*csdl.log((rk[3] + rk[0] - dij[3])/(rk[3] + rk[0] + dij[3] + 1.e-12))
        )

        v = sigma/(4*np.pi) * (
            dpij[0][0]/(dij[0] + 1.e-12)*csdl.log((rk[0] + rk[1] - dij[0])/(rk[0] + rk[1] + dij[0] + 1.e-12)) + 
            dpij[1][0]/(dij[1] + 1.e-12)*csdl.log((rk[1] + rk[2] - dij[1])/(rk[1] + rk[2] + dij[1] + 1.e-12)) + 
            dpij[2][0]/(dij[2] + 1.e-12)*csdl.log((rk[2] + rk[3] - dij[2])/(rk[2] + rk[3] + dij[2] + 1.e-12)) + 
            dpij[3][0]/(dij[3] + 1.e-12)*csdl.log((rk[3] + rk[0] - dij[3])/(rk[3] + rk[0] + dij[3] + 1.e-12))
        ) * -1. # NOTE: THIS -1 IS BECAUSE OF THE DIFFERENT SIGN IN THE POTENTIAL EQUATION

        w = sigma/(4*np.pi) * (
            csdl.arctan((mij[0]*ek[0]-hk[0])/(dz[0]*rk[0]+1.e-12)) - csdl.arctan((mij[0]*ek[1]-hk[1])/(dz[0]*rk[1]+1.e-12)) + 
            csdl.arctan((mij[1]*ek[1]-hk[1])/(dz[1]*rk[1]+1.e-12)) - csdl.arctan((mij[1]*ek[2]-hk[2])/(dz[1]*rk[2]+1.e-12)) + 
            csdl.arctan((mij[2]*ek[2]-hk[2])/(dz[2]*rk[2]+1.e-12)) - csdl.arctan((mij[2]*ek[3]-hk[3])/(dz[2]*rk[3]+1.e-12)) + 
            csdl.arctan((mij[3]*ek[3]-hk[3])/(dz[3]*rk[3]+1.e-12)) - csdl.arctan((mij[3]*ek[0]-hk[0])/(dz[3]*rk[0]+1.e-12))
        )
        
        return u, v, w
    
def compute_source_influence_new(A, AM, B, BM, SL, SM, A1, PN, S, l=None, m=None, n=None, mode='potential', mu=1.):
    '''
    Function to compute induced potential of a source panel based on the 
    VSAERO documentation found here: https://ntrs.nasa.gov/citations/19900004884

    This function takes inputs related to panel parameters and data about
    the point where potential is induced.

    Each input is a list (all of the same length), where the number of panel
    sides equals the length of the lists.
    '''
    if mode == 'potential':

        panel_segment_potential = []
        num_sides = len(A)
        
        for i in range(num_sides):

            PA = PN[i]**2*SL[i] + A1[i]*AM[i]
            PB = PN[i]**2*SL[i] + A1[i]*BM[i]

            ## ==== WITH NUMERICAL SOFTENING ====
            # RNUM = SM[i]*PN[i]*(B[i]*PA - A[i]*PB) + 1.e-24
            # DNOM = PA*PB + PN[i]**2*A[i]*B[i]*SM[i]**2 + 1.e-24
    
            # atan_term = csdl.arctan(RNUM/DNOM) # NOTE: add some numerical softening here
            # atan_term = 2*csdl.arctan(((RNUM**2 + DNOM**2)**0.5 - DNOM) / (RNUM+1.e-24)) # half angle formula
            # ========

            ## ==== WITHOUT NUMERICAL SOFTENING ====
            RNUM = SM[i]*PN[i]*(B[i]*PA - A[i]*PB) + 1.e-24
            DNOM = PA*PB + PN[i]**2*A[i]*B[i]*SM[i]**2 + 1.e-24
    
            # atan_term = 2*csdl.arctan(((RNUM**2 + DNOM**2)**0.5 - DNOM) / (RNUM+1.e-24)) # half angle formula

            atan_term = csdl.arctan2(RNUM, DNOM)
            # ========

            GL = (1/(S[i]+1.e-12)) * csdl.log((A[i]+B[i]+(S[i]+1.e-12))/(A[i]+B[i]-(S[i]+1.e-12)))

            side_potential = A1[i]*GL - PN[i]*atan_term

            panel_segment_potential.append(side_potential)

        source_potential = mu/(4*np.pi) * sum(panel_segment_potential)
        return source_potential
    
    elif mode == 'velocity':
        panel_segment_velocity = []
        num_sides = len(A)
        
        for i in range(num_sides):

            PA = PN[i]**2*SL[i] + A1[i]*AM[i]
            PB = PN[i]**2*SL[i] + A1[i]*BM[i]
    
            RNUM = SM[i]*PN[i]*(B[i]*PA - A[i]*PB) + 1.e-24
            DNOM = PA*PB + PN[i]**2*A[i]*B[i]*SM[i]**2 + 1.e-24
    
            # atan_term = csdl.arctan(RNUM/DNOM) # NOTE: add some numerical softening here
            atan_term = 2*csdl.arctan(((RNUM**2 + DNOM**2)**0.5 - DNOM) / (RNUM+1.e-24)) # half angle formula

            GL = (1/(S[i]+1.e-12)) * csdl.log((A[i]+B[i]+S[i])/(A[i]+B[i]-S[i]))

            print(GL.shape)
            print(l.shape)
            print(SM[i].shape)
            # exit()
            
            side_velocity = GL * (SM[i]*l - SL[i]*m) + atan_term*n # Cjk is the atan term

            panel_segment_velocity.append(side_velocity)

        source_velocity = mu/(4*np.pi) * sum(panel_segment_velocity)

        return source_velocity
    
def compute_source_velocity():
    return