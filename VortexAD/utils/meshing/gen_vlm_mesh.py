import numpy as np 
import csdl_alpha as csdl 



def gen_vlm_mesh(ns, nc, b, c_root, taper=1., sweep=0., frame='default'):
    '''
    Generates a standard mesh for vlm analysis
    Inputs:
    - ns, nc: spanwise and chordwise discretization
    - b: wing span
    - c_root: root chord length
    - taper: taper ratio, default of 1.
    - sweep: sweep angle of wing in degrees, default of 0.
    '''
    c_tip = c_root*taper
    le_tip_x = np.tan(sweep*np.pi/180)*b/2
    span_array = np.linspace(-b/2, b/2, ns)

    mesh = np.zeros((nc,ns,3))
    if frame == 'default':
        for i in range(ns):
            mesh[:,i,0] = np.linspace(0, c_root, nc) # RECTANGULAR WING
            mesh[:,i,1] = span_array[i]

    elif frame  == 'caddee':
        '''
        This uses the body-fixed frame: x points forward, z points down, y points to the right wing
        For simple wing, this essentially flips the sign of x and y
        '''
        for i in range(ns):
            mesh[:,i,0] = np.linspace(0, -c_root, nc) # RECTANGULAR WING
            mesh[:,i,1] = span_array[i]
    else:
        raise ValueError('Invalid input for frame. Options are default or caddee')

    return mesh