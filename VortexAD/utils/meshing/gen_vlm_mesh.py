import numpy as np 
import matplotlib.pyplot as plt

def gen_vlm_mesh(ns, nc, b, c_root, taper=1., sweep=0., airfoil_path=False, plot=False, frame='default'):
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

    if airfoil_path:
        airfoil_data = np.genfromtxt(airfoil_path, skip_header=1)
        '''
        Data goes from TE to LE to TE --> the TE is duplicated
        '''
        num_pts = airfoil_data.shape[0]
        nc_airfoil = num_pts // 2 + 1

        print(num_pts, nc_airfoil)
        nondim_x = airfoil_data[:,0]
        nondim_z = airfoil_data[:,1]
        if plot:
            plt.plot(nondim_x[:nc_airfoil], nondim_z[:nc_airfoil], '*')
            plt.tight_layout()
            plt.show()
            exit()
        
        # LE TO TE
        upper_surf = airfoil_data[:nc_airfoil,:][::-1,:]
        lower_surf = airfoil_data[nc_airfoil:]

        camber_line = (upper_surf + lower_surf)/2.


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