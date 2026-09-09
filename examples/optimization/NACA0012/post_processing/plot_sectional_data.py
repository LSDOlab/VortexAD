import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

def ellipse(x, a, b):
    '''
    x: x coordinates
    a: semi-major axis
    b: semi-minor axis
    '''
    y = b * np.sqrt(1 - x**2/a**2)
    return y

file_names = [
    'NACA0012_opt_demo',
]

rho = 1.225
V_inf = 50.
area = 10.

full_dict = {}
for i, fname in enumerate(file_names):
    with open(f'sectional_data_plots/{fname}_sectional_data.pkl', 'rb') as file:
        data = pickle.load(file)
    # computing geometric twist
    pc_u = data['pc_u']
    pc_l = data['pc_l']
    LE = (pc_u[0,:] + pc_l[0,:])/2.
    TE = (pc_u[-1,:] + pc_l[-1,:])/2.

    camber_line = TE-LE
    chord = np.linalg.norm(camber_line, axis=1)
    geom_twist = np.arctan(camber_line[:,2]/-camber_line[:,0]) * 180/np.pi

    data['geom_twist'] = geom_twist
    data['inflow_angle'] = geom_twist + data['pitch']

    data['dCLdb'] = data['dLdb']/(0.5*rho*V_inf**2*chord)

    full_dict[i+1] = data



cases = list(full_dict.keys())
case_names = ['P1', 'P2']
num_cases = len(cases)
colors = ['r', 'g', 'b', 'k']
# colors = [
#     '#9467bd',
#     '#8c564b',
#     '#e377c2',
# ]

# subplot of sectional lift and sectional twist
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(15,7))
spans = [10, 15]
for i, case in enumerate(cases):
    case_name = case_names[i]
    case_dict = full_dict[case]
    sec_center = case_dict['sec_center']
    sec_center[0] = 0
    dLdb = case_dict['dLdb']
    inflow_angle = case_dict['inflow_angle']
    a = spans[i]/2
    b = dLdb[0]
    case_ellipse = ellipse(np.abs(sec_center), a, b)
    axs[0].plot(sec_center, dLdb, colors[i], markersize=5, linewidth=3, label=f'{case_name}')
    axs[0].plot(sec_center, case_ellipse, colors[i], linestyle='--', linewidth=2, label=f'{case_name} ellipse')
    axs[1].plot(sec_center, inflow_angle, colors[i], markersize=5, linewidth=3, label=f'{case_name}')

ymin, ymax = ymin, ymax = axs[0].get_ylim()

for i, case in enumerate(cases):
    case_name = case_names[i]
    case_dict = full_dict[case]
    dLdb = case_dict['dLdb']
    inflow_angle = case_dict['inflow_angle']
    sec_center = case_dict['sec_center']

axs[0].set_ylabel(r'Sectional lift (N/m)', fontsize=15)
axs[0].tick_params(axis='y', labelsize=13, rotation=45)
axs[0].set_ylim([ymin, ymax])
axs[0].grid()
axs[0].legend(fontsize=15)

axs[1].set_xlabel(r'Spanwise location (m)', fontsize=15)
axs[1].set_ylabel(r'Local angle of attack $(^{\circ})$', fontsize=15)
axs[1].tick_params(axis='y', labelsize=13, rotation=45)
axs[1].tick_params(axis='x', labelsize=13)
axs[1].grid()


plt.savefig('sectional_data_plots/spanwise_lift_twist_dist.pdf')
plt.savefig('sectional_data_plots/spanwise_lift_twist_dist.png')
# plt.show()
# exit()