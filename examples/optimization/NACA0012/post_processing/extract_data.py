import numpy as np
import csdl_alpha as csdl
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.getcwd()))

file_names = [
    'NACA0012_opt_demo',
]

extracted_data_dir = 'extracted_data/'

recorder = csdl.Recorder(inline=True)
recorder.start()

final_iter_list = []
data_dicts = []

for i, fname in enumerate(file_names):
    print(f'======== working on file {fname} ========')
    iter_count = 0
    # fname = '../' + fname
    while True:
        try:
            # asdf = csdl.inline_import('../' + fname + '.hdf5', f'iteration_{iter_count}')
            asdf = csdl.inline_import('../' +fname + '.hdf5', f'iteration_{iter_count}')
        except Exception:
            break
        iter_count += 1
    final_iter = iter_count
    final_iter_list.append(final_iter)

    data_dict = {}
    for j in range(final_iter):
        print(f'filename {fname}, iter {j}')
        vars = csdl.inline_import(
            # '../' + fname+'.hdf5',
            '../' +fname+'.hdf5',
            f'iteration_{j}'
        )
        if j == 0:
            data_dict = {key:[] for key in vars.keys()}

        for key in vars.keys():
            data_dict[key].append(vars[key].value)

    for key in vars.keys():
        data_dict[key] = np.array(data_dict[key])

    with open(extracted_data_dir + fname + '.pkl', 'wb') as file:
        pickle.dump(data_dict, file)
        file.close()

    data_dicts.append(data_dict)
    