import numpy as np
import pickle

file_name = 'data_mixed_False.pkl'
with open(file_name, 'rb') as file:
    og_data = pickle.load(file)

file_name = 'data_mixed_True.pkl'
with open(file_name, 'rb') as file:
    mixed_data = pickle.load(file)


for key in og_data.keys():
    error = np.linalg.norm(mixed_data[key] - og_data[key])/np.linalg.norm(og_data[key])
    print(f'{key} error (%): {error*100.}')