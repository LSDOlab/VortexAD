import numpy as np
import matplotlib.pyplot as plt
import pickle

file_name = 'NACA0012_opt_demo.pkl'

with open(f'extracted_data/{file_name}', 'rb') as file:
    data = pickle.load(file)