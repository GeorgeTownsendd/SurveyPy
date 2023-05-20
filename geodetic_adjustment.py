import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from AdjustmentNetwork import load_dataset, SingleTargetObservation, MultiTargetObservation
from LeastSquaresAdjustment import adjust_network
from NetworkVisualisation import generate_transition_frames, plot_adjustment_process, plot_error_ellipses, visualize_network

dataset_list = [
    'SURV2230_Example', # Working
    'surv3350_tutorial67', # Working
    'SURV3350_Assignment1', # Working!
    'SURV3350_A1_integerestimate', # Working
]
dataset_name = dataset_list[0]


#visualize_adjustments(dataset_name, [2, 4, 5, 6], side_length=10, coord_range=((0, 10), (0, 10)), max_iterations=100)
#generate_transition_frames(dataset_name, 0.25, 10, 40, 7, value_type='weight')

network = load_dataset(dataset_name=dataset_name)
A, L, vhat = adjust_network(network, max_iterations=10, return_type='final_matrices')

#plot_adjustment_process(network)
#plot_error_ellipses(network)

visualize_network(network)

'''
A = network.final_adjustment_state['A']
G = np.linalg.inv(network.final_adjustment_state['P'])
N = network.final_adjustment_state['N']

v_terms = network.final_adjustment_state['V'] ** 2
s_terms = np.diag(G)

for i in range(len(v_terms)):
    print(network.observations[i], v_terms[i], s_terms[i], (v_terms[i] / s_terms[i]))

significance_level = 0.01
alpha = significance_level / 2 #two tailed

E_vhat = G - (A @ np.linalg.inv(N)) @ A.T #variance-covariance of corrections V
V_variances = np.diag(E_vhat)
V_std = np.sqrt(V_variances)

v_threshold_levels = [norm.ppf(1-alpha, 0, vi) for vi in V_std]

for i in range(len(v_terms)):
    print(V_std[i], v_threshold_levels[i])
    print(V_std[i] > v_threshold_levels[i])'''