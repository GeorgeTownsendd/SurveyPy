import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from AdjustmentNetwork import load_dataset
from LeastSquaresAdjustment import adjust_network
from NetworkVisualisation import generate_transition_frames, plot_adjustment_process, plot_error_ellipses, visualize_adjustments

dataset_list = [
    'SURV2230_Example', # Working
    'surv3350_tutorial67', # Working
    'SURV3350_Assignment1', # Working!
    'SURV3350_A1_integerestimate', # Working
]
dataset_name = dataset_list[-2]


#visualize_adjustments(dataset_name, [2, 4, 5, 6], side_length=10, coord_range=((0, 10), (0, 10)), max_iterations=100)
#generate_transition_frames(dataset_name, 0.25, 10, 40, 7, value_type='weight')

network = load_dataset(dataset_name=dataset_name)
A, L, vhat = adjust_network(network, max_iterations=5, return_type='final_matrices')

#plot_adjustment_process(network)
plot_error_ellipses(network)