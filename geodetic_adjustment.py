import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AdjustmentNetwork import load_dataset
from LeastSquaresAdjustment import adjust_network
from NetworkVisualisation import generate_transition_frames, plot_adjustment_process

dataset_list = [
    'SURV2230_Example', # Working
    'surv3350_tutorial67', # Working
    'SURV3350_Assignment1', # Working!
    'SURV3350_A1_integerestimate', # Working
    'resection_4ray', #Not working
    'direction_distance_lab' #Not working - some stations have a fixed E but varied N
]
dataset_name = dataset_list[2]
#generate_transition_frames(dataset_name, 0.25, 10, 40, 7, value_type='weight')

network = load_dataset(dataset_name=dataset_name)
A, L, vhat = adjust_network(network, max_iterations=10)

plot_adjustment_process(network)