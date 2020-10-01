"""
Script to help run multiple tests, of the arm exploration model, on different parameter sets.
"""

import Run_armModel
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm


# Parameter set
parameters = {
    'n_arms':               3,      # 1, 2, 3
    'RA_size':              30,     # 20, 30, 50, 100
    'ntrials':              6000,   # 6000, 8000, 10000, 30000
    'eta_RL':               0.15,   # 0.15, 0.1, 0.05
    'pPos':                 0.0015,  # /10, /50, /100, /150
    'pDec':                 0.0015,  # pPos, pPos / 2
    'r_sigma':              0.35,   # 0.35, 0.3, 0.25
    'noise_lim':            0.5,    # 0.5, 0.4, 0.3
    'n_score':              0,
    'percent_score':        0,
    'strict_score':         0
}

# Storing test results of parameter grid search
res_path = 'Results/'
Test_results = {}
if not os.path.exists(res_path):
    os.mkdir(res_path)

# Run test
score = Run_armModel.run_test(parameters, res_path)

# Store results
parameters['n_score'] = int(score[0])
parameters['percent_score'] = score[1]
parameters['strict_score'] = int(score[2])

with open(res_path+'parameters.txt', 'w') as outfile:
    json.dump(parameters, outfile, indent=4)