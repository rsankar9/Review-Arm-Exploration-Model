"""
Script to help run multiple simulations, of the arm exploration model, on a given parameter set.
"""

import ArmExploration
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json

def run_test(parameters, res_path):
    """ Run multiple simulations on given test parameter set."""

    # Randomisation seeds
    rSeedOverall = np.random.randint(0, 1e7)
    np.random.seed(rSeedOverall)

    # No. of simulations
    Nruns = 200
    rSeeds = np.random.randint(1e7, size=Nruns)

    # Recording
    E_simul = np.zeros(Nruns)
    E_late = np.zeros(Nruns)
    E_onlyRL = np.zeros(Nruns)
    RL_conv = np.zeros(Nruns)
    E_init = np.zeros(Nruns)

    # Multiple simulations
    f = 0
    for run in range(Nruns):
        f = f + 1
        rSeed = rSeeds[run]
        print('Simulation #', run)
        print('Condition 0')
        resFile = res_path + str(f) + '_' + str(rSeed) + '_onlyRL'
        e_onlyRL = ArmExploration.SPTModel(resFile, rSeed, 0, 0, parameters)
        print('Condition 1')
        resFile = res_path + str(f) + '_' + str(rSeed) + '_lateHL'
        e_late = ArmExploration.SPTModel(resFile, rSeed, 1, 0, parameters)
        print('Condition 2')
        resFile = res_path + str(f) + '_' + str(rSeed) + '_simulHL'
        e_simul = ArmExploration.SPTModel(resFile, rSeed, 1, 1, parameters)

        E_simul[run] = e_simul[0]
        E_late[run] = e_late[0]
        E_onlyRL[run] = e_onlyRL[1]
        E_init[run] = e_late[2]
        RL_conv[run] = e_onlyRL[3]

    # Plot error in test phase over all simulations
    fig, [ax0, ax1] = plt.subplots(2,figsize=(15,15))
    x = np.arange(1, Nruns+1)
    ax0.plot(x, E_simul, label='Simul HL', color='red', marker='x', linewidth=0)
    ax0.plot(x, E_late, label='Late HL', color='black', marker='o', linewidth=0)
    # ax0.plot(x, E_onlyRL, label='No HL', color='green', marker='|', linewidth=0)
    ax0.axhline(y=0.2, linestyle='--', color='grey')
    # ax0.set_aspect(1.0)
    ax0.set_xlabel('Simulation #')
    ax0.set_ylabel('Error')
    ax0.set_ylim(top=1, bottom=0)
    ax0.set_title('Mean error during the test phase of each simulation')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.set_xticks([])

    ## Error vs difficulty
    ax1.plot(E_init, E_simul, label='Simul HL', color='red', marker='x', linewidth=0, alpha=0.9)
    ax1.plot(E_init, E_late, label='Late HL', color='black', marker='o', linewidth=0, alpha=0.2)
    ax1.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), color='grey', linestyle='--', alpha=0.5)
    ax1.set_aspect(1.0)
    ax1.set_xlabel('Initial error')
    ax1.set_ylabel('Final error')
    ax1.set_ylim(top=1.1, bottom=-0.1)
    ax1.set_xlim(-0.1,1.1)
    ax1.set_title('Initial point vs error')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.legend()
    plt.savefig(res_path+'OverallResults_' + str(rSeedOverall) +  '.png')
    # plt.show()
    plt.close()


    # Stat tests
    ttest = stats.ttest_rel(E_simul, E_late)
    desc = np.array(['Mean', 'Std', 'Var', 'Median'])
    stats_simul = [np.mean(E_simul), np.std(E_simul), np.var(E_simul), np.median(E_simul)]
    stats_delay = [np.mean(E_late), np.std(E_late), np.var(E_late), np.median(E_late)]
    stats_onlyRL = [np.mean(E_onlyRL), np.std(E_onlyRL), np.var(E_onlyRL), np.median(E_onlyRL)]
    stats_conv = [np.mean(RL_conv), np.std(RL_conv), np.var(RL_conv), np.median(RL_conv)]

    stat_summary = {
        'Ttest' :   {
            't-statistic':  ttest[0],
            'p-value': ttest[1]
        },
        'Simul':   {
                        'Mean':    stats_simul[0],
                        'Std':     stats_simul[1],
                        'Var':     stats_simul[2],
                        'Median':  stats_simul[3]
        },
        'Delay':   {
                        'Mean':    stats_delay[0],
                        'Std':     stats_delay[1],
                        'Var':     stats_delay[2],
                        'Median':  stats_delay[3]
        },
        'OnlyRL':   {
                        'Mean':    stats_onlyRL[0],
                        'Std':     stats_onlyRL[1],
                        'Var':     stats_onlyRL[2],
                        'Median':  stats_onlyRL[3]
        },
        'RLconv':   {
                        'Mean':    stats_conv[0],
                        'Std':     stats_conv[1],
                        'Var':     stats_conv[2],
                        'Median':  stats_conv[3]
        }
    }

    with open(res_path + 'stats_summary.txt', 'w') as outfile:
        json.dump(stat_summary, outfile, indent=4)

    # Scoring
    score = np.sum(E_simul > E_late)
    strict_score = np.sum((E_simul > E_late) & (E_simul >= 0.2) & (E_late <= 0.2))

    return score, score/Nruns, strict_score