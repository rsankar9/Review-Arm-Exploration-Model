

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def sigmoid(x, m=5, a=0.5):
    """Support for the activation function for neurons. This function works as a sigmoidal."""

    return 1 / (1 + np.exp(-1 * (x - a) * m))

def sliding_window(X, window):
    """This fucntion returns a running average over the recent entries."""

    Y = np.array([X[i:i + 10] for i in range(len(X) - window + 1)])
    return Y

def SPTModel(arg_resFile, arg_rSeed, arg_HL, arg_early, arg_params):
    """Proof of concept model using arm exploration."""


    # Parameters #
    # --------------- #
    np.random.seed(arg_rSeed)

    # Arm parameters
    n_arms = arg_params['n_arms']
    min_a, max_a = 0, 2*np.pi                                                       # Rotation constraints
    lengths = np.ones(n_arms)/n_arms
    total_l = np.sum(lengths)
    pos = np.zeros((n_arms, 2))
    output_range = 2 * total_l                                                      # Maximum error possible

    # Simulation parameters
    nTrainTrials = arg_params['ntrials']
    nTestTrials = int(nTrainTrials/10)
    nTotalTrials  = nTrainTrials + nTestTrials

    # Task parameters
    theta = np.random.uniform(0*np.pi, 2*np.pi)
    l = np.random.uniform(0, total_l)
    target = np.array([l * np.cos(theta), l * np.sin(theta)])

    # Learning parameters
    HL_start = 0
    HL, RL = arg_early, 1
    delay = int(nTrainTrials/3)

    # Network parameters
    HVC_size, RA_size, MO_size = 3, arg_params['RA_size'], n_arms
    eta = arg_params['eta_RL']
    reward_sigma = arg_params['r_sigma']
    noise_lim = arg_params['noise_lim']
    pPos = arg_params['pPos']
    pDec = arg_params['pDec']


    # Simulation data structures #
    # --------------- #

    # For recording purposes
    R = np.zeros(nTotalTrials)
    E = np.zeros(nTotalTrials)
    positions = np.zeros((nTotalTrials, n_arms, 2))
    RL_angles = np.zeros((nTotalTrials, n_arms))
    HL_angles = np.zeros((nTotalTrials, n_arms))
    T_angles = np.zeros((nTotalTrials, n_arms))

    # Angle initialisations
    angles_rl = np. zeros(n_arms)
    angles_hl = np. zeros(n_arms)

    # Weight limits
    Wmin_RL, Wmax_RL = -1.0 , 1.0
    Wmin_HL, Wmax_HL = -1.0 , 1.0
    Wmin_MO, Wmax_MO = 0.0, 25.0

    # Activation function
    arg_steepFactor = 100.


    # Model structure #
    # --------------- #

    # Network layers
    HVC = np.zeros(HVC_size)
    RA = np.zeros(RA_size)
    MO = np.zeros(MO_size)

    # Network weights
    W_HVC_RA_HL = np.zeros((HVC_size, RA_size), float) #+ 0.05
    W_HVC_RA_RL = np.random.uniform(Wmin_RL + 0.05, Wmax_RL - 0.05, (HVC_size, RA_size))
    W_RA_MO = np.random.uniform(Wmin_MO + 0.05, Wmax_MO - 0.05, (RA_size, MO_size))

    # For recording purposes
    S_W_HR_HL = np.zeros((nTotalTrials, W_HVC_RA_HL[0].size))
    S_W_HR_RL = np.zeros((nTotalTrials, W_HVC_RA_RL[0].size))
    S_MO = np.zeros((nTotalTrials, MO.size))
    S_dW_HL = np.zeros((nTotalTrials, RA_size))
    S_dW_RL = np.zeros((nTotalTrials, RA_size))


    # Testing model limits #
    # ------------ #
    RA_min = np.zeros(RA_size)
    RA_max = np.zeros(RA_size)

    MO_sig_slope = 1

    # calculate RA slope
    HVC = np.zeros(HVC_size)
    HVC[0] = 1
    W_HVC_RA_temp = np.zeros((HVC_size, RA_size)) + Wmin_HL

    RA_min[...] = np.dot(HVC, W_HVC_RA_temp) / HVC_size
    min_RA_sig_in = np.min(RA_min)

    HVC = np.zeros(HVC_size)
    HVC[0] = 1
    W_HVC_RA_temp = np.zeros((HVC_size, RA_size)) + Wmax_HL

    RA_max[...] = np.dot(HVC, W_HVC_RA_temp) / HVC_size
    max_RA_sig_in = np.max(RA_max)

    RA_sig_mid = (min_RA_sig_in + max_RA_sig_in) / 2.0
    RA_sig_slope = arg_steepFactor / (max_RA_sig_in - min_RA_sig_in)

    # calculate output range
    RA = sigmoid(RA_min, RA_sig_slope, RA_sig_mid)
    MO[...] = np.dot(RA, W_RA_MO) / RA_size

    min_possible_output = min(MO)

    RA = sigmoid(RA_max, RA_sig_slope, RA_sig_mid)
    MO[...] = np.dot(RA, W_RA_MO) / RA_size

    max_possible_output = max(MO)

    motor_output_range = max_possible_output - min_possible_output

    # print("Actual output range:", motor_output_range, min_possible_output, max_possible_output)
    # print("Modified output range:", motor_output_range, min_possible_output/np.pi - 1 , 'pi, ', max_possible_output/np.pi - 1, 'pi')


    # Model simulation #
    # ------------ #
    # print('----Simulations running----')

    HVC[0] = 1

    for nt in range(nTrainTrials):
        # Activate path A after delay in condition 1
        if nt == delay:
            HL = arg_HL
            HL_start  = nt

        # Introducing noise for RL
        noise_RL = np.random.uniform(-noise_lim, noise_lim, (HVC_size, RA_size))
        W_HVC_RA_RL_temp = W_HVC_RA_RL + noise_RL

        # Compute RA activity
        RA = np.zeros(RA_size)
        RA[...] += np.dot(HVC, W_HVC_RA_HL) / HVC_size * HL
        RA[...] += np.dot(HVC, W_HVC_RA_RL_temp) / HVC_size * RL
        RA = sigmoid(RA, RA_sig_slope, RA_sig_mid)

        # Compute MO activity
        MO[...] = ((np.dot(RA, W_RA_MO) / RA_size) - np.pi)

        # Checking if random initialisation is within bounds
        if nt==0:
            if np.any(MO<0) and np.any(MO>2*np.pi): print('WARNING: MO initialisation out of bounds.')

        # Calculating position of arm
        pos[:,0], pos[:,1] = lengths * np.cos(MO), lengths * np.sin(MO)
        pos[...] = np.cumsum(pos, axis=0)
        output = pos[-1]

        # Compute error and reward
        error = np.sqrt(((output - target) ** 2).sum())
        E[nt] = error / output_range
        R[nt] = np.exp(-E[nt] ** 2 / reward_sigma ** 2)
        R_prev = 0

        # Compute weight update
        dW1 = pPos * HVC.reshape(HVC_size, 1) * (RA) * HL
        dW2 = pDec * (1 - HVC.reshape(HVC_size, 1)) * (RA) * HL
        dW3 = pDec * (HVC.reshape(HVC_size, 1)) * (1 - RA) * HL
        dW4 = dW1 * 0.0
        if nt > 25:
            R_prev = R[nt - 25: nt].sum() / 25.
            dW4 = eta * noise_RL * (R[nt] - R_prev) * HVC.reshape(HVC_size,1) * (RA) * RL

        dW_HL = ((dW1 - dW2 - dW3) * (Wmax_HL - W_HVC_RA_HL) * (W_HVC_RA_HL - Wmin_HL) )[0,:]
        dW_RL = (dW4 * (Wmax_RL - W_HVC_RA_RL) * (W_HVC_RA_RL - Wmin_RL))[0,:]

        # Soft limits
        W_HVC_RA_HL += (dW1-dW2-dW3) * (Wmax_HL - W_HVC_RA_HL) * (W_HVC_RA_HL - Wmin_HL)
        W_HVC_RA_RL += dW4 * (Wmax_RL - W_HVC_RA_RL) * (W_HVC_RA_RL - Wmin_RL)

        # Keeping track for plotting purposes
        positions[nt] = pos
        T_angles[nt] = MO % (2*np.pi)
        S_W_HR_HL[nt] = W_HVC_RA_HL[0,:]
        S_W_HR_RL[nt] = W_HVC_RA_RL[0,:]
        S_MO[nt] = MO.ravel()
        S_dW_HL[nt] = dW_HL
        S_dW_RL[nt] = dW_RL


    # print('Stats at end of training')
    # print('Target angle:', theta)
    # print('Initial angle MO:', T_angles[0])
    # print('Final angle MO:', MO)
    # print('Final output:', output)
    # print('Final reward:', R[nTrainTrials-1])
    # print('Final error:', E[nTrainTrials-1])
    # print('Final angles RL:', angles_rl)
    # print('Final angles HL:', angles_hl)

    # print('----Testing----')

    for nt in range(nTrainTrials,nTotalTrials):
        # Compute RA activity
        RA = np.zeros(RA_size)
        RA[...] += np.dot(HVC, W_HVC_RA_HL) / HVC_size * HL
        RA = sigmoid(RA, RA_sig_slope, RA_sig_mid)

        # Compute MO activity
        MO[...] = ((np.dot(RA, W_RA_MO) / RA_size) - np.pi)

        # Calculating position of arm
        pos[:,0], pos[:,1] = lengths * np.cos(MO), lengths * np.sin(MO)
        pos[...] = np.cumsum(pos, axis=0)
        output = pos[-1]

        # Compute error and reward
        error = np.sqrt(((output - target) ** 2).sum())
        E[nt] = error / output_range
        R[nt] = np.exp(-E[nt] ** 2 / reward_sigma ** 2)

        # For keeping track
        positions[nt] = pos
        T_angles[nt] = MO % (2*np.pi)

        # Keeping track for plotting purposes
        S_W_HR_HL[nt] = W_HVC_RA_HL[0,:]
        S_W_HR_RL[nt] = W_HVC_RA_RL[0,:]
        S_MO[nt] = MO.ravel()


    # Plotting results #
    # ------------ #

    # print('----Plotting----')

    plt.rcParams.update({'font.size': 15})

    fig1 = plt.figure(figsize=(10,10))
    fig1.subplots_adjust(hspace=0.6)
    gs = GridSpec(2, 3, figure=fig1)

    # Setting plot titles
    if arg_HL==0 and arg_early==0:
        fig1.suptitle('Solely Reinforcement learning', fontsize=23)
        subfigure_label = 'C'
        HL_start = -10000
    elif arg_HL == 1 and arg_early == 0:
        fig1.suptitle('Delayed motor pathway', fontsize=23)
        subfigure_label = 'A'
    elif arg_HL == 1 and arg_early == 1:
        HL_start = 0
        fig1.suptitle('Simultaneous motor and tutor pathways', fontsize=23)
        subfigure_label = 'B'

    # Subfigure label
    fig1.text(0.08, 0.955, subfigure_label, style='oblique', fontsize=28)

    # Testing convergence
    conv = np.ones(100)
    E_avgd = np.convolve(E, conv, 'valid')/100
    RL_conv = 6000
    min_E_avgd = np.min(E_avgd)
    for nt in np.arange(500,nTrainTrials):
        if np.std(E_avgd[nt-500:nt]) < 0.003 and E_avgd[nt] < min_E_avgd+0.01:
            RL_conv = nt
            break


    # Panel 1: Plotting error, averaged over 100 trials
    ax = fig1.add_subplot(gs[:1, :-1])
    x = np.arange(nTotalTrials-100+1)

    ax.plot(x, E_avgd, label='Error', marker=',', color='brown')
    ax.plot(x, E[99:], alpha=0.1, marker=',', color='brown')
    ax.axvline(x=nTrainTrials, linestyle='--', color='black', label='Testing phase')
    if arg_HL==0 and arg_early==0:          ax.axvline(x=RL_conv, linestyle='--', color='grey', label='Convergence')
    ax.annotate('Test\nphase', xy=(nTrainTrials, 1), xytext=(nTrainTrials+10, 1), horizontalalignment='left', verticalalignment='top')
    if arg_HL == 1 and arg_early == 0:      ax.axvline(x=HL_start, linestyle='-.', color='grey', label='HL start')
    if arg_HL == 1:     ax.annotate('Path A\nactive', xy=(HL_start, 1), xytext=(HL_start + 10, 1), horizontalalignment='left', verticalalignment='top')
    if arg_HL == 0:     ax.annotate('Convergence', xy=(RL_conv, 1), xytext=(RL_conv - 10, 1), horizontalalignment='right', verticalalignment='top')

    ax.set_ylim(top=1, bottom=0)
    ax.set_ylabel('Error')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])


    # Panel 2: Plotting weights over the trials
    ax = fig1.add_subplot(gs[1:2, :-1])

    line1 = ax.plot(S_W_HR_HL, color='brown', alpha=0.5, label='Path A weights')
    line2 = ax.plot(S_W_HR_RL, color='grey', alpha=0.5, label='Path B weights')
    ax.axvline(x=nTrainTrials, linestyle='--', color='black', label='Testing phase')
    if arg_HL == 1 and arg_early == 0:
        ax.axvline(x=HL_start, linestyle='-.', color='grey', label='HL start')
    ax.set_ylabel('Weights')
    ax.set_xlabel('Trials')
    ax.set_ylim(Wmin_RL, Wmax_RL)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # Panel 3: Plotting position after every 100th trials, along with the last arm
    ax = fig1.add_subplot(gs[:2, -1:])

    r = np.linspace(0,2*np.pi,1000)                                                                                                                             # Plotting outer circle
    x, y = np.cos(r) * total_l, np.sin(r) * total_l
    line3 = ax.plot(x,y,linestyle=':', label='Reach', color='black')
    for nt in range(0,nTotalTrials,250):
        ax.plot([0,positions[nt, 0, 0]], [0,positions[nt, 0, 1]], color='brown', alpha=0.2)
        ax.plot(positions[nt, :, 0], positions[nt, :, 1], color='brown', alpha=0.2)
    line4 = ax.plot(positions[::250,-1,0], positions[::250,-1,1], marker='.', linewidth=0., markersize=8, label = 'Arm position', color='brown', alpha = 0.2)   # Plotting farthest arm position
    line5 = ax.plot(0,0,color='black',marker='.', markersize=8, linewidth=0, label='Origin')                                                                    # Plotting (0,0)
    line6 = ax.plot(positions[0,-1,0], positions[0,-1,1], marker='o', linewidth=0., markersize=8, label = 'Initial point', color='black')                       # Initial point
    line7 = ax.plot(target[0], target[1],color='black',marker='X', markersize=10, markeredgewidth=0.2,  linewidth=0, label='Target')                            # Plotting target point

    ax.set_ylim((-1.001,1.001))
    ax.set_xlim((-1.001,1.001))
    ax.set_aspect(1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Customising legend
    lines = [line1[0]] + [line2[0]] + line6 + line7
    labels = [l.get_label() for l in lines]
    if arg_early == 1:
        ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.4,-1.3), frameon=False, fontsize=16)

    # Save
    plt.savefig(arg_resFile+'.pdf', rasterized=True, bbox_inches = 'tight', pad_inches = 0)
    # plt.show()
    plt.close()

    # Return initial and end error stats
    return np.mean(E[-nTestTrials:]), np.mean(E[nTrainTrials-nTestTrials:nTrainTrials]), E[0], RL_conv