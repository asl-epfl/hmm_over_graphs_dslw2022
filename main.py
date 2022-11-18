import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import random
import networkx as nx
from functions import *

#%%
mpl.style.use('seaborn-colorblind')
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}'
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'

#%%
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)

#%% Setup
N = 10
M = 2
N_ITER = 100
theta = np.array([0., 1.5])
var = 1
np.random.seed(0)
delta = 0.1

# %% Transition Matrix
T = np.array([[1 - delta, delta], [delta, 1 - delta]])

############## Build Network Topology ##############
G = np.random.choice([0, 1], p=[.7, .3], size=(N,N))
G = G.T + G
G[G > 0] = 1
deg = G.sum(axis=0)

# Metropolis Combination Rule
A = np.array([[1 / max(deg[i], deg[j]) for i in range(N)] for j in range(N)])
A[G == 0] = 0
for i in range(N):
    A[i, i] = 0
    A[i, i] = 1 - np.sum(A, 0)[i]

#%% Plot Network
Gr = nx.from_numpy_array(A)
pos = nx.spring_layout(Gr, scale=1.1)
f, ax = plt.subplots(1, 1, figsize=(5, 2.5))
plt.axis('off')
plt.xlim([-1.3, 1.1])
plt.ylim([-1.1, 1.1])
nx.draw_networkx_nodes(Gr, pos=pos, node_color='C5',nodelist=range(0, N), node_size=400, edgecolors='k', linewidths=.5)
nx.draw_networkx_labels(Gr, pos, {i: i+1 for i in range(N)}, font_size=14, font_color='black', alpha=1)
nx.draw_networkx_edges(Gr, pos=pos, node_size=400, alpha=1, arrowsize=6, width=1)
plt.tight_layout()
f.savefig(FIG_PATH + 'fig1.pdf', bbox_inches='tight', pad_inches = 0)


############## Compare Different Methods ##############
# %% Initial Beliefs
mu_0 = np.ones((N, M))
mu_0 = mu_0 / np.sum(mu_0, axis=1)[:, None]

# %% Generate Markovian Hidden State
np.random.seed(100)
theta_markov = []
for i in range(N_ITER):
    if i == 0:
        theta_markov.append(np.random.randint(0, 2))
    else:
        theta_markov.append(np.random.choice([0, 1], 1, p=list(T[theta_markov[i-1]]))[0])

#%% Generate Observations
np.random.seed(12)
random.seed(0)
csi=[]
for l in range(0, N):
    csi.append(np.array(theta[theta_markov]) + np.sqrt(var) * sp.truncnorm.rvs((a - np.array(theta[theta_markov]))/np.sqrt(var), (b - np.array(theta[theta_markov]))/np.sqrt(var), size=N_ITER))
csi = np.array(csi)

# %% Run Simulations
gamma = N

# Distributed HMM gamma = K
MU = asl_markov(mu_0, csi, A, N_ITER, T, theta, var, gamma)
# gamma = 1
MU_1 = asl_markov(mu_0, csi, A, N_ITER, T, theta, var, 1)
# gamma = K
MU_2 = asl_markov(mu_0, csi, A, N_ITER, T, theta, var, 2)

# Centralized HMM
MU_c = centralized_markov(np.ones((1, M))/M, csi, N_ITER, T,theta, var)
# ASL
MU_adapt = asl(mu_0, csi, A, N_ITER, theta, var, delta)

# %% Plot comparison between ASL and centralized HMM
fig = plt.figure(constrained_layout=False, figsize=(4, 7))
gs1 = fig.add_gridspec(nrows=2, ncols=1, left=0.15, right=0.9, hspace=0.5, bottom=0.48, top=.95)

theta_markov = np.array(theta_markov)
theta1_idx = np.arange(len(MU)-1)[theta_markov > 0]
theta2_idx = np.arange(len(MU)-1)[theta_markov == 0]

ax1 = fig.add_subplot(gs1[0])
ax1.scatter(theta1_idx, np.ones(len(theta1_idx)), s=4, color='k')
ax1.scatter(theta2_idx, np.zeros(len(theta2_idx)), s=4, color='k')
ax1.set_xlabel(r'$i$', fontsize=11)
ax1.set_xlim([0, len(MU)-1])
ax1.set_ylabel(r'Hidden State', fontsize=10, labelpad=12)
ax1.set_ylim([-.2, 1.2])
ax1.set_yticks([0, 1])
ax1.tick_params(which='major', labelsize=10)

ax2 = fig.add_subplot(gs1[1])
ax2.set_xlim([0, len(MU)-1])
ax2.set_ylim([-.1, 1.1])
ax2.plot([x[0][1] for x in MU_c], '-', color='k', linewidth=1.5, alpha=.9)
ax2.plot([x[0][1] for x in MU], '--', color='C2', linewidth=1.5, dashes=(2, 1))
ax2.plot([x[0][1] for x in MU_adapt], '-', color='C0', linewidth=1.5)
ax2.tick_params(which='major', labelsize=10)
ax2.set_xlabel(r'$i$', fontsize=11)
ax2.set_ylabel('Belief Evolution', fontsize=10)
ax2.legend([r'cHMM', r'dHMM', r'ASL'], ncol=3, loc='center',bbox_to_anchor=(.5,-.47), fontsize=11, handlelength=1.2)

gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.15, right=0.9, hspace=0.2, bottom=0.12, top=.3)
ax3 = fig.add_subplot(gs2[0])
ax3.plot([x[0][1] for x in MU_1], '-', color='C5', linewidth = 1.5)
ax3.plot([x[0][1] for x in MU_2], '-', color='C1', linewidth = 1.5)
ax3.plot([x[0][1] for x in MU], '-', color='C2', linewidth = 1.5)
ax3.set_title('Diffusion HMM Filter', fontsize=12)
ax3.set_xlim([0, len(MU)-1])
ax3.set_ylim([-.1,1.1])
ax3.tick_params(which='major', labelsize=10)
ax3.set_xlabel(r'$i$', fontsize=11)
ax3.set_ylabel('Belief Evolution', fontsize=10)
ax3.legend([r'$\gamma=1$', r'$\gamma=2$', r'$\gamma=10$'], ncol=3, loc='center', bbox_to_anchor=(.5, -.5), fontsize=12, handlelength=1.2)
fig.savefig(FIG_PATH+'fig2.pdf',bbox_inches='tight')


############## Examine the Role of the Network on the Risk ##############
#%% Generate Topologies:

# 1. Very sparse
np.random.seed(0)
G1 = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i == j + 1 or j == i + 1 or i == j:
            G1[i, j] = 1
deg = G1.sum(axis=0)

# Metropolis Combination Matrix
A1 = np.array([[1 / max(deg[i], deg[j]) for i in range(N)] for j in range(N)])
A1[G1 == 0] = 0
for i in range(N):
    A1[i, i] = 0
    A1[i, i] = 1 - np.sum(A1, 0)[i]

# 2. Fully connected
G2 = np.ones((N, N))
A2 = G2 / G2.sum(axis=0)

#%% Monte Carlo Simulations
N_MC = 1000
mu_c_mc, mu_d_mc, mu_d_mc1, mu_d_mc2 = [], [], [], []
for n in range(N_MC):
    theta_markov = []
    for i in range(N_ITER):
        if i == 0:
            theta_markov.append(np.random.randint(0, 2))
        else:
            theta_markov.append(np.random.choice([0, 1], 1, p=list(T[theta_markov[i - 1]]))[0])
    theta_repeat = np.repeat(np.array(theta_markov)[None], N, axis=0)
    csi = np.array(theta[theta_repeat]) + np.sqrt(var) * sp.truncnorm.rvs((a - theta[theta_repeat])/np.sqrt(var), (b - theta[theta_repeat])/np.sqrt(var), size=(N, N_ITER))

    gamma = N

    MU = asl_markov(mu_0, csi, A, N_ITER, T,theta, var, gamma) # Matrix 1
    MU_1 = asl_markov(mu_0, csi, A1, N_ITER, T,theta, var, gamma) # Matrix 2
    MU_2 = asl_markov(mu_0, csi, A2, N_ITER, T,theta, var, gamma) # Matrix 3
    MU_c = centralized_markov(np.ones((1, M))/M, csi, N_ITER, T,theta, var)

    mu_d_mc.append(MU)
    mu_d_mc1.append(MU_1)
    mu_d_mc2.append(MU_2)
    mu_c_mc.append(MU_c)

#%%
mu_d_mc_aux = np.array(mu_d_mc)
mu_c_mc_aux = np.array(mu_c_mc)
mu_d_mc_aux1 = np.array(mu_d_mc1)
mu_d_mc_aux2 = np.array(mu_d_mc2)

#%%
# Compute KL divergence
DKL = np.sum(mu_c_mc_aux * np.log(mu_c_mc_aux / mu_d_mc_aux), axis=3)
DKL1 = np.sum(mu_c_mc_aux * np.log(mu_c_mc_aux / mu_d_mc_aux1), axis=3)
DKL2 = np.sum(mu_c_mc_aux * np.log(mu_c_mc_aux / mu_d_mc_aux2), axis=3)

# Compute average KL divergence
Edkl = np.mean(DKL, axis=0)
Edkl1 = np.mean(DKL1, axis=0)
Edkl2 = np.mean(DKL2, axis=0)

#%% Compute rho_2
l2 = np.sort(np.abs(np.linalg.eig(A)[0]))[-2]
l2_1 = np.sort(np.abs(np.linalg.eig(A1)[0]))[-2]
l2_2 = np.sort(np.abs(np.linalg.eig(A2)[0]))[-2]

#%% Plot risk evolution for different networks
f, ax = plt.subplots(1, 1, figsize=(4.5, 2.8), gridspec_kw={'bottom': .4, 'left': .2})
ax.plot(Edkl1[:,0], '-', linewidth=1.8, color='C0')
ax.plot(Edkl[:,0], '-', linewidth=1.8, color='C5')
ax.plot(Edkl2[:,0], '-', linewidth=1.8, color='teal')
ax.set_xlim([0, len(Edkl)-1])
ax.set_xlabel(r'$i$', fontsize=14)
ax.set_ylabel(r'$J_i(\bm{\mu}_{k,i})$', fontsize=14)
ax.legend([r'$\rho_2 = %1.2f$' % l2_1, r'$\rho_2 = %1.2f$' % l2, r'$\rho_2=%1.2f$' % l2_2], ncol=3, fontsize=12, handlelength=1, loc='center', bbox_to_anchor=(.5, -.58))
ax.tick_params(which='major', labelsize=12)
f.savefig(FIG_PATH + 'fig3.pdf', bbox_inches='tight')
