import numpy as np
import scipy.stats as sp

a = -1
b = 2


def trunc_gaussian(x, m, var, a, b):
    '''
    Computes the truncated Gaussian pdf value at x.
    x: value at which the pdf is computed
    m: mean
    var: variance
    '''
    p = sp.truncnorm((a - m) / np.sqrt(var), (b - m) / np.sqrt(var), m , np.sqrt(var)).pdf(x)
    return p

def bayesian_update(L, mu):
    '''
    Computes the Bayesian update.
    L: likelihoods matrix
    mu: beliefs matrix
    '''
    aux = L*mu
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu

def asl_bayesian_update(L, mu, delta):
    '''
    Computes the adaptive Bayesian update.
    L: likelihoods matrix
    mu: beliefs matrix
    delta: step size
    '''
    aux = L*mu**(1-delta)
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu


def asl(mu_0, csi, A, N_ITER, theta, var, delta = 0):
    '''
    Executes the adaptive social learning algorithm with truncated Gaussian likelihoods.
    mu_0: initial beliefs
    csi: observations
    A: Combination matrix
    N_ITER: number of iterations
    theta: vector of means for the Gaussian likelihoods
    var: variance of Gaussian likelihoods
    delta: step size
    '''
    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([trunc_gaussian(csi[:, i], t, var, a, b) for t in theta]).T
        psi = asl_bayesian_update(L_i, mu, delta)
        decpsi = np.log(psi)
        mu = np.exp((A.T).dot(decpsi))/np.sum(np.exp((A.T).dot(decpsi)),axis =1)[:,None]
        MU.append(mu)
    return MU

def asl_markov(mu_0, csi, A, N_ITER, T, theta, var, gamma):
    '''
    Executes the distributed HMM filtering algorithm with truncated Gaussian likelihoods.
    mu_0: initial beliefs
    csi: observations
    A: Combination matrix
    N_ITER: number of iterations
    T: transition matrix
    theta: vector of means for the Gaussian likelihoods
    var: variance of Gaussian likelihoods
    gamma: exponent parameter
    '''
    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([trunc_gaussian(csi[:,i], t, var, a, b) for t in theta]).T
        psi = bayesian_update(L_i**gamma, mu)
        decpsi = np.log(psi)
        mu = np.exp((A.T).dot(decpsi))/np.sum(np.exp((A.T).dot(decpsi)),axis =1)[:,None]
        MU.append(mu)
        mu = mu @ T
    return MU


def centralized_markov(mu_0, csi, N_ITER, T, theta, var):
    '''
    Executes the centralized HMM filtering algorithm with truncated Gaussian likelihoods.
    mu_0: initial beliefs
    csi: observations
    N_ITER: number of iterations
    T: transition matrix
    theta: vector of means for the Gaussian likelihoods
    var: variance of Gaussian likelihoods
    '''
    mu = mu_0.copy()
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([trunc_gaussian(csi[:,i], t, var, a, b) for t in theta]).T
        L_i = np.prod(L_i, axis = 0)
        mu = bayesian_update(L_i, mu)
        MU.append(mu)
        mu = mu @ T
    return MU
