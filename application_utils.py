import sys
import os
import sqlite3
import itertools

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson, geom, uniform
from scipy.special import polygamma, gammaln
from sklearn.cluster import KMeans

import pyhsmm
import pyhsmm.basic.distributions as distributions
from pyhsmm import util
from pyhsmm.util.general import rle
from pyhsmm.util.text import progprint_xrange
 

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

#################################################################
###  Functions to estimate the hidden states with a HDP-HSMM  ###
#################################################################

def train_model(posteriormodel,data,iterator):
    cuts = np.abs(np.diff(data))
    changepoints = util.general.indicators_to_changepoints(np.concatenate(((0,),cuts > cuts.max()/(5*posteriormodel.num_states))))
    posteriormodel.add_data(data,changepoints=list(changepoints))
    for idx in iterator:
        posteriormodel.resample_model()

def print_model(posteriormodel):
    # Print the distributions and their parameters of the posterior model
    print('---------------------------------')
    print('Posterior model: ')
    print('\nObservations distributions:')
    print(posteriormodel.obs_distns)
    #for state in set(posteriormodel.stateseqs[0]):
        #print('weights: {}'.format(posteriormodel.obs_distns[state].weights.weights))
        #print('params: {}'.format([c.mu for c in posteriormodel.obs_distns[state].components]))
    print('\nDurations distributions:')
    for dist in posteriormodel.dur_distns:
        if type(dist) == pyhsmm.basic.distributions.PoissonDuration:
            print('Poisson(lambda={})'.format(dist.lmbda))
        elif type(dist) == pyhsmm.basic.distributions.NegativeBinomialFixedRDuration:
            print('NegativeBinomialFixedR(p={})'.format(dist.p))
        elif type(dist) == pyhsmm.basic.distributions.GeometricDuration:
            print('Geometric(p={})'.format(dist.p))

def plot_states(posteriormodel,data=None,powers_hat=None,cmap='summer'):
    plt.rcParams['image.cmap'] = cmap
    Nb_obs = posteriormodel.datas[0].shape[0]
    Nb_segment = int(np.ceil(Nb_obs / 1000.))
    
    fig = plt.figure(figsize=(16,(5+2)*Nb_segment))
    height_ratios = ([5]+[2])*Nb_segment
    gs = GridSpec(2*Nb_segment,2,width_ratios=[15, 1],height_ratios=height_ratios,wspace=0.05) 
    
    for t in np.arange(0,Nb_obs,1000):
        
        # We plot the data with the estimated one in red dashes 
        data_ax = plt.subplot(gs[(t//1000)*2,0])
        data_ax.plot(data,color='red',label='true powers')
        data_ax.plot(powers_hat,'b--',label='estimated powers')
        data_ax.set_xlim((t,t+1000))
        data_ax.legend(loc='best')
        
        rect = data_ax.axis() # xmin xmax ymin ymax
        #data_ax.vlines([c[1] for c in posteriormodel.states_list[0].changepoints[:-1]],rect[2],rect[3],color='black',linestyles='dashed')

    stateseq = posteriormodel.stateseqs[0]
    num_states = len(set(stateseq))
    # Colors
    state_usages = np.bincount(stateseq,minlength=num_states)
    freqs = state_usages / state_usages.sum()
    used_states = sorted(set(stateseq), key=lambda x: posteriormodel.obs_distns[x].mu)
    #used_states = sorted(set(stateseq), key=lambda x: np.mean([c.mu for c in posteriormodel.obs_distns[x].components]))
    unused_states = [idx for idx in range(num_states) if idx not in used_states]

    colorseq = np.linspace(0,1,num_states)
    state_colors = dict((idx, v) for idx, v in zip(used_states,colorseq))

    for state in unused_states:
        state_colors[state] = 1.
                
    # Colorbar
    unique_states = np.sort(list(set(stateseq)))
    n = len(unique_states)
    C_bar = np.atleast_2d([state_colors[state] for state in unique_states])
    x_bar, y_bar = np.array([0.,1.]), np.arange(n+1)/n
        
    # State sequence 
    stateseq_norep, durations = rle(stateseq)
    x, y = np.hstack((0,durations.cumsum())), np.array([0.,1.])
    C = np.atleast_2d([state_colors[state] for state in stateseq_norep])
                
    for t in np.arange(0,Nb_obs,1000):
            
        stateseq_ax = plt.subplot(gs[(t//1000)*2 + 1,0])
        
        colorbar_ax = plt.subplot(gs[(t//1000)*2 + 1,1])
        colorbar_ax.set_ylim((0.,1.))
        colorbar_ax.set_xlim((0.,1.))
        colorbar_ax.set_xticks([])
        colorbar_ax.yaxis.set_label_position("right")
        colorbar_ax.get_yaxis().set_ticks([])
                
        # State sequence 
        stateseq_ax.pcolormesh(x,y,C,cmap=cmap,vmin=0,vmax=1,alpha=0.5)
        stateseq_ax.set_ylim((0.,1.))
        stateseq_ax.set_xlim((t,t+1000))
        stateseq_ax.set_yticks([])
        stateseq_ax.set_title('Hidden states sequence')
            
        # We plot a colorbar to indicate the color of each state
        colorbar_ax.pcolormesh(x_bar,y_bar,C_bar.T,vmin=0,vmax=1,alpha=0.5,cmap=cmap)
        for j in range(n):
            colorbar_ax.text(.5, (1+j*2)/(2*n), str(unique_states[j]), ha='center', va='center')
        colorbar_ax.get_yaxis().labelpad = 15
        colorbar_ax.set_ylabel('States', rotation=270)
        
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.97]) 
    return fig

# This class help us to get the index in the labels_list coresponding to the observation we are looking at
class stateToLabel:
    
    def __init__(self,Nb_states):
        self.idxState_list = [-1]*Nb_states
        
    def increment(self,componentstate):
        self.idxState_list[state] += 1
        return self.idxState_list[state]
    
def show_result(posteriormodel,data=None,draw=False,cmap='summer',pp=None,title_text=None):
    # We print the models params and we plot the hidden states
    print('---------------------------------')
    true_powers = data
    powers_hat = np.array([posteriormodel.obs_distns[state].mu for state in posteriormodel.stateseqs[0]])
    #idxLabel = stateToLabel(posteriormodel.num_states)
    #powers_hat = np.array([posteriormodel.obs_distns[state].components[posteriormodel.obs_distns[state].labels_list[0].z[idxLabel.increment(state)]].mu for state in posteriormodel.stateseqs[0]])
    T = len(true_powers)
    print('Mean absolute error: {}'.format(np.sum(np.abs(true_powers - powers_hat))/T))
    print_model(posteriormodel)
    
    if draw or pp is not None:
        fig = plot_states(posteriormodel,data,powers_hat,cmap)
        if pp is not None:
            # Log file
            print('Saving on the log file...')
            fig.suptitle(title_text,fontsize=24)
            fig.savefig(pp, format='pdf')
        if not draw:
            plt.close(fig)
        
def estimate_hidden_states(data,Nb_states,Nb_samples,verbose=0,pp=None,title_text=None):
    ### define uninformed priors
    prior_means = np.linspace(data.min(),data.max(),Nb_states)
    gauss = [{'mu_0': mean, 'sigmasq': 1.,'tausq_0': 1.} for mean in prior_means]
    #NegBin = {'alpha_0': 1.,'beta_0': 1.,'r': 1.}
    #normal = distributions.ScalarGaussianFixedvar(mu_0=np.mean(data),tausq_0=100.,sigmasq=100.)
    gaussMixture = [distributions.ScalarGaussianFixedvarMixture(mu_0_list=prior_means,tausq_0_list=[1.]*Nb_states,mixture_weights_0=[1./Nb_states]*Nb_states,sigmasq=1.) for _ in range(Nb_states)]
    #gaussMixture = [distributions.MixtureDistribution(components=[distributions.ScalarGaussianFixedvar(**gauss_params) for gauss_params in gauss],alpha_0=Nb_states) for _ in range(Nb_states)]
    #poi_dist = distributions.PoissonDuration(alpha_0=10.,beta_0=2.)
    #ng_dist = distributions.NegativeBinomialFixedRDuration(r=1.,alpha_0=1.,beta_0=1.)
    pnbMixture = [distributions.MixtureDurationDistribution(components=[distributions.PoissonDuration(alpha_0=10.,beta_0=2.),distributions.NegativeBinomialFixedRDuration(r=1.,alpha_0=1.,beta_0=1.)],alpha_0=Nb_states) for _ in range(Nb_states)]

    ### construct model
    posterior_HDPHSMM = pyhsmm.models.WeakLimitHDPHSMMPossibleChangepoints(
        init_state_distn='uniform',
        alpha=20.,gamma=20.,
        obs_distns=gaussMixture,
        dur_distns=pnbMixture)
    # obs_distns=[distributions.ScalarGaussianFixedvar(**gauss_params) for gauss_params in gauss]
    
    ### train model
    if verbose == 0:
        iterator = range(Nb_samples)
    elif verbose == 1 or verbose == 2:
        iterator = progprint_xrange(Nb_samples)
    train_model(posterior_HDPHSMM,data,iterator)
    
    ### show result
    if verbose == 1:
        show_result(posterior_HDPHSMM,data,pp=pp,title_text=title_text)
    elif verbose == 2:
        show_result(posterior_HDPHSMM,data,draw=True,pp=pp,title_text=title_text)
        
    return posterior_HDPHSMM.stateseqs[0]

#######################################################
###  Functions to estimate priors hyper parameters  ###
#######################################################

def compute_estimators_plot(data,distribution,dur=False):
    # We compute the estimators (for the plot estimation) coresponding to the distribution name given
    x_bar = np.mean(data)
    v_bar = np.var(data)
    s_bar = np.std(data)
    #print('xbar',x_bar,'vbar',v_bar,'sbar',s_bar)
    if dur and distribution!='geometric':
        x_bar = x_bar - 1
    
    if distribution == 'gaussian':
        estimators = {'loc':x_bar,'scale':s_bar}
    elif distribution == 'nbinom':
        if x_bar > v_bar:
            print('The mean is superior to the variance so we cannot compute estimators for the negative binomial')
            return None
        if v_bar == 0.:
            print('The variance is equal to 0 so we cannot compute estimators for the negative binomial')
            return None
        estimators = {'n':(x_bar**2)/(v_bar-x_bar),'p':1-((v_bar-x_bar) / v_bar)}
    elif distribution == 'poisson':
        estimators = {'mu':x_bar}
    elif distribution == 'geometric':
        if x_bar == 0.:
            print('The mean is equal to 0 so we cannot compute estimators for the geometric')
            return None
        estimators = {'p':1./x_bar}
    elif distribution == 'pnbMixture':
        estimators = EM(10**(-14),50,10**(-7),10**(-14),50,verbose=0).fit(data)

    if dur and distribution!='geometric':
        estimators['loc'] = 1
    return estimators

def compute_estimators_priors(data,distribution,dur=False):
    # We compute the estimators (for the priors hyperparams) coresponding to the distribution name given
    x_bar = np.mean(data)
    v_bar = np.var(data)
    s_bar = np.std(data)

    if dur and distribution!='geometric':
        x_bar = x_bar - 1

    if distribution == 'gaussian':
        estimators = {'theta_hat':x_bar,'sigmasq':v_bar}
    elif distribution == 'nbinom':
        if x_bar > v_bar:
            print('The mean is superior to the variance so we cannot compute estimators for the negative binomial')
            return None
        if v_bar == 0.:
            print('The variance is equal to 0 so we cannot compute estimators for the negative binomial')
            return None
        estimators = {'r_hat':(x_bar**2)/(v_bar-x_bar),'p_hat':(v_bar-x_bar) / v_bar}
    elif distribution == 'poisson':
        estimators = {'lambda_hat':x_bar}
    elif distribution == 'geometric':
        if x_bar == 0.:
            print('The mean is equal to 0 so we cannot compute estimators for the geometric')
            return None
        estimators = {'p_hat':1./x_bar}
    elif distribution == 'pnbMixture':
        w = EM(10**(-14),50,10**(-7),10**(-14),50,verbose=0).fit(data)
        estimators = {params.split('_')[0]+'_hat':params_value for params, params_value in w.items()}

    return estimators

def get_distribution(distribution): 
    # We get the scipy.stats distribution class coresponding to the name we give
    return {
  'gaussian': norm,
  'nbinom': nbinom,
  'poisson': poisson,
  'geometric': geom,
  'pnbMixture': NegBinPoissonMixtureDuration()
}[distribution]

def plot_estimation(ax,data,distribution=None,dur=False,title=None):
    data_max = np.max(data)
    if title is None:
        title = ''
        
    # We plot our data
    if dur:
        if data_max>100:
            bins_values, _, _ = ax.hist(data,normed=1,bins=data_max)
            ax.set_ylim((0,np.max(bins_values)*1.1))
        else:
            df = pd.DataFrame(data)
            values = df[0].value_counts().sort_index()/len(data)
            ax.set_ylim((0,np.max(values.values)*1.1))
            ax.bar(values.index,values.values,width=0.5);
        x = np.arange(np.max(data)+1)
    else:
        ax.hist(data,normed=1);
        x = np.linspace(np.min(data),np.max(data),100)
    
    # We plot the estimated distribution
    if distribution is None:
        if dur:       
            for dist, color in zip(['nbinom','poisson','geometric','pnbMixture'],['red','green','orange','magenta']):
                estimators = compute_estimators_plot(data,dist,dur)
                if estimators is None:
                    print(title+'We cannot plot the {} because the estimators cannot be computed'.format(dist))
                    continue
                ax.plot(x,get_distribution(dist).pmf(x,**estimators),color,label=dist)
    else:
        estimators = compute_estimators_plot(data,distribution,dur)
        if estimators is None:
            print(title+'We cannot plot the distribution if the estimators cannot be computed')
            return
        if dur:
            ax.plot(x,get_distribution(distribution).pmf(x,**estimators),'red',label=distribution)
        else:
            ax.plot(x,get_distribution(distribution).pdf(x,**estimators),'red',label=distribution)
            
    ax.legend(loc='best')
    ax.set_title(title)

def estimate_hypparams(data,hidden_states,dur_distributions,obs_distributions,draw=False,pp=None,title_text=None):

    df = pd.DataFrame({'power': data, 'mode': hidden_states})
    # we create a column to indicate each time there is a change in mode between two rows
    df['changepoint'] = np.abs(df['mode'].diff()) > 0
    # we create a column to indicate the segment of each row, a segment is a list of consecutive rows with a same mode
    df['group'] = df['changepoint'].cumsum()

    priors_hypparams = {}
    list_sigmasq = []
    Nb_states = len(set(hidden_states))
    if draw or pp is not None:
        fig_params, ax_params = plt.subplots(Nb_states*2,1,figsize=(15,10*Nb_states))
    for i in range(Nb_states):
        priors_hypparams['mode '+str(i)] = {}
    
        # Durations
        D = df.loc[df['mode']==i,:].groupby('group').size().values # We compute the length of each segment
        if dur_distributions[i] is not None:
            priors_hypparams['mode '+str(i)]['Dur '+dur_distributions[i]] = compute_estimators_priors(D,dur_distributions[i],dur=True)

        # Observations
        obs = df.loc[df['mode']==i,'power'] # We get all the observations from mode i
        if obs_distributions[i] is not None:
            priors_hypparams['mode '+str(i)]['Obs '+obs_distributions[i]] = compute_estimators_priors(obs,obs_distributions[i],dur=False)

        # Transitions
        trans = [x for j, x in enumerate(df['mode'][1:],start=1) if df['mode'][j-1]==i and x!=i ]
        trans_counts = np.bincount(trans,minlength=Nb_states)
        #print('mode {0}, trans counts: {1}'.format(i,trans_counts))
        priors_hypparams['mode '+str(i)]['Trans Dirichlet'] = trans_counts / np.sum(trans_counts)
        
        if draw or pp is not None:
            plot_estimation(ax_params[i*2],D,None,dur=True,title='Dur mode: {} '.format(i))
            plot_estimation(ax_params[i*2+1],obs,obs_distributions[i],dur=False,title='Obs mode: {} '.format(i))
    if pp is not None:
        fig_params.suptitle(title_text,fontsize=24)
        fig_params.savefig(pp, format='pdf')
    if not draw:
        plt.close(fig_params)
    # we sort our priors so that theta_hat increases with the mode number
    priors_hypparams_sorted = {}
    theta_hat_list = np.zeros(Nb_states,dtype=np.double)
    for mode_name, mode in priors_hypparams.items():
        theta_hat_list[int(mode_name.split(' ')[1])] = mode['Obs gaussian']['theta_hat']
    theta_order = np.argsort(theta_hat_list)
    #print('theta order: {}'.format(theta_order))
    for sorted_mode_number, old_mode_number in enumerate(theta_order):
        priors_hypparams_sorted['mode {}'.format(sorted_mode_number)] = priors_hypparams['mode {}'.format(old_mode_number)]
        priors_hypparams_sorted['mode {}'.format(sorted_mode_number)]['Trans Dirichlet'] = [priors_hypparams['mode {}'.format(old_mode_number)]['Trans Dirichlet'][j] for j in theta_order]
    
    return priors_hypparams_sorted

def aggregate_priors(data,dist_name):
    # We aggregate the parameters of every houses with the mean
    # and we compute estimators for the hyperparameters
    hyperparams = {}
    
    if dist_name=='gaussian':
        hyperparams['mu_0'] = np.mean(data['theta_hat'])
        hyperparams['tausq_0'] = np.var(data['theta_hat'])
        hyperparams['sigmasq'] = np.mean(data['sigmasq'])
    elif dist_name=='nbinom':
        x_bar = np.mean(data['p_hat'])
        v_bar = np.var(data['p_hat'])
        hyperparams['r'] = np.mean(data['r_hat'])
        hyperparams['alpha_0'] = x_bar*((x_bar*(1-x_bar)/v_bar)-1)
        hyperparams['beta_0'] = (1-x_bar)*((x_bar*(1-x_bar)/v_bar)-1)
    elif dist_name=='poisson':
        x_bar = np.mean(data['lambda_hat'])
        v_bar = np.var(data['lambda_hat'])
        hyperparams['alpha_0'] = (x_bar**2)/v_bar
        hyperparams['beta_0'] = x_bar/v_bar
    elif dist_name=='pnbMixture':
        # Proportion
        x_bar_pi = np.mean(data['pi_hat'])
        v_bar_pi = np.var(data['pi_hat'])
        hyperparams['alpha_pi'] = x_bar_pi*((x_bar_pi*(1-x_bar_pi)/v_bar_pi)-1)
        hyperparams['beta_pi'] = (1-x_bar_pi)*((x_bar_pi*(1-x_bar_pi)/v_bar_pi)-1)
        # NegBin component
        x_bar_ng = np.mean(data['p_hat'])
        v_bar_ng = np.var(data['p_hat'])
        hyperparams['r'] = np.mean(data['r_hat'])
        hyperparams['alpha_nb'] = x_bar_ng*((x_bar_ng*(1-x_bar_ng)/v_bar_ng)-1)
        hyperparams['beta_nb'] = (1-x_bar_ng)*((x_bar_ng*(1-x_bar_ng)/v_bar_ng)-1)
        # Poisson component
        x_bar_p = np.mean(data['lambda_hat'])
        v_bar_p = np.var(data['lambda_hat'])
        hyperparams['alpha_p'] = (x_bar_p**2)/v_bar_p
        hyperparams['beta_p'] = x_bar_p/v_bar_p
        
    return hyperparams

#########################################################################
###  Classes to sample and fit a Poisson-Negative Binomial Mixture  ###
#########################################################################

class  NegBinPoissonMixtureDuration:
    
    def __init__(self,pi=0.5,lambda_0=1,r=1,p=0.5):
        self.pi = pi
        self.lambda_0 = lambda_0
        self.r = r
        self.p = p
        
    def sample(self,n):
        data = np.zeros(n,dtype=np.int32)
        U = uniform.rvs(size=n)
        poisson_index = U <= self.pi
        data[poisson_index] = poisson.rvs(mu=self.lambda_0,loc=1,size=np.sum(poisson_index))
        data[np.invert(poisson_index)] = nbinom.rvs(n=self.r,p=1-self.p,loc=1,size=n-np.sum(poisson_index))
        return data

    def pmf(self,data,pi=None,lambda_0=None,r=None,p=None,loc=None):
        pi = pi if pi is not None else self.pi
        lambda_0 = lambda_0 if lambda_0 is not None else self.lambda_0
        r = r if r is not None else self.r
        p = p if p is not None else self.p
        loc = loc if loc is not None else 0
        
        return pi*poisson.pmf(data,mu=lambda_0,loc=loc)+(1-pi)*nbinom.pmf(data,n=r,p=1-p,loc=loc)

class EM:
    
    def __init__(self,EM_epsilon,EM_maxIter,Newton_tol,Newton_epsilon,Newton_maxIter,verbose=0):
        self.EM_epsilon = EM_epsilon
        self.EM_maxIter = EM_maxIter
        self.Newton_tol = Newton_tol
        self.Newton_epsilon = Newton_epsilon
        self.Newton_maxIter = Newton_maxIter
        self.verbose = verbose
        
    def initialize_params(self,data):
        kmeans = KMeans(n_clusters=2).fit(data.reshape(-1,1))
        data_0 = data[kmeans.labels_ == 0]
        data_1 = data[kmeans.labels_ == 1]
        if data_0.size == 0:
            U = uniform.rvs(size=data_1.size)
            data_poisson = data_1[U>0.5]
            data_nbinom = data_1[U<=0.5]
        elif data_1.size == 0:
            U = uniform.rvs(size=data_0.size)
            data_poisson = data_0[U>0.5]
            data_nbinom = data_0[U<=0.5]
        else:
            if np.abs(np.var(data_0)-np.mean(data_0)) <= np.abs(np.var(data_1)-np.mean(data_1)):
                data_poisson = data_0
                data_nbinom = data_1
            else:
                data_poisson = data_1
                data_nbinom = data_0
            
        self.pi = data_poisson.size / data.size
        self.lambda_0 = np.mean(data_poisson)
        x_bar = np.mean(data_nbinom)
        v_bar = np.var(data_nbinom)
        self.r = (x_bar**2)/(v_bar-x_bar) if v_bar > x_bar else x_bar**2
        self.p = (v_bar-x_bar) / v_bar if v_bar > x_bar else uniform.rvs()

    def fit(self,data):
        EM_converged = False
        j = 1
        self.initialize_params(data)
        T_current = self.E_step(data)
        poisson_likelihood = np.sum(T_current[0,:]*(np.log(self.pi)-self.lambda_0+(data-1)*np.log(self.lambda_0)-np.array([np.sum(np.log(np.arange(1,d))) for d in data-1])))
        nbinom_likelihood = np.sum(T_current[1,:]*(np.log(1-self.pi)+(data-1)*np.log(1-self.p)+gammaln(data-1+self.r)-gammaln(self.r)-np.array([np.sum(np.log(np.arange(1,d))) for d in data-1])))
        Q_current = poisson_likelihood + nbinom_likelihood
        
        self.M_step(data,T_current)
        poisson_likelihood = np.sum(T_current[0,:]*(np.log(self.pi)-self.lambda_0+(data-1)*np.log(self.lambda_0)-np.array([np.sum(np.log(np.arange(1,d))) for d in data-1])))
        nbinom_likelihood = np.sum(T_current[1,:]*(np.log(1-self.pi)+(data-1)*np.log(1-self.p)+gammaln(data-1+self.r)-gammaln(self.r)-np.array([np.sum(np.log(np.arange(1,d))) for d in data-1])))
        Q_next = poisson_likelihood + nbinom_likelihood
        if Q_next <= Q_current + self.EM_epsilon:
            EM_converged = True
            
        while not EM_converged and j < self.EM_maxIter:
            Q_current = Q_next
            T_current = self.E_step(data)
            self.M_step(data,T_current)
            poisson_likelihood = np.sum(T_current[0,:]*(np.log(self.pi)-self.lambda_0+(data-1)*np.log(self.lambda_0)-np.array([np.sum(np.log(np.arange(1,d))) for d in data-1])))
            nbinom_likelihood = np.sum(T_current[1,:]*(np.log(1-self.pi)+(data-1)*np.log(1-self.p)+gammaln(data-1+self.r)-gammaln(self.r)-np.array([np.sum(np.log(np.arange(1,d))) for d in data-1])))
            Q_next = poisson_likelihood + nbinom_likelihood
            if Q_next <= Q_current + self.EM_epsilon:
                EM_converged = True
                
            j += 1
        
        if self.verbose == 1:
            if EM_converged:
                print("The EM algo has converged")
            else:
                print("The EM algo has not converged because of maxIter")
        
        theta = {'pi': self.pi, 'lambda_0': self.lambda_0, 'r': self.r, 'p': self.p}
        return theta
        
    def E_step(self,data):
        T = np.vstack((self.pi * poisson.pmf(data,mu=self.lambda_0,loc=1) , (1-self.pi) * nbinom.pmf(data,n=self.r,p=1-self.p,loc=1)))
        # if the two likelihood are so small that they are rounded to zero, we don't know which one is the more likely
        # so we put 0.5 probability for both of them
        T[:,T.sum(0)==0.] = np.array([0.5,0.5]).reshape(2,1)
        T = T / T.sum(0)
        return T

    def M_step(self,data,T):
        self.pi = np.maximum(np.minimum(np.mean(T[0,:]),1.-self.EM_epsilon),self.EM_epsilon)
    
        if np.sum(T[0,:]) != 0.:
            self.lambda_0 = np.maximum(np.sum((data-1)*T[0,:]) / np.sum(T[0,:]),self.EM_epsilon)
        
        if np.sum(T[1,:]) != 0.:
            if self.verbose == 1 and np.var(data)<=np.mean(data):
                print("Warning: the sample var is less than the sample mean")
            self.r = self.Newton(self.r,data,T,self.verbose)
            self.p = np.minimum(np.sum((data-1)*T[1,:]) / np.sum((data-1+self.r)*T[1,:]),1.-self.EM_epsilon)
        
    def Newton(self,r,data,T,verbose):
        i = 1
        converged = False
        prob_sum = np.sum(T[1,:])
        prob_sum_data = np.sum((data-1)*T[1,:])
        
        # Initial step
        prob_sum_r = np.sum((data-1+self.r)*T[1,:])
        g = np.sum(T[1,:]*(np.log(1-(prob_sum_data/prob_sum_r))+polygamma(0,data-1+self.r)-polygamma(0,self.r)))
        gprime = np.sum(T[1,:]*(polygamma(1,data-1+self.r)-polygamma(1,self.r)+np.exp(np.log(prob_sum)+np.log(prob_sum_data)-np.log(self.r)-np.log(prob_sum)-np.log(prob_sum_r))))
        if np.abs(gprime) <= self.Newton_epsilon:
            return self.r
        r_current = np.maximum(self.r - (g/gprime),0.0001)
        if np.abs(r_current - self.r) <= self.Newton_tol*np.abs(r_current):
            converged = True
        
        while not converged and i < self.Newton_maxIter:
            prob_sum_r = np.sum((data-1+r_current)*T[1,:])
            g = np.sum(T[1,:]*(np.log(1-(prob_sum_data/prob_sum_r))+polygamma(0,data-1+r_current)-polygamma(0,r_current)))
            gprime = np.sum(T[1,:]*(polygamma(1,data-1+r_current)-polygamma(1,r_current)+np.exp(np.log(prob_sum)+np.log(prob_sum_data)-np.log(self.r)-np.log(prob_sum)-np.log(prob_sum_r))))
            
            if np.abs(gprime) <= self.Newton_epsilon:
                break
            
            r_next = np.maximum(r_current - (g/gprime),0.0001)
            if np.abs(r_next - r_current) <= self.Newton_tol*np.abs(r_next):
                converged = True
            
            r_current = r_next
            i += 1
        
        if verbose == 1:
            if converged:
                print("The Newton's method has converged for r")
            elif i == self.EM_maxIter:
                print("The Newton's method has not converged for r because of maxIter")
            else:
                print("The Newton's method has not converged for r because of epsilon")
        
        return r_current
