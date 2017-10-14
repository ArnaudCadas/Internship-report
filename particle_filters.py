import numpy as np
import pandas as pd
from scipy.stats import norm, uniform, dirichlet

import sys, time
import itertools
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def timeprint(iterator,plot=False):
    times = []
    idx = 1
    for thing in iterator:
        prev_time = time.time()
        yield thing
        times.append(time.time() - prev_time)
        sys.stdout.write('Step {} done in {:.4f}sec   \r'.format(idx,times[idx-1]))
        idx += 1
        sys.stdout.flush()
    print('\n{:.4f}sec avg, {:.4f}sec total\n'.format(np.mean(times),np.sum(times)))
    
    if plot:
        plt.boxplot(times)
        plt.ylabel('time in sec')
        plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
        plt.title('boxplot of time for each step');

# Stops iterating through the list as soon as it finds the value
def getIndexOfTuple(l, value):
    for pos,t in enumerate(l):
        if t == value:
            return pos

    # Matches behavior of list.index
    raise ValueError("list.index(x): x not in list")

#######################
####### SIR HMM #######
#######################

class SIR_HMM_class:

    def __init__(self,N,buffer_length,Nb_states,alpha,mu_0,tausq_0,sigmasq):
        ### Create storage ###
        self.N = N # number of particles
        self.Nb_states = Nb_states # number of possible states
        self.buffer_length = buffer_length # the length of the state sequence we will keep in memory
        
        # state space
        self.particles = np.zeros((buffer_length,N),dtype=np.int64) # hidden states of the hmm
        
        # importance weights
        self.weights = np.zeros((buffer_length,N),dtype=np.double)
        self.v = np.zeros(N,dtype=np.double)
        
        # bayesian sufficient statistics
        self.transition_counts = np.zeros((N,Nb_states,Nb_states),dtype=np.int64) # number of transitions from one hidden state to an other
        self.obs_sum = np.zeros((N,Nb_states),dtype=np.double) # the sum of all observations until current time for each particle and each possible state
        self.Nb_obs = np.zeros((N,Nb_states),dtype=np.int64) # the number of observations until current time for each particle and each possible state
        
        # parameters
        self.transition_mat = np.zeros((N,Nb_states,Nb_states),dtype=np.double) # transition matrix for each particle
        self.theta = np.zeros((N,Nb_states),dtype=np.double) # mean of the gaussian distribution put on observations for each particle and each possible state
        
        # priors
        assert len(alpha)==Nb_states and len(mu_0)==Nb_states and len(tausq_0)==Nb_states #and len(sigmasq)==Nb_states
        self.alpha = alpha
        self.mu_0 = mu_0
        self.tausq_0 = tausq_0
        self.sigmasq = sigmasq
        
        # posteriors
        self.mu_n = np.tile(mu_0,N).reshape(N,Nb_states)
        self.tausq_n = np.tile(tausq_0,N).reshape(N,Nb_states)
        
        ### Initialize parameters from priors ###
        self.initialize_params()
        
    def initialize_params(self):
        # Initalize parameters from priors
        for i in range(self.N):
            self.transition_mat[i,:,:] = dirichlet.rvs(alpha=self.alpha,size=self.Nb_states)
            self.theta[i,:] = [norm.rvs(loc=self.mu_0[state],scale=np.sqrt(self.tausq_0[state])) for state in range(self.Nb_states)]
        
    def compute_offsprings(self,W):
        weights_cumsum = np.hstack((0,np.cumsum(W)))
        offsprings = np.zeros(self.N,dtype=np.int64)
        U = uniform.rvs(scale=1./self.N)
        U_inc = 0
        for i in np.arange(1,self.N+1):
            while U_inc < 1 and U + U_inc >= weights_cumsum[i-1] and U + U_inc <= weights_cumsum[i]:
                offsprings[i-1] = offsprings[i-1] + 1
                U_inc = U_inc + 1./self.N
        return offsprings    
    
    def add_data(self,observations):
        # the data must be 1D
        _observations = np.atleast_1d(observations)
        assert _observations.ndim == 1
        # if there was already data we concatenate it, else we create the attribut
        if hasattr(self, 'obs'):
            self.obs = np.hstack((self.obs,_observations))
        else:
            self.obs = _observations
        
    def initial_step(self):
        self.step = 0
        for i in range(self.N):
            # Sample particles
            _transition_unnorm = [norm.pdf(self.obs[0],loc=self.theta[i,state],scale=np.sqrt(self.sigmasq[state])) for state in range(self.Nb_states)]
            self.particles[0,i] = np.random.choice(self.Nb_states,p=_transition_unnorm/np.sum(_transition_unnorm))
            
            # Compute unnormalize weights
            self.v[i] = np.sum(_transition_unnorm)
            
            # Compute bayesian sufficient statistics
            self.obs_sum[i,self.particles[0,i]] += self.obs[0]
            self.Nb_obs[i,self.particles[0,i]] += 1
            
            # Sample parameters
            self.tausq_n[i,self.particles[0,i]] = 1./(1./self.tausq_0[self.particles[0,i]] + self.Nb_obs[i,self.particles[0,i]]/self.sigmasq[self.particles[0,i]])
            self.mu_n[i,self.particles[0,i]] = (self.mu_0[self.particles[0,i]]/self.tausq_0[self.particles[0,i]] + self.obs_sum[i,self.particles[0,i]]/self.sigmasq[self.particles[0,i]])*self.tausq_n[i,self.particles[0,i]]
            self.theta[i,:] = [norm.rvs(loc=self.mu_n[i,state],scale=np.sqrt(self.tausq_n[i,state])) for state in range(self.Nb_states)]
        
        # Compute weights
        self.weights[0,:] = self.v/np.sum(self.v)
        
        ### Resample particles ###
        # compute offsprings
        offsprings = self.compute_offsprings(self.weights[0,:])
        new_particles_idx = np.repeat(range(self.N),offsprings)
        # resample
        self.particles[0,:] = self.particles[0,new_particles_idx]
        self.transition_mat = self.transition_mat[new_particles_idx,:,:]
        self.obs_sum = self.obs_sum[new_particles_idx,:]
        self.Nb_obs = self.Nb_obs[new_particles_idx,:]
        self.mu_n = self.mu_n[new_particles_idx,:]
        self.tausq_n = self.tausq_n[new_particles_idx,:]
        self.theta = self.theta[new_particles_idx,:]
        self.weights[0,:] = np.repeat(1./self.N,self.N)
        
        return self.compute_estimator(self.particles[0,:],self.weights[0,:])
        
    def next_step(self,verbose):
        self.step += 1
        k = min(self.step,self.buffer_length-1)
        # we move the particles and weights
        if self.step > self.buffer_length:
            _temp_particles = np.zeros((self.buffer_length,self.N),dtype=np.int64)
            _temp_particles[0:k,:] = self.particles[1:(k+1),:]
            self.particles = _temp_particles
            _temp_weights = np.zeros((self.buffer_length,self.N),dtype=np.double)
            _temp_weights[0:k,:] = self.weights[1:(k+1),:]
            self.weights = _temp_weights
        #print('particles step {}'.format(self.step),self.particles)
        ### Resample particles (and weights) ###
        if verbose == 2:
            print('\n\n-------- step {} ---------'.format(self.step))
            print('particles before resampling',self.particles[k-1,:])
            print('theta before resampling',self.theta)
        # compute weights 
        for i in range(self.N):
            _transition_unnorm = self.transition_mat[i,self.particles[k-1,i],:] * [norm.pdf(self.obs[self.step],loc=self.theta[i,state],scale=np.sqrt(self.sigmasq[state])) for state in range(self.Nb_states)]
            self.v[i] = np.sum(_transition_unnorm)
        #print('v',self.v)
        if np.sum(self.v) == 0.:
            #print('v nul')
            # case when all particles are unlikely (we put uniform weights on all particles)
            self.weights[k-1,:] = np.repeat(1./self.N,self.N)
        else:
            #print('v non nul')
            self.weights[k-1,:] = self.v/np.sum(self.v)
        # compute offsprings
        #print('weights compute', self.v/np.sum(self.v))
        #print('weights',self.weights[k-1,:])
        offsprings = self.compute_offsprings(self.weights[k-1,:])
        new_particles_idx = np.repeat(range(self.N),offsprings)
        #print('new_id',new_particles_idx)
        # resample
        self.particles[0:k,:] = self.particles[0:k,new_particles_idx]
        self.transition_counts = self.transition_counts[new_particles_idx,:,:]
        self.transition_mat = self.transition_mat[new_particles_idx,:,:]
        self.obs_sum = self.obs_sum[new_particles_idx,:]
        self.Nb_obs = self.Nb_obs[new_particles_idx,:]
        self.mu_n = self.mu_n[new_particles_idx,:]
        self.tausq_n = self.tausq_n[new_particles_idx,:]
        self.theta = self.theta[new_particles_idx,:]
        self.weights[k-1,:] = np.repeat(1./self.N,self.N)
        
        if verbose == 2:
            print('offsprings',offsprings)
            print('particles after resampling',self.particles[k-1,:])
            print('theta after resampling',self.theta)
        
        # Sample new particles
        for i in range(self.N):
            _transition_unnorm = self.transition_mat[i,self.particles[k-1,i],:] * [norm.pdf(self.obs[self.step],loc=self.theta[i,state],scale=np.sqrt(self.sigmasq[state])) for state in range(self.Nb_states)]
            self.particles[k,i] = np.random.choice(self.Nb_states,p=_transition_unnorm/np.sum(_transition_unnorm))    
            
            # Update sufficient statistics
            self.transition_counts[i,self.particles[k-1,i],self.particles[k,i]]+=1
            self.obs_sum[i,self.particles[k,i]]+=self.obs[self.step]
            self.Nb_obs[i,self.particles[k,i]]+=1
            
        # Sample parameters
        for i in range(self.N):
            self.tausq_n[i,self.particles[k,i]] = 1./(1./self.tausq_0[self.particles[k,i]] + self.Nb_obs[i,self.particles[k,i]]/self.sigmasq[self.particles[k,i]])
            self.mu_n[i,self.particles[k,i]] = (self.mu_0[self.particles[k,i]]/self.tausq_0[self.particles[k,i]] + self.obs_sum[i,self.particles[k,i]]/self.sigmasq[self.particles[k,i]])*self.tausq_n[i,self.particles[k,i]]
            self.theta[i,:] = [norm.rvs(loc=self.mu_n[i,state],scale=np.sqrt(self.tausq_n[i,state])) for state in range(self.Nb_states)]
            for j in range(self.Nb_states):
                self.transition_mat[i,j,:] = dirichlet.rvs(alpha=self.alpha+self.transition_counts[i,j,:])
              
        if verbose == 2:
            print('new particles ',self.particles[k,:])
            print('tausq_n',self.tausq_n)
            print('mu_n',self.mu_n)
            print('new theta ', self.theta)
            
        return self.compute_estimator(self.particles[k,:],self.weights[k,:])
        
    def estimate_state(self,particles,weights):
        set_particles = set(particles)
        unique_id_weights = [np.where(particles == particle)[0] for particle in set_particles]
        unique_weights = [np.sum(weights[idx]) for idx in unique_id_weights]
        unique_id_particles = [idx[0] for idx in unique_id_weights]
        unique_particles = particles[unique_id_particles]
        estimated_state = unique_particles[np.argmax(unique_weights)]
        return estimated_state
    
    def compute_estimator(self,particles,weights):
        state_hat = self.estimate_state(particles,weights)
        transition_mat_hat = np.mean(self.transition_mat,axis=0)
        theta_hat = np.mean(self.theta,axis=0)
        return state_hat, transition_mat_hat, theta_hat

    def state_accuracy(self,true_states,estimated_states):
        n = len(true_states)
        return np.sum(estimated_states == true_states) / float(n)
            
    def fit(self,observations,print_result=True,plot_dur=False,verbose=1):
        n = len(observations)
        if plot_dur and verbose == 0:
            verbose = 1
        if verbose == 1 or verbose == 2:
            iterator = timeprint(np.arange(1,n),plot_dur)
        else:
            iterator = np.arange(1,n)
            
        # initialize estimators
        estimated_obs = np.zeros(n,dtype=np.int64)
        estimated_states = np.zeros(n,dtype=np.int64)
        estimated_transition_mat = np.zeros((n,self.Nb_states,self.Nb_states),dtype=np.double)
        estimated_theta = np.zeros((n,self.Nb_states),dtype=np.double)
            
        # load data
        self.add_data(observations)
        
        # first step
        estimated_states[0], estimated_transition_mat[0,:,:], estimated_theta[0,:] = self.initial_step()
        estimated_obs[0] = estimated_theta[0,estimated_states[0]]
          
        # iterations
        for step in iterator:
            estimated_states[step], estimated_transition_mat[step,:,:], estimated_theta[step,:] = self.next_step(verbose)
            estimated_obs[step] = estimated_theta[step,estimated_states[step]]
        
        # print the result
        if print_result:
            print("Estimators of the transition matrix at the end: ")
            print(estimated_transition_mat[n-1,:,:])
            print("\n Estimators of the observations means at the end: ")
            print(estimated_theta[n-1,:])
            
            # plot results
            nb_plots = (n // 1000) + 1
            fig, axes = plt.subplots(nb_plots,1,figsize=(17,5*nb_plots))

            for t in np.arange(0,nb_plots*1000,1000):
                axes[t//1000].plot(observations[t:(t+1000)],color='red',label='true observations')
                axes[t//1000].plot(estimated_obs[t:(t+1000)],linestyle='--',color='blue',label='estimated observations')
                axes[t//1000].legend(loc='best');
            
        return estimated_obs, estimated_states, estimated_transition_mat, estimated_theta

#################################
####### SIR Factorial HMM #######
#################################

class SIR_Factorial_HMM:

    def __init__(self,N,buffer_length,Nb_chains,Nb_states,alpha,mu_0,tausq_0,sigmasq):
        ### Create storage ###
        self.N = N # number of particles
        self.Nb_chains = Nb_chains # number of chains
        assert len(Nb_states)==Nb_chains
        self.Nb_states = Nb_states # number of possible states for each chain
        self.possible_states = [state for state in itertools.product(*[range(i) for i in Nb_states])]
        self.Nb_possible_states = len(self.possible_states)
        self.buffer_length = buffer_length # the length of the state space (and weights) sequence we will keep in memory
        
        # state space
        self.particles = np.zeros((Nb_chains,buffer_length,N),dtype=np.int64) # hidden states of each chains
        self.obs = np.zeros((Nb_chains,buffer_length,N),dtype=np.int64) # hidden observations of each chains
        
        # importance weights
        self.weights = np.zeros((buffer_length,N),dtype=np.double)
        self.v = np.zeros(N,dtype=np.double)
        
        self.transition_counts = np.empty(Nb_chains,dtype=object)
        self.obs_sum = np.empty(Nb_chains,dtype=object)
        self.Nb_obs = np.empty(Nb_chains,dtype=object)
        self.transition_mat = np.empty(Nb_chains,dtype=object)
        self.theta = np.empty(Nb_chains,dtype=object)
        self.alpha = np.empty(Nb_chains,dtype=object)
        self.mu_0 = np.empty(Nb_chains,dtype=object)
        self.tausq_0 = np.empty(Nb_chains,dtype=object)
        self.sigmasq = np.empty(Nb_chains,dtype=object)
        self.mu_n = np.empty(Nb_chains,dtype=object)
        self.tausq_n = np.empty(Nb_chains,dtype=object)
        assert len(alpha)==Nb_chains and len(mu_0)==Nb_chains and len(tausq_0)==Nb_chains  and len(sigmasq)==Nb_chains 
        for j in range(Nb_chains):
            # bayesian sufficient statistics
            self.transition_counts[j] = np.zeros((N,Nb_states[j],Nb_states[j]),dtype=np.int64) # number of transitions from one hidden state to an other
            self.obs_sum[j] = np.zeros((N,Nb_states[j]),dtype=np.double) # the sum of all observations until current time for each particle and each possible state
            self.Nb_obs[j] = np.zeros((N,Nb_states[j]),dtype=np.int64) # the number of observations until current time for each particle and each possible state
        
            # parameters
            self.transition_mat[j] = np.zeros((N,Nb_states[j],Nb_states[j]),dtype=np.double) # transition matrix for each particle
            self.theta[j] = np.zeros((N,Nb_states[j]),dtype=np.double) # mean of the gaussian distribution put on observations for each particle and each possible state
        
            # priors
            assert len(alpha[j])==Nb_states[j] and len(mu_0[j])==Nb_states[j] and len(tausq_0[j])==Nb_states[j] and len(sigmasq[j])==Nb_states[j]
            self.alpha[j] = alpha[j]
            self.mu_0[j] = mu_0[j]
            self.tausq_0[j] = tausq_0[j]
            self.sigmasq[j] = sigmasq[j]
        
            # posteriors
            self.mu_n[j] = np.tile(mu_0[j],N).reshape(N,Nb_states[j])
            self.tausq_n[j] = np.tile(tausq_0[j],N).reshape(N,Nb_states[j])
        
        ### Initialize parameters from priors ###
        self.initialize_params()
        
    def initialize_params(self):
        # Initalize parameters from priors
        for j in range(self.Nb_chains):
            for i in range(self.N):
                self.transition_mat[j][i,:,:] = dirichlet.rvs(alpha=self.alpha[j],size=self.Nb_states[j])
                self.theta[j][i,:] = [norm.rvs(loc=self.mu_0[j][state],scale=np.sqrt(self.tausq_0[j][state])) for state in range(self.Nb_states[j])]
        
    def compute_offsprings(self,W):
        weights_cumsum = np.hstack((0,np.cumsum(W)))
        offsprings = np.zeros(self.N,dtype=np.int64)
        U = uniform.rvs(scale=1./self.N)
        U_inc = 0
        for i in np.arange(1,self.N+1):
            while U_inc < 1 and U + U_inc >= weights_cumsum[i-1] and U + U_inc <= weights_cumsum[i]:
                offsprings[i-1] = offsprings[i-1] + 1
                U_inc = U_inc + 1./self.N
        return offsprings    
    
    def add_data(self,observations):
        # the data must be 1D
        _observations = np.atleast_1d(observations)
        assert _observations.ndim == 1
        # if there was already data we concatenate it, else we create the attribut
        if hasattr(self, 'agg_obs'):
            self.agg_obs = np.hstack((self.agg_obs,_observations))
        else:
            self.agg_obs = _observations
        
    def initial_step(self):
        self.step = 0
        # Compute gaussian parameters
        self.sum_theta = np.zeros((self.N,len(self.possible_states)),dtype=np.double)
        self.sum_sigmasq = np.zeros(len(self.possible_states),dtype=np.double)
        for idx, state in enumerate(self.possible_states):
            for j in range(self.Nb_chains):
                self.sum_sigmasq[idx] += self.sigmasq[j][state[j]]
                for i in range(self.N):
                    self.sum_theta[i,idx] += self.theta[j][i,state[j]]
        for i in range(self.N):
            # Sample particles
            _transition_unnorm = [norm.pdf(self.agg_obs[0],
                                           loc=self.sum_theta[i,idx],
                                           scale=np.sqrt(self.sum_sigmasq[idx])) for idx in range(self.Nb_possible_states)]
            self.particles[:,0,i] = self.possible_states[np.random.choice(self.Nb_possible_states,p=_transition_unnorm/np.sum(_transition_unnorm))]
            
            # Sample hidden observations
            state_idx = getIndexOfTuple(self.possible_states,tuple(self.particles[:,0,i]))
            mu_bar = [self.theta[j][i,self.particles[j,0,i]]+self.sigmasq[j][self.particles[j,0,i]]*(self.agg_obs[0]-self.sum_theta[i,state_idx])/self.sum_sigmasq[state_idx] for j in range(self.Nb_chains)]
            Sigma_bar = np.diag([self.sigmasq[j][self.particles[j,0,i]] for j in range(self.Nb_chains)]) - np.array([[self.sigmasq[j][self.particles[j,0,i]]*self.sigmasq[l][self.particles[l,0,i]]/self.sum_sigmasq[state_idx] for l in range(self.Nb_chains)] for j in range(self.Nb_chains)])
            self.obs[:,0,i] = np.random.multivariate_normal(mean=mu_bar,cov=Sigma_bar)
            
            # Compute unnormalize weights
            self.v[i] = np.sum(_transition_unnorm)
            
            for j in range(self.Nb_chains):
                # Compute bayesian sufficient statistics
                self.obs_sum[j][i,self.particles[j,0,i]] += self.obs[j,0,i]
                self.Nb_obs[j][i,self.particles[j,0,i]] += 1
            
                # Sample parameters
                self.tausq_n[j][i,self.particles[j,0,i]] = 1./(1./self.tausq_0[j][self.particles[j,0,i]] + self.Nb_obs[j][i,self.particles[j,0,i]]/self.sigmasq[j][self.particles[j,0,i]])
                self.mu_n[j][i,self.particles[j,0,i]] = (self.mu_0[j][self.particles[j,0,i]]/self.tausq_0[j][self.particles[j,0,i]] + self.obs_sum[j][i,self.particles[j,0,i]]/self.sigmasq[j][self.particles[j,0,i]])*self.tausq_n[j][i,self.particles[j,0,i]]
                self.theta[j][i,:] = [norm.rvs(loc=self.mu_n[j][i,state],scale=np.sqrt(self.tausq_n[j][i,state])) for state in range(self.Nb_states[j])]
        # Compute weights
        self.weights[0,:] = self.v/np.sum(self.v)
        
        ### Resample particles ###
        # compute offsprings
        offsprings = self.compute_offsprings(self.weights[0,:])
        new_particles_idx = np.repeat(range(self.N),offsprings)
        # resample
        self.particles[:,0,:] = self.particles[:,0,new_particles_idx]
        self.obs[:,0,:] = self.obs[:,0,new_particles_idx]
        for j in range(self.Nb_chains):
            self.transition_mat[j] = self.transition_mat[j][new_particles_idx,:,:]
            self.obs_sum[j] = self.obs_sum[j][new_particles_idx,:]
            self.Nb_obs[j] = self.Nb_obs[j][new_particles_idx,:]
            self.theta[j] = self.theta[j][new_particles_idx,:]
            self.mu_n[j] = self.mu_n[j][new_particles_idx,:]
            self.tausq_n[j] = self.tausq_n[j][new_particles_idx,:]
        self.weights[0,:] = np.repeat(1./self.N,self.N)
        return self.compute_estimator(self.particles[:,0,:],self.weights[0,:])
        
    def next_step(self,verbose):
        self.step += 1
        k = min(self.step,self.buffer_length-1)
        # we move the particles and weights
        if self.step > self.buffer_length:
            _temp_particles = np.zeros((self.Nb_chains,self.buffer_length,self.N),dtype=np.int64)
            _temp_particles[:,0:k,:] = self.particles[:,1:(k+1),:]
            self.particles = _temp_particles
            _temp_obs = np.zeros((self.Nb_chains,self.buffer_length,self.N),dtype=np.int64)
            _temp_obs[:,0:k,:] = self.obs[:,1:(k+1),:]
            self.obs = _temp_obs
            _temp_weights = np.zeros((self.buffer_length,self.N),dtype=np.double)
            _temp_weights[0:k,:] = self.weights[1:(k+1),:]
            self.weights = _temp_weights
        #print('particles step {}'.format(self.step),self.particles)
        ### Resample particles (and weights) ###
        if verbose == 2:
            print('\n\n-------- step {} ---------'.format(self.step))
            print('particles before resampling',self.particles[:,k-1,:])
            print('theta before resampling',self.theta)
        # compute weights 
        for i in range(self.N):
            _transition_unnorm = [np.prod([self.transition_mat[j][i,self.particles[j,k-1,i],state[j]] for j in range(self.Nb_chains)]) \
                                  *norm.pdf(self.agg_obs[self.step],
                                           loc=self.sum_theta[i,idx],
                                           scale=np.sqrt(self.sum_sigmasq[idx])) for idx, state in enumerate(self.possible_states)]
            self.v[i] = np.sum(_transition_unnorm)
        #print('v',self.v)
        if np.sum(self.v) == 0.:
            #print('v nul')
            # case when all particles are unlikely (we put uniform weights on all particles)
            self.weights[k-1,:] = np.repeat(1./self.N,self.N)
        else:
            #print('v non nul')
            self.weights[k-1,:] = self.v/np.sum(self.v)
        # compute offsprings
        #print('weights compute', self.v/np.sum(self.v))
        #print('weights',self.weights[k-1,:])
        offsprings = self.compute_offsprings(self.weights[k-1,:])
        new_particles_idx = np.repeat(range(self.N),offsprings)
        #print('new_id',new_particles_idx)
        # resample
        self.particles[:,0:k,:] = self.particles[:,0:k,new_particles_idx]
        self.obs[:,0:k,:] = self.obs[:,0:k,new_particles_idx]
        for j in range(self.Nb_chains):
            self.transition_counts[j] = self.transition_counts[j][new_particles_idx,:,:]
            self.transition_mat[j] = self.transition_mat[j][new_particles_idx,:,:]
            self.obs_sum[j] = self.obs_sum[j][new_particles_idx,:]
            self.Nb_obs[j] = self.Nb_obs[j][new_particles_idx,:]
            self.theta[j] = self.theta[j][new_particles_idx,:]
            self.mu_n[j] = self.mu_n[j][new_particles_idx,:]
            self.tausq_n[j] = self.tausq_n[j][new_particles_idx,:]
        self.weights[k-1,:] = np.repeat(1./self.N,self.N)
        
        if verbose == 2:
            print('offsprings',offsprings)
            print('particles after resampling',self.particles[:,k-1,:])
            print('theta after resampling',self.theta)
        
        # Compute gaussian parameters
        self.sum_theta = np.zeros((self.N,len(self.possible_states)),dtype=np.double)
        self.sum_sigmasq = np.zeros(len(self.possible_states),dtype=np.double)
        for idx, state in enumerate(self.possible_states):
            for j in range(self.Nb_chains):
                self.sum_sigmasq[idx] += self.sigmasq[j][state[j]]
                for i in range(self.N):
                    self.sum_theta[i,idx] += self.theta[j][i,state[j]]
        
        # Sample new particles
        for i in range(self.N):
            _transition_unnorm = [np.prod([self.transition_mat[j][i,self.particles[j,k-1,i],state[j]] for j in range(self.Nb_chains)]) \
                                  *norm.pdf(self.agg_obs[self.step],
                                           loc=self.sum_theta[i,idx],
                                           scale=np.sqrt(self.sum_sigmasq[idx])) for idx, state in enumerate(self.possible_states)]
            self.particles[:,k,i] = self.possible_states[np.random.choice(self.Nb_possible_states,p=_transition_unnorm/np.sum(_transition_unnorm))]
            
            # Sample hidden observations
            state_idx = getIndexOfTuple(self.possible_states,tuple(self.particles[:,k,i]))
            mu_bar = [self.theta[j][i,self.particles[j,k,i]]+self.sigmasq[j][self.particles[j,k,i]]*(self.agg_obs[self.step]-self.sum_theta[i,state_idx])/self.sum_sigmasq[state_idx] for j in range(self.Nb_chains)]
            Sigma_bar = np.diag([self.sigmasq[j][self.particles[j,k,i]] for j in range(self.Nb_chains)]) - np.array([[self.sigmasq[j][self.particles[j,k,i]]*self.sigmasq[l][self.particles[l,k,i]]/self.sum_sigmasq[state_idx] for l in range(self.Nb_chains)] for j in range(self.Nb_chains)])
            self.obs[:,k,i] = np.random.multivariate_normal(mean=mu_bar,cov=Sigma_bar)
            
            for j in range(self.Nb_chains):
                # Update bayesian sufficient statistics
                self.transition_counts[j][i,self.particles[j,k-1,i],self.particles[j,k,i]]+=1
                self.obs_sum[j][i,self.particles[j,k,i]] += self.obs[j,k,i]
                self.Nb_obs[j][i,self.particles[j,k,i]] += 1
            
                # Sample parameters
                self.tausq_n[j][i,self.particles[j,k,i]] = 1./(1./self.tausq_0[j][self.particles[j,k,i]] + self.Nb_obs[j][i,self.particles[j,k,i]]/self.sigmasq[j][self.particles[j,k,i]])
                self.mu_n[j][i,self.particles[j,k,i]] = (self.mu_0[j][self.particles[j,k,i]]/self.tausq_0[j][self.particles[j,k,i]] + self.obs_sum[j][i,self.particles[j,k,i]]/self.sigmasq[j][self.particles[j,k,i]])*self.tausq_n[j][i,self.particles[j,k,i]]
                self.theta[j][i,:] = [norm.rvs(loc=self.mu_n[j][i,state],scale=np.sqrt(self.tausq_n[j][i,state])) for state in range(self.Nb_states[j])]
                for k in range(self.Nb_states[j]):
                    self.transition_mat[j][i,k,:] = dirichlet.rvs(alpha=self.alpha[j]+self.transition_counts[j][i,k,:])
    
        if verbose == 2:
            print('new particles ',self.particles[:,k,:])
            print('tausq_n',self.tausq_n)
            print('mu_n',self.mu_n)
            print('new theta ', self.theta)
            
        return self.compute_estimator(self.particles[:,k,:],self.weights[k,:])
        
    def estimate_state(self,particles,weights):
        set_particles = set(particles)
        unique_id_weights = [np.where(particles == particle)[0] for particle in set_particles]
        unique_weights = [np.sum(weights[idx]) for idx in unique_id_weights]
        unique_id_particles = [idx[0] for idx in unique_id_weights]
        unique_particles = particles[unique_id_particles]
        estimated_state = unique_particles[np.argmax(unique_weights)]
        return estimated_state
    
    def compute_estimator(self,particles,weights):
        state_hat = [self.estimate_state(particles[j,:],weights) for j in range(self.Nb_chains)]
        transition_mat_hat = [np.mean(self.transition_mat[j],axis=0) for j in range(self.Nb_chains)]
        theta_hat = [np.mean(self.theta[j],axis=0) for j in range(self.Nb_chains)]
        return state_hat, transition_mat_hat, theta_hat

    def state_accuracy(self,true_states,estimated_states):
        n = len(true_states)
        return np.sum(estimated_states == true_states) / float(n)
            
    def fit(self,observations,true_values=None,print_result=True,plot_dur=False,verbose=1):
        n = len(observations)
        if plot_dur and verbose == 0:
            verbose = 1
        if verbose == 1 or verbose == 2:
            iterator = timeprint(np.arange(1,n),plot_dur)
        else:
            iterator = np.arange(1,n)
            
        # initialize estimators
        estimated_agg_obs = np.zeros(n,dtype=np.int64)
        estimated_obs = np.zeros((n,self.Nb_chains),dtype=np.int64)
        estimated_states = np.zeros((n,self.Nb_chains),dtype=np.int64)
        estimated_transition_mat = np.empty(self.Nb_chains,dtype=object)
        estimated_theta = np.empty(self.Nb_chains,dtype=object)
        for j in range(self.Nb_chains):
            estimated_transition_mat[j] = np.zeros((n,self.Nb_states[j],self.Nb_states[j]),dtype=np.double)
            estimated_theta[j] = np.zeros((n,self.Nb_states[j]),dtype=np.double)
            
        # load data
        self.add_data(observations)
        
        # first step
        estimated_states[0,:], _estimated_transition_mat, _estimated_theta = self.initial_step()
        for j in range(self.Nb_chains):
            estimated_transition_mat[j][0,:,:] = _estimated_transition_mat[j]
            estimated_theta[j][0,:] = _estimated_theta[j]
            estimated_obs[0,j] = estimated_theta[j][0,estimated_states[0,j]]
            estimated_agg_obs[0] += estimated_obs[0,j]
          
        # iterations
        for step in iterator:
            estimated_states[step,:], _estimated_transition_mat, _estimated_theta = self.next_step(verbose)
            for j in range(self.Nb_chains):
                estimated_transition_mat[j][step,:,:] = _estimated_transition_mat[j]
                estimated_theta[j][step,:] = _estimated_theta[j]
                estimated_obs[step,j] = estimated_theta[j][step,estimated_states[step,j]]
                estimated_agg_obs[step] += estimated_obs[step,j]
        
        # print the result
        if print_result:
            print("--- Estimators of the transition matrix at the end ---")
            for j in range(self.Nb_chains):
                print("Chain {}".format(j))
                print(estimated_transition_mat[j][n-1,:,:])
            print("\n--- Estimators of the observations means at the end ---")
            for j in range(self.Nb_chains):
                print("Chain {}".format(j))
                print(estimated_theta[j][n-1,:])
            
            # plot results
            assert true_values is not None 
            true_values = np.atleast_2d(true_values)
            assert true_values.ndim==2
            
            interval = min(n,1000)
            
            if n%1000 == 0:
                nb_plots = (n // 1000)*self.Nb_chains
            else:
                nb_plots = ((n // 1000) + 1)*self.Nb_chains
            fig, axes = plt.subplots(nb_plots,1,figsize=(17,5*nb_plots))

            for t in np.arange(0,(nb_plots//self.Nb_chains)*1000,1000):
                for j in range(self.Nb_chains):
                    axes[(t//1000)*self.Nb_chains+j].plot(np.arange(t,t+interval),true_values[t:(t+1000),j],color='red',label='true observations')
                    axes[(t//1000)*self.Nb_chains+j].plot(np.arange(t,t+interval),estimated_obs[t:(t+1000),j],linestyle='--',color='blue',label='estimated observations')
                    axes[(t//1000)*self.Nb_chains+j].set_title('Chain {}'.format(j))
                    axes[(t//1000)*self.Nb_chains+j].legend(loc='best');
            
        return estimated_agg_obs, estimated_obs, estimated_states, estimated_transition_mat, estimated_theta
