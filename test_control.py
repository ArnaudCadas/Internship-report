#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:34:10 2017

@author: acadas
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import particle_filters as pf

import pyhsmm
import pyhsmm.basic.distributions as distributions

from pyhsmm.plugins.factorial.models import Factorial, FactorialComponentHSMM
from pyhsmm.internals.transitions import HDPHSMMTransitions
from pyhsmm.internals.initial_state import Uniform

import ast

#### Power Simulator ####

# we load our parameters estimations
hyperparams_estimations = pd.read_table('parameters_estimations.tsv', index_col=[0,1,2])
hyperparams_estimations.sort_index(inplace=True)

class Power_simulator:
    priors = hyperparams_estimations
    
    def __init__(self,T):
        self.controlled_devices = ['air']
        self.uncontrolled_devices = ['refrigerator','furnace','dishwasher']
        # We choose a house at random
        self.house = np.random.choice(list(set(self.priors.index.get_level_values('house_id'))))
        # We construct our models
        self.controlled_params = self.construct_controlled_params()
        self.model = self.construct_model()
        
        self.t = 0
        self.nominal_power, self.uncontrolled_power = self.generate_aggregated_power(N=T)
        
    def construct_controlled_params(self):
        controlled_params = {}
        for device in self.controlled_devices:
            modes = self.priors.loc[(device,self.house,slice(None))].index.get_level_values('mode')
            Nb_states = len(modes)
            controlled_params[device] = {'mu': np.zeros(Nb_states), 'sigmasq': np.zeros(Nb_states)}
            # We set up the params for each operating mode
            for mode in modes:
                # We set up the observations params
                obs_params = ast.literal_eval(self.priors.loc[(device,self.house,mode),'Obs params'])
                controlled_params[device]['mu'][int(mode.split(' ')[1])] = obs_params['theta_hat']
                controlled_params[device]['sigmasq'][int(mode.split(' ')[1])] = obs_params['sigmasq']
        return controlled_params
            
        
    def construct_model(self):
        # We construct a Factorial model for all devices 
        devices = self.controlled_devices + self.uncontrolled_devices
        component_model_list = []
        
        for device in devices:
            modes = self.priors.loc[(device,self.house,slice(None))].index.get_level_values('mode')
            Nb_states = len(modes)
            obs_distns = []
            dur_distns = []
            trans_mat = []
            # We set up the params for each operating mode
            for mode in modes:
                trans_mat.append(ast.literal_eval(self.priors.loc[(device,self.house,mode),'Trans params']))
                # We set up the observations params
                obs_params = ast.literal_eval(self.priors.loc[(device,self.house,mode),'Obs params'])
                theta_hat = obs_params['theta_hat']
                sigmasq = obs_params['sigmasq']
                gaussMixture = distributions.ScalarGaussianFixedvarMixture(mu_0_list=[theta_hat]*Nb_states,
                                                                   tausq_0_list=[sigmasq]*Nb_states,
                                                                   mixture_weights_0=[1./Nb_states]*Nb_states,
                                                                   sigmasq=sigmasq,
                                                                   mu=theta_hat)
                obs_distns.append(gaussMixture)

                # We set up the durations params
                dur_params = ast.literal_eval(self.priors.loc[(device,self.house,mode),'Dur params'])
                lambda_hat = dur_params['lambda_hat']
                p_hat = dur_params['p_hat']
                pi_hat = dur_params['pi_hat']
                r_hat = dur_params['r_hat']
                pnbMixture = distributions.MixtureDurationDistribution(  \
                    components=[distributions.PoissonDuration(alpha_0=1.,beta_0=1.,lmbda=lambda_hat),
                            distributions.NegativeBinomialFixedRDuration(alpha_0=0.5,beta_0=0.5,r=r_hat,p=p_hat)],
                    alpha_0=Nb_states,weights=np.array([pi_hat, 1-pi_hat]))
                dur_distns.append(pnbMixture)
        
            # We set up the transitions params
            trans_mat = np.array(trans_mat)
            trans_distn = HDPHSMMTransitions(state_dim=Nb_states,alpha=20.,gamma=20.,A=trans_mat,fullA=trans_mat,beta=[1./Nb_states]*Nb_states)

            # We set up the model for the device
            component_model = FactorialComponentHSMM(init_state_concentration=Nb_states,trans_distn=trans_distn,obs_distns=obs_distns,dur_distns=dur_distns)
            component_model_list.append(component_model)
    
        # We set up the Factorial model
        Factorial_model = Factorial(component_model_list)
        return Factorial_model
    
    def generate_aggregated_power(self,N=1):
        total_obs, all_obs, all_states = self.model.generate(N)
        uncontrolled_obs = all_obs[:,len(self.controlled_devices):].sum(1)
        return total_obs, uncontrolled_obs
    
    def generate_controlled_power(self,new_states,N=1):
        all_obs = np.zeros((N,len(self.controlled_devices)))
        for j, device in enumerate(self.controlled_devices):
            mu = self.controlled_params[device]['mu'][new_states[device]]
            sigmasq = self.controlled_params[device]['sigmasq'][new_states[device]]
            all_obs[:,j] = np.maximum(np.sqrt(sigmasq)*np.random.normal(size=N)+mu,0.).T
        total_obs = all_obs.sum(1)
        return np.double(total_obs)
    
    def measure_aggregated_power(self,new_states):
        # We compute the aggregated power at time t
        agg_power = self.uncontrolled_power[self.t] + self.generate_controlled_power(new_states)
        # We increment time
        self.t += 1
        # We send the aggregated power
        return agg_power
    
#### Disaggregation System ####

disagg_priors = pd.read_table('priors.tsv', index_col=[0,1,2])
disagg_priors.sort_index(inplace=True)

HC = pd.read_table('houses_classes.tsv', index_col=[0])
HC.sort_index(inplace=True)

class Disaggregation_system:
    SIR_priors = disagg_priors
    houses_classes = HC
    
    def __init__(self,Nb_particles,T,theta_max,theta_min,theta_off,kappa,rho):
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.theta_off = theta_off
        self.kappa = kappa
        self.rho = rho
        
        self.theta_hist = np.zeros(2,dtype=np.int64) # vector that gives the previous theta (theta_hist[0]) and the current one (theta_hist[1])
        self.state = np.random.choice([0,1]) # The operating mode for the device, 0: OFF, 1: ON
        
        # Associed power simulator 
        self.PS = Power_simulator(T)
        self.agg_power = self.PS.measure_aggregated_power({'air': self.state})
        
        # Disaggregation algorithm
        self.disaggregation_algo = self.construct_disagg_model(Nb_particles)
        
        # We start the disaggreation system for the first observation
        #self.initial_fit()
        
    def construct_disagg_model(self,Nb_particles):
        devices = self.PS.controlled_devices + self.PS.uncontrolled_devices
        classes = ['class {}'.format(c) for c in self.houses_classes.loc[self.PS.house,:].values[:4]]
        params = {'Nb_states': [],'alpha': [],'mu_0':[],'tausq_0':[],'sigmasq':[]}
        L = {device: 1 for device in devices}
        for device, device_class in zip(devices,classes):
            list_modes = self.SIR_priors.loc[(device,device_class,slice(None)),].index.get_level_values(self.SIR_priors.index.names[2])
            Nb_states = len(list_modes)
            params['Nb_states'].append(Nb_states*L[device])
            params['alpha'].append(np.repeat(1./(Nb_states*L[device]),Nb_states*L[device]))
            params['mu_0'].append(np.repeat([ast.literal_eval(self.SIR_priors.loc[(device,device_class,mode),'Obs params'])['mu_0'] for mode in list_modes],L[device]))
            params['tausq_0'].append(np.repeat([ast.literal_eval(self.SIR_priors.loc[(device,device_class,mode),'Obs params'])['tausq_0'] for mode in list_modes],L[device]))
            params['sigmasq'].append(np.repeat([ast.literal_eval(self.SIR_priors.loc[(device,device_class,mode),'Obs params'])['sigmasq'] for mode in list_modes],L[device]))
        
        disagg_model = pf.SIR_Factorial_HMM(N=Nb_particles,buffer_length=5,Nb_chains=4,**params)
        return disagg_model
        
    def cdf_off(self,theta):
        if theta >= self.theta_max:
            return 1.
        elif theta < self.theta_min:
            return 0.
        else:
            return (1-self.rho)*np.power(np.maximum(0,(theta-self.theta_off))/(self.theta_max-self.theta_off),self.kappa)

    def cdf_on(self,theta):
        if theta >= self.theta_max:
            return 1.
        elif theta <= self.theta_min:
            return 0.
        else:
            return 1.-self.cdf_off(self.theta_max+self.theta_min-theta)
        
    def p_off(self,theta_current=None,theta_previous=None):
        assert (theta_current is None and theta_previous is None) or (theta_current is not None and theta_previous is not None)
        if theta_current is None and theta_previous is None:
            theta_current=self.theta_hist[1]
            theta_previous=self.theta_hist[0]
        return np.maximum(0,self.cdf_off(theta_current)-self.cdf_off(theta_previous))/(1.-self.cdf_off(theta_previous))
    
    def p_on(self,theta_current=None,theta_previous=None):
        assert (theta_current is None and theta_previous is None) or (theta_current is not None and theta_previous is not None)
        if theta_current is None and theta_previous is None:
            theta_current=self.theta_hist[1]
            theta_previous=self.theta_hist[0]
        return np.maximum(0,self.cdf_on(theta_previous)-self.cdf_on(theta_current))/self.cdf_on(theta_previous)
    
    def p_off_zeta(self,zeta,theta_current=None,theta_previous=None):
        assert (theta_current is None and theta_previous is None) or (theta_current is not None and theta_previous is not None)
        if theta_current is None and theta_previous is None:
            theta_current=self.theta_hist[1]
            theta_previous=self.theta_hist[0]
        if zeta == 0.:
            return self.p_off(theta_current,theta_previous)
        else:
            return self.p_off(theta_current,theta_previous)/(self.p_off(theta_current,theta_previous)+(1.-self.p_off(theta_current,theta_previous))*np.exp(zeta))
    
    def p_on_zeta(self,zeta,theta_current=None,theta_previous=None):
        assert (theta_current is None and theta_previous is None) or (theta_current is not None and theta_previous is not None)
        if theta_current is None and theta_previous is None:
            theta_current=self.theta_hist[1]
            theta_previous=self.theta_hist[0]
        if zeta == 0.:
            return self.p_on(theta_current,theta_previous)
        else:
            return self.p_on(theta_current,theta_previous)*np.exp(zeta)/(self.p_on(theta_current,theta_previous)*np.exp(zeta)+1.-self.p_on(theta_current,theta_previous))
    
    def trans_mat_nominal(self,n=None):
        if n is None:
            n = int(np.floor(self.theta_max)) - int(np.ceil(self.theta_min))
            
        discrete_theta = np.linspace(self.theta_min,self.theta_max,n)
        trans_mat_off = np.array([[self.p_off(j,i) for j in discrete_theta]+[self.p_off(j,i) for j in discrete_theta] for i in discrete_theta])
        
    def disaggregate_power(self):
        verbose = 0
        self.disaggregation_algo.add_data(self.agg_power)
        estimated_states, _estimated_transition_mat, _estimated_theta = self.disaggregation_algo.next_step(verbose)
        self.state = estimated_states[0]
        
    def initial_fit(self):
        zeta = 0.
        self.disaggregation_algo.add_data(self.agg_power)
        estimated_states, _estimated_transition_mat, _estimated_theta = self.disaggregation_algo.initial_step()
        self.state = estimated_states[0]
        
        if self.state == 0:
            if np.random.uniform() <= self.p_on_zeta(zeta):
                self.state = 1
        elif self.state == 1:
            if np.random.uniform() <= self.p_off_zeta(zeta):
                self.state = 0
                
        self.agg_power = self.PS.measure_aggregated_power({'air': self.state})
        
    
    def fit(self,zeta):
        self.disaggregate_power()
        
        if self.state == 0:
            if np.random.uniform() <= self.p_on_zeta(zeta):
                self.state = 1
        elif self.state == 1:
            if np.random.uniform() <= self.p_off_zeta(zeta):
                self.state = 0
                
        self.agg_power = self.PS.measure_aggregated_power({'air': self.state})
        
#### Test ####
D = Disaggregation_system(50,100,120.,100.,105.,4.,0.8)
D.agg_power
D.disaggregation_algo.add_data(D.agg_power)
_ = D.disaggregation_algo.initial_step()