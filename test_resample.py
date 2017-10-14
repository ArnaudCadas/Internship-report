from __future__ import division
import numpy as np
np.seterr(divide='ignore')

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.plot_factorial import plot_model

from pyhsmm.plugins.factorial import models
from pyhsmm.plugins.factorial import util as futil

import pandas as pd
import sys
import os
import sqlite3
import pickle

house_id = 946

# we get the data for the house_id
conn = sqlite3.connect('../Data/powers_082016.db')
cursor = conn.cursor()
query = """SELECT * FROM powers WHERE dataid=%d""" % house_id
data = pd.read_sql_query(query, conn)
conn.close()

# we get all the columns associated with devices
devices = data.columns[2:]
T = 1000 # we keep only T observations to reduce computational time
# we stock the data for these columns and we change the unit to Watts
powers = data.loc[0:T,devices]*1000

# we aggregate devices with the same name (for example: we sum lights_plugs1, lights_plugs2, lights_plugs3 etc...)
# because we'll use the same models for devices with the same name as they should behave similarly
# so we need the aggregate values to compute proportions and usage to then choose the devices and the houses
for k in ['6','5','4','3','2']:
    for name in [x.split(k)[0] for x in powers.columns if k in x]:
        powers[name] = 0.
        for e in range(int(k)):
            name_num = name+str(e+1)
            powers[name] += powers[name_num]
            del powers[name_num]
# we get rid of the 1 in the devices names
powers.rename(columns=lambda x: x.split('1')[0], inplace=True)

devices_colors={"air":"blue", 
                "refrigerator":"red", 
                "dishwasher":"green",
                "drye":"orange",
                "furnace":"cyan", 
                "use":"black"}

devices_list = ['air','furnace','refrigerator','dishwasher']

dict_file = open( "priors.dict", "rb" ) 
hypparamss = pickle.load( dict_file  )
dict_file.close()

def get_distribution(distribution_name,params):
    if distribution_name=='Obs gaussian':
        return pyhsmm.basic.distributions.ScalarGaussianFixedvar(**params)
    elif distribution_name=='Dur nbinom':
        return pyhsmm.basic.distributions.NegativeBinomialFixedRDuration(**params)
    elif distribution_name=='Dur poisson':
        return pyhsmm.basic.distributions.PoissonDuration(**params)

L = 2 # Number of times we repeat the distribution for each mode
    
distns = {}
for device in devices_list:
    distns[device] = {'Obs': [],'Dur': []}
    for mode, mode_value in hypparamss[device].items():
        for dist_name in mode_value.keys():
            for _ in range(L):
                distns[device][dist_name.split(' ')[0]].append(get_distribution(dist_name,hypparamss[device][mode][dist_name]))

### construct posterior model
posteriormodel = models.Factorial([models.FactorialComponentHSMM(
        init_state_concentration=2.,
        alpha=1.,gamma=4.,
        obs_distns=distns[device]['Obs'],
        dur_distns=distns[device]['Dur'],
        trunc=200)
    for device in devices_list])

posteriormodel.add_data(data=powers['use'].values)

nsubiter=25
for itr in progprint_xrange(20):
    posteriormodel.resample_model(min_extra_noise=0.1,max_extra_noise=100.**2,niter=nsubiter)
        
