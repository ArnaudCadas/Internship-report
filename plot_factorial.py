from __future__ import division
import numpy as np
np.seterr(divide='ignore')

import pyhsmm
from pyhsmm.plugins.factorial import models
from pyhsmm.plugins.factorial import util as futil

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import cm

def plot_stateseq(s,ax,Interval):
    num_states = s.state_dim
    data = s.data
    stateseq = s.stateseq
    # Colors
    cmap = cm.get_cmap()
    state_usages = np.bincount(stateseq,minlength=num_states)
    freqs = state_usages / state_usages.sum()
    #used_states = sorted(set(stateseq), key=lambda x: freqs[x], reverse=True)
    used_states = sorted(set(stateseq), key=lambda x: s.obs_distns[x].mu)
    unused_states = [idx for idx in range(num_states) if idx not in used_states]

    colorseq = np.linspace(0,1,num_states)
    state_colors = dict((idx, v) for idx, v in zip(used_states,colorseq))

    for state in unused_states:
        state_colors[state] = cmap(1.)

    # State sequence colors
    from pyhsmm.util.general import rle

    stateseq_norep, durations = rle(stateseq)
    datamin, datamax = data.min(), data.max()

    x, y = np.hstack((0,durations.cumsum())), np.array([datamin,datamax])
    C = np.atleast_2d([state_colors[state] for state in stateseq_norep])

    ax.pcolormesh(x,y,C,cmap=cm.get_cmap('summer'),vmin=0,vmax=1,alpha=0.8)
    ax.set_ylim((datamin,datamax))
    ax.set_xlim((Interval[0],Interval[1]))
    ax.set_yticks([])


def plot_model(posteriormodel,true_values=None,T=None,Interval=None,devices_colors=None):
    # Figure
    if T is None:
        T = len(posteriormodel.states_list[0].data)
    if Interval is None:
        Interval=[0,T]
    nb_components = len(posteriormodel.component_models)
    height = 6
    fig = plt.figure(figsize=(height*2,height*nb_components))
    
    # Axes
    gs = GridSpec(nb_components,1)
    cgs = []
    scgs = []
    for i in range(nb_components):
        cgs.append(GridSpecFromSubplotSpec(2,1,subplot_spec=gs[i],
                                           height_ratios=[0.8,0.2],
                                           hspace=0.3))
        scgs.append(GridSpecFromSubplotSpec(len(posteriormodel.states_list),1,
                                            subplot_spec=cgs[i][1]))
    
        feature_ax = plt.subplot(cgs[i][0])
        stateseq_axs = [plt.subplot(scgs[i][idx]) for idx in range(len(posteriormodel.states_list))]

        for ax in stateseq_axs:
            ax.grid('off')

        # Plot the power estimated values
        if true_values is not None:
            device_name=true_values.columns[i]
            feature_ax.plot(posteriormodel.states_list[0].museqs[:,i],linestyle='--',color=devices_colors[device_name],label="Estimated")
            feature_ax.plot(true_values.iloc[:,i],color=devices_colors[device_name],label="True")
            feature_ax.set_title(device_name)
            feature_ax.set_ylabel("Power (Watts)")
            feature_ax.legend(loc='best')
        else:
            feature_ax.plot(posteriormodel.states_list[0].museqs[:,i])
            feature_ax.set_title("Estimation of component {}".format(i+1))
        feature_ax.set_xlim([Interval[0],Interval[1]])

        # Plot the hidden states sequence
        for s,ax,data in zip(posteriormodel.states_list,stateseq_axs,[s.data for s in posteriormodel.states_list]):
            plot_stateseq(s.states_list[i],ax,Interval)
            ax.set_title("Hidden states sequence")
        plt.draw()
