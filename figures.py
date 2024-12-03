#############  PACKAGES  #############
import sys, os
from copy import deepcopy
from importlib import reload

import numpy as np

import scipy as sp
import scipy.stats as st
import scipy.interpolate as sp_interpolate
from scipy.interpolate import interp1d

import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

import seaborn as sns

import mplot as mp

import re
import math
from dataclasses import dataclass
import json
############# PARAMETERS #############

# Standard color scheme
BKCOLOR  = '#252525'
C_BEN    = '#EB4025'
C_BEN_LT = '#F08F78'
C_NEU    = '#969696'
C_NEU_LT = '#E8E8E8'
C_DEL    = '#3E8DCF'
C_DEL_LT = '#78B4E7'
C_group  = ['#32b166', '#e5a11c', '#a48cf4', '#ff69b4', '#ff8c00', '#36ada4', '#f0e54b',
            '#f77189', '#f7754f', '#dc8932', '#c39532', '#ae9d31', '#97a431', '#77ab31', 
            '#31b33e', '#33b07a', '#35ae93', '#37abb4', '#38a9c5', '#3aa5df', '#6e9bf4', 
            '#cc7af4', '#f45cf2', '#f565cc', '#f66bad']

# Plot conventions

def cm2inch(x): return float(x)/2.54
SINGLE_COLUMN   = cm2inch(8.5)
DOUBLE_COLUMN   = cm2inch(17.4)

# paper style
FONTFAMILY   = 'Arial'
SIZESUBLABEL = 8
SIZELABEL    = 6
SIZETICK     = 6
SMALLSIZEDOT = 6.
SIZELINE     = 0.6

GOLDR        = (1.0 + np.sqrt(5)) / 2.0
TICKLENGTH   = 3
TICKPAD      = 3
AXWIDTH      = 0.4

FIGPROPS = {
    'transparent' : True,
    #'bbox_inches' : 'tight'
}

DEF_ERRORPROPS = {
    'mew'        : AXWIDTH,
    'markersize' : SMALLSIZEDOT/2,
    'fmt'        : 'o',
    'elinewidth' : SIZELINE/2,
    'capthick'   : 0,
    'capsize'    : 0
}

DEF_LABELPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZELABEL,
    'color'  : BKCOLOR
}

DEF_SUBLABELPROPS = {
    'family'  : FONTFAMILY,
    'size'    : SIZESUBLABEL+1,
    'weight'  : 'bold',
    'ha'      : 'center',
    'va'      : 'center',
    'color'   : 'k',
    'clip_on' : False
}

# GLOBAL VARIABLES -- simulation
NUC = ['-', 'A', 'C', 'G', 'T']

# GitHub
SIM_DIR = 'data/simulation'
# HIV_DIR = 'data/HIV'
FIG_DIR = 'figures'

############# PLOTTING  FUNCTIONS #############
def plot_simple(**pdata):
    """
    Example evolutionary trajectory for a binary 20-site system
    """

    # unpack passed data
    sim_dir       = pdata['sim_dir']            # 'simple'
    name          = pdata['name']           # '0'
    output        = pdata['output']         # 'output'

    seq_length    = pdata['seq_length']     # 20
    generations   = pdata['generations']    # 500
    ytick_t       = pdata['ytick_t']
    yminorticks_t = pdata['yminorticks_t']

    bene          = pdata['bene']           # [0,1]
    dele          = pdata['dele']           # [4,5]
    p_1           = pdata['p_1']            # [6,7] , special sites 1
    p_2           = pdata['p_2']            # [8,9] , special sites 2

    fB            = pdata['s_ben']          # 0.02
    fD            = pdata['s_del']          # -0.02
    fi_1          = pdata['fi_1']           # time-varying selection coefficient for special sites 1
    fi_2          = pdata['fi_2']           # time-varying selection coefficient for special sites 2

    savepdf       = pdata['savepdf']         # True

    # get data
    data        = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR,sim_dir,name.split('_')[0]))
    timepoints  = int(generations) + 1
    times       = np.linspace(0,generations,timepoints)
    
    data_full   = np.load('%s/%s/%s/c_%s.npz'%(SIM_DIR,sim_dir,output,name), allow_pickle="True")
    sc_full     = data_full['selection']
    TimeVaryingSC = [np.average(sc_full[i]) for i in range(seq_length)]

    # Allele frequency x
    x     = []
    for t in range(timepoints):
        idx    = data.T[0]==times[t]
        t_data = data[idx].T[2:].T
        t_num  = data[idx].T[1].T
        t_freq = np.einsum('i,ij->j', t_num, t_data) / float(np.sum(t_num))
        x.append(t_freq)
    x = np.array(x).T # get allele frequency (binary case)

    # set up figure grid
    fig   = plt.figure(figsize=(SINGLE_COLUMN, SINGLE_COLUMN*1.1),dpi=500)

    box_tra = dict(left=0.15, right=0.92, bottom=0.72, top=0.95)
    box_tc  = dict(left=0.15, right=0.92, bottom=0.38, top=0.61)
    box_le  = dict(left=0.10, right=0.42, bottom=0.05, top=0.25)
    box_sc  = dict(left=0.55, right=0.92, bottom=0.05, top=0.25)

    gs_tra  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra)
    gs_tc   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)
    gs_le   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_le)
    gs_sc   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc)

    ax_tra  = plt.subplot(gs_tra[0, 0])
    ax_tc   = plt.subplot(gs_tc[0, 0])
    ax_le   = plt.subplot(gs_le[0, 0])
    ax_sc   = plt.subplot(gs_sc[0, 0])

    dx = -0.08
    dy =  0.02

    # color for time-varying mutations
    c_sin = 5 # index for sin mutation
    c_cos = 2 # index for cos mutation

    ## a -- allele frequencies
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'ylim':        [0, 1.10],
               'yticks':      [0, 1.00],
               'yticklabels' :[0, 1],
               'yminorticks': [0.25, 0.5, 0.75,1],
               'nudgey':      1,
               'xlabel':      'Generation',
               'ylabel':      'Allele\nfrequency, ' + r'$x$',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1 },
               'axoffset':    0.1,
               'theme':       'open'}

    # all individual sites
    for i in range(seq_length):
        pprops['plotprops']['alpha'] = 1

        if i in bene:
            mp.line(ax=ax_tra, x=[times], y=[x[i]], colors=[C_BEN], **pprops)
        elif i in dele:
            mp.line(ax=ax_tra, x=[times], y=[x[i]], colors=[C_DEL], **pprops)
        elif i in p_1:
            mp.line(ax=ax_tra, x=[times], y=[x[i]], colors=[C_group[c_sin]], **pprops)
        elif i in p_2:
            mp.line(ax=ax_tra, x=[times], y=[x[i]], colors=[C_group[c_cos]], **pprops)
        else:
            mp.line(ax=ax_tra, x=[times], y=[x[i]], colors=[C_NEU], **pprops)

    pprops['plotprops'] = {'lw': SIZELINE, 'ls': '-', 'alpha': 0 }
    mp.plot(type='line',ax=ax_tra, x=[[0,500]], y=[[1,1]], colors=[C_NEU], **pprops)

    ax_tra.text( box_tra['left']+dx,  box_tra['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## b -- time-varying selection coefficients (sin/cos)
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'ylim':        [ytick_t[0], ytick_t[-1]],
               'yticks':      ytick_t,
               'yminorticks': yminorticks_t,
               'yticklabels': [int(i*100) for i in ytick_t],
               'nudgey':      1,
               'xlabel':      'Generation',
               'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1 },
               'axoffset':    0.1,
               'theme':       'open'}

    for ii in p_1:
        sc_p = sc_full[ii]
        mp.line(ax=ax_tc, x=[times], y=[sc_p], colors=[C_group[c_sin]], **pprops)
    for ii in p_2:
        sc_p = sc_full[ii]
        mp.line(ax=ax_tc, x=[times], y=[sc_p], colors=[C_group[c_cos]], **pprops)

    pprops['plotprops']['ls'] = ':'
    mp.line(            ax=ax_tc, x=[times], y=[fi_1], colors=[C_group[c_sin]], **pprops)
    mp.plot(type='line',ax=ax_tc, x=[times], y=[fi_2], colors=[C_group[c_cos]], **pprops)

    ax_tc.text(box_tc['left']+dx, box_tc['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ##  add legend
    sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' }
    lprops = { 'lw' : SIZELINE, 'ls' : '-', 'alpha': 1 }
    pprops = { 'xlim':        [ -1 ,    6],
               'ylim':        [-0.04, 0.04],
               'yticks':      [],
               'xticks':      [],
               'theme':       'open',
               'hide':        ['left','bottom'] }

    legend_x  = 1.5
    legend_dx = 0.8
    x_dot     = legend_x - legend_dx
    x_line    = [legend_x - 1.4*legend_dx, legend_x - 0.6*legend_dx]
    legend_y  = 0.035
    legend_dy = -0.015

    # constant labels
    c_coe1         = [C_BEN, C_NEU, C_DEL]
    legend_t  = ['Beneficial', 'Neutral', 'Deleterious']
    for k in range(len(legend_t)):
        mp.scatter(ax=ax_le, x=[[x_dot]], y=[[legend_y + (k *legend_dy)]],colors=[c_coe1[k]],plotprops=sprops,**pprops)
        ax_le.text(legend_x, legend_y + (k*legend_dy), legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    # time-varying labels
    yy_sin = legend_y + 2.9 * legend_dy
    yy_cos = legend_y + 3.1 * legend_dy
    mp.line(ax=ax_le, x=[x_line], y=[[yy_sin, yy_sin]], colors=[C_group[c_sin]], plotprops=lprops, **pprops)
    mp.line(ax=ax_le, x=[x_line], y=[[yy_cos, yy_cos]], colors=[C_group[c_cos]], plotprops=lprops, **pprops)
    ax_le.text(legend_x, legend_y + (3*legend_dy), 'Time varying', ha='left', va='center', **DEF_LABELPROPS)

    # true coefficient labels
    lprops['ls'] = ':'
    yy =  [legend_y + 4.0 * legend_dy, legend_y + 4.2 * legend_dy, legend_y + 4.4 * legend_dy]
    mp.line(ax=ax_le, x=[x_line], y=[[yy[0], yy[0]]], colors=[C_group[c_sin]], plotprops=lprops, **pprops)
    mp.line(ax=ax_le, x=[x_line], y=[[yy[1], yy[1]]], colors=[C_group[c_cos]], plotprops=lprops, **pprops)
    mp.plot(type='line',ax=ax_le, x=[x_line], y=[[yy[2], yy[2]]], colors=[BKCOLOR], plotprops=lprops, **pprops)
    ax_le.text(legend_x, yy[1], 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ## c -- constant selection coefficients (beneficial/neutral/deleterious)
    sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' }
    pprops = { 'xlim':        [ -0.3,    6],
               'ylim':        [-0.04, 0.04],
               'yticks':      [-0.04, 0, 0.04],
               'yminorticks': [-0.03,-0.02, -0.01, 0.01, 0.02, 0.03],
               'yticklabels': [-4, 0, 4],
               'xticks':      [],
               'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'theme':       'open',
               'hide':        ['bottom'] }
    
    nB        = len(bene)
    nD        = len(dele)
    nN        = seq_length-nB-nD-len(p_1)-len(p_2)

    x_ben = np.random.normal(1, 0.08, nB)
    x_neu = np.random.normal(3, 0.16, nN)
    x_del = np.random.normal(5, 0.08, nD)
    x_bar = np.hstack([x_ben,x_neu,x_del])

    for i in range(seq_length):
        if i not in p_1 and i not in p_2:
            xdat = [x_bar[i]]
            ydat = [TimeVaryingSC[i]]
            if i in bene:
                mp.scatter(ax=ax_sc, x=[xdat], y=[ydat],colors=[C_BEN],plotprops=sprops,**pprops)
            elif i in dele:
                mp.scatter(ax=ax_sc, x=[xdat], y=[ydat],colors=[C_DEL],plotprops=sprops,**pprops)
            else:
                mp.scatter(ax=ax_sc, x=[xdat], y=[ydat],colors=[C_NEU],plotprops=sprops,**pprops)

    mp.line(ax=ax_sc, x=[[0.5, 1.5]], y=[[fB,fB]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    mp.line(ax=ax_sc, x=[[2, 4]], y=[[0,0]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    mp.plot(type ='line',ax=ax_sc,x=[[4.5, 5.5]], y=[[fD,fD]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    ax_sc.text(box_sc['left']+dx, box_sc['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    if savepdf==True:
        plt.savefig('%s/fig-%s.pdf' % (FIG_DIR,sim_dir), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    else:
        plt.savefig('%s/%s/%s.jpg' % (FIG_DIR,sim_dir,name), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_trait(**pdata):
    """
    Example evolutionary trajectory for a binary 20-site system
    """

    # unpack passed data
    sim_dir       = pdata['sim_dir']            # 'trait'
    output        = pdata['output']         # 'output'
    name          = pdata['name']           #'1-con'

    seq_length    = pdata['seq_length']     # 20
    generations   = pdata['generations']    # 500
    ytick_e       = pdata['ytick_e']
    yminorticks_e = pdata['yminorticks_e']
    ytick_f       = pdata['ytick_f']
    yminorticks_f = pdata['yminorticks_f']

    escape_group  = pdata['escape_group']   # escape group, random generated
    p_sites       = pdata['p_sites']        # special sites, random generated

    bene          = pdata['bene']           # [0,1,2,3]
    dele          = pdata['dele']           # [16,17,18,19]
    fB            = pdata['s_ben']          # 0.02
    fD            = pdata['s_del']          # -0.02
    fn            = pdata['fn']             # time-varying escape coefficient
    fi            = pdata['fi']             # time-varying selection coefficient

    savepdf       = pdata['savepdf']         # True

    # get data
    data        = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR,sim_dir,name.split('_')[0]))
    ne          = len(escape_group)
    timepoints  = int(generations) + 1
    times       = np.linspace(0,generations,timepoints)

    data_full   = np.load('%s/%s/output%s/c_%s.npz'%(SIM_DIR,sim_dir,output,name), allow_pickle="True")
    sc_full     = data_full['selection']
    TimeVaryingSC = [np.average(sc_full[i]) for i in range(seq_length)]
    TimeVaryingTC = sc_full[-ne:]

    nB        = len(bene)
    nD        = len(dele)
    nN        = seq_length-nB-nD

    # Allele frequency x
    x     = []
    for t in range(timepoints):
        idx    = data.T[0]==times[t]
        t_data = data[idx].T[2:].T
        t_num  = data[idx].T[1].T
        t_freq = np.einsum('i,ij->j', t_num, t_data) / float(np.sum(t_num))
        x.append(t_freq)
    x = np.array(x).T # get allele frequency (binary case)

    # Escape group frequency y
    y    = []
    for t in range(timepoints):
        idx    = data.T[0]==times[t]
        t_num  = data[idx].T[1].T
        t_fre     = []
        for n in range(len(escape_group)):
            t_data_n  = t_num*0
            for nn in escape_group[n]:
                t_data_n += data[idx].T[nn+2]
            for k in range(len(t_data_n)):
                if t_data_n[k] != 0:
                    t_data_n[k] = 1
            t_freq_n = np.einsum('i,i', t_num, t_data_n) / float(np.sum(t_num))
            t_fre.append(t_freq_n)
        y.append(t_fre)
    y = np.array(y).T # get polygenic frequency

    def find_in_nested_list(a, i):
        for index, sublist in enumerate(a):
            if i in sublist:
                return True, index
        return False, None

    # set up figure grid
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_tra1 = dict(left=0.10, right=0.34, bottom=0.60, top=0.95)
    box_tra2 = dict(left=0.42, right=0.66, bottom=0.60, top=0.95)
    box_tra3 = dict(left=0.71, right=0.95, bottom=0.60, top=0.95)
    box_lab  = dict(left=0.05, right=0.15, bottom=0.10, top=0.45)
    box_sc   = dict(left=0.24, right=0.40, bottom=0.10, top=0.45)
    box_sp   = dict(left=0.48, right=0.67, bottom=0.10, top=0.45)
    box_tc   = dict(left=0.76, right=0.95, bottom=0.10, top=0.45)

    gs_tra1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra1)
    gs_tra2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra2)
    gs_tra3 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra3)
    gs_lab  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_lab)
    gs_sc   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc)
    gs_sp   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sp)
    gs_tc   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)

    ax_tra1 = plt.subplot(gs_tra1[0, 0])
    ax_tra2 = plt.subplot(gs_tra2[0, 0])
    ax_tra3 = plt.subplot(gs_tra3[0, 0])
    ax_lab  = plt.subplot(gs_lab[0, 0])
    ax_sc   = plt.subplot(gs_sc[0, 0])
    ax_sp   = plt.subplot(gs_sp[0, 0])
    ax_tc   = plt.subplot(gs_tc[0, 0])

    dx = -0.04
    dy =  0.02

    C_tv = C_group[2]

    ## a,b -- allele frequencies
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'ylim':        [0, 1.10],
               'yticks':      [0, 1.00],
               'yticklabels' :[0, 1],
               'yminorticks': [0.25, 0.5, 0.75,1],
               'nudgey':      1,
               'xlabel':      'Generation',
               'ylabel':      'Allele\nfrequency, ' + r'$x$',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1 },
               'axoffset':    0.1,
               'theme':       'open'}

    # all individual sites
    for i in range(seq_length):
        if i not in p_sites:
            if i < len(bene):
                mp.line(ax=ax_tra1, x=[times], y=[x[i]], colors=[C_BEN], **pprops)
            elif i >= seq_length-len(dele):
                mp.line(ax=ax_tra1, x=[times], y=[x[i]], colors=[C_DEL], **pprops)
            else:
                mp.line(ax=ax_tra1, x=[times], y=[x[i]], colors = [C_NEU], **pprops)
        else:
            mp.line(ax=ax_tra2, x=[times], y=[x[i]], colors = [C_tv], **pprops)
    
    pprops['plotprops'] = {'lw': SIZELINE, 'ls': '-', 'alpha': 0 }
    pprops['ylabel'] = 'Constant allele\nfrequency, ' + r'$x$'
    mp.plot(type='line',ax=ax_tra1, x=[[0,1000]], y=[[1,1]], colors=[C_NEU], **pprops)
    pprops['ylabel'] = 'Time-varying allele\nfrequency, ' + r'$x$'
    mp.plot(type='line',ax=ax_tra2, x=[[0,1000]], y=[[1,1]], colors=[C_NEU], **pprops)

    ax_tra1.text( box_tra1['left']+dx,  box_tra1['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_tra2.text( box_tra2['left']+dx,  box_tra2['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c -- allele frequencies - binary trait and its alleles
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'ylim':        [0, 1.10],
               'yticks':      [0, 1.00],
               'yticklabels' :[0, 1],
               'yminorticks': [0.25, 0.5, 0.75,1],
               'nudgey':      1,
               'xlabel':      'Generation',
               'ylabel':      'Allele\nfrequency, ' + r'$x$',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1 },
               'axoffset':    0.1,
               'theme':       'open'}

    pprops['plotprops']['alpha'] = 0.4
    for i in range(seq_length):
        # if the site is escape site, plot it in figure b
        found, group = find_in_nested_list(escape_group, i)
        if found:
            mp.line(ax=ax_tra3, x=[times], y=[x[i]], colors=[C_group[group]], **pprops)

    # escape group
    pprops['plotprops']['alpha'] = 1.0
    for n in range(ne):
        # mp.line(ax=ax_tra3, x=[times], y=[y[n]], colors=[C_group[n]], **pprops_c)
        mp.plot(type='line',ax=ax_tra3, x=[times], y=[y[n]], colors=[C_group[n]], **pprops)

    ax_tra3.text( box_tra3['left']+dx,  box_tra3['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ##  add legend
    sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' }
    lprops = { 'lw' : SIZELINE, 'ls' : ':', 'alpha': 1 }
    pprops = { 'xlim':        [ -1 ,    6],
               'ylim':        [-0.03, 0.03],
               'yticks':      [],
               'xticks':      [],
               'theme':       'open',
               'hide':        ['left','bottom'] }

    # individual loci label
    legend_x  =  0
    legend_dx = -0.8
    legend_y  = 0.028 
    legend_dy = -0.008
    c_coe1    = [C_BEN, C_NEU, C_DEL, C_tv]
    legend_t  = ['Beneficial', 'Neutral', 'Deleterious','Time-varying']
    for k in range(len(legend_t)):
        mp.scatter(ax=ax_lab, x=[[legend_x+legend_dx]], y=[[legend_y + (k *legend_dy)]],colors=[c_coe1[k]],plotprops=sprops,**pprops)
        ax_lab.text(legend_x, legend_y + (k*legend_dy), legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    # binary traits-related label
    mp.scatter(ax=ax_lab, x=[[legend_x+legend_dx]], y=[[legend_y + (4 *legend_dy)]],colors=[C_group[0]],plotprops=sprops,**pprops)
    sprops['alpha'] = 0.6
    mp.scatter(ax=ax_lab, x=[[legend_x+legend_dx]], y=[[legend_y + (5 *legend_dy)]],colors=[C_group[0]],plotprops=sprops,**pprops)
    ax_lab.text(legend_x, legend_y + (4*legend_dy), 'Binary trait', ha='left', va='center', **DEF_LABELPROPS)
    ax_lab.text(legend_x, legend_y + (5*legend_dy), 'Escape locus', ha='left', va='center', **DEF_LABELPROPS)

    # true coefficient labels
    xx = [legend_x+1.5*legend_dx, legend_x+0.5*legend_dx]
    yy = legend_y + 6.2 * legend_dy
    mp.plot(type='line',ax=ax_lab, x=[xx], y=[[yy,yy]], colors=[BKCOLOR], plotprops=lprops, **pprops)
    ax_lab.text(legend_x, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ## d -- individual beneficial/neutral/deleterious selection coefficients

    pprops = { 'xlim':        [ -0.3,    6],
               'ylim':        [-0.03, 0.03],
               'yticks':      [-0.03, 0, 0.03],
               'yminorticks': [-0.02, -0.01, 0.01, 0.02],
               'yticklabels': [-3, 0, 3],
               'xticks':      [],
               'ylabel':      'Inferred constant\nselection coefficient, ' + r'$\hat{s}$' + ' (%)',
               'theme':       'open',
               'hide':        ['bottom'] }
    
    x_ben = np.random.normal(1, 0.08, nB)
    x_neu = np.random.normal(3, 0.16, nN)
    x_del = np.random.normal(5, 0.08, nD)
    x_bar = np.hstack([x_ben,x_neu,x_del])
    
    sprops['alpha'] = 1.0
    for i in range(seq_length):
        found, group = find_in_nested_list(escape_group, i)
        if i not in p_sites:
            xdat = [x_bar[i]]
            ydat = [TimeVaryingSC[i]]
            if found:
                sprops['alpha'] = 0.6
                mp.scatter(ax=ax_sc, x=[xdat], y=[ydat],colors=[C_group[group]],plotprops=sprops,**pprops)
                sprops['alpha'] = 1.0
            else:
                if i < nB:
                    mp.scatter(ax=ax_sc, x=[xdat], y=[ydat],colors=[C_BEN],plotprops=sprops,**pprops)
                elif i >= seq_length-nD:
                    mp.scatter(ax=ax_sc, x=[xdat], y=[ydat],colors=[C_DEL],plotprops=sprops,**pprops)
                else:
                    mp.scatter(ax=ax_sc, x=[xdat], y=[ydat],colors=[C_NEU],plotprops=sprops,**pprops)

    mp.line(ax=ax_sc, x=[[0.5, 1.5]], y=[[fB,fB]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    mp.line(ax=ax_sc, x=[[2, 4]], y=[[0,0]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    mp.plot(type ='line',ax=ax_sc,x=[[4.5, 5.5]], y=[[fD,fD]], colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    
    ax_sc.text(box_sc['left']+dx, box_sc['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## e -- special sites selection coefficients
    pprops = {  'xticks':      [0, 200, 400, 600, 800, 1000],
                'ylim':        [ytick_f[0], ytick_f[-1]],
                'yticks':      ytick_f,
                'yminorticks': yminorticks_f,
                'yticklabels': [int(i*100) for i in ytick_f],
                'nudgey':      1,
                'xlabel':      'Generation',
                'ylabel':      'Inferred time-varying\nselection coefficient, ' + r'$\hat{s}$' + ' (%)',
                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1 },
                'axoffset':    0.1,
                'theme':       'open'}

    for ii in range(len(p_sites)):
        p_index = p_sites[ii]
        sc_p = sc_full[p_index]
        mp.line(ax=ax_sp, x=[times], y=[sc_p], colors=[C_tv], **pprops)

    pprops['plotprops']['ls'] = ':'
    mp.plot(type='line',ax=ax_sp, x=[times], y=[fi], colors=[C_tv], **pprops)

    ax_sp.text(box_sp['left']+dx, box_sp['top']+dy, 'e'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## f -- trait coefficients
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'ylim':        [ytick_e[0], ytick_e[-1]],
               'yticks':      ytick_e,
               'yminorticks': yminorticks_e,
               'yticklabels': [int(i*100) for i in ytick_e],
               'nudgey':      1,
               'xlabel':      'Generation',
               'ylabel':      'Inferred time-varying\ntrait coefficient, ' + r'$\hat{s}$' + ' (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1 },
               'axoffset':    0.1,
               'theme':       'open'}

    yy =  0.05
    for n in range(ne):
        pprops['plotprops']['ls'] = ':'
        mp.line(ax=ax_tc, x=[times], y=[fn], colors=[C_group[n]], **pprops)

        pprops['plotprops']['ls'] = '-'
        mp.plot(type='line',ax=ax_tc, x=[times], y=[TimeVaryingTC[n]], colors=[C_group[n]], **pprops)

    ax_tc.text(box_tc['left']+dx, box_tc['top']+dy, 'f'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    if savepdf:
        plt.savefig('%s/fig-%s.pdf' % (FIG_DIR,sim_dir), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()
    else:
        plt.savefig('%s/%s/%s.jpg' % (FIG_DIR,sim_dir,name), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    
def plot_simple_his(**pdata):

    """
    histogram of selection coefficients and trait coefficients
    """

    # unpack passed data
    sim_dir       = pdata['sim_dir']        # 'simple'
    out_dir       = pdata['out_dir']         # 'output'
    beta          = pdata['beta']          # 4
    generations   = pdata['generations']    # 500
    xtick_h       = pdata['xtick_h']
    ytick_h       = pdata['ytick_h']
    ytick_t       = pdata['ytick_t']
    yminorticks_t = pdata['yminorticks_t']

    p_1           = pdata['p_1']            # [6,7] , special sites 1
    p_2           = pdata['p_2']            # [8,9] , special sites 2
    fB            = pdata['s_ben']          # 0.02
    fD            = pdata['s_del']          # -0.02
    fi_1          = pdata['fi_1']           # time-varying selection coefficient for special sites 1
    fi_2          = pdata['fi_2']           # time-varying selection coefficient for special sites 2

    savepdf       = pdata['savepdf']           # True

    timepoints  = int(generations) + 1
    times       = np.linspace(0,generations,timepoints)

    # data for selection coefficients for different simulations
    df       = pd.read_csv('%s/%s/mpl_collected_%s.csv' % (SIM_DIR,sim_dir,beta), memory_map=True)
    ben_cols = ['sc_%d' % i for i in [0,1]]
    neu_cols = ['sc_%d' % i for i in [2,3]]
    del_cols = ['sc_%d' % i for i in [4,5]]

    # get data for inference results for different simulations
    tc_all_1   = np.zeros((100,len(p_1),generations+1))
    tc_all_2   = np.zeros((100,len(p_2),generations+1))

    for k in range(100):
        name = str(k)
        data_full     = np.load('%s/%s/%s/c_%s.npz'%(SIM_DIR,sim_dir,out_dir,name), allow_pickle="True")
        sc_full       = data_full['selection']
        for ii in p_1:
            tc_all_1[k][p_1.index(ii)] = sc_full[ii]
        for ii in p_2:
            tc_all_2[k][p_2.index(ii)] = sc_full[ii]
        
    tc_ave_1 = np.zeros((len(p_1),generations+1))
    tc_1     = np.swapaxes(tc_all_1, 0, 2)
    for n in range(len(p_1)):
        for t in range(len(tc_all_1[0][0])):
            tc_ave_1[n][t] = np.average(tc_1[t][n])

    tc_ave_2 = np.zeros((len(p_2),generations+1))
    tc_2     = np.swapaxes(tc_all_2, 0, 2)
    for n in range(len(p_2)):
        for t in range(len(tc_all_2[0][0])):
            tc_ave_2[n][t] = np.average(tc_2[t][n])

    # PLOT FIGURE
    ## set up figure grid

    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.8
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_se  = dict(left=0.10, right=0.92, bottom=0.65, top=0.95)
    box_lab = dict(left=0.05, right=0.15, bottom=0.10, top=0.45)
    box_tc1 = dict(left=0.24, right=0.52, bottom=0.10, top=0.50)
    box_tc2 = dict(left=0.60, right=0.92, bottom=0.10, top=0.50)

    gs_se  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_se)
    gs_lab = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_lab)
    gs_tc1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc1)
    gs_tc2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc2)

    ax_se  = plt.subplot(gs_se[0, 0])
    ax_lab = plt.subplot(gs_lab[0, 0])
    ax_tc1 = plt.subplot(gs_tc1[0, 0])
    ax_tc2 = plt.subplot(gs_tc2[0, 0])

    dx = -0.04
    dy =  0.03

    c_sin = 5
    c_cos = 2

    ## a -- histogram for selection coefficients
    dashlineprops = { 'lw' : SIZELINE * 1.5, 'ls' : ':', 'alpha' : 0.5, 'color' : BKCOLOR }
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim':        [xtick_h[0], xtick_h[-1]],
               'xticks':      xtick_h,
               'xticklabels': [int(i*100) for i in xtick_h],
               'ylim':        [ytick_h[0], ytick_h[-1]],
               'yticks':      ytick_h,
               'xlabel'      : 'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)',
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(-0.04, 0.04, 0.001),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    colors     = [C_BEN, C_NEU, C_DEL]
    tags       = ['beneficial', 'neutral', 'deleterious']
    cols       = [ben_cols, neu_cols, del_cols]
    s_true_loc = [fB, 0, fD]

    for i in range(len(tags)):
        x = [np.array(df[cols[i]]).flatten()]
        tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
        ax_se.text(s_true_loc[i], ytick_h[-1]*1.04, r'$s_{%s}$' % (tags[i]), color=colors[i], **tprops)
        dashlineprops['color'] = colors[i]
        ax_se.axvline(x=s_true_loc[i], **dashlineprops)
        if i<len(tags)-2: mp.hist(             ax=ax_se, x=x, colors=[colors[i]], **pprops)
        else:             mp.plot(type='hist', ax=ax_se, x=x, colors=[colors[i]], **pprops)

    ax_se.text(  box_se['left']+dx,  box_se['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ##  add legend
    pprops = { 'xlim':        [ -1 ,    6],
               'ylim':        [-0.05, 0.05],
               'yticks':      [],
               'xticks':      [],
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 },
               'theme':       'open',
               'hide':        ['left','bottom'] }

    yy =  -0.021
    coef_legend_dy = 0.021
    xx_line = [-0.9, 1.3]
    yy_line = np.zeros((3,2))
    for i in range(3):
        for j in range(2):
            yy_line[i][j] = yy + coef_legend_dy*((2 - i)+ 0.2 * j - 0.1)
    c_cols = [C_group[c_sin], C_group[c_cos]]

    mp.line(ax=ax_lab, x=[xx_line], y=[[yy_line[0][0], yy_line[0][0]]], colors=[c_cols[0]], **pprops)
    mp.line(ax=ax_lab, x=[xx_line], y=[[yy_line[0][1], yy_line[0][1]]], colors=[c_cols[1]], **pprops)
    ax_lab.text(2, yy+coef_legend_dy*2, 'Inferred \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['alpha'] = 1.0
    pprops['plotprops']['lw'] = SIZELINE*3
    mp.line(ax=ax_lab, x=[xx_line], y=[[yy_line[1][0], yy_line[1][0]]], colors=[c_cols[0]], **pprops)
    mp.line(ax=ax_lab, x=[xx_line], y=[[yy_line[1][1], yy_line[1][1]]], colors=[c_cols[1]], **pprops)
    ax_lab.text(2, yy+coef_legend_dy, 'Average \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['ls'] = ':'
    mp.line(ax=ax_lab, x=[xx_line], y=[[yy_line[2][0], yy_line[2][0]]], colors=[c_cols[0]], **pprops)
    mp.plot(type='line',ax=ax_lab, x=[xx_line], y=[[yy_line[2][1], yy_line[2][1]]], colors=[c_cols[1]], **pprops)
    ax_lab.text(2, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ## b  -- escape coefficients
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'ylim':        [ytick_t[0], ytick_t[-1]],
               'yticks':      ytick_t,
               'yminorticks': yminorticks_t,
               'yticklabels': [int(i*100) for i in ytick_t],
               'nudgey':      1,
               'xlabel':      'Generation',
               'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.15 },
               'axoffset':    0.1,
               'theme':       'open'}

    for n in range(len(p_1)):
        pprops['plotprops']['alpha'] = 0.15
        pprops['plotprops']['lw'] = SIZELINE
        for k in range(100):
            mp.line(ax=ax_tc1, x=[times], y=[tc_all_1[k][n]], colors=[C_group[c_sin]], **pprops)

        pprops['plotprops']['alpha'] = 1
        pprops['plotprops']['lw'] = SIZELINE*3
        mp.line(ax=ax_tc1, x=[times], y=[tc_ave_1[n]], colors=[C_group[c_sin]], **pprops)

    for n in range(len(p_2)):
        pprops['plotprops']['alpha'] = 0.15
        pprops['plotprops']['lw'] = SIZELINE
        for k in range(100):
            mp.line(ax=ax_tc2, x=[times], y=[tc_all_2[k][n]], colors=[C_group[c_cos]], **pprops)

        pprops['plotprops']['alpha'] = 1
        pprops['plotprops']['lw'] = SIZELINE*3
        mp.line(ax=ax_tc2, x=[times], y=[tc_ave_2[n]], colors=[C_group[c_cos]], **pprops)

    pprops['plotprops']['ls'] = ':'
    mp.plot(type='line',ax=ax_tc1, x=[times], y=[fi_1], colors=[C_group[c_sin]], **pprops)
    mp.plot(type='line',ax=ax_tc2, x=[times], y=[fi_2], colors=[C_group[c_cos]], **pprops)

    ax_tc1.text(box_tc1['left']+dx, box_tc1['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_tc2.text(box_tc2['left']+dx, box_tc2['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    if savepdf:
        plt.savefig('%s/simple_his.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
   
def plot_trait_his(**pdata):

    """
    histogram of selection coefficients and trait coefficients
    """

    # unpack passed data
    sim_dir           = pdata['sim_dir']            # 'sim'
    output        = pdata['output']         # ''
    seq_length    = pdata['seq_length']     # 20
    generations   = pdata['generations']    # 500
    xtick_h       = pdata['xtick_h']
    ytick_h       = pdata['ytick_h']
    ytick_e       = pdata['ytick_e']
    yminorticks_e = pdata['yminorticks_e']
    ytick_f       = pdata['ytick_f']
    yminorticks_f = pdata['yminorticks_f']

    seq_length    = pdata['seq_length']     # 20
    p_sites       = pdata['p_sites']        # [9,10]
    fB            = pdata['s_ben']          # 0.02
    fD            = pdata['s_del']          # -0.02
    fn            = pdata['fn']             # time-varying selection coefficient
    fi            = pdata['fi']             # time-varying selection coefficient
    savepdf       = pdata['savepdf']         # True

    with open("%s/%s/escape_groups.dat"%(SIM_DIR,sim_dir), 'r') as file:
        escape_groups = json.load(file)

    ne          = len(escape_groups[0])
    timepoints  = int(generations) + 1
    times       = np.linspace(0,generations,timepoints)

    # data for selection coefficients for different simulations
    df       = pd.read_csv('%s/%s/mpl_collected%s.csv' % (SIM_DIR,sim_dir,output), memory_map=True)
    ben_cols = ['sc_%d' % i for i in [0,1,2,3]]
    neu_cols = ['sc_%d' % i for i in [4,5,6,7,8,11,12,13,14,15]]
    del_cols = ['sc_%d' % i for i in [16,17,18,19]]

    # get data for inference results for different simulations
    tc_all   = np.zeros((100,ne,generations+1))
    sc_p_all = np.zeros((100,len(p_sites),generations+1))

    for k in range(100):
        name = str(k)
        data_full     = np.load('%s/%s/output%s/c_%s.npz'%(SIM_DIR,sim_dir,output,name), allow_pickle="True")
        sc_full       = data_full['selection']
        TimeVaryingTC = sc_full[seq_length:]
        for ii in range(len(p_sites)):
            p_index = p_sites[ii]
            sc_p_all[k][ii] = sc_full[p_index]
        
        for n in range(ne):
            tc_all[k][n] = TimeVaryingTC[n]

    tc_average = np.zeros((ne,generations+1))
    tc_all_n   = np.swapaxes(tc_all, 0, 2)

    sc_average = np.zeros((len(p_sites),generations+1))
    sc_p_all_n   = np.swapaxes(sc_p_all, 0, 2)

    for n in range(ne):
        for t in range(len(tc_all[0][0])):
            tc_average[n][t] = np.average(tc_all_n[t][n])

    for n in range(len(p_sites)):
        for t in range(len(sc_p_all[0][0])):
            sc_average[n][t] = np.average(sc_p_all_n[t][n])

    # PLOT FIGURE
    ## set up figure grid

    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.8
    fig   = plt.figure(figsize=(w, goldh),dpi=500)

    box_se  = dict(left=0.10, right=0.92, bottom=0.65, top=0.95)
    box_lab = dict(left=0.05, right=0.15, bottom=0.10, top=0.45)
    box_sc  = dict(left=0.24, right=0.52, bottom=0.10, top=0.50)
    box_tc  = dict(left=0.60, right=0.92, bottom=0.10, top=0.50)

    gs_se  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_se)
    gs_lab = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_lab)
    gs_tc  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)
    gs_sc  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc)

    ax_se  = plt.subplot(gs_se[0, 0])
    ax_lab = plt.subplot(gs_lab[0, 0])
    ax_tc  = plt.subplot(gs_tc[0, 0])
    ax_sc  = plt.subplot(gs_sc[0, 0])

    dx = -0.04
    dy =  0.03

    ## a -- histogram for selection coefficients

    dashlineprops = { 'lw' : SIZELINE * 1.5, 'ls' : ':', 'alpha' : 0.5, 'color' : BKCOLOR }
    histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
    pprops = { 'xlim':        [xtick_h[0], xtick_h[-1]],
               'xticks':      xtick_h,
               'xticklabels': [int(i*100) for i in xtick_h],
               'ylim':        [ytick_h[0], ytick_h[-1]],
               'yticks':      ytick_h,
               'xlabel'      : 'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)',
               'ylabel'      : 'Frequency',
               'bins'        : np.arange(-0.04, 0.04, 0.001),
               'combine'     : True,
               'plotprops'   : histprops,
               'axoffset'    : 0.1,
               'theme'       : 'boxed' }

    colors     = [C_BEN, C_NEU, C_DEL]
    tags       = ['beneficial', 'neutral', 'deleterious']
    cols       = [ben_cols, neu_cols, del_cols]
    s_true_loc = [fB, 0, fD]

    for i in range(len(tags)):
        x = [np.array(df[cols[i]]).flatten()]
        tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
        ax_se.text(s_true_loc[i], ytick_h[-1]*1.04, r'$s_{%s}$' % (tags[i]), color=colors[i], **tprops)
        ax_se.axvline(x=s_true_loc[i], **dashlineprops)
        if i<len(tags)-2: mp.hist(             ax=ax_se, x=x, colors=[colors[i]], **pprops)
        else:             mp.plot(type='hist', ax=ax_se, x=x, colors=[colors[i]], **pprops)

    ax_se.text(  box_se['left']+dx,  box_se['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ##  add legend
    pprops = { 'xlim':        [ -1 ,    6],
               'ylim':        [-0.05, 0.05],
               'yticks':      [],
               'xticks':      [],
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.3 },
               'theme':       'open',
               'hide':        ['left','bottom'] }

    yy =  -0.021
    coef_legend_dy = 0.021
    c_epitope = C_group[0]
    c_tv      = C_group[2]

    xx_line = [-0.9, 1.3]
    yy_line = np.zeros((3,2))
    for i in range(3):
        for j in range(2):
            yy_line[i][j] = yy + coef_legend_dy*((2 - i)+ 0.2 * j - 0.1)

    mp.line(ax=ax_lab, x=[xx_line], y=[[yy_line[0][0],yy_line[0][0]]], colors=[c_epitope], **pprops)
    mp.line(ax=ax_lab, x=[xx_line], y=[[yy_line[0][1],yy_line[0][1]]], colors=[c_tv], **pprops)
    ax_lab.text(2, yy+coef_legend_dy*2, 'Inferred \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['alpha'] = 1
    pprops['plotprops']['lw'] = SIZELINE*3
    mp.line(ax=ax_lab, x=[xx_line], y=[[yy_line[1][0],yy_line[1][0]]], colors=[c_epitope], **pprops)
    mp.line(ax=ax_lab, x=[xx_line], y=[[yy_line[1][1],yy_line[1][1]]], colors=[c_tv], **pprops)
    ax_lab.text(2, yy+coef_legend_dy, 'Average \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['ls'] = ':'
    mp.line(            ax=ax_lab, x=[xx_line], y=[[yy_line[2][0],yy_line[2][0]]], colors=[c_epitope], **pprops)
    mp.plot(type='line',ax=ax_lab, x=[xx_line], y=[[yy_line[2][1],yy_line[2][1]]], colors=[c_tv], **pprops)
    ax_lab.text(2, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ## b  -- selection coefficients for special sites
    ax_tc.text(box_tc['left']+dx, box_tc['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    pprops = {  'xticks':      [0, 200, 400, 600, 800, 1000],
                'ylim':        [ytick_f[0], ytick_f[-1]],
                'yticks':      ytick_f,
                'yminorticks': yminorticks_f,
                'yticklabels': [int(i*100) for i in ytick_f],
                'nudgey':      1,
                'xlabel':      'Generation',
                'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.1 },
                'axoffset':    0.15,
                'theme':       'open'}

    for n in range(len(p_sites)):
        p_index = p_sites[n]

        pprops['plotprops']['alpha'] = 0.15
        pprops['plotprops']['lw'] = SIZELINE
        for k in range(100):
            mp.line(ax=ax_sc, x=[times], y=[sc_p_all[k][n]], colors=[c_tv], **pprops)

        pprops['plotprops']['alpha'] = 0.4
        pprops['plotprops']['lw'] = SIZELINE*1.2
        mp.line(ax=ax_sc, x=[times], y=[sc_p_all[0][n]], colors=[c_tv], **pprops)

        pprops['plotprops']['alpha'] = 1
        pprops['plotprops']['lw'] = SIZELINE*3
        mp.line(ax=ax_sc, x=[times], y=[sc_average[n]], colors=[c_tv], **pprops)

    pprops['plotprops']['ls'] = ':'
    mp.plot(type='line',ax=ax_sc, x=[times], y=[fi], colors=[c_tv], **pprops)

    ax_sc.text(box_sc['left']+dx, box_sc['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c  -- escape coefficients 
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'ylim':        [ytick_e[0], ytick_e[-1]],
               'yticks':      ytick_e,
               'yminorticks': yminorticks_e,
               'yticklabels': [int(i*100) for i in ytick_e],
               'nudgey':      1,
               'xlabel':      'Generation',
               'ylabel':      'Inferred trait\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.15 },
               'axoffset':    0.1,
               'theme':       'open'}

    for n in range(ne):
        pprops['plotprops']['alpha'] = 0.15
        pprops['plotprops']['lw'] = SIZELINE
        for k in range(100):
            mp.line(ax=ax_tc, x=[times], y=[tc_all[k][n]], colors=[C_group[n]], **pprops)

        pprops['plotprops']['alpha'] = 1
        pprops['plotprops']['lw'] = SIZELINE*3
        mp.line(ax=ax_tc, x=[times], y=[tc_average[n]], colors=[C_group[n]], **pprops)

        pprops['plotprops']['ls'] = ':'
        mp.plot(type='line',ax=ax_tc, x=[times], y=[fn], colors=[C_group[n]], **pprops)

    # SAVE FIGURE
    if savepdf:
        plt.savefig('%s/%s_his%s.pdf' % (FIG_DIR,sim_dir,output), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

# def plot_epitope_theta(**pdata):

#     # unpack passed data
#     tag      = pdata['tag']
#     sim_dir     = pdata['sim_dir']
#     HIV_DIR  = pdata['HIV_DIR']
#     FIG_DIR  = pdata['FIG_DIR']
#     xtick      = pdata['xtick']
#     xminortick = pdata['xminortick']
#     ytick      = pdata['ytick']
#     yminortick = pdata['yminortick']
#     theta      = pdata['theta']
#     savepdf    = pdata['savepdf']


#     # information for escape group
#     data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
#     escape_group = data_pro['escape_group']
#     sample_times = data_pro['sample_times']
#     times        = data_pro['times']
#     time_step    = data_pro['time_step']
#     ne           = len(escape_group)

#     if ne == 0:
#         print(f'CH{tag[6:]} does not contain any trait group')
#         return

#     # import data with extended time
#     data_sc = np.load('%s/%s/c_%s_%d_%s.npz'%(HIV_DIR,sim_dir,tag,time_step,theta), allow_pickle="True")
#     sc_all_ex   = data_sc['selection']# time range:times

#     time_index = []
#     for t in range(len(sample_times)):
#         index = list(times).index(sample_times[t])
#         time_index.append(index)

#     sc_sample    = np.zeros((len(sc_all_ex),len(sample_times)))#time range:sample_times
#     sc_sample_ex = np.zeros_like(sc_sample)
#     for i in range(len(sample_times)):
#         index          = time_index[i]
#         sc_sample_ex[:,i] = sc_all_ex[:,index]

#     # PLOT FIGURE
#     # set up figure grid
#     fig   = plt.figure(figsize=(2,1.5*ne),dpi=500)
#     gs = gridspec.GridSpec(ne, 1, width_ratios=[1])
#     if ne > 1:
#         gs.update(left=0.1, right=0.9, bottom=0.10, top=0.95)
#     else:
#         gs.update(left=0.1, right=0.9, bottom=0.20, top=0.95)
#     ax  = [[plt.subplot(gs[n, 0])] for n in range(ne)]

#     if len(xtick) == 0:
#         xtick = [int(i) for i in sample_times]

#     var_ec = np.zeros(ne)
    
#     # escape coefficients without extended time (time range:times)
#     pprops = { 'xlim':        [ 0,    max(sample_times)+5],
#                'xticks':      xtick,
#                'xticklabels': [],
#                'ylabel':      'Inferred escape \ncoefficient, ' + r'$\hat{s}$' + ' (%)'}

#     for n in range(ne):
#         if len(ytick) != ne or len(ytick[0]) == 0:
#             ymax = max(max(var_ec[n], max(sc_sample_ex[-(ne-n),:])) * 1.25,0.02)
#             ymin = min(min(sc_sample_ex[-(ne-n),:]), -0.02)
#             pprops['ylim']        = [ymin, ymax]
#             pprops['yticks']      = [round(ymin/0.01)*0.01,  0, round(var_ec[n]*100)/100 ,round(ymax/0.01)*0.01]
#             pprops['yticklabels'] = [round(ymin/0.01)     ,  0, round(var_ec[n]*100)     ,round(ymax/0.01)*1]
#         else:
#             dy   = (ytick[n][-1] - ytick[n][0])*0.01
#             ymax = ytick[n][-1] + dy
#             ymin = ytick[n][ 0] + dy
#             pprops['ylim']        = [ymin, ymax]
#             pprops['yticks']      = ytick[n]
#             pprops['yticklabels'] = [int(i*100) for i in ytick[n]]
#             pprops['yminorticks'] = yminortick[n]

#         if n == ne - 1:
#             pprops['xticklabels'] = xtick
#             pprops['xlabel']      = 'Days after Fiebig I/II'+ theta

#         lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
#         mp.line(ax=ax[n][0], x=[times], y=[sc_all_ex[-(ne-n),:]], colors=[C_group[n]],plotprops=lprops, **pprops)

#         sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
#         mp.plot(type='scatter', ax=ax[n][0], x=[sample_times], y=[sc_sample_ex[-(ne-n),:]], colors=[C_group[n]],plotprops=sprops, **pprops)

#         ax[n][0].axhline(y=0, ls='--', lw=SIZELINE/2, color=BKCOLOR)
    
#     if savepdf:
#         plt.savefig('%s/fig-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
#         plt.show()
#     else:
#         plt.savefig('%s/CH%s-%s.jpg' % (FIG_DIR,tag[-5:],theta), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_HIV(**pdata):

    # unpack passed data
    out_dir  = pdata['dir']
    HIV_DIR  = pdata['HIV_DIR']
    FIG_DIR  = pdata['FIG_DIR']
    xtick      = pdata['xtick']
    xminortick = pdata['xminortick']
    ytick      = pdata['ytick']
    yminortick = pdata['yminortick']
    savepdf    = pdata['savepdf']

    tags = ['703010256-3','700010040-3']
    epi_indexes = [1,0] # index for epitope RN9 (1) and AR9 (0)
    sample_times_all = [] # raw time points
    times_all        = [] # interpolated time points
    var_ec_all       = [] # escape coefficients for constant case
    traj_var_all     = [] # frequencies for individual escape sites
    traj_group_all   = [] # frequencies for escape groups
    tc_sample_ex_all = [] # selection coefficients for raw time points
    tc_all_ex_all    = [] # selection coefficients for interpolated time points
    
    for i in range(len(tags)):
        tag = tags[i]
        epi_index = epi_indexes[i]
        FData = GetFigureData(out_dir,tag,HIV_DIR)
        ne = FData.ne
        sample_times_all.append(FData.sample_times)
        times_all.append(FData.times)
        var_ec_all.append(FData.var_ec[epi_index])
        traj_var_all.append(FData.traj_var[epi_index])
        traj_group_all.append(FData.traj_group[epi_index])
        tc_sample_ex_all.append(FData.sc_sample_ex[-(ne-epi_index)])
        tc_all_ex_all.append(FData.sc_all_ex[-(ne-epi_index)])

    # PLOT FIGURE
    # set up figure grid
    w     = SINGLE_COLUMN
    goldh = w / 1.4
    fig   = plt.figure(figsize=(w, goldh),dpi=500)

    box_tra1 = dict(left=0.15, right=0.48, bottom=0.62, top=0.92)
    box_tra2 = dict(left=0.63, right=0.96, bottom=0.62, top=0.92)
    box_tc1  = dict(left=0.15, right=0.48, bottom=0.16, top=0.44)
    box_tc2  = dict(left=0.63, right=0.96, bottom=0.16, top=0.44)

    gs_tra1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra1)
    gs_tra2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra2)
    gs_tc1  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc1)
    gs_tc2  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc2)

    ax_tra1 = plt.subplot(gs_tra1[0, 0])
    ax_tra2 = plt.subplot(gs_tra2[0, 0])
    ax_tc1  = plt.subplot(gs_tc1[0, 0])
    ax_tc2  = plt.subplot(gs_tc2[0, 0])

    dx = -0.10
    dy =  0.04

    ## a -- escape group frequencies for RN9 (CH256-3)
    pprops = { 'xticks':      xtick[0],
               'xminorticks': xminortick[0],
               'xticklabels': [],
               'yticks':      [0, 1.0],
               'yminorticks': [0.25, 0.5, 0.75],
               'ylabel':      'RN9 escape\n variant frequency',
               'nudgey':      1.1,
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.4 },
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}
      
    # plot frequencies for individual escape sites
    for nn in range(len(traj_var_all[0])):
        pprops['plotprops']['alpha'] = 0.4
        mp.line(ax=ax_tra1, x=[sample_times_all[0]], y=[traj_var_all[0][nn]], colors=[C_group[0]], **pprops)

    # plot frequencies for individual group
    pprops['plotprops']['alpha'] = 1
    mp.plot(type='line', ax=ax_tra1, x=[sample_times_all[0]], y=[traj_group_all[0]], colors=[C_group[0]], **pprops)
        
    arrow = FancyArrowPatch((30, -0.28), (30, -0.05), arrowstyle='->,head_width=0.8,head_length=0.8', lw=SIZELINE, clip_on=False)
    ax_tra1.add_patch(arrow)
    ax_tra1.text(30, -0.4, 'T cell response\ndrops dramatically', ha='left', va='center', **DEF_LABELPROPS)

    ax_tra1.text(box_tra1['left']+dx,  box_tra1['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    ## b -- escape group frequencies for AR9 (CH040-3)
    pprops['xticks'] = xtick[1]
    pprops['xminorticks'] = xminortick[1]
    pprops['ylabel'] = 'AR9 escape\n variant frequency'

    # plot frequencies for individual escape sites
    for nn in range(len(traj_var_all[1])):
        pprops['plotprops']['alpha'] = 0.4
        mp.line(ax=ax_tra2, x=[sample_times_all[1]], y=[traj_var_all[1][nn]], colors=[C_group[1]], **pprops)

    # plot frequencies for individual group
    pprops['plotprops']['alpha'] = 1
    mp.plot(type='line', ax=ax_tra2, x=[sample_times_all[1]], y=[traj_group_all[1]], colors=[C_group[1]], **pprops)

    arrow = FancyArrowPatch((100, -0.28), (100, -0.05), arrowstyle='->,head_width=0.8,head_length=0.8', lw=SIZELINE, clip_on=False)
    ax_tra2.add_patch(arrow)
    ax_tra2.text(100, -0.4, 'Peak intensity \nof T cell response', ha='left', va='center', **DEF_LABELPROPS)

    ax_tra2.text(box_tra2['left']+dx,  box_tra2['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # c -- escape coefficients for RN9 (CH256-3)
    pprops = { 'xticks':      xtick[0],
               'xminorticks': xminortick[0],
               'ylim':        [ytick[0][0], ytick[0][-1]],
               'yticks':      ytick[0],
               'yticklabels': [int(i*100) for i in ytick[0]],
               'yminorticks': yminortick[0],
               'xlabel':      'Days after Fiebig I/II',
               'ylabel':      'Inferred escape \ncoefficient, ' + r'$\hat{s}$' + ' (%)'}

    lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
    mp.line(ax=ax_tc1, x=[times_all[0]], y=[tc_all_ex_all[0]], colors=[C_group[0]],plotprops=lprops, **pprops)

    sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
    mp.plot(type='scatter', ax=ax_tc1, x=[sample_times_all[0]], y=[tc_sample_ex_all[0][:]], colors=[C_group[0]],plotprops=sprops, **pprops)
    
    ax_tc1.axhline(y=0, ls='--', lw=SIZELINE/2, color=BKCOLOR)
    ax_tc1.axhline(y=var_ec_all[0], ls=':', lw=SIZELINE, color=C_group[0])

    ax_tc1.text(box_tc1['left']+dx,  box_tc1['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    # d -- escape coefficients for AR9 (CH040-3)
    pprops = { 'xticks':      xtick[1],
               'xminorticks': xminortick[1],
               'ylim':        [ytick[1][0], ytick[1][-1]],
               'yticks':      ytick[1],
               'yticklabels': [int(i*100) for i in ytick[1]],
               'yminorticks': yminortick[1],
               'xlabel':      'Days after Fiebig I/II',
               'ylabel':      'Inferred escape \ncoefficient, ' + r'$\hat{s}$' + ' (%)'}

    lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
    mp.line(ax=ax_tc2, x=[times_all[1]], y=[tc_all_ex_all[1]], colors=[C_group[1]],plotprops=lprops, **pprops)

    sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
    mp.plot(type='scatter', ax=ax_tc2, x=[sample_times_all[1]], y=[tc_sample_ex_all[1]], colors=[C_group[1]],plotprops=sprops, **pprops)
    
    ax_tc2.axhline(y=0, ls='--', lw=SIZELINE/2, color=BKCOLOR)
    ax_tc2.axhline(y=var_ec_all[1], ls=':', lw=SIZELINE, color=C_group[1])

    ax_tc2.text(box_tc2['left']+dx,  box_tc2['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    if savepdf:
        plt.savefig('%s/fig-epitope.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()


# def plot_epitope(**pdata):

#     # unpack passed data
#     tag      = pdata['tag']
#     dirs     = pdata['dirs']
#     HIV_DIR  = pdata['HIV_DIR']
#     FIG_DIR  = pdata['FIG_DIR']
#     xtick      = pdata['xtick']
#     xminortick = pdata['xminortick']
#     ytick      = pdata['ytick']
#     yminortick = pdata['yminortick']
#     savepdf    = pdata['savepdf']

#     # information for escape group
#     data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
#     escape_group = data_pro['escape_group']
#     sample_times = data_pro['sample_times']
#     times        = data_pro['times']
#     time_step    = data_pro['time_step']
#     ne           = len(escape_group)

#     if ne == 0:
#         print(f'CH{tag[6:]} does not contain any trait group')
#         return

#     # Import data
#     data_sc = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,dirs[0],tag,time_step), allow_pickle="True")

#     # Get the index for sample times
#     time_index = []
#     for t in range(len(sample_times)):
#         index = list(times).index(sample_times[t])
#         time_index.append(index)

#     # Get information for escape group
#     df_escape   = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv'%(HIV_DIR,tag), memory_map=True)
#     epitopes = df_escape['epitope'].unique()

#     var_ec = np.zeros(ne)
#     traj_var   = [] # frequencies for individual escape sites
#     traj_group = [] # frequencies for escape groups
#     var_tag    = [] # name for epitope
#     for n in range(len(epitopes)):
#         df_esc  = df_escape[(df_escape.epitope==epitopes[n])]
#         df_row  = df_esc.iloc[0]
#         var_ec.append(df_esc.iloc[0].tc_MPL)

#         # get the name for epitopes
#         epi_nuc = ''.join(epitopes[n])
#         var_tag.append(epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc)))

#         # get frequencies for escape sites and groups
#         traj_group.append([df_row['xp_at_%d' % t] for t in sample_times])
#         var_traj = []
#         for df_iter, df_entry in df_esc.iterrows():
#             if df_entry.nucleotide != '-': # not include '-' variants
#                 var_traj.append([df_entry['f_at_%d' % t] for t in sample_times])
#         traj_var.append(var_traj)

#     # import data for VL-dependent r and constant r
#     # index: time-varying gamma, 0 : VL-dependent r,  1 : constant r 
#     # index: constant gamma,     2 : VL-dependent r,  3 : constant r
#     sc_all     = [[] for n in range(len(dirs) * 2)]
#     sc_sample  = np.zeros((len(dirs)*2,len(data_sc['selection']),len(sample_times)))

#     for ii in range(len(dirs)):
#         out_dir = dirs[ii]
#         # VL-dependent r
#         data_sc = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,out_dir,tag,time_step), allow_pickle="True")
#         sc_all[2*ii]   = data_sc['selection']# time range:times

#         for i in range(len(sample_times)):
#             index          = time_index[i]
#             for j in range(len(sc_all[0])):
#                 sc_sample[2*ii][j][i] = sc_all[2*ii][j][index]

#         # Constant r
#         dir_cr = 'cr/' + out_dir
#         data_sc_cr = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,dir_cr,tag,time_step), allow_pickle="True")
#         sc_all[2*ii+1]  = data_sc_cr['selection']# time range:times

#         for i in range(len(sample_times)):
#             index          = time_index[i]
#             for j in range(len(sc_all[0])):
#                 sc_sample[2*ii+1][j][i] = sc_all[2*ii+1][j][index]

#     # Setting ticks
#     if len(ytick) != ne or len(ytick[0]) == 0:
#         # set xticks automatically
#         mat_time = sample_times[-1]
#         if mat_time> 300:
#             dt = 200
#         elif mat_time < 100:
#             dt = 20
#         else:
#             dt = 100
            
#         xtick = []
#         for n in range(math.ceil(mat_time/dt)):
#             xtick.append(n*dt)
#         xtick.append((n+1)*dt)

#         # set yticks automatically
#         ytick  = [[] for n in range(ne)]
#         for n in range(ne):
#             max_var = max([max(sc_all[i][-(ne-n)]) for i in range(len(sc_all))])
#             min_var = min([min(sc_all[i][-(ne-n)]) for i in range(len(sc_all))])
#             ymax = max(max(var_ec[n], max_var) * 1.25,0.02)
#             ymin = min(min_var, -0.02)
#             ytick[n] = [round(ymin/0.01)*0.01,  0, round(var_ec[n]*100)/100 ,round(ymax/0.01)*0.01]

#     # PLOT FIGURE
#     # set up figure grid
#     fig   = plt.figure(figsize=(6,1.4*ne),dpi=500)
#     gs = gridspec.GridSpec(ne, 3, width_ratios=[1,1,1],wspace=0.4,hspace=0.3)
#     if ne > 1:
#         gs.update(left=0.1, right=0.9, bottom=0.10, top=0.95)
#     else:
#         gs.update(left=0.1, right=0.9, bottom=0.20, top=0.95)
#     ax  = [[plt.subplot(gs[n, 0]), plt.subplot(gs[n, 1]), plt.subplot(gs[n,2])] for n in range(ne)]

#     if len(xtick) == 0:
#         xtick = [int(i) for i in sample_times]

#     # a -- escape group frequencies
#     pprops = { 'xticks':      xtick,
#                'xminorticks': xminortick,
#                'xticklabels': [],
#                'yticks':      [0, 1.0],
#                'yminorticks': [0.25, 0.5, 0.75],
#                'nudgey':      1.1,
#                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.4 },
#                'axoffset':    0.1,
#                'theme':       'open',
#                'combine'     : True}

#     for n in range(ne):
#         pprops['ylabel']          = var_tag[n]
#         # add x axis and label at the bottom
#         if n == ne - 1:
#             pprops['xticklabels'] = xtick
#             pprops['xlabel']      = 'Days after Fiebig I/II'

#         # plot frequencies for individual escape sites
#         for nn in range(len(traj_var[n])):
#             pprops['plotprops']['alpha'] = 0.4
#             mp.line(ax=ax[n][0], x=[sample_times], y=[traj_var[n][nn]], colors=[C_group[n]], **pprops)

#         # plot frequencies for individual group
#         pprops['plotprops']['alpha'] = 1
#         mp.plot(type='line', ax=ax[n][0], x=[sample_times], y=[traj_group[n]], colors=[C_group[n]], **pprops)

#     # b.c -- escape coefficients for time-varying r and constant r
#     pprops = { 'xticks':      xtick,
#                'xticklabels': [],
#                'ylabel':      'Inferred escape \ncoefficient, ' + r'$\hat{s}$' + ' (%)'}

#     for n in range(ne):
#         pprops['ylim']       = [ytick[n][0]-0.005,ytick[n][-1]+0.005]
#         pprops['yticks']      = ytick[n]
#         pprops['yticklabels'] = [int(i*100) for i in ytick[n]]

#         if len(yminortick) > 0:
#             pprops['yminorticks']  = yminortick[n]

#         if n == ne - 1:
#             pprops['xticklabels'] = xtick
#             pprops['xlabel']      = 'Days after Fiebig I/II'

#         # VL-dependent r
#         lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
#         sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':0.5}
#         mp.line(ax=ax[n][1], x=[times], y=[sc_all[0][-(ne-n)]], colors=[C_group[n]],plotprops=lprops, **pprops)
#         mp.scatter(ax=ax[n][1], x=[sample_times], y=[sc_sample[0,-(ne-n),:]], colors=[C_group[n]],plotprops=sprops, **pprops)
#         mp.line(ax=ax[n][2], x=[times], y=[sc_all[2][-(ne-n)]], colors=[C_group[n]],plotprops=lprops, **pprops)
#         mp.scatter(ax=ax[n][2], x=[sample_times], y=[sc_sample[2,-(ne-n),:]], colors=[C_group[n]],plotprops=sprops, **pprops)

#         # Constant r
#         lprops = {'lw': SIZELINE, 'ls': '--', 'alpha': 0.3 }
#         sprops = { 'lw' : 0, 's' : 6, 'marker' : '^','alpha':0.3}
#         mp.line(ax=ax[n][1], x=[times], y=[sc_all[1][-(ne-n)]], colors=[C_group[n]],plotprops=lprops, **pprops)
#         mp.plot(type='scatter', ax=ax[n][1], x=[sample_times], y=[sc_sample[1,-(ne-n),:]], colors=[C_group[n]],plotprops=sprops, **pprops)
#         mp.line(ax=ax[n][2], x=[times], y=[sc_all[3][-(ne-n)]], colors=[C_group[n]],plotprops=lprops, **pprops)
#         mp.plot(type='scatter', ax=ax[n][2], x=[sample_times], y=[sc_sample[3,-(ne-n),:]], colors=[C_group[n]],plotprops=sprops, **pprops)

#        ax[n][1].axhline(y=var_ec[n], ls=':', lw=SIZELINE, color=C_group[n])
#        ax[n][2].axhline(y=var_ec[n], ls=':', lw=SIZELINE, color=C_group[n])

#         ax[n][1].axhline(y=0, ls=':', lw=SIZELINE, color=BKCOLOR)
#         ax[n][2].axhline(y=0, ls=':', lw=SIZELINE, color=BKCOLOR)

#     legend_x = (xtick[0] + xtick[-1])/2
#     legend = 'Time-varying ' + r'$\gamma^{\prime}$'
#     ax[0][1].text(legend_x, ytick[0][-1], legend, ha='center', va='center', **DEF_LABELPROPS)
#     legend = 'Constant ' + r'$\gamma^{\prime} = 200$'
#     ax[0][2].text(legend_x, ytick[0][-1], legend, ha='center', va='center', **DEF_LABELPROPS)
#     # ax[0][1].text(legend_x, ytick[0][-1], 'VL-dependent r', ha='center', va='center', **DEF_LABELPROPS)
#     # legend = 'Constant r (' + r'$1.4 \times 10^{-5}$' + ')'

#     if savepdf:
#         plt.savefig('%s/fig-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
#         plt.show()
#     else:
#         plt.savefig('%s/HIV/CH%s.jpg' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_special_site(**pdata):

    # unpack passed data
    tag        = pdata['tag']
    HIV_DIR    = pdata['HIV_DIR']
    FIG_DIR    = pdata['FIG_DIR']
    xtick      = pdata['xtick']
    xminortick = pdata['xminortick']
    ytick      = pdata['ytick_sp']
    yminortick = pdata['yminor_sp']
    savepdf    = pdata['savepdf']

    out_dir    = HIV_DIR.split('/')[-1]
    # information for special sites
    data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    muVec    = data_pro['muVec']

    sample_times = data_pro['sample_times']
    # times        = data_pro['times']
    # time_step    = data_pro['time_step']
    special_sites    = data_pro['special_sites']
    if len(special_sites) == 0:
        print(f'CH{tag[6:]} has no special site')
        return

    # import data with extended time
    try:
        data_sc  = np.load('%s/output/sc_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
        # data_sc  = np.load('%s/output/c_%s_%d.npz'%(HIV_DIR,tag,time_step), allow_pickle="True")
    except FileNotFoundError:
        print(f'No data for CH{tag[-5:]}')
        return
    # data_sc = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,out_dir,tag,time_step), allow_pickle="True")
    sc_all  = data_sc['selection']# time range:times
    time_all = np.linspace(0, sample_times[-1], int(sample_times[-1]+1))

    df_sc   = pd.read_csv('%s/constant/analysis/%s-analyze.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
    
    index_s =  [] # variants name
    for i in special_sites:
        df_i  = df_sc[(df_sc.polymorphic_index==i) & (df_sc.nucleotide!=df_sc.TF ) & (df_sc.nucleotide!= '-')]
        for df_iter, df_entry in df_i.iterrows():
            # get variant name
            site    = int(df_entry.polymorphic_index)
            variant = str(site)+df_entry.nucleotide
            index_s.append(variant)

    '''get selection coefficient for time-varying case'''
    sc_old  =  np.zeros(len(index_s))                 # selection coefficient for constant case
    sc_a    =  np.zeros((len(index_s),len(time_all)))    # time varying sc (time range:times)
    traj_sp =  [] # frequencies for special sites
    for i in range(len(index_s)):
        site_i = int(index_s[i][:-1])
        nuc_i  = str(index_s[i][-1])
        df_sp  = df_sc[(df_sc.polymorphic_index==site_i) & (df_sc.nucleotide == nuc_i)]

        if len(df_sp) != 1:
            print(f'error for variant {index_s[i]}')

        df_i   = df_sp.iloc[0]
        # get selection coefficient for constant case
        sc_old[i] = df_sp.sc_MPL

        q_index  = NUC.index(df_i.nucleotide)
        TF_index = NUC.index(df_i.TF)
        traj_sp.append([df_i['f_at_%d' % t] for t in sample_times])

        # use muVec to get the real position of this variant
        index_mu    = muVec[site_i,q_index]
        index_TF    = muVec[site_i,TF_index]
        if index_mu == -1 or index_TF == -1:
            print('CH%s error, %d'%(tag[-5:],site))
        else:
            sc_a[i] = sc_all[int(index_mu)] - sc_all[int(index_TF)]

    sc_s  = np.zeros((len(index_s),len(sample_times)))
    # selection coefficients for sampled time points
    index = 0
    for ti, t in enumerate(time_all):
        if t in sample_times:
            sc_s[:,index] = sc_a[:,ti]
            index += 1

    # PLOT FIGURE
    # set up figure grid
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1,0.5,1],wspace = 0.3,)
    gs.update(left=0.1, right=0.9, bottom=0.20, top=0.95)
    ax  = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])]

    if len(xtick) == 0:
        xtick = [int(i) for i in sample_times]

    # frequencies
    pprops = { 'xticks':      xtick,
               'xminorticks': xminortick,
               'yticks':      [0, 1.0],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True,
               'xlabel':      'Days after Fiebig I/II',
               'ylabel':      'Frequencies'}

    for i in range(len(index_s)):
        pprops['plotprops']['alpha'] = 0.8
        # plot frequencies for special sites
        if i == len(index_s) - 1:
            mp.plot(type='line',ax=ax[0], x=[sample_times], y=[traj_sp[i]], colors=[C_group[i]], **pprops)
        else:
            mp.line(ax=ax[0], x=[sample_times], y=[traj_sp[i]], colors=[C_group[i]], **pprops)

    # label for different special variants
    n_sp = len(index_s)
    pprops = { 'xlim': [0,  5],
               'ylim': [0,  n_sp],
               'xticks': [],
               'yticks': [],
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
               'theme': 'open',
               'hide' : ['top', 'bottom', 'left', 'right'] }

    traj_legend_x  =  1.5
    traj_legend_y  = [0.5+i for i in range(n_sp)]

    for k in range(len(traj_legend_y)):
        x1 = traj_legend_x-2.0
        x2 = traj_legend_x-0.5
        yy = traj_legend_y[k]
        ax[1].text(traj_legend_x, traj_legend_y[k], index_s[k], ha='left', va='center', **DEF_LABELPROPS)
        mp.plot(type='line',ax=ax[1], x=[[x1, x2]], y=[[yy, yy]], colors=[C_group[k]], **pprops)

    # escape coefficients with extended time (time range:times)
    pprops = { 'xlim':        [ 0,    max(sample_times)+5],
               'xticks':      xtick,
               'xminorticks': xminortick,
               'xlabel':      'Days after Fiebig I/II',
               'ylabel':      'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)'}

    if len(ytick) == 0:
        ymax = max(max(max(sc_old), sc_a.max()) * 1.25, 0.02)
        ymin = min(min(min(sc_old), sc_a.min()) * 1.25,-0.02)
        pprops['ylim']        = [ymin, ymax]
        pprops['yticks']      = [round(ymin/0.01)*0.01,  0, round(ymax/0.01)*0.01]
        pprops['yticklabels'] = [round(ymin/0.01)     ,  0, round(ymax/0.01)*1]
    else:
        pprops['yticks'] = ytick
        pprops['yticklabels'] = [int(i*100) for i in ytick]
        pprops['yminorticks'] = yminortick

    lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 }

    for i in range(len(index_s)):
        # 0 and previous result
        ax[2].axhline(y=sc_old[i], ls=':', lw=SIZELINE, color=C_group[i])
        ax[2].axhline(y=0, ls='--', lw=SIZELINE/2, color=BKCOLOR)

        # sc-time (time range:times)
        mp.line(ax=ax[2], x=[time_all], y=[sc_a[i]], colors=[C_group[i]],plotprops=lprops, **pprops)

        # sc-time(time range:sample_times)
        if i == len(index_s) - 1:
            sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
            mp.plot(type='scatter', ax=ax[2], x=[sample_times], y=[sc_s[i]], colors=[C_group[i]],plotprops=sprops, **pprops)
        else:
            sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
            mp.scatter(ax=ax[2], x=[sample_times], y=[sc_s[i]], colors=[C_group[i]],plotprops=sprops, **pprops)

    if savepdf:
        plt.savefig('%s/fig-sp-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()
    else:
        plt.savefig('%s/%s/sp-CH%s.jpg' % (FIG_DIR,out_dir,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)


@dataclass
class FData:
    ne:0
    sample_times:[]
    var_ec:[]
    traj_var:[]
    traj_group:[]
    sc_sample_ex:[]
    times:[]
    sc_all_ex:[]
    var_tag:[]

def GetFigureData(out_dir,tag,HIV_DIR):
    data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    escape_group = data_pro['escape_group']
    sample_times = data_pro['sample_times']
    ne           = len(escape_group)
    sample_times = data_pro['sample_times']

    # import data with extended time
    # data_tc     = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,out_dir,tag,time_step), allow_pickle="True")
    data_tc     = np.load('%s/output/sc_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    sc_all_ex   = data_tc['selection']# time range:times
    
    df_escape   = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv'%(HIV_DIR,tag), memory_map=True)
    epitopes    = df_escape['epitope'].unique()

    time_all = np.linspace(0, sample_times[-1], int(sample_times[-1]+1))
    sc_sample_ex  = np.zeros((len(sc_all_ex),len(sample_times)))
    # selection coefficients for sampled time points
    for i, ti in enumerate(time_all):
        if ti in sample_times:
            index = list(sample_times).index(ti)
            for j in range(len(sc_all_ex)):
                sc_sample_ex[j][index] = sc_all_ex[j][i]

    var_ec     = [] # escape coefficients for constant case
    traj_var   = [] # frequencies for individual escape sites
    traj_group = [] # frequencies for escape groups
    var_tag    = [] # name for epitope
    for n in range(len(epitopes)):
        df_esc  = df_escape[(df_escape.epitope==epitopes[n])]
        df_row  = df_esc.iloc[0]
        var_ec.append(df_esc.iloc[0].tc_MPL)

        # get the name for epitopes
        epi_nuc = ''.join(epitopes[n])
        var_tag.append(epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc)))

        # get frequencies for escape sites and groups
        traj_group.append([df_row['xp_at_%d' % t] for t in sample_times])
        var_traj = []
        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.nucleotide != '-': # not include '-' variants
                var_traj.append([df_entry['f_at_%d' % t] for t in sample_times])
        traj_var.append(var_traj)

    return FData(ne,sample_times,var_ec,traj_var,traj_group,sc_sample_ex,time_all,sc_all_ex,var_tag)

def plot_all_epitopes(**pdata):

    # unpack passed data
    tags     = pdata['tags']
    name     = pdata['name']
    out_dir  = pdata['out_dir']
    HIV_DIR  = pdata['HIV_DIR']
    output   = pdata['output']

    ne_all           = []
    sample_times_all = []
    traj_group_all   = []
    sc_sample_ex_all = []
    times_all        = []
    sc_all_ex_all    = []

    # information for escape group
    for tag in tags:
        data_pro     = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
        escape_group = data_pro['escape_group']
        ne           = len(escape_group)
        if ne == 0:
            print(f'CH{tag[6:]} does not contain any trait group')
        else:
            FData = GetFigureData(out_dir,tag,HIV_DIR)
            ne_all.append(ne)
            sample_times_all.append(FData.sample_times)
            traj_group_all.append(FData.traj_group)
            sc_sample_ex_all.append(FData.sc_sample_ex)
            times_all.append(FData.times)
            sc_all_ex_all.append(FData.sc_all_ex)

    # PLOT FIGURE
    # set up figure grid
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3.6
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_traj = dict(left=0.10, right=0.47, bottom=0.18, top=0.92)
    box_tc   = dict(left=0.58, right=0.95, bottom=0.18, top=0.92)

    gs_traj  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_traj)
    gs_tc    = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)
    
    ax_traj  = plt.subplot(gs_traj[0, 0])
    ax_tc    = plt.subplot(gs_tc[0, 0])

    dx = -0.04
    dy =  0.03

    # a escape group frequencies short time
    pprops =  { 'xticks':      [ 0,  np.log(11),np.log(51),np.log(101), np.log(201),np.log(401),np.log(701)],
                'xticklabels': [ 0, 10, 50, 100, 200, 400, 700],
                'yticks':      [0, 1.0],
                'yminorticks': [0.25, 0.5, 0.75],
                'nudgey':      1.1,
                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.8 },
                'xlabel':      'Days after Fiebig I/II',
                'ylabel':      'Trait frequencies',
                'axoffset':    0.1,
                'theme':       'open',
                'combine':     True}
    
    for n in range(len(sample_times_all)):
        traj_group   = traj_group_all[n]
        sample_times = [sample_times_all[n] for k in range(len(traj_group))]
        log_sample_times = [np.log(i+1) for i in sample_times]
        color_n      = [C_NEU for k in range(len(traj_group))]
        mp.plot(type='line', ax=ax_traj, x=log_sample_times, y=traj_group, colors=color_n, **pprops)

    ax_traj.text( box_traj['left']+dx,  box_traj['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b escape coefficients with extended time (time range:times)
    pprops = {  'xticks':      [ 0,  np.log(11),np.log(51),np.log(101), np.log(201),np.log(401),np.log(701)],
                'xticklabels': [ 0, 10, 50, 100, 200, 400, 700],
                'ylim':        [-0.02,0.42],
                'yticks':      [-0.1,0,0.1,0.2,0.3,0.4],
                'yminorticks': [-0.05,0.05,0.15,0.25,0.35],
                'yticklabels': [-10,0,10,20,30,40],
                'xlabel':      'Days after Fiebig I/II',
                'ylabel':      'Inferred escape \ncoefficient, ' + r'$\hat{s}$' + ' (%)'}

    lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
    sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}

    for n in range(len(sample_times_all)):
        ne           = ne_all[n]
        sc_all_ex    = sc_all_ex_all[n][-ne:,:]
        sc_sample_ex = sc_sample_ex_all[n][-ne:,:]
        sample_times = [sample_times_all[n] for k in range(len(sc_sample_ex))]
        times        = [times_all[n] for k in range(len(sc_all_ex))]
        log_sample_times = [np.log(i+1) for i in sample_times]
        log_times        = [np.log(i+1) for i in times]
        color_n      = [C_NEU for k in range(len(sc_sample_ex))]

        mp.line(ax=ax_tc, x=log_times, y=sc_all_ex, colors=color_n,plotprops=lprops, **pprops)
        mp.scatter(ax=ax_tc, x=log_sample_times, y=sc_sample_ex, colors=color_n,plotprops=sprops, **pprops)
        
    lprops['ls'] = ':'
    lprops['alpha'] = 1.0
    mp.plot(type='line',ax=ax_tc, x=[[0,np.log(701)]],y=[[0,0]],colors=[BKCOLOR],plotprops=lprops, **pprops)

    ax_tc.text( box_tc['left']+dx,  box_tc['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    plt.savefig('%s/fig-tc.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_gamma_prime(**pdata):

    """
    histogram of trait coefficients with different gamma^{prime} values
    """

    # unpack passed data
    sim_dir       = pdata['sim_dir']        # 'simple'
    generations   = pdata['generations']    # 500
    ytick_t       = pdata['ytick_t']
    yminorticks_t = pdata['yminorticks_t']
    ytick_d       = pdata['ytick_d']
    yminorticks_d = pdata['yminorticks_d']

    p_1           = pdata['p_1']            # [6,7] , special sites 1
    p_2           = pdata['p_2']            # [8,9] , special sites 2
    fi_1          = pdata['fi_1']           # time-varying selection coefficient for special sites 1
    fi_2          = pdata['fi_2']           # time-varying selection coefficient for special sites 2

    savepdf       = pdata['savepdf']           # True

    timepoints  = int(generations) + 1
    times       = np.linspace(0,generations,timepoints)

    # data for selection coefficients for different simulations
    output_suffix = ['_0.25', '_1', '']
    tc_all = []
    for ii in range(len(output_suffix)):
        # get data for inference results for different simulations
        tc_all_1   = np.zeros((100,len(p_1),generations+1))
        tc_all_2   = np.zeros((100,len(p_2),generations+1))
        output = output_suffix[ii]
        for k in range(100):
            name = str(k)
            data_full     = np.load('%s/%s/output%s/c_%s.npz'%(SIM_DIR,sim_dir,output,name), allow_pickle="True")
            sc_full       = data_full['selection']
            for ii in p_1:
                tc_all_1[k][p_1.index(ii)] = sc_full[ii]

            for ii in p_2:
                tc_all_2[k][p_2.index(ii)] = sc_full[ii]
    
        tc_ave_1 = np.zeros((len(p_1),generations+1))
        tc_dev_1 = np.zeros((len(p_1),generations+1))
        
        tc_1     = np.swapaxes(tc_all_1, 0, 2)
        for n in range(len(p_1)):
            for t in range(len(tc_all_1[0][0])):
                tc_ave_1[n][t] = np.average(tc_1[t][n])
                mse      = np.mean((tc_1[t][n] - fi_1[t]) ** 2)
                tc_dev_1[n][t] = np.sqrt(mse) #mse/variance

        tc_ave_2 = np.zeros((len(p_2),generations+1))
        tc_dev_2 = np.zeros((len(p_1),generations+1))
        tc_2     = np.swapaxes(tc_all_2, 0, 2)
        for n in range(len(p_2)):
            for t in range(len(tc_all_2[0][0])):
                tc_ave_2[n][t] = np.average(tc_2[t][n])
                mse      = np.mean((tc_2[t][n] - fi_2[t]) ** 2)
                tc_dev_2[n][t] = np.sqrt(mse) #mse/variance
        
        tc_all.append([tc_all_1,tc_all_2,tc_ave_1,tc_ave_2,tc_dev_1,tc_dev_2])
        
    ## gamma value
    T_ex   = int(round(times[-1]*0.5/10)*10)
    ex_gap  = 10
    etleft  = np.linspace(-T_ex,-ex_gap,int(T_ex/ex_gap))
    etright = np.linspace(times[-1]+ex_gap,times[-1]+T_ex,int(T_ex/ex_gap))
    ExTimes = np.concatenate((etleft, times, etright))

    gamma_p = []
    beta = [0.25, 1, 4]
    for ii in range(len(output_suffix)):
        gamma_t = np.zeros(len(ExTimes))
        last_time = times[-1]
        tv_range = int(round(times[-1]*0.1/10)*10)
        alpha  = np.log(beta[ii]) / tv_range
        for ti, t in enumerate(ExTimes): # loop over all time points, ti: index, t: time
            if t <= 0:
                gamma_t[ti] = beta[ii]
            elif t >= last_time:
                gamma_t[ti] = beta[ii]
            elif 0 < t and t <= tv_range:
                gamma_t[ti] = beta[ii] * np.exp(-alpha * t)
            elif last_time - tv_range < t and t <= last_time:
                gamma_t[ti] = 1 * np.exp(alpha * (t - last_time + tv_range))
            else:
                gamma_t[ti] = 1

        gamma_p.append(gamma_t)

    # PLOT FIGURE
    ## set up figure grid
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box = dict(left=0.08, right=0.98, bottom=0.08, top=0.95)
    gs = gridspec.GridSpec(3, 3, width_ratios=[1,1,1],wspace=0.3,hspace=0.25,**box)
    ax  = [[plt.subplot(gs[n, 0]), plt.subplot(gs[n, 1]), plt.subplot(gs[n, 2])] for n in range(3)]

    c_sin = 5
    c_cos = 2

    ## a -- histogram for selection coefficients
    pprops = { 'xticks':      [-500, 0, 500, 1000, 1500],
               'xticklabels': [],
               'ylim':        [0, 4.2],
               'yticks':      [0,1,4],
               'yticklabels': ['0','g','4g'],
               'nudgey':      1,
               'xlabel':      '',
               'ylabel':      '$\gamma^{\prime}$',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
               'axoffset':    0.1,
               'theme':       'open'}

    mp.plot(type='line',ax=ax[0][0], x=[ExTimes], y=[gamma_p[0]], colors=[BKCOLOR], **pprops)
    mp.plot(type='line',ax=ax[1][0], x=[ExTimes], y=[gamma_p[1]], colors=[BKCOLOR], **pprops)
    pprops['xlabel'] = 'Generation'
    pprops['xticklabels'] = [-500, 0, 500, 1000, 1500]
    mp.plot(type='line',ax=ax[2][0], x=[ExTimes], y=[gamma_p[2]], colors=[BKCOLOR], **pprops)

    # legend
    yy =  2
    coef_legend_dy = 0.6
    xx_line = [-150, 100]
    yy_line = np.zeros((3,2))
    for i in range(3):
        for j in range(2):
            yy_line[i][j] = yy + coef_legend_dy*((2 - i)+ (2 * j - 1)* (0.1+0.025*i))
    c_cols = [C_group[c_sin], C_group[c_cos]]

    pprops['plotprops']['alpha'] = 0.5
    mp.line(ax=ax[0][0], x=[xx_line], y=[[yy_line[0][0], yy_line[0][0]]], colors=[c_cols[0]], **pprops)
    mp.line(ax=ax[0][0], x=[xx_line], y=[[yy_line[0][1], yy_line[0][1]]], colors=[c_cols[1]], **pprops)
    ax[0][0].text(300, yy+coef_legend_dy*2, 'Inferred coefficients', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['alpha'] = 1.0
    pprops['plotprops']['lw'] = SIZELINE*3
    mp.line(ax=ax[0][0], x=[xx_line], y=[[yy_line[1][0], yy_line[1][0]]], colors=[c_cols[0]], **pprops)
    mp.line(ax=ax[0][0], x=[xx_line], y=[[yy_line[1][1], yy_line[1][1]]], colors=[c_cols[1]], **pprops)
    ax[0][0].text(300, yy+coef_legend_dy, 'Average coefficients', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['ls'] = ':'
    mp.plot(type='line',ax=ax[0][0], x=[xx_line], y=[[yy,yy]], colors=[BKCOLOR], **pprops)
    ax[0][0].text(300, yy, 'True coefficients', ha='left', va='center', **DEF_LABELPROPS)

    dy = 0.03
    ax[0][0].text(                     box['left']-0.04, box['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax[0][1].text((2*box['left']/3+box['right']/3)-0.02, box['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax[0][2].text((box['left']/3+2*box['right']/3)+0.01, box['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # add background
    cBG = '#F5F5F5'
    ddx = 0.025
    ddy = 0.15
    rec_xy = [box['left']+ddx, (2*box['top']/3+box['bottom']/3)+ddy]
    rec = matplotlib.patches.Rectangle(xy=( rec_xy[0], rec_xy[1]),
                                            width=(box['right']-box['left'])/3*0.65,height=(box['top']-box['bottom'])/3*0.4, 
                                            transform=fig.transFigure, ec=None, fc=cBG, clip_on=False, zorder=-100)
    rec = ax[0][0].add_patch(rec)

    ## b  -- escape coefficients
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'xticklabels': [],
               'ylim':        [ytick_t[0], ytick_t[-1]],
               'yticks':      ytick_t,
               'yminorticks': yminorticks_t,
               'yticklabels': [int(i*100) for i in ytick_t],
               'nudgey':      1,
               'xlabel':      '',
               'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.1},
               'axoffset':    0.1,
               'theme':       'open'}

    for ii in range(len(output_suffix)):
        tc_all_1 = tc_all[ii][0]
        tc_all_2 = tc_all[ii][1]
        tc_ave_1 = tc_all[ii][2]
        tc_ave_2 = tc_all[ii][3]

        if ii == 2:
            pprops['xticklabels'] = [0,200,400,600,800,1000]
            pprops['xlabel'] = 'Generation'

        pprops['plotprops']['ls'] = '-'
        for n in range(len(p_1)):
            pprops['plotprops']['alpha'] = 0.1
            pprops['plotprops']['lw'] = SIZELINE
            for k in range(100):
                mp.line(ax=ax[ii][1], x=[times], y=[tc_all_1[k][n]], colors=[C_group[c_sin]], **pprops)

            pprops['plotprops']['alpha'] = 1
            pprops['plotprops']['lw'] = SIZELINE*3
            mp.line(ax=ax[ii][1], x=[times], y=[tc_ave_1[n]], colors=[C_group[c_sin]], **pprops)

        for n in range(len(p_2)):
            pprops['plotprops']['alpha'] = 0.1
            pprops['plotprops']['lw'] = SIZELINE
            for k in range(100):
                mp.line(ax=ax[ii][1], x=[times], y=[tc_all_2[k][n]], colors=[C_group[c_cos]], **pprops)

            pprops['plotprops']['alpha'] = 1
            pprops['plotprops']['lw'] = SIZELINE*3
            mp.line(ax=ax[ii][1], x=[times], y=[tc_ave_2[n]], colors=[C_group[c_cos]], **pprops)

        pprops['plotprops']['ls'] = ':'
        pprops['plotprops']['alpha'] = 0.5
        pprops['plotprops']['lw'] = SIZELINE*2
        mp.line(            ax=ax[ii][1], x=[times], y=[fi_1], colors=[BKCOLOR], **pprops)
        mp.plot(type='line',ax=ax[ii][1], x=[times], y=[fi_2], colors=[BKCOLOR], **pprops)

    ## c  -- mean squared error
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'xticklabels': [],
               'ylim':        [ytick_d[0], ytick_d[-1]],
               'yticks':      ytick_d,
               'yminorticks': yminorticks_d,
               'yticklabels': [int(i*100) for i in ytick_d],
               'nudgey':      1,
               'xlabel':      '',
               'ylabel':      'Root mean square error (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1},
               'axoffset':    0.1,
               'theme':       'open'}

    for ii in range(len(output_suffix)):
        tc_dev_1 = tc_all[ii][4]
        tc_dev_2 = tc_all[ii][5]

        if ii == 2:
            pprops['xticklabels'] = [0,200,400,600,800,1000]
            pprops['xlabel'] = 'Generation'

        pprops['plotprops']['ls'] = '-'
        pprops['plotprops']['alpha'] = 1.0
        for n in range(len(p_1)):
            mp.line(ax=ax[ii][2], x=[times], y=[tc_dev_1[n]], colors=[C_group[c_sin]], **pprops)

        for n in range(len(p_2)):
            if n != len(p_2) - 1:
                mp.line(ax=ax[ii][2], x=[times], y=[tc_dev_2[n]], colors=[C_group[c_cos]], **pprops)
            else:
                mp.plot(type='line',ax=ax[ii][2], x=[times], y=[tc_dev_2[n]], colors=[C_group[c_cos]], **pprops)

    # # SAVE FIGURE
    if savepdf:
        plt.savefig('%s/gamma_t_new.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def mpl(x_mu,sample_times):
    
    x        = np.stack(np.array((np.ones(len(x_mu))-x_mu,x_mu))).T # wild type and mutant type
    xx       = np.zeros((len(x),len(x[0]),len(x[0])))
    x_length = len(x[0])

    ## regularization parameter
    gamma_1s       = 1/sample_times[-1]
    gamma_2tv      = 200

    ## functions
    def diffusion_matrix_at_t(x,xx):
        x_length = len(x)
        C = np.zeros([x_length,x_length])
        for i in range(x_length):
            C[i,i] = x[i] - x[i] * x[i]
            for j in range(int(i+1) ,x_length):
                C[i,j] = xx[i,j] - x[i] * x[j]
                C[j,i] = xx[i,j] - x[i] * x[j]
        return C

    def cal_delta_x(single_freq,times):
        delta_x = np.zeros((len(x),x_length))   # difference between the frequency at time t and time t-1s
        # Calculate manually
        for tt in range(len(single_freq)-1):
            h = times[tt+1]-times[tt]
            delta_x[tt] = (single_freq[tt+1] - single_freq[tt])/h
        
        # dt for the last time point, make sure the expected x[t+1] is less than 1
        dt_last = times[-1] - times[-2]
        for ii in range(x_length):
            if single_freq[-1,ii] + delta_x[-1,ii]*dt_last> 1:
                delta_x[-1,ii] = (1 - single_freq[-1,ii])/dt_last
            else:
                delta_x[-1,ii] = delta_x[-2,ii]
        return delta_x
    
    ################################################################################
    ######################## time varying inference (without extend)################

    # extend the time range
    t_extend   = int(round(sample_times[-1]*0.5/10)*10)
    etleft  = np.linspace(-t_extend,-10,int(t_extend/10)) # time added before the beginning time (dt=10)
    etright = np.linspace(sample_times[-1]+10,sample_times[-1]+t_extend,int(t_extend/10))
    ExTimes = np.concatenate((etleft, sample_times, etright))

    # individual site: gamma_2c, escape group and special site: gamma_2tv
    # gamma 2 is also time varying, it is smaller at the boundary
    gamma_t = np.ones(len(ExTimes))
    last_time = sample_times[-1]
    tv_range = max(int(round(last_time*0.1/10)*10),1)
    beta   = 4
    alpha  = np.log(beta) / tv_range
    for ti, t in enumerate(ExTimes): # loop over all time points, ti: index, t: time
        if t <= 0:
            gamma_t[ti] = beta
        elif t >= last_time:
            gamma_t[ti] = beta
        elif 0 < t and t <= tv_range:
            gamma_t[ti] = beta * np.exp(-alpha * t)
        elif last_time - tv_range < t and t <= last_time:
            gamma_t[ti] = 1 * np.exp(alpha * (t - last_time + tv_range))
        else:
            gamma_t[ti] = 1
    gamma_2 = gamma_t * gamma_2tv
    
    # get dx
    delta_x_raw = cal_delta_x(x, sample_times)

    # Use linear interpolates to get the input arrays at any given time point
    interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_dx  = interp1d(sample_times, delta_x_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_g2  = interp1d(ExTimes, gamma_2, axis=0, kind='linear', bounds_error=False, fill_value=0)
    
    def fun(time,s):
        """ Function defining the right-hand side of the system of ODE's"""
        s1                 = s[:x_length,:]   # the actual selection coefficients s1 = s
        s2                 = s[x_length:,:]   # the derivatives of the selection coefficients, s2 = s'
        dsdt               = np.zeros_like(s)  # the RHS of the system of ODE's

        # s' = s2
        dsdt[:x_length, :] = s2

        # s2'(t) = A(t)s1(t) + b(t)
        for ti, t in enumerate(time): # loop over all time points, ti: index, t: time
            
            gamma_2     = interp_g2(t)

            if t < 0 or t > sample_times[-1]:
                # outside the range, only gamma
                A_t = np.diag(gamma_1s * np.ones(x_length))
                b_t = np.zeros(x_length)

            else:
                # calculate the frequency at time t
                single_freq = interp_x(t)
                double_freq = interp_xx(t)
                delta_x     = interp_dx(t)

                # calculate A(t) = C(t) + gamma_1 * I
                C_t = diffusion_matrix_at_t(single_freq, double_freq) # covariance matrix
                A_t = C_t + np.diag(gamma_1s * np.ones(x_length))

                # calculate b(t)
                b_t      = - delta_x

            # s'' = A(t)s(t) + b(t)
            dsdt[x_length:, ti] = A_t @ s1[:, ti] / gamma_2 + b_t / gamma_2

        return dsdt

    # Boundary conditions
    # solution to the system of differential equation with the derivative of the selection coefficients zero at the endpoints
    def bc(b1,b2):
        return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints

    # initial guess for the selection coefficients
    ss_extend = np.zeros((2*x_length,len(ExTimes)))

    # solve the system of differential equations
    solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=100000, tol=1e-3)
    
    sc_sample         = solution.sol(sample_times)
    desired_sc_sample = sc_sample[:x_length,:]

    return desired_sc_sample[1]-desired_sc_sample[0]

def plot_single_mutation(**pdata):
    """
    inference result for a single mutation
    case a : frequency increase linearly
    case b : sigmoidal increase
    """

    # unpack passed data
    generations   = pdata['generations']    # 200
    s_coef        = pdata['s_coef']         # 0.04
    savepdf       = pdata['savepdf']           # True

    tps   = int(generations) + 1
    times = np.linspace(0,generations,tps)
    x_2   = np.exp(s_coef * (times-generations/2)) / (np.exp(s_coef * (times-generations/2)) + 1)

    # x_2 = np.zeros(tps)
    # x_mut = 10
    # x_wt  = 90
    # for t in range(tps):
    #     x_mut_0 = x_mut
    #     x_mut = (1 + s_coef) * x_mut_0
    #     x_2[t] = x_mut/(x_mut+x_wt)

    x_1   = np.linspace(x_2[0],x_2[-1],tps)

    # get inference result for a single mutation
    y_1 = mpl(x_1,times)
    y_2 = mpl(x_2,times)

    # PLOT FIGURE
    ## set up figure grid
    w     = SINGLE_COLUMN
    goldh = w / 1.5
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    box_tra1 = dict(left=0.12, right=0.46, bottom=0.59, top=0.93)
    box_tra2 = dict(left=0.12, right=0.46, bottom=0.15, top=0.49)
    box_sc1  = dict(left=0.61, right=0.95, bottom=0.59, top=0.93)
    box_sc2  = dict(left=0.61, right=0.95, bottom=0.15, top=0.49)

    gs_tra1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra1)
    gs_tra2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra2)
    gs_sc1  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc1)
    gs_sc2  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc2)

    ax_tra1 = plt.subplot(gs_tra1[0, 0])
    ax_tra2 = plt.subplot(gs_tra2[0, 0])
    ax_sc1  = plt.subplot(gs_sc1[0, 0])
    ax_sc2  = plt.subplot(gs_sc2[0, 0])

    dx = -0.08
    dy =  0.02

    ## a,c -- allele frequency for single mutation
    pprops = { 'xticks':      [0, 100, 200],
               'yticks':      [0, 1],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1,
               'ylabel':      'Allele frequency',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
               'axoffset':    0.1,
               'theme':       'open'}

    mp.plot(type='line',ax=ax_tra1, x=[times], y=[x_1], colors=[BKCOLOR], **pprops)
    pprops['xlabel'] = 'Generation (days)'
    mp.plot(type='line',ax=ax_tra2, x=[times], y=[x_2], colors=[BKCOLOR], **pprops)

    ax_tra1.text(box_tra1['left']+dx, box_tra1['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_tra2.text(box_tra2['left']+dx, box_tra2['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## b -- inferred selection coefficients
    pprops = { 'xticks':      [0, 100, 200],
               'yticks':      [0, 0.02,0.04,0.06],
               'yticklabels': [0, 2, 4, 6],
               'nudgey':      1,
               'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0},
               'axoffset':    0.1,
               'theme':       'open'}

    mp.plot(type='line',ax=ax_sc1, x=[times], y=[y_1], colors=[BKCOLOR], **pprops)
    ax_sc1.text(box_sc1['left']+dx, box_sc1['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## d  -- inferred selection coefficients
    pprops = { 'xticks':      [0, 100, 200],
               'yticks':      [0, 0.02,0.04,0.06],
               'yticklabels': [0, 2, 4, 6],
               'nudgey':      1,
               'xlabel':      'Generation (days)',
               'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0},
               'axoffset':    0.1,
               'theme':       'open'}
    
    mp.plot(type='line',ax=ax_sc2, x=[times], y=[y_2], colors=[BKCOLOR], **pprops)
    ax_sc2.text(box_sc2['left']+dx, box_sc2['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    if savepdf:
        plt.savefig('%s/single_mutation.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)



# def plot_gamma_prime_HIV(**pdata):

#     """
#     histogram of trait coefficients with different gamma^{prime} values
#     """
    
#     dirs        = pdata['dirs']
#     HIV_DIR     = pdata['HIV_DIR']
#     FIG_DIR     = pdata['FIG_DIR']
#     tag        = pdata['tag']
#     epi_index = pdata['epi_index']
#     epi_name  = pdata['epi_name']
#     xtick_g     = pdata['xtick_g']
#     xtick       = pdata['xtick']
#     xminortick  = pdata['xminortick']
#     ytick       = pdata['ytick']
#     yminortick  = pdata['yminortick']
#     c_index     = pdata['c_index']
#     savepdf     = pdata['savepdf']

#     FData = GetFigureData(dirs[0],tag,HIV_DIR)
#     ne               = FData.ne
#     sample_times     = FData.sample_times
#     times            = FData.times
#     var_ec           = FData.var_ec[epi_index] # escape coefficients for constant case

#     ## gamma value
#     T_ex   = int(round(times[-1]*0.5/10)*10)
#     ex_gap  = 10
#     etleft  = np.linspace(-T_ex,-ex_gap,int(T_ex/ex_gap))
#     etright = np.linspace(times[-1]+ex_gap,times[-1]+T_ex,int(T_ex/ex_gap))
#     ExTimes = np.concatenate((etleft, times, etright))

#     beta = 4
#     gamma_t = np.zeros(len(ExTimes))
#     last_time = times[-1]
#     tv_range = int(round(times[-1]*0.1/10)*10)
#     alpha  = np.log(beta) / tv_range
#     for ti, t in enumerate(ExTimes): # loop over all time points, ti: index, t: time
#         if t <= 0:
#             gamma_t[ti] = beta
#         elif t >= last_time:
#             gamma_t[ti] = beta
#         elif 0 < t and t <= tv_range:
#             gamma_t[ti] = beta * np.exp(-alpha * t)
#         elif last_time - tv_range < t and t <= last_time:
#             gamma_t[ti] = 1 * np.exp(alpha * (t - last_time + tv_range))
#         else:
#             gamma_t[ti] = 1

#     # PLOT FIGURE
#     ## set up figure grid
#     w     = DOUBLE_COLUMN #SLIDE_WIDTH
#     goldh = w / 1.5
#     fig   = plt.figure(figsize=(w, goldh),dpi=1000)

#     box = dict(left=0.08, right=0.98, bottom=0.08, top=0.98)
#     gs = gridspec.GridSpec(3, 3, width_ratios=[1,1,1],wspace=0.4,hspace=0.25,**box)
#     ax  = [[plt.subplot(gs[n, 0]), plt.subplot(gs[n, 1]), plt.subplot(gs[n, 2])] for n in range(3)]

#     dx = -0.04
#     dy =  0.02

#     c_sin = 5
#     c_cos = 2

#     ## a -- gamma^{prime}
#     pprops = { 'xticks':      xtick_g,
#                'ylim':        [0, 1000],
#                'yticks':      [0,50,200,800],
#                'nudgey':      1,
#                'xlabel':      '',
#                'ylabel':      '$\gamma^{\prime}$',
#                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
#                'axoffset':    0.1,
#                'theme':       'open'}

#     mp.plot(type='line',ax=ax[0][0], x=[[ExTimes[0],ExTimes[-1]]], y=[[200,200]], colors=[BKCOLOR], **pprops)
#     mp.plot(type='line',ax=ax[1][0], x=[ExTimes], y=[gamma_t*50], colors=[BKCOLOR], **pprops)
    
#     pprops['xlabel'] = 'Generation'
#     mp.plot(type='line',ax=ax[2][0], x=[ExTimes], y=[gamma_t*200], colors=[BKCOLOR], **pprops)

#     ## b -- inferred trait coefficients
#     pprops = { 'xticks':      xtick,
#                'xminorticks': xminortick,
#                'xticklabels': [],
#                'yticks':      ytick,
#                'yminorticks': yminortick,
#                'ylabel':      'Inferred escape \n coefficient for ' + epi_name,
#                'nudgey':      1.1,
#                'axoffset':    0.1,
#                'theme':       'open',
#                'combine'     : True}
    
#     for i in range(3):
#         if i == 2:
#             pprops['xticklabels'] = xtick
#             pprops['xlabel'] = 'Days after Fiebig I/II'

#         out_dir = dirs[i]

#         FData = GetFigureData(out_dir,tag,HIV_DIR)
#         tc_sample_ex = FData.sc_sample_ex[-(ne-epi_index)]
#         tc_all_ex    = FData.sc_all_ex[-(ne-epi_index)]

#         print(tc_sample_ex)

#         lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
#         mp.line(ax=ax[i][1], x=[times], y=[tc_all_ex], colors=[C_group[c_index]],plotprops=lprops, **pprops)

#         sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
#         mp.scatter(ax=ax[i][1], x=[sample_times], y=[tc_sample_ex[:]], colors=[C_group[c_index]],plotprops=sprops, **pprops)

#         lprops['ls'] = '--'
#         ax[i][1].axhline(y=0, ls='--', lw=SIZELINE/2, color=BKCOLOR)
#         mp.plot(type ='line',ax=ax[i][1], x=[[xtick[0],xtick[-1]]], y=[[var_ec,var_ec]], colors=[C_group[c_index]],plotprops=lprops, **pprops)

#     legend_x = (xtick[0] + xtick[-1])/2

#     ax[0][1].text(legend_x, ytick[-1], 'VL-dependent r', ha='center', va='center', **DEF_LABELPROPS)

#     ## c -- inferred trait coefficients
#     pprops = { 'xticks':      xtick,
#                'xminorticks': xminortick,
#                'xticklabels': [],
#                'yticks':      ytick,
#                'yminorticks': yminortick,
#                'ylabel':      'Inferred escape \n coefficient for ' + epi_name,
#                'nudgey':      1.1,
#                'axoffset':    0.1,
#                'theme':       'open',
#                'combine'     : True}
    
#     for i in range(3):
#         if i == 2:
#             pprops['xticklabels'] = xtick
#             pprops['xlabel'] = 'Days after Fiebig I/II'

#         out_dir = 'cr/' + dirs[i]

#         FData = GetFigureData(out_dir,tag,HIV_DIR)
#         tc_sample_ex = FData.sc_sample_ex[-(ne-epi_index)]
#         tc_all_ex    = FData.sc_all_ex[-(ne-epi_index)]

#         print(tc_sample_ex)

#         lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
#         mp.line(ax=ax[i][2], x=[times], y=[tc_all_ex], colors=[C_group[c_index]],plotprops=lprops, **pprops)

#         sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
#         mp.scatter(ax=ax[i][2], x=[sample_times], y=[tc_sample_ex[:]], colors=[C_group[c_index]],plotprops=sprops, **pprops)

#         lprops['ls'] = '--'
#         ax[i][2].axhline(y=0, ls='--', lw=SIZELINE/2, color=BKCOLOR)
#         mp.plot(type ='line',ax=ax[i][2], x=[[xtick[0],xtick[-1]]], y=[[var_ec,var_ec]], colors=[C_group[c_index]],plotprops=lprops, **pprops)

#     legend = 'Constant r (' + r'$1.4 \times 10^{-5}$' + ')'
#     ax[0][2].text(legend_x, ytick[-1], legend, ha='center', va='center', **DEF_LABELPROPS)

#     # # SAVE FIGURE
#     if savepdf:
#         plt.savefig('%s/gamma_t_HIV.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_traits(**pdata):

    # unpack passed data
    tag      = pdata['tag']
    name     = pdata['name']
    HIV_DIR  = pdata['HIV_DIR']
    FIG_DIR  = pdata['FIG_DIR']
    xtick      = pdata['xtick']
    xminortick = pdata['xminortick']
    ytick      = pdata['ytick']
    yminortick = pdata['yminortick']
    ytick_es   = pdata['ytick_es']
    yminor_es  = pdata['yminor_es']
    savepdf    = pdata['savepdf']

    out_dir    = HIV_DIR.split('/')[-1]
    # information for escape group
    data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    escape_group = data_pro['escape_group']
    sample_times = data_pro['sample_times']
    # times        = data_pro['times']
    # time_step    = data_pro['time_step']
    muVec        = data_pro['muVec']
    ne           = len(escape_group)

    if ne == 0:
        print(f'CH{tag[-5:]} has no binary trait')
        return
    
    # Get information for escape group
    df_escape = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv'%(HIV_DIR,tag), memory_map=True)
    
    epitopes = df_escape['epitope'].unique()

    var_ec     = [] # escape coefficients for constant case
    traj_var   = [] # frequencies for individual escape sites
    traj_group = [] # frequencies for escape groups
    var_tag    = [] # name for epitope
    var_sc     = [] # selection coefficients for constant case
    sc_index   = [] # index for escape sites (0: polymorphic index, 1: nucleotide, 2: TF)
    for n in range(len(epitopes)):
        df_esc  = df_escape[(df_escape.epitope==epitopes[n])]
        df_row  = df_esc.iloc[0]
        var_ec.append(df_esc.iloc[0].tc_MPL)

        # get the name for epitopes
        epi_nuc = ''.join(epitopes[n])
        var_tag.append(epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc)))

        # get frequencies for escape sites and groups
        traj_group.append([df_row['xp_at_%d' % t] for t in sample_times])
        traj_var_epi = []
        var_sc_epi   = []
        sc_index_epi = []
        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.nucleotide != '-': # not include '-' variants
                traj_var_epi.append([df_entry['f_at_%d' % t] for t in sample_times])
                var_sc_epi.append(df_entry['sc_MPL'])
                index_es = [df_entry['polymorphic_index'], df_entry['nucleotide'], df_entry['TF']]
                sc_index_epi.append(index_es)
        traj_var.append(traj_var_epi)
        var_sc.append(var_sc_epi)
        sc_index.append(sc_index_epi)
    
    # Import data for VL-dependent r
    try:
        data_sc  = np.load('%s/output/sc_%s%s.npz'%(HIV_DIR,tag,name), allow_pickle="True")
        # data_sc  = np.load('%s/output/c_%s_%d.npz'%(HIV_DIR,tag,time_step), allow_pickle="True")

    except FileNotFoundError:
        print(f'No data for CH{tag[-5:]}')
        return
    
    sc_all   = data_sc['selection']
    time_all = np.linspace(0, sample_times[-1], int(sample_times[-1]+1))
    sc_sample  = np.zeros((len(sc_all),len(sample_times)))
    sc_es_c  = [] # selection coefficients for escape sites for constant case
    sc_es_tv = [] # selection coefficients for escape sites for time-varying case
    # selection coefficients for sampled time points
    for i, ti in enumerate(time_all):
        if ti in sample_times:
            index = list(sample_times).index(ti)
            for j in range(len(sc_all)):
                sc_sample[j][index] = sc_all[j][i]

    # selection coefficients
    for n in range(len(var_sc)):
        var_sc_es_n  = np.zeros((2,len(var_sc[n]))) # 0: constant case, 1: time-varying case
        for i in range(len(var_sc[n])):
            # constant case
            var_sc_es_n[0][i] = var_sc[n][i]
            # time-varying case
            site   = int(sc_index[n][i][0])
            allele = NUC.index(sc_index[n][i][1])
            TF     = NUC.index(sc_index[n][i][2])
            index_mu    = int(muVec[site,allele])
            index_TF    = int(muVec[site,TF])
            var_sc_es_n[1][i] = np.average(sc_all[index_mu] - sc_all[index_TF])

        sc_es_c.append(var_sc_es_n[0])
        sc_es_tv.append(var_sc_es_n[1])
    
    '''Setting ticks'''
    if len(ytick) != ne or len(ytick[0]) == 0:
        autotick = True
        # set xticks automatically
        mat_time = sample_times[-1]
        if mat_time> 300:
            dt = 200
        elif mat_time < 100:
            dt = 20
        else:
            dt = 100
            
        xtick = []
        for n in range(math.ceil(mat_time/dt)):
            xtick.append(n*dt)
        xtick.append((n+1)*dt)

        # set yticks automatically
        ytick    = [[] for n in range(ne)]
        ytick_es = [[] for n in range(ne)]
        for n in range(ne):
            # epitope
            max_var = max(sc_all[-(ne-n)])
            min_var = min(sc_all[-(ne-n)])
            ymax = max(max(var_ec[n], max_var) * 1.25,0.02)
            ymin = min(min_var, -0.02)
            ytick[n] = [round(ymin/0.01)*0.01,  0, round(var_ec[n]*100)/100 ,round(ymax/0.01)*0.01]
            # escape sites
            max_var = max(max(sc_es_c[n]), max(sc_es_tv[n]))
            min_var = min(min(sc_es_c[n]), min(sc_es_tv[n]))
            ymax = max(max_var * 1.25,0.01)
            ymin = min(min_var, -0.01)
            ytick_es[n] = [round(ymin/0.01)*0.01,  0, round(ymax/0.01)*0.01]
    else:
        autotick = False

    # PLOT FIGURE
    # set up figure grid
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 4.3 * ne
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    if ne > 1:
        box = dict(left=0.07, right=0.98, bottom=0.10, top=0.95)
    else:
        box = dict(left=0.07, right=0.98, bottom=0.20, top=0.95)
    gs = gridspec.GridSpec(ne, 3, width_ratios=[1,1,1],wspace=0.4,hspace=0.3,**box)
    ax  = [[plt.subplot(gs[n, 0]), plt.subplot(gs[n, 1]), plt.subplot(gs[n,2])] for n in range(ne)]

    if len(xtick) == 0:
        xtick = [int(i) for i in sample_times]

    # a -- escape group frequencies
    pprops = { 'xticks':      xtick,
               'xminorticks': xminortick,
               'xticklabels': [],
               'yticks':      [0, 1.0],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.4 },
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    for n in range(ne):
        pprops['ylabel']          = var_tag[n]
        # add x axis and label at the bottom
        if n == ne - 1:
            pprops['xticklabels'] = xtick
            pprops['xlabel']      = 'Days after Fiebig I/II'

        # plot frequencies for individual escape sites
        for nn in range(len(traj_var[n])):
            pprops['plotprops']['alpha'] = 0.4
            mp.line(ax=ax[n][0], x=[sample_times], y=[traj_var[n][nn]], colors=[C_group[n]], **pprops)

        # plot frequencies for individual group
        pprops['plotprops']['alpha'] = 1
        mp.plot(type='line', ax=ax[n][0], x=[sample_times], y=[traj_group[n]], colors=[C_group[n]], **pprops)

    dy = 0.03
    ax[0][0].text(                     box['left']-0.04, box['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax[0][1].text((2*box['left']/3+box['right']/3)-0.02, box['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax[0][2].text((box['left']/3+2*box['right']/3)+0.01, box['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b. -- escape coefficients for time-varying r and constant r
    lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
    sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':0.5}
    pprops = { 'xticks':      xtick,
               'xticklabels': [],
               'ylabel':      'Inferred coefficient, ' + r'$\hat{s}$' + ' (%)'}

    for n in range(ne):
        pprops['ylim']       = [ytick[n][0], ytick[n][-1]]
        pprops['yticks']      = ytick[n]
        pprops['yticklabels'] = [int(i*100) for i in ytick[n]]

        if len(yminortick) > 0:
            pprops['yminorticks']  = yminortick[n]

        if n == ne - 1:
            pprops['xticklabels'] = xtick
            pprops['xlabel']      = 'Days after Fiebig I/II'

        # VL-dependent r
        mp.line(               ax=ax[n][1], x=[time_all],     y=[sc_all[-(ne-n)]],    colors=[C_group[n]], plotprops=lprops, **pprops)
        mp.plot(type='scatter',ax=ax[n][1], x=[sample_times], y=[sc_sample[-(ne-n)]], colors=[C_group[n]], plotprops=sprops, **pprops)
        
        ax[n][1].axhline(y=var_ec[n], ls=':', lw=SIZELINE, color=C_group[n])
        ax[n][1].axhline(y=0, ls=':', lw=SIZELINE, color=BKCOLOR)

    legend_x = (xtick[0] + xtick[-1])/2
    legend = 'Selection coefficients\n for epitopes'
    ax[0][1].text(legend_x, ytick[0][-1], legend, ha='center', va='center', **DEF_LABELPROPS)
    
    ## c -- constant selection coefficients for escape sites
    sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' ,'alpha':0.5}
    pprops = { 'xlim':        [ -0.3,    6],
               'xticks':      [],
               'ylabel':      'Inferred coefficient, ' + r'$\hat{s}$' + ' (%)',
               'theme':       'open',
               'hide':        ['bottom'] }

    for n in range(ne):
        n_es = len(sc_es_c[n]) # number of escape variants in this epitope

        # set xticks automatically
        pprops['ylim']        = [ytick_es[n][0],ytick_es[n][-1]]
        pprops['yticks']      = ytick_es[n]
        pprops['yticklabels'] = [int(i*100) for i in ytick_es[n]]
        if not autotick:
            pprops['yminorticks'] = yminor_es[n]

        if n_es == 1:
            x_bar = [3]
            d_x   = 4
        else:
            x_bar = np.linspace(0.5, 5.5, n_es)
            d_x   = x_bar[1] - x_bar[0]

        # plot selection coefficients for escape sites
        for i in range(n_es):
            xdat = np.random.normal(x_bar[i], 0.08, 2)
            mp.scatter(ax=ax[n][2], x=[xdat[0]], y=[sc_es_tv[n][i]],colors=[C_group[n]],plotprops=sprops,**pprops)
            
            x_dat = [x_bar[i]-d_x*0.4 , x_bar[i]+d_x*0.4]
            y_dat = [sc_es_c[n][i], sc_es_c[n][i]]

            if i == n_es - 1:
                mp.plot(type='line', ax=ax[n][2], x=[x_dat], y=[y_dat], colors=[C_group[n]], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
            else:
                mp.line(ax=ax[n][2], x=[x_dat], y=[y_dat], colors=[C_group[n]], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)

    legend = 'Selection coefficients\n for escape variants'
    ax[0][2].text(3, ytick_es[0][-1], legend, ha='center', va='center', **DEF_LABELPROPS)

    if savepdf:
        plt.savefig('%s/fig-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()
    else:
        plt.savefig('%s/%s/CH%s%s.jpg' % (FIG_DIR,out_dir,tag[-5:],name), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)


def plot_traits_compare(**pdata):

    # unpack passed data
    tag      = pdata['tag']
    HIV_DIR  = pdata['HIV_DIR']
    FIG_DIR  = pdata['FIG_DIR']
    out_dir    = pdata['out_dir']
    xtick      = pdata['xtick']
    xminortick = pdata['xminortick']
    ytick      = pdata['ytick']
    yminortick = pdata['yminortick']
    savepdf    = pdata['savepdf']

    # information for escape group
    data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    escape_group = data_pro['escape_group']
    sample_times = data_pro['sample_times']
    times        = data_pro['times']
    time_step    = data_pro['time_step']
    ne           = len(escape_group)

    if ne == 0:
        print(f'CH{tag[-5:]} has no binary trait')
        return
    
    # Get information for escape group
    df_escape = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv'%(HIV_DIR,tag), memory_map=True)
    
    epitopes = df_escape['epitope'].unique()

    var_ec     = [] # escape coefficients for constant case
    traj_var   = [] # frequencies for individual escape sites
    traj_group = [] # frequencies for escape groups
    var_tag    = [] # name for epitope
    for n in range(len(epitopes)):
        df_esc  = df_escape[(df_escape.epitope==epitopes[n])]
        df_row  = df_esc.iloc[0]
        var_ec.append(df_esc.iloc[0].tc_MPL)

        # get the name for epitopes
        epi_nuc = ''.join(epitopes[n])
        var_tag.append(epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc)))

        # get frequencies for escape sites and groups
        traj_group.append([df_row['xp_at_%d' % t] for t in sample_times])
        traj_var_epi = []
        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.nucleotide != '-': # not include '-' variants
                traj_var_epi.append([df_entry['f_at_%d' % t] for t in sample_times])
        traj_var.append(traj_var_epi)

    
    ''' Import data for old VL-dependent r '''
    data_sc  = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,out_dir,tag,time_step), allow_pickle="True")
    sc_all   = data_sc['selection']# time range:times
    sc_sample  = np.zeros((len(sc_all),len(sample_times)))
    # selection coefficients for sampled time points
    for i, ti in enumerate(times):
        if ti in sample_times:
            index = list(sample_times).index(ti)
            for j in range(len(sc_all)):
                sc_sample[j][index] = sc_all[j][i]
    
    
    ''' Import data for new VL-dependent r '''
    data_sc  = np.load('%s/output/sc_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    sc_new   = data_sc['selection']
    time_new = np.linspace(0, sample_times[-1], int(sample_times[-1]+1))
    sc_sample_new  = np.zeros((len(sc_new),len(sample_times)))
    # selection coefficients for sampled time points
    for i, ti in enumerate(time_new):
        if ti in sample_times:
            index = list(sample_times).index(ti)
            for j in range(len(sc_new)):
                sc_sample_new[j][index] = sc_new[j][i]
    
    '''Setting ticks'''
    if len(ytick) != ne or len(ytick[0]) == 0:
        autotick = True
        # set xticks automatically
        mat_time = sample_times[-1]
        if mat_time> 300:
            dt = 200
        elif mat_time < 100:
            dt = 20
        else:
            dt = 100
            
        xtick = []
        for n in range(math.ceil(mat_time/dt)):
            xtick.append(n*dt)
        xtick.append((n+1)*dt)

        # set yticks automatically
        ytick    = [[] for n in range(ne)]
        for n in range(ne):
            # epitope
            max_var = max(max(sc_all[-(ne-n)]), max(sc_new[-(ne-n)]))
            min_var = min(min(sc_all[-(ne-n)]), min(sc_new[-(ne-n)]))
            ymax = max(max(var_ec[n], max_var) * 1.25,0.02)
            ymin = min(min_var, -0.02)
            ytick[n] = [round(ymin/0.01)*0.01,  0, round(var_ec[n]*100)/100 ,round(ymax/0.01)*0.01]
    else:
        autotick = False

    # PLOT FIGURE
    # set up figure grid
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3.0 * ne
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    if ne > 1:
        box = dict(left=0.07, right=0.98, bottom=0.10, top=0.95)
    else:
        box = dict(left=0.07, right=0.98, bottom=0.20, top=0.95)
    gs = gridspec.GridSpec(ne, 2, width_ratios=[1,1],wspace=0.4,hspace=0.3,**box)
    ax  = [[plt.subplot(gs[n, 0]), plt.subplot(gs[n, 1])] for n in range(ne)]

    if len(xtick) == 0:
        xtick = [int(i) for i in sample_times]

    # a -- escape group frequencies
    pprops = { 'xticks':      xtick,
               'xminorticks': xminortick,
               'xticklabels': [],
               'yticks':      [0, 1.0],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.4 },
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    for n in range(ne):
        pprops['ylabel']          = var_tag[n]
        # add x axis and label at the bottom
        if n == ne - 1:
            pprops['xticklabels'] = xtick
            pprops['xlabel']      = 'Days after Fiebig I/II'

        # plot frequencies for individual escape sites
        for nn in range(len(traj_var[n])):
            pprops['plotprops']['alpha'] = 0.4
            mp.line(ax=ax[n][0], x=[sample_times], y=[traj_var[n][nn]], colors=[C_group[n]], **pprops)

        # plot frequencies for individual group
        pprops['plotprops']['alpha'] = 1
        mp.plot(type='line', ax=ax[n][0], x=[sample_times], y=[traj_group[n]], colors=[C_group[n]], **pprops)

    dy = 0.03
    ax[0][0].text(                 box['left']-0.04, box['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax[0][1].text((box['left']+box['right'])/2-0.02, box['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b. -- escape coefficients for time-varying r and constant r
    pprops = { 'xticks':      xtick,
               'xticklabels': [],
               'ylabel':      'Inferred coefficient, ' + r'$\hat{s}$' + ' (%)'}

    for n in range(ne):
        pprops['ylim']       = [ytick[n][0], ytick[n][-1]]
        pprops['yticks']      = ytick[n]
        pprops['yticklabels'] = [int(i*100) for i in ytick[n]]

        if len(yminortick) > 0:
            pprops['yminorticks']  = yminortick[n]

        if n == ne - 1:
            pprops['xticklabels'] = xtick
            pprops['xlabel']      = 'Days after Fiebig I/II'

        # old VL-dependent r
        lprops = {'lw': SIZELINE, 'ls': ':', 'alpha': 0.6 }
        sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':0.8}
        mp.line(   ax=ax[n][1], x=[times],        y=[sc_all[-(ne-n)]],    colors=[C_group[n]], plotprops=lprops, **pprops)
        mp.scatter(ax=ax[n][1], x=[sample_times], y=[sc_sample[-(ne-n)]], colors=[C_group[n]], plotprops=sprops, **pprops)

        # # new VL-dependent r version 1
        # lprops = {'lw': SIZELINE, 'ls': '-.', 'alpha': 0.6 }
        # sprops = {'lw' : 0, 's' : 6, 'marker' : '^','alpha':0.8}
        # mp.line(               ax=ax[n][1], x=[time_new],     y=[sc_new_1[-(ne-n)]],        colors=[C_group[n]], plotprops=lprops, **pprops)
        # mp.plot(type='scatter',ax=ax[n][1], x=[sample_times], y=[sc_sample_1[-(ne-n)]], colors=[C_group[n]], plotprops=sprops, **pprops)

        # new VL-dependent r
        lprops = {'lw': SIZELINE, 'ls': '--', 'alpha': 0.6 }
        sprops = {'lw' : 0, 's' : 6, 'marker' : '*','alpha':0.8}
        mp.line(               ax=ax[n][1], x=[time_new],     y=[sc_new[-(ne-n)]],        colors=[C_group[n]], plotprops=lprops, **pprops)
        mp.plot(type='scatter',ax=ax[n][1], x=[sample_times], y=[sc_sample_new[-(ne-n)]], colors=[C_group[n]], plotprops=sprops, **pprops)

        ax[n][1].axhline(y=var_ec[n], ls=':', lw=SIZELINE, color=C_group[n])
        ax[n][1].axhline(y=0, ls=':', lw=SIZELINE, color=BKCOLOR)

    legend_x = (xtick[0] + xtick[-1])/2
    legend = 'Selection coefficients\n for epitopes'
    ax[0][1].text(legend_x, ytick[0][-1], legend, ha='center', va='center', **DEF_LABELPROPS)
    
    if savepdf:
        plt.savefig('%s/compare-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()
    else:
        plt.savefig('%s/HIV/compare-CH%s.jpg' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

# def plot_special_site_compare(**pdata):

#     # unpack passed data
#     tag        = pdata['tag']
#     HIV_DIR    = pdata['HIV_DIR']
#     FIG_DIR    = pdata['FIG_DIR']
#     out_dir    = pdata['out_dir']
#     xtick      = pdata['xtick']
#     xminortick = pdata['xminortick']
#     ytick      = pdata['ytick_sp']
#     yminortick = pdata['yminor_sp']
#     savepdf    = pdata['savepdf']

#     # information for special sites
#     data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
#     muVec    = data_pro['muVec']
#     sample_times  = data_pro['sample_times']
#     times         = data_pro['times']
#     time_step     = data_pro['time_step']
#     special_sites = data_pro['special_sites']
#     if len(special_sites) == 0:
#         print(f'CH{tag[6:]} has no special site')
#         return

#     ''' Get information for special sites in constant case'''
#     df_sc   = pd.read_csv('%s/constant/analysis/%s-analyze.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
    
#     index_s =  [] # variants name
#     for i in special_sites:
#         df_i  = df_sc[(df_sc.polymorphic_index==i) & (df_sc.nucleotide!=df_sc.TF ) & (df_sc.nucleotide!= '-')]
#         for df_iter, df_entry in df_i.iterrows():
#             # get variant name
#             site    = int(df_entry.polymorphic_index)
#             variant = str(site)+df_entry.nucleotide
#             index_s.append(variant)

#     sc_old  =  np.zeros(len(index_s))                 # selection coefficient for constant case
#     for i in range(len(index_s)):
#         site_i = int(index_s[i][:-1])
#         nuc_i  = str(index_s[i][-1])
#         df_sp  = df_sc[(df_sc.polymorphic_index==site_i) & (df_sc.nucleotide == nuc_i)]

#         if len(df_sp) != 1:
#             print(f'error for variant {index_s[i]}')

#         df_i   = df_sp.iloc[0]
#         # get selection coefficient for constant case
#         sc_old[i] = df_sp.sc_MPL

#     ''' Import data for old VL-dependent r '''
#     data_sc = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,out_dir,tag,time_step), allow_pickle="True")
#     sc_all  = data_sc['selection']# time range:times

#     sc_a    =  np.zeros((len(index_s),len(times)))    # time varying sc (time range:times)
#     traj_sp =  [] # frequencies for special sites
#     for i in range(len(index_s)):
#         site_i = int(index_s[i][:-1])
#         nuc_i  = str(index_s[i][-1])
#         df_sp  = df_sc[(df_sc.polymorphic_index==site_i) & (df_sc.nucleotide == nuc_i)]

#         if len(df_sp) != 1:
#             print(f'error for variant {index_s[i]}')

#         df_i   = df_sp.iloc[0]
#         # get selection coefficient for constant case
#         sc_old[i] = df_sp.sc_MPL

#         q_index  = NUC.index(df_i.nucleotide)
#         TF_index = NUC.index(df_i.TF)
#         traj_sp.append([df_i['f_at_%d' % t] for t in sample_times])

#         # use muVec to get the real position of this variant
#         index_mu    = muVec[site_i,q_index]
#         index_TF    = muVec[site_i,TF_index]
#         if index_mu == -1 or index_TF == -1:
#             print('error, %d'%(site))
#         else:
#             sc_a[i] = sc_all[int(index_mu)] - sc_all[int(index_TF)]

#     time_index = []
#     for t in range(len(sample_times)):
#         index = list(times).index(sample_times[t])
#         time_index.append(index)

#     sc_s    = np.zeros((len(index_s),len(sample_times)))#time range:sample_times
#     for i in range(len(sample_times)):
#         index        = time_index[i]
#         sc_s[:,i]    = sc_a[:,index]

#     # PLOT FIGURE
#     # set up figure grid
#     w     = DOUBLE_COLUMN #SLIDE_WIDTH
#     goldh = w / 3
#     fig   = plt.figure(figsize=(w, goldh),dpi=1000)
#     gs  = gridspec.GridSpec(1, 3, width_ratios=[1,0.5,1],wspace = 0.3,)
#     gs.update(left=0.1, right=0.9, bottom=0.20, top=0.95)
#     ax  = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])]

#     if len(xtick) == 0:
#         xtick = [int(i) for i in sample_times]

#     # frequencies
#     pprops = { 'xticks':      xtick,
#                'xminorticks': xminortick,
#                'yticks':      [0, 1.0],
#                'yminorticks': [0.25, 0.5, 0.75],
#                'nudgey':      1.1,
#                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
#                'axoffset':    0.1,
#                'theme':       'open',
#                'combine'     : True,
#                'xlabel':      'Days after Fiebig I/II',
#                'ylabel':      'Frequencies'}

#     for i in range(len(index_s)):
#         pprops['plotprops']['alpha'] = 0.8
#         # plot frequencies for special sites
#         if i == len(index_s) - 1:
#             mp.plot(type='line',ax=ax[0], x=[sample_times], y=[traj_sp[i]], colors=[C_group[i]], **pprops)
#         else:
#             mp.line(ax=ax[0], x=[sample_times], y=[traj_sp[i]], colors=[C_group[i]], **pprops)

#     # label for different special variants
#     n_sp = len(index_s)
#     pprops = { 'xlim': [0,  5],
#                'ylim': [0,  n_sp],
#                'xticks': [],
#                'yticks': [],
#                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 },
#                'theme': 'open',
#                'hide' : ['top', 'bottom', 'left', 'right'] }

#     traj_legend_x  =  1.5
#     traj_legend_y  = [0.5+i for i in range(n_sp)]

#     for k in range(len(traj_legend_y)):
#         x1 = traj_legend_x-2.0
#         x2 = traj_legend_x-0.5
#         yy = traj_legend_y[k]
#         ax[1].text(traj_legend_x, traj_legend_y[k], index_s[k], ha='left', va='center', **DEF_LABELPROPS)
#         mp.plot(type='line',ax=ax[1], x=[[x1, x2]], y=[[yy, yy]], colors=[C_group[k]], **pprops)

#     # escape coefficients with extended time (time range:times)
#     pprops = { 'xlim':        [ 0,    max(sample_times)+5],
#                'xticks':      xtick,
#                'xminorticks': xminortick,
#                'xlabel':      'Days after Fiebig I/II',
#                'ylabel':      'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)'}

#     if len(ytick) == 0:
#         ymax = max(max(max(sc_old), sc_a.max()) * 1.25, 0.02)
#         ymin = min(min(min(sc_old), sc_a.min()) * 1.25,-0.02)
#         pprops['ylim']        = [ymin, ymax]
#         pprops['yticks']      = [round(ymin/0.01)*0.01,  0, round(ymax/0.01)*0.01]
#         pprops['yticklabels'] = [round(ymin/0.01)     ,  0, round(ymax/0.01)*1]
#     else:
#         pprops['yticks'] = ytick
#         pprops['yticklabels'] = [int(i*100) for i in ytick]
#         pprops['yminorticks'] = yminortick

#     lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 1.0 }

#     for i in range(len(index_s)):
#         # 0 and previous result
#         ax[2].axhline(y=sc_old[i], ls=':', lw=SIZELINE, color=C_group[i])
#         ax[2].axhline(y=0, ls='--', lw=SIZELINE/2, color=BKCOLOR)

#         # sc-time (time range:times)
#         mp.line(ax=ax[2], x=[times], y=[sc_a[i]], colors=[C_group[i]],plotprops=lprops, **pprops)

#         # sc-time(time range:sample_times)
#         if i == len(index_s) - 1:
#             sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
#             mp.plot(type='scatter', ax=ax[2], x=[sample_times], y=[sc_s[i]], colors=[C_group[i]],plotprops=sprops, **pprops)
#         else:
#             sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
#             mp.scatter(ax=ax[2], x=[sample_times], y=[sc_s[i]], colors=[C_group[i]],plotprops=sprops, **pprops)

#     if savepdf:
#         plt.savefig('%s/fig-sp-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
#         plt.show()
#     else:
#         plt.savefig('%s/HIV/sp-CH%s.jpg' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_traits_dt(**pdata):

    # unpack passed data
    tag        = pdata['tag']
    HIV_DIR1   = pdata['HIV_DIR1'] # directory for long dt
    HIV_DIR2   = pdata['HIV_DIR2']
    FIG_DIR    = pdata['FIG_DIR']
    xtick      = pdata['xtick']
    xminortick = pdata['xminortick']
    ytick      = pdata['ytick']
    yminortick = pdata['yminortick']
    savepdf    = pdata['savepdf']

    # information for escape group
    data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR1,tag), allow_pickle="True")
    escape_group = data_pro['escape_group']
    sample_time1 = data_pro['sample_times']
    ne           = len(escape_group)

    if ne == 0:
        print(f'CH{tag[-5:]} has no binary trait')
        return
    
    # Get information for escape group
    df_escape = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv'%(HIV_DIR1,tag), memory_map=True)
    epitopes = df_escape['epitope'].unique()
    var_ec     = [] # escape coefficients for constant case
    traj_var   = [] # frequencies for individual escape sites
    traj_group = [] # frequencies for escape groups
    var_tag    = [] # name for epitope
    for n in range(len(epitopes)):
        df_esc  = df_escape[(df_escape.epitope==epitopes[n])]
        df_row  = df_esc.iloc[0]
        var_ec.append(df_esc.iloc[0].tc_MPL)

        # get the name for epitopes
        epi_nuc = ''.join(epitopes[n])
        var_tag.append(epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc)))

        # get frequencies for escape sites and groups
        traj_group.append([df_row['xp_at_%d' % t] for t in sample_time1])
        traj_var_epi = []
        for df_iter, df_entry in df_esc.iterrows():
            if df_entry.nucleotide != '-': # not include '-' variants
                traj_var_epi.append([df_entry['f_at_%d' % t] for t in sample_time1])
        traj_var.append(traj_var_epi)

    ''' Import data for long dt'''
    data_sc  = np.load('%s/output/sc_%s.npz'%(HIV_DIR1,tag), allow_pickle="True")
    sc_1   = data_sc['selection']
    time_1 = np.linspace(0, sample_time1[-1], int(sample_time1[-1]+1))
    sc_sample_1  = np.zeros((len(sc_1),len(sample_time1)))
    # selection coefficients for sampled time points
    for i, ti in enumerate(time_1):
        if ti in sample_time1:
            index = list(sample_time1).index(ti)
            for j in range(len(sc_1)):
                sc_sample_1[j][index] = sc_1[j][i]

    ''' Import data for short dt'''
    data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR2,tag), allow_pickle="True")
    sample_time2 = data_pro['sample_times']
    escape_group = data_pro['escape_group']
    ne_2         = len(escape_group)

    data_sc  = np.load('%s/output/sc_%s.npz'%(HIV_DIR2,tag), allow_pickle="True")
    sc_2   = data_sc['selection']
    time_2 = np.linspace(0, sample_time2[-1], int(sample_time2[-1]+1))
    sc_sample_2  = np.zeros((len(sc_2),len(sample_time2)))
    
    # selection coefficients for sampled time points
    for i, ti in enumerate(time_2):
        if ti in sample_time2:
            index = list(sample_time2).index(ti)
            for j in range(len(sc_2)):
                sc_sample_2[j][index] = sc_2[j][i]
    
    ''''Make sure epitopes in both cases match each other'''
    if ne != ne_2:
        df_escape = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv'%(HIV_DIR2,tag), memory_map=True)
        epitopes = df_escape['epitope'].unique()
        var_ec_2     = [] # escape coefficients for constant case
        var_tag_2    = [] # name for epitope
        for n in range(len(epitopes)):
            df_esc  = df_escape[(df_escape.epitope==epitopes[n])]
            df_row  = df_esc.iloc[0]
            var_ec_2.append(df_esc.iloc[0].tc_MPL)

            # get the name for epitopes
            epi_nuc = ''.join(epitopes[n])
            var_tag_2.append(epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc)))
    else:
        var_ec_2 = var_ec
        var_tag_2 = var_tag

    '''Setting ticks'''
    if len(ytick) != ne or len(ytick[0]) == 0:
        autotick = True
        # set xticks automatically
        mat_time = sample_time1[-1]
        if mat_time> 300:
            dt = 200
        elif mat_time < 100:
            dt = 20
        else:
            dt = 100
            
        xtick = []
        for n in range(math.ceil(mat_time/dt)):
            xtick.append(n*dt)
        xtick.append((n+1)*dt)

        # set yticks automatically
        ytick    = [[] for n in range(ne)]
        for n in range(ne):
            # epitope
            max_var = max(max(sc_1[-(ne-n)]), max(sc_2[-(ne-n)]))
            min_var = min(min(sc_1[-(ne-n)]), min(sc_2[-(ne-n)]))
            ymax = max(max(var_ec[n], max_var) * 1.25,0.02)
            ymin = min(min_var, -0.02)
            ytick[n] = [round(ymin/0.01)*0.01,  0, round(var_ec[n]*100)/100 ,round(ymax/0.01)*0.01]
    else:
        autotick = False

    # PLOT FIGURE
    # set up figure grid
    w     = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3.0 * ne
    fig   = plt.figure(figsize=(w, goldh),dpi=1000)

    if ne > 1:
        box = dict(left=0.07, right=0.98, bottom=0.10, top=0.95)
    else:
        box = dict(left=0.07, right=0.98, bottom=0.20, top=0.95)
    gs = gridspec.GridSpec(ne, 2, width_ratios=[1,1],wspace=0.4,hspace=0.3,**box)
    ax  = [[plt.subplot(gs[n, 0]), plt.subplot(gs[n, 1])] for n in range(ne)]

    if len(xtick) == 0:
        xtick = [int(i) for i in sample_time1]

    # a -- escape group frequencies
    pprops = { 'xticks':      xtick,
               'xminorticks': xminortick,
               'xticklabels': [],
               'yticks':      [0, 1.0],
               'yminorticks': [0.25, 0.5, 0.75],
               'nudgey':      1.1,
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.4 },
               'axoffset':    0.1,
               'theme':       'open',
               'combine'     : True}

    for n in range(ne):
        pprops['ylabel']          = var_tag[n]
        # add x axis and label at the bottom
        if n == ne - 1:
            pprops['xticklabels'] = xtick
            pprops['xlabel']      = 'Days after Fiebig I/II'

        # plot frequencies for individual escape sites
        for nn in range(len(traj_var[n])):
            pprops['plotprops']['alpha'] = 0.4
            mp.line(ax=ax[n][0], x=[sample_time1], y=[traj_var[n][nn]], colors=[C_group[n]], **pprops)

        # plot frequencies for individual group
        pprops['plotprops']['alpha'] = 1
        mp.plot(type='line', ax=ax[n][0], x=[sample_time1], y=[traj_group[n]], colors=[C_group[n]], **pprops)

    dy = 0.03
    ax[0][0].text(                 box['left']-0.04, box['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax[0][1].text((box['left']+box['right'])/2-0.02, box['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b. -- escape coefficients for time-varying r and constant r
    pprops = { 'xticks':      xtick,
               'xticklabels': [],
               'ylabel':      'Inferred coefficient, ' + r'$\hat{s}$' + ' (%)'}

    for n in range(ne):
        pprops['ylim']       = [ytick[n][0], ytick[n][-1]]
        pprops['yticks']      = ytick[n]
        pprops['yticklabels'] = [int(i*100) for i in ytick[n]]

        if len(yminortick) > 0:
            pprops['yminorticks']  = yminortick[n]

        if n == ne - 1:
            pprops['xticklabels'] = xtick
            pprops['xlabel']      = 'Days after Fiebig I/II'

        # short dt VL-dependent r
        if var_tag[n] in var_tag_2:
            n_2 = var_tag_2.index(var_tag[n])
            lprops = {'lw': SIZELINE, 'ls': '--', 'alpha': 0.6 }
            sprops = {'lw' : 0, 's' : 6, 'marker' : '*','alpha':0.8}
            mp.line(   ax=ax[n][1], x=[time_2],     y=[sc_2[-(ne_2-n_2)]],        colors=[C_group[n]], plotprops=lprops, **pprops)
            mp.scatter(ax=ax[n][1], x=[sample_time2], y=[sc_sample_2[-(ne_2-n_2)]], colors=[C_group[n]], plotprops=sprops, **pprops)
            ax[n][1].axhline(y=var_ec_2[n_2], ls='--', lw=SIZELINE, color=C_group[n])

        # long dt VL-dependent r
        lprops = {'lw': SIZELINE, 'ls': ':', 'alpha': 0.6 }
        sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':0.8}
        mp.line(               ax=ax[n][1], x=[time_1],        y=[sc_1[-(ne-n)]],    colors=[C_group[n]], plotprops=lprops, **pprops)
        mp.plot(type='scatter',ax=ax[n][1], x=[sample_time1], y=[sc_sample_1[-(ne-n)]], colors=[C_group[n]], plotprops=sprops, **pprops)
        
        ax[n][1].axhline(y=var_ec[n], ls=':', lw=SIZELINE, color=C_group[n])
        ax[n][1].axhline(y=0, ls=':', lw=SIZELINE, color=BKCOLOR)

    legend_x = (xtick[0] + xtick[-1])/2
    legend = 'Selection coefficients\n for epitopes'
    ax[0][1].text(legend_x, ytick[0][-1], legend, ha='center', va='center', **DEF_LABELPROPS)
    
    if savepdf:
        plt.savefig('%s/compare-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()
    else:
        plt.savefig('%s/compare-CH%s.jpg' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
