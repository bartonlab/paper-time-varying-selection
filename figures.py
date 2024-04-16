#############  PACKAGES  #############
import sys, os
from copy import deepcopy
from importlib import reload

import numpy as np

import scipy as sp
import scipy.stats as st

import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

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
C_group  = ['#32b166','#e5a11c', '#a48cf4','#ff69b4','#ff8c00','#36ada4','#f0e54b']

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
HIV_DIR = 'data/HIV'
FIG_DIR = 'figures'

############# PLOTTING  FUNCTIONS #############
def plot_simple(**pdata):
    """
    Example evolutionary trajectory for a binary 20-site system
    """

    # unpack passed data
    dir           = pdata['dir']            # 'simple'
    name          = pdata['name']           # '0'

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
    bc_n          = pdata['bc_n']          # neutral site color

    # get data
    data        = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR,dir,name.split('_')[0]))
    timepoints  = int(generations) + 1
    times       = np.linspace(0,generations,timepoints)

    if bc_n: # Neumann boundary condition
        data_full   = np.load('%s/%s/output/c_%s.npz'%(SIM_DIR,dir,name), allow_pickle="True")
    else: # Dirichlet boundary condition
        data_full   = np.load('%s/%s/output_d/c_%s.npz'%(SIM_DIR,dir,name), allow_pickle="True")
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

    def find_in_nested_list(a, i):
        for index, sublist in enumerate(a):
            if i in sublist:
                return True, index
        return False, None

    # set up figure grid
    fig   = plt.figure(figsize=(SINGLE_COLUMN, SINGLE_COLUMN*1.2),dpi=500)

    box_tra = dict(left=0.15, right=0.92, bottom=0.72, top=0.95)
    box_sc  = dict(left=0.15, right=0.92, bottom=0.41, top=0.64)
    box_tc  = dict(left=0.15, right=0.92, bottom=0.10, top=0.33)

    gs_tra  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra)
    gs_sc   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc)
    gs_tc   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)

    ax_tra  = plt.subplot(gs_tra[0, 0])
    ax_sc   = plt.subplot(gs_sc[0, 0])
    ax_tc   = plt.subplot(gs_tc[0, 0])

    dx = -0.04
    dy =  0.03

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

    c_sin = -2 # index for sin mutation
    c_cos = -5 # index for cos mutation

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

    # ##  add legend
    # sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' }

    # pprops = { 'xlim':        [ -1 ,    6],
    #            'ylim':        [-0.03, 0.03],
    #            'yticks':      [],
    #            'xticks':      [],
    #            'theme':       'open',
    #            'hide':        ['left','bottom'] }

    # coef_legend_x  =  0
    # coef_legend_d  = -0.6
    # coef_legend_dy = -0.011
    # c_coe1         = [C_BEN, C_NEU, C_DEL, C_group[0]]
    # coef_legend_t  = ['Beneficial', 'Neutral', 'Deleterious','Escape sites']
    # for k in range(len(coef_legend_t)):
    #     mp.scatter(ax=ax_lab, x=[[coef_legend_x+coef_legend_d]], y=[[0.021 + (k *coef_legend_dy)]],colors=[c_coe1[k]],plotprops=sprops,**pprops)
    #     ax_lab.text(coef_legend_x, 0.021 + (k*coef_legend_dy), coef_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    # yy =  0.021 + 4.2 * coef_legend_dy
    # mp.plot(type='line',ax=ax_lab, x=[[coef_legend_x-0.9, coef_legend_x-0.3]], y=[[yy, yy]], \
    # colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    # ax_lab.text(coef_legend_x, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    # ax_sc.text(box_tra1['left']+dx, box_lab['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## b -- constant selection coefficients (beneficial/neutral/deleterious)
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
    ax_sc.text(box_sc['left']+dx, box_sc['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    
    ## c -- time-varying selection coefficients (sin/cos)
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
    mp.line(            ax=ax_tc, x=[times], y=[fi_2], colors=[C_group[c_cos]], **pprops)
    mp.plot(type='line',ax=ax_tc, x=[[0,times[-1]]], y=[[0,0]], colors=[BKCOLOR], **pprops)

    ax_tc.text(box_tc['left']+dx, box_tc['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    if savepdf==True:
        plt.savefig('%s/fig-%s.pdf' % (FIG_DIR,dir), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    else:
        if bc_n: # Neumann boundary condition
            plt.savefig('%s/%s/%s.jpg' % (FIG_DIR,dir,name), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        else: # Dirichlet boundary condition
            plt.savefig('%s/%s/%s_d.jpg' % (FIG_DIR,dir,name), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_trait(**pdata):
    """
    Example evolutionary trajectory for a binary 20-site system
    """

    # unpack passed data
    dir           = pdata['dir']            # 'trait'
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

    nB            = pdata['n_ben']          # 4
    nD            = pdata['n_del']          # 0.02
    fB            = pdata['s_ben']          # 4
    fD            = pdata['s_del']          # 0.02
    fn            = pdata['fn']             # time-varying escape coefficient
    fi            = pdata['fi']             # time-varying selection coefficient

    savepdf       = pdata['savepdf']         # True

    # get data
    data        = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR,dir,name.split('_')[0]))
    ne          = len(escape_group)
    timepoints  = int(generations) + 1
    times       = np.linspace(0,generations,timepoints)

    data_full   = np.load('%s/%s/output%s/c_%s.npz'%(SIM_DIR,dir,output,name), allow_pickle="True")
    sc_full     = data_full['selection']
    TimeVaryingSC = [np.average(sc_full[i]) for i in range(seq_length)]
    TimeVaryingTC = sc_full[-ne:]

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
    fig   = plt.figure(figsize=(6, 3),dpi=500)

    box_tra1 = dict(left=0.10, right=0.32, bottom=0.60, top=0.95)
    box_tra2 = dict(left=0.40, right=0.62, bottom=0.60, top=0.95)
    box_tra3 = dict(left=0.70, right=0.92, bottom=0.60, top=0.95)
    box_lab  = dict(left=0.05, right=0.15, bottom=0.10, top=0.45)
    box_sc   = dict(left=0.24, right=0.40, bottom=0.10, top=0.45)
    box_tc   = dict(left=0.46, right=0.64, bottom=0.10, top=0.45)
    box_sp   = dict(left=0.74, right=0.92, bottom=0.10, top=0.45)
    # box_sc   = dict(left=0.22, right=0.32, bottom=0.10, top=0.45)
    # box_tc   = dict(left=0.40, right=0.62, bottom=0.10, top=0.45)
    # box_sp   = dict(left=0.70, right=0.92, bottom=0.10, top=0.45)

    gs_tra1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra1)
    gs_tra2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra2)
    gs_tra3 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tra3)
    gs_lab  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_lab)
    gs_sc   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc)
    gs_tc   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)
    gs_sp   = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sp)

    ax_tra1 = plt.subplot(gs_tra1[0, 0])
    ax_tra2 = plt.subplot(gs_tra2[0, 0])
    ax_tra3 = plt.subplot(gs_tra3[0, 0])
    ax_lab  = plt.subplot(gs_lab[0, 0])
    ax_sc   = plt.subplot(gs_sc[0, 0])
    ax_tc   = plt.subplot(gs_tc[0, 0])
    ax_sp   = plt.subplot(gs_sp[0, 0])

    dx = -0.04
    dy =  0.03

    ## a,b,c -- allele frequencies - individual sites,  escape groups, special sites
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
        if i not in p_sites:
            if i < nB:
                mp.line(ax=ax_tra1, x=[times], y=[x[i]], colors=[C_BEN], **pprops)
            elif i >= seq_length-nD:
                mp.line(ax=ax_tra1, x=[times], y=[x[i]], colors=[C_DEL], **pprops)
            else:
                mp.line(ax=ax_tra1, x=[times], y=[x[i]], colors = [C_NEU], **pprops)

        else:
            # all special sites
            # if i < nB:
            #     mp.line(ax=ax_tra3, x=[times], y=[x[i]], colors=[C_BEN], **pprops)
            # elif i >= seq_length-nD:
            #     mp.plot(type='line',ax=ax_tra3, x=[times], y=[x[i]], colors=[C_DEL], **pprops)
            # else:
            #     mp.line(ax=ax_tra3, x=[times], y=[x[i]], colors = [C_NEU], **pprops)
            mp.line(ax=ax_tra3, x=[times], y=[x[i]], colors = [C_group[-2]], **pprops)

        # if the site is escape site, plot it in figure b
        found, group = find_in_nested_list(escape_group, i)
        if found:
            pprops['plotprops']['alpha'] = 0.4
            mp.line(ax=ax_tra2, x=[times], y=[x[i]], colors=[C_group[group]], **pprops)

    # escape group
    pprops['plotprops']['alpha'] = 1
    for n in range(ne):
        mp.line(ax=ax_tra2, x=[times], y=[y[n]], colors=[C_group[n]], **pprops)

    pprops['plotprops'] = {'lw': SIZELINE, 'ls': '-', 'alpha': 0 }
    mp.plot(type='line',ax=ax_tra1, x=[[0,1000]], y=[[1,1]], colors=[C_NEU], **pprops)
    mp.plot(type='line',ax=ax_tra2, x=[[0,1000]], y=[[1,1]], colors=[C_NEU], **pprops)
    mp.plot(type='line',ax=ax_tra3, x=[[0,1000]], y=[[1,1]], colors=[C_NEU], **pprops)

    ax_tra1.text( box_tra1['left']+dx,  box_tra1['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_tra2.text( box_tra2['left']+dx,  box_tra2['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    ax_tra3.text( box_tra3['left']+dx,  box_tra3['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ##  add legend
    sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' }

    pprops = { 'xlim':        [ -1 ,    6],
               'ylim':        [-0.03, 0.03],
               'yticks':      [],
               'xticks':      [],
               'theme':       'open',
               'hide':        ['left','bottom'] }

    coef_legend_x  =  0
    coef_legend_d  = -0.6
    coef_legend_dy = -0.011
    c_coe1         = [C_BEN, C_NEU, C_DEL, C_group[0]]
    coef_legend_t  = ['Beneficial', 'Neutral', 'Deleterious','Escape sites']
    for k in range(len(coef_legend_t)):
        mp.scatter(ax=ax_lab, x=[[coef_legend_x+coef_legend_d]], y=[[0.021 + (k *coef_legend_dy)]],colors=[c_coe1[k]],plotprops=sprops,**pprops)
        ax_lab.text(coef_legend_x, 0.021 + (k*coef_legend_dy), coef_legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    yy =  0.021 + 4.2 * coef_legend_dy
    mp.plot(type='line',ax=ax_lab, x=[[coef_legend_x-0.9, coef_legend_x-0.3]], y=[[yy, yy]], \
    colors=[BKCOLOR], plotprops=dict(lw=SIZELINE, ls=':'), **pprops)
    ax_lab.text(coef_legend_x, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ax_sc.text(box_tra1['left']+dx, box_lab['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## d -- individual beneficial/neutral/deleterious selection coefficients

    pprops = { 'xlim':        [ -0.3,    6],
               'ylim':        [-0.03, 0.03],
               'yticks':      [-0.03, 0, 0.03],
               'yminorticks': [-0.02, -0.01, 0.01, 0.02],
               'yticklabels': [-3, 0, 3],
               'xticks':      [],
               'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'theme':       'open',
               'hide':        ['bottom'] }

    nN        = seq_length-nB-nD

    x_ben = np.random.normal(1, 0.08, nB)
    x_neu = np.random.normal(3, 0.16, nN)
    x_del = np.random.normal(5, 0.08, nD)
    x_bar = np.hstack([x_ben,x_neu,x_del])

    for i in range(seq_length):
        found, group = find_in_nested_list(escape_group, i)
        if i not in p_sites:
            xdat = [x_bar[i]]
            ydat = [TimeVaryingSC[i]]
            if found:
                mp.scatter(ax=ax_sc, x=[xdat], y=[ydat],colors=[C_group[group]],plotprops=sprops,**pprops)
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

    ## e -- trait coefficients
    pprops = { 'xticks':      [0, 200, 400, 600, 800, 1000],
               'ylim':        [ytick_e[0], ytick_e[-1]],
               'yticks':      ytick_e,
               'yminorticks': yminorticks_e,
               'yticklabels': [int(i*100) for i in ytick_e],
               'nudgey':      1,
               'xlabel':      'Generation',
               'ylabel':      'Inferred trait\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1 },
               'axoffset':    0.1,
               'theme':       'open'}

    yy =  0.05
    for n in range(ne):
        pprops['plotprops']['ls'] = ':'
        mp.line(ax=ax_tc, x=[times], y=[fn], colors=[C_group[n]], **pprops)
        # mp.line(ax=ax_tc, x=[[150, 250]], y=[[yy+0.01*n, yy+0.01*n]], colors=[C_group[n]], **pprops)
        pprops['plotprops']['ls'] = '-'
        mp.plot(type='line',ax=ax_tc, x=[times], y=[TimeVaryingTC[n]], colors=[C_group[n]], **pprops)
        # ax_tc.text(300, yy+0.01*n, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ax_tc.text(box_tc['left']+dx, box_tc['top']+dy, 'e'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## f -- special sites selection coefficients
    pprops = {  'xticks':      [0, 200, 400, 600, 800, 1000],
                'ylim':        [ytick_f[0], ytick_f[-1]],
                'yticks':      ytick_f,
                'yminorticks': yminorticks_f,
                'yticklabels': [int(i*100) for i in ytick_f],
                'nudgey':      1,
                'xlabel':      'Generation',
                'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 1 },
                'axoffset':    0.1,
                'theme':       'open'}

    for ii in range(len(p_sites)):
        p_index = p_sites[ii]
        sc_p = sc_full[p_index]
        # if p_index < nB:
        #     mp.line(ax=ax_sp, x=[times], y=[sc_p], colors=[C_BEN], **pprops)
        # elif p_index >= seq_length-nD:
        #     mp.line(ax=ax_sp, x=[times], y=[sc_p], colors=[C_DEL], **pprops)
        # else:
        #     mp.line(ax=ax_sp, x=[times], y=[sc_p], colors=[C_NEU], **pprops)
        mp.line(ax=ax_sp, x=[times], y=[sc_p], colors=[C_group[-2]], **pprops)

    pprops['plotprops']['ls'] = ':'
    mp.plot(type='line',ax=ax_sp, x=[times], y=[fi], colors=[C_group[-2]], **pprops)

    ax_sp.text(box_sp['left']+dx, box_sp['top']+dy, 'f'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    if savepdf:
        plt.savefig('%s/fig-%s.pdf' % (FIG_DIR,dir), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()
    else:
        plt.savefig('%s/%s/%s.jpg' % (FIG_DIR,dir,name), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
    
# def plot_simple_his_try(**pdata):

#     """
#     histogram of selection coefficients and trait coefficients
#     """

#     # unpack passed data
#     dir           = pdata['dir']            # 'simple'
#     generations   = pdata['generations']    # 500
#     xtick_h       = pdata['xtick_h']
#     ytick_h       = pdata['ytick_h']
#     ytick_t       = pdata['ytick_t']
#     yminorticks_t = pdata['yminorticks_t']

#     p_1           = pdata['p_1']            # [6,7] , special sites 1
#     p_2           = pdata['p_2']            # [8,9] , special sites 2
#     fB            = pdata['s_ben']          # 0.02
#     fD            = pdata['s_del']          # -0.02
#     fi_1          = pdata['fi_1']           # time-varying selection coefficient for special sites 1
#     fi_2          = pdata['fi_2']           # time-varying selection coefficient for special sites 2

#     bc_n          = pdata['bc_n']           # True

#     timepoints  = int(generations) + 1
#     times       = np.linspace(0,generations,timepoints)
#     TLeft   = int(round(times[-1]*2/10)*10) # time range added before the beginning time
#     TRight  = int(round(times[-1]*2/10)*10) # time range added after the ending time
#     etleft  = np.linspace(-TLeft,-40,int(TLeft/40)) # time added before the beginning time (dt=10)
#     etright = np.linspace(times[-1]+40,times[-1]+TRight,int(TRight/40))
#     ExTimes = np.concatenate((etleft, times, etright))

#     # data for selection coefficients for different simulations
#     if bc_n: # Neumann boundary condition
#         df       = pd.read_csv('%s/%s/mpl_collected.csv' % (SIM_DIR,dir), memory_map=True)
#     else: # Dirichlet boundary condition
#         df       = pd.read_csv('%s/%s/mpl_collected_d.csv' % (SIM_DIR,dir), memory_map=True)
#     ben_cols = ['sc_%d' % i for i in [0,1]]
#     neu_cols = ['sc_%d' % i for i in [2,3]]
#     del_cols = ['sc_%d' % i for i in [4,5]]

#     # get data for inference results for different simulations
#     tc_all_1   = np.zeros((100,len(p_1),len(ExTimes)))
#     tc_all_2   = np.zeros((100,len(p_2),len(ExTimes)))

#     for k in range(100):
#         name = str(k)
#         if bc_n: # Neumann boundary condition
#             # data_full     = np.load('%s/%s/output/c_%s.npz'%(SIM_DIR,dir,name), allow_pickle="True")
#             data_full     = np.load('%s/%s/output-1-2/c_%s.npz'%(SIM_DIR,dir,name), allow_pickle="True")
#         else: # Dirichlet boundary condition
#             data_full     = np.load('%s/%s/output_d/c_%s.npz'%(SIM_DIR,dir,name), allow_pickle="True")
#         sc_full       = data_full['all']
#         for ii in p_1:
#             tc_all_1[k][p_1.index(ii)] = sc_full[ii]
#         for ii in p_2:
#             tc_all_2[k][p_2.index(ii)] = sc_full[ii]
        
#     tc_ave_1 = np.zeros((len(p_1),len(ExTimes)))
#     tc_1     = np.swapaxes(tc_all_1, 0, 2)
#     for n in range(len(p_1)):
#         for t in range(len(tc_all_1[0][0])):
#             tc_ave_1[n][t] = np.average(tc_1[t][n])

#     tc_ave_2 = np.zeros((len(p_2),len(ExTimes)))
#     tc_2     = np.swapaxes(tc_all_2, 0, 2)
#     for n in range(len(p_2)):
#         for t in range(len(tc_all_2[0][0])):
#             tc_ave_2[n][t] = np.average(tc_2[t][n])

#     # PLOT FIGURE
#     ## set up figure grid

#     w     = DOUBLE_COLUMN #SLIDE_WIDTH
#     goldh = w / 1.8
#     fig   = plt.figure(figsize=(w, goldh),dpi=1000)

#     box_se  = dict(left=0.10, right=0.92, bottom=0.65, top=0.95)
#     box_lab = dict(left=0.05, right=0.15, bottom=0.10, top=0.45)
#     box_tc1 = dict(left=0.24, right=0.52, bottom=0.07, top=0.50)
#     box_tc2 = dict(left=0.60, right=0.92, bottom=0.07, top=0.50)

#     gs_se  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_se)
#     gs_lab = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_lab)
#     gs_tc1 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc1)
#     gs_tc2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc2)

#     ax_se  = plt.subplot(gs_se[0, 0])
#     ax_lab = plt.subplot(gs_lab[0, 0])
#     ax_tc1 = plt.subplot(gs_tc1[0, 0])
#     ax_tc2 = plt.subplot(gs_tc2[0, 0])

#     dx = -0.04
#     dy =  0.03

#     ## a -- histogram for selection coefficients

#     dashlineprops = { 'lw' : SIZELINE * 1.5, 'ls' : ':', 'alpha' : 0.5, 'color' : BKCOLOR }
#     histprops = dict(histtype='bar', lw=SIZELINE/2, rwidth=0.8, ls='solid', alpha=0.6, edgecolor='none')
#     pprops = { 'xlim':        [xtick_h[0], xtick_h[-1]],
#                'xticks':      xtick_h,
#                'xticklabels': [int(i*100) for i in xtick_h],
#                'ylim':        [ytick_h[0], ytick_h[-1]],
#                'yticks':      ytick_h,
#                'xlabel'      : 'Inferred selection coefficient, ' + r'$\hat{s}$' + ' (%)',
#                'ylabel'      : 'Frequency',
#                'bins'        : np.arange(-0.04, 0.04, 0.001),
#                'combine'     : True,
#                'plotprops'   : histprops,
#                'axoffset'    : 0.1,
#                'theme'       : 'boxed' }

#     colors     = [C_BEN, C_NEU, C_DEL]
#     tags       = ['beneficial', 'neutral', 'deleterious']
#     cols       = [ben_cols, neu_cols, del_cols]
#     s_true_loc = [fB, 0, fD]

#     for i in range(len(tags)):
#         x = [np.array(df[cols[i]]).flatten()]
#         tprops = dict(ha='center', va='center', family=FONTFAMILY, size=SIZELABEL, clip_on=False)
#         ax_se.text(s_true_loc[i], ytick_h[-1]*1.04, r'$s_{%s}$' % (tags[i]), color=colors[i], **tprops)
#         dashlineprops['color'] = colors[i]
#         ax_se.axvline(x=s_true_loc[i], **dashlineprops)
#         if i<len(tags)-2: mp.hist(             ax=ax_se, x=x, colors=[colors[i]], **pprops)
#         else:             mp.plot(type='hist', ax=ax_se, x=x, colors=[colors[i]], **pprops)

#     ax_se.text(  box_se['left']+dx,  box_se['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

#     ##  add legend
#     sprops = { 'lw' : 0, 's' : 9., 'marker' : 'o' }

#     pprops = { 'xlim':        [ -1 ,    6],
#                'ylim':        [-0.05, 0.05],
#                'yticks':      [],
#                'xticks':      [],
#                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.15 },
#                'theme':       'open',
#                'hide':        ['left','bottom'] }

#     yy =  -0.021
#     coef_legend_dy = 0.021
#     c_epitope = C_group[0]

#     pprops['plotprops']['alpha'] = 0.15
#     mp.line(ax=ax_lab, x=[[-0.9, 1.3]], y=[[yy+coef_legend_dy*2, yy+coef_legend_dy*2]], colors=[c_epitope], **pprops)
#     ax_lab.text(2, yy+coef_legend_dy*2, 'Inferred \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

#     pprops['plotprops']['alpha'] = 1
#     pprops['plotprops']['lw'] = SIZELINE*3
#     mp.line(ax=ax_lab, x=[[-0.9, 1.3]], y=[[yy+coef_legend_dy, yy+coef_legend_dy]], colors=[c_epitope], **pprops)
#     ax_lab.text(2, yy+coef_legend_dy, 'Average \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

#     pprops['plotprops']['ls'] = ':'
#     mp.plot(type='line',ax=ax_lab, x=[[-0.9, 1.3]], y=[[yy, yy]], colors=[c_epitope], **pprops)
#     ax_lab.text(2, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

#     ## b  -- escape coefficients
#     pprops = { 'xticks':      [-2000, -1000, 0, 1000, 2000, 3000],
#                'ylim':        [ytick_t[0], ytick_t[-1]],
#                'yticks':      ytick_t,
#                'yminorticks': yminorticks_t,
#                'yticklabels': [int(i*100) for i in ytick_t],
#                'nudgey':      1,
#                'xlabel':      'Generation',
#                'ylabel':      'Inferred selection\ncoefficient, ' + r'$\hat{s}$' + ' (%)',
#                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.15 },
#                'axoffset':    0.1,
#                'theme':       'open'}

#     c_sin = -2
#     c_cos = -5

#     for n in range(len(p_1)):
#         pprops['plotprops']['alpha'] = 0.15
#         pprops['plotprops']['lw'] = SIZELINE
#         for k in range(100):
#             mp.line(ax=ax_tc1, x=[ExTimes], y=[tc_all_1[k][n]], colors=[C_group[c_sin]], **pprops)

#         pprops['plotprops']['alpha'] = 1
#         pprops['plotprops']['lw'] = SIZELINE*3
#         mp.line(ax=ax_tc1, x=[ExTimes], y=[tc_ave_1[n]], colors=[C_group[c_sin]], **pprops)

#     for n in range(len(p_2)):
#         pprops['plotprops']['alpha'] = 0.15
#         pprops['plotprops']['lw'] = SIZELINE
#         for k in range(100):
#             mp.line(ax=ax_tc2, x=[ExTimes], y=[tc_all_2[k][n]], colors=[C_group[c_cos]], **pprops)

#         pprops['plotprops']['alpha'] = 1
#         pprops['plotprops']['lw'] = SIZELINE*3
#         mp.line(ax=ax_tc2, x=[ExTimes], y=[tc_ave_2[n]], colors=[C_group[c_cos]], **pprops)

#     pprops['plotprops']['ls'] = ':'
#     mp.plot(type='line',ax=ax_tc1, x=[times], y=[fi_1], colors=[C_group[c_sin]], **pprops)
#     mp.plot(type='line',ax=ax_tc2, x=[times], y=[fi_2], colors=[C_group[c_cos]], **pprops)

#     ax_tc1.text(box_tc1['left']+dx, box_tc1['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
#     ax_tc2.text(box_tc2['left']+dx, box_tc2['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

#     # # SAVE FIGURE
#     # if bc_n:
#     #     plt.savefig('%s/simple_his_1-2.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
#     # else:
#     #     plt.savefig('%s/simple_his_d.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_simple_his(**pdata):

    """
    histogram of selection coefficients and trait coefficients
    """

    # unpack passed data
    dir           = pdata['dir']            # 'simple'
    output        = pdata['output']           # ''
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
    df       = pd.read_csv('%s/%s/mpl_collected%s.csv' % (SIM_DIR,dir,output), memory_map=True)
    ben_cols = ['sc_%d' % i for i in [0,1]]
    neu_cols = ['sc_%d' % i for i in [2,3]]
    del_cols = ['sc_%d' % i for i in [4,5]]

    # get data for inference results for different simulations
    tc_all_1   = np.zeros((100,len(p_1),generations+1))
    tc_all_2   = np.zeros((100,len(p_2),generations+1))

    for k in range(100):
        name = str(k)
        data_full     = np.load('%s/%s/output%s/c_%s.npz'%(SIM_DIR,dir,output,name), allow_pickle="True")
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
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.15 },
               'theme':       'open',
               'hide':        ['left','bottom'] }

    yy =  -0.021
    coef_legend_dy = 0.021
    c_epitope = C_group[0]

    pprops['plotprops']['alpha'] = 0.15
    mp.line(ax=ax_lab, x=[[-0.9, 1.3]], y=[[yy+coef_legend_dy*2, yy+coef_legend_dy*2]], colors=[c_epitope], **pprops)
    ax_lab.text(2, yy+coef_legend_dy*2, 'Inferred \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['alpha'] = 1
    pprops['plotprops']['lw'] = SIZELINE*3
    mp.line(ax=ax_lab, x=[[-0.9, 1.3]], y=[[yy+coef_legend_dy, yy+coef_legend_dy]], colors=[c_epitope], **pprops)
    ax_lab.text(2, yy+coef_legend_dy, 'Average \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['ls'] = ':'
    mp.plot(type='line',ax=ax_lab, x=[[-0.9, 1.3]], y=[[yy, yy]], colors=[c_epitope], **pprops)
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

    c_sin = -2
    c_cos = -5

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
        plt.savefig('%s/simple_his%s.pdf' % (FIG_DIR,output), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
   
def plot_trait_his(**pdata):

    """
    histogram of selection coefficients and trait coefficients
    """

    # unpack passed data
    dir           = pdata['dir']            # 'sim'
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
    fB            = pdata['s_ben']          # 0.02
    fD            = pdata['s_del']          # -0.02
    fn            = pdata['fn']             # time-varying selection coefficient
    fi            = pdata['fi']             # time-varying selection coefficient
    savepdf       = pdata['savepdf']         # True

    with open("%s/%s/escape_groups.dat"%(SIM_DIR,dir), 'r') as file:
        escape_groups = json.load(file)

    with open("%s/%s/special_groups.dat"%(SIM_DIR,dir), 'r') as file:
        special_groups = json.load(file)

    ne          = len(escape_groups[0])
    timepoints  = int(generations) + 1
    times       = np.linspace(0,generations,timepoints)

    # data for selection coefficients for different simulations
    df       = pd.read_csv('%s/%s/mpl_collected%s.csv' % (SIM_DIR,dir,output), memory_map=True)
    ben_cols = ['sc_%d' % i for i in [0,1,2,3]]
    neu_cols = ['sc_%d' % i for i in [4,5,6,7,8,9,10,11,12,13,14,15]]
    del_cols = ['sc_%d' % i for i in [16,17,18,19]]

    # get data for inference results for different simulations
    tc_all   = np.zeros((100,ne,generations+1))
    sc_p_all = np.zeros((100,len(special_groups[0]),generations+1))

    for k in range(100):
        name = str(k)
        p_sites       = special_groups[k]

        data_full     = np.load('%s/%s/output%s/c_%s.npz'%(SIM_DIR,dir,output,name), allow_pickle="True")
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
    box_tc  = dict(left=0.24, right=0.52, bottom=0.10, top=0.50)
    box_sc2 = dict(left=0.60, right=0.92, bottom=0.10, top=0.50)

    gs_se  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_se)
    gs_lab = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_lab)
    gs_tc  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc)
    gs_sc2 = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_sc2)

    ax_se  = plt.subplot(gs_se[0, 0])
    ax_lab = plt.subplot(gs_lab[0, 0])
    ax_tc  = plt.subplot(gs_tc[0, 0])
    ax_sc2 = plt.subplot(gs_sc2[0, 0])

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
               'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.15 },
               'theme':       'open',
               'hide':        ['left','bottom'] }

    yy =  -0.021
    coef_legend_dy = 0.021
    c_epitope = C_group[0]

    pprops['plotprops']['alpha'] = 0.15
    mp.line(ax=ax_lab, x=[[-0.9, 1.3]], y=[[yy+coef_legend_dy*2, yy+coef_legend_dy*2]], colors=[c_epitope], **pprops)
    ax_lab.text(2, yy+coef_legend_dy*2, 'Inferred \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['alpha'] = 1
    pprops['plotprops']['lw'] = SIZELINE*3
    mp.line(ax=ax_lab, x=[[-0.9, 1.3]], y=[[yy+coef_legend_dy, yy+coef_legend_dy]], colors=[c_epitope], **pprops)
    ax_lab.text(2, yy+coef_legend_dy, 'Average \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    pprops['plotprops']['ls'] = ':'
    mp.plot(type='line',ax=ax_lab, x=[[-0.9, 1.3]], y=[[yy, yy]], colors=[c_epitope], **pprops)
    ax_lab.text(2, yy, 'True \ncoefficient', ha='left', va='center', **DEF_LABELPROPS)

    ## b  -- escape coefficients
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


    ax_tc.text(box_tc['left']+dx, box_tc['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    ## c  -- selection coefficients for special sites
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
            mp.line(ax=ax_sc2, x=[times], y=[sc_p_all[k][n]], colors=[C_group[-2]], **pprops)

        pprops['plotprops']['alpha'] = 0.4
        pprops['plotprops']['lw'] = SIZELINE*1.2
        mp.line(ax=ax_sc2, x=[times], y=[sc_p_all[0][n]], colors=[C_group[-2]], **pprops)

        pprops['plotprops']['alpha'] = 1
        pprops['plotprops']['lw'] = SIZELINE*3
        mp.line(ax=ax_sc2, x=[times], y=[sc_average[n]], colors=[C_group[-2]], **pprops)

    pprops['plotprops']['ls'] = ':'
    mp.plot(type='line',ax=ax_sc2, x=[times], y=[fi], colors=[C_group[-2]], **pprops)

    ax_sc2.text(box_sc2['left']+dx, box_sc2['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # SAVE FIGURE
    plt.savefig('%s/trait_his%s.pdf' % (FIG_DIR,output), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_epitope_theta(**pdata):

    # unpack passed data
    tag      = pdata['tag']
    dir     = pdata['dir']
    HIV_DIR  = pdata['HIV_DIR']
    FIG_DIR  = pdata['FIG_DIR']
    xtick      = pdata['xtick']
    xminortick = pdata['xminortick']
    ytick      = pdata['ytick']
    yminortick = pdata['yminortick']
    theta      = pdata['theta']
    savepdf    = pdata['savepdf']


    # information for escape group
    data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    escape_group = data_pro['escape_group']
    sample_times = data_pro['sample_times']
    times        = data_pro['times']
    time_step    = data_pro['time_step']
    ne           = len(escape_group)

    if ne == 0:
        print(f'CH{tag[6:]} does not contain any trait group')
        return

    # import data with extended time
    data_sc = np.load('%s/%s/c_%s_%d_%s.npz'%(HIV_DIR,dir,tag,time_step,theta), allow_pickle="True")
    sc_all_ex   = data_sc['selection']# time range:times

    time_index = []
    for t in range(len(sample_times)):
        index = list(times).index(sample_times[t])
        time_index.append(index)

    sc_sample    = np.zeros((len(sc_all_ex),len(sample_times)))#time range:sample_times
    sc_sample_ex = np.zeros_like(sc_sample)
    for i in range(len(sample_times)):
        index          = time_index[i]
        sc_sample_ex[:,i] = sc_all_ex[:,index]

    # PLOT FIGURE
    # set up figure grid
    fig   = plt.figure(figsize=(2,1.5*ne),dpi=500)
    gs = gridspec.GridSpec(ne, 1, width_ratios=[1])
    if ne > 1:
        gs.update(left=0.1, right=0.9, bottom=0.10, top=0.95)
    else:
        gs.update(left=0.1, right=0.9, bottom=0.20, top=0.95)
    ax  = [[plt.subplot(gs[n, 0])] for n in range(ne)]

    if len(xtick) == 0:
        xtick = [int(i) for i in sample_times]

    var_ec = np.zeros(ne)
    # escape coefficients without extended time (time range:times)
    pprops = { 'xlim':        [ 0,    max(sample_times)+5],
               'xticks':      xtick,
               'xticklabels': [],
               'ylabel':      'Inferred escape \ncoefficient, ' + r'$\hat{s}$' + ' (%)'}

    for n in range(ne):
        if len(ytick) != ne or len(ytick[0]) == 0:
            ymax = max(max(var_ec[n], max(sc_sample_ex[-(ne-n),:])) * 1.25,0.02)
            ymin = min(min(sc_sample_ex[-(ne-n),:]), -0.02)
            pprops['ylim']        = [ymin, ymax]
            pprops['yticks']      = [round(ymin/0.01)*0.01,  0, round(var_ec[n]*100)/100 ,round(ymax/0.01)*0.01]
            pprops['yticklabels'] = [round(ymin/0.01)     ,  0, round(var_ec[n]*100)     ,round(ymax/0.01)*1]
        else:
            dy   = (ytick[n][-1] - ytick[n][0])*0.01
            ymax = ytick[n][-1] + dy
            ymin = ytick[n][ 0] + dy
            pprops['ylim']        = [ymin, ymax]
            pprops['yticks']      = ytick[n]
            pprops['yticklabels'] = [int(i*100) for i in ytick[n]]
            pprops['yminorticks'] = yminortick[n]

        if n == ne - 1:
            pprops['xticklabels'] = xtick
            pprops['xlabel']      = 'Times (days)'+ theta

        lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
        mp.line(ax=ax[n][0], x=[times], y=[sc_all_ex[-(ne-n),:]], colors=[C_group[n]],plotprops=lprops, **pprops)

        sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
        mp.plot(type='scatter', ax=ax[n][0], x=[sample_times], y=[sc_sample_ex[-(ne-n),:]], colors=[C_group[n]],plotprops=sprops, **pprops)

        ax[n][0].axhline(y=0, ls='--', lw=SIZELINE/2, color=BKCOLOR)
    
    if savepdf:
        plt.savefig('%s/fig-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()
    else:
        plt.savefig('%s/CH%s-%s.jpg' % (FIG_DIR,tag[-5:],theta), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_epitope(**pdata):

    # unpack passed data
    tag      = pdata['tag']
    dir     = pdata['dir']
    HIV_DIR  = pdata['HIV_DIR']
    FIG_DIR  = pdata['FIG_DIR']
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
        print(f'CH{tag[6:]} does not contain any trait group')
        return

    # import data with extended time
    data_sc = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,dir,tag,time_step), allow_pickle="True")
    sc_all_ex   = data_sc['selection']# time range:times

    # # get ExTimes (extended time after interpolation)
    # TLeft = int(round(times[-1]*0.5/10)*10)
    # TRight = int(round(times[-1]*0.5/10)*10)
    # etleft  = np.linspace(-TLeft,-10,int(TLeft/10))
    # etright = np.linspace(times[-1]+10,times[-1]+TRight,int(TRight/10))

    time_index = []
    for t in range(len(sample_times)):
        index = list(times).index(sample_times[t])
        time_index.append(index)

    sc_sample    = np.zeros((len(sc_all_ex),len(sample_times)))#time range:sample_times
    sc_sample_ex = np.zeros_like(sc_sample)
    for i in range(len(sample_times)):
        index          = time_index[i]
        sc_sample_ex[:,i] = sc_all_ex[:,index]

    df_escape   = pd.read_csv('%s/epitopes/escape_group-%s.csv'%(HIV_DIR,tag), memory_map=True)
    epitopes = df_escape['epitope'].unique()

    # var_ec     = [] # escape coefficients for constant case
    traj_var   = [] # frequencies for individual escape sites
    traj_group = [] # frequencies for escape groups
    var_tag    = [] # name for epitope
    for n in range(len(epitopes)):
        df_esc  = df_escape[(df_escape.epitope==epitopes[n])]
        df_row  = df_esc.iloc[0]
        # var_ec.append(df_esc.iloc[0].tc_MPL)

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

    var_ec = np.zeros(ne)

    # PLOT FIGURE
    # set up figure grid
    fig   = plt.figure(figsize=(4,1.5*ne),dpi=500)
    gs = gridspec.GridSpec(ne, 2, width_ratios=[1,1],wspace=0.4,hspace=0.4)
    if ne > 1:
        gs.update(left=0.1, right=0.9, bottom=0.10, top=0.95)
    else:
        gs.update(left=0.1, right=0.9, bottom=0.20, top=0.95)
    ax  = [[plt.subplot(gs[n, 0]), plt.subplot(gs[n, 1])] for n in range(ne)]

    if len(xtick) == 0:
        xtick = [int(i) for i in sample_times]

    # escape group frequencies
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
            pprops['xlabel']      = 'Times (days)'

        # plot frequencies for individual escape sites
        for nn in range(len(traj_var[n])):
            pprops['plotprops']['alpha'] = 0.4
            mp.line(ax=ax[n][0], x=[sample_times], y=[traj_var[n][nn]], colors=[C_group[n]], **pprops)

        # plot frequencies for individual group
        pprops['plotprops']['alpha'] = 1
        mp.plot(type='line', ax=ax[n][0], x=[sample_times], y=[traj_group[n]], colors=[C_group[n]], **pprops)

    # escape coefficients without extended time (time range:times)
    pprops = { 'xlim':        [ 0,    max(sample_times)+5],
               'xticks':      xtick,
               'xticklabels': [],
               'ylabel':      'Inferred escape \ncoefficient, ' + r'$\hat{s}$' + ' (%)'}

    for n in range(ne):
        if len(ytick) != ne or len(ytick[0]) == 0:
            ymax = max(max(var_ec[n], max(sc_sample_ex[-(ne-n),:])) * 1.25,0.02)
            ymin = min(min(sc_sample_ex[-(ne-n),:]), -0.02)
            pprops['ylim']        = [ymin, ymax]
            pprops['yticks']      = [round(ymin/0.01)*0.01,  0, round(var_ec[n]*100)/100 ,round(ymax/0.01)*0.01]
            pprops['yticklabels'] = [round(ymin/0.01)     ,  0, round(var_ec[n]*100)     ,round(ymax/0.01)*1]
        else:
            dy   = (ytick[n][-1] - ytick[n][0])*0.01
            ymax = ytick[n][-1] + dy
            ymin = ytick[n][ 0] + dy
            pprops['ylim']        = [ymin, ymax]
            pprops['yticks']      = ytick[n]
            pprops['yticklabels'] = [int(i*100) for i in ytick[n]]
            pprops['yminorticks'] = yminortick[n]

        if n == ne - 1:
            pprops['xticklabels'] = xtick
            pprops['xlabel']      = 'Times (days)'

        lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
        mp.line(ax=ax[n][1], x=[times], y=[sc_all_ex[-(ne-n),:]], colors=[C_group[n]],plotprops=lprops, **pprops)

        sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
        mp.plot(type='scatter', ax=ax[n][1], x=[sample_times], y=[sc_sample_ex[-(ne-n),:]], colors=[C_group[n]],plotprops=sprops, **pprops)

        # ax[n][1].axhline(y=var_ec[n], ls=':', lw=SIZELINE, color=C_group[n])
        ax[n][1].axhline(y=0, ls='--', lw=SIZELINE/2, color=BKCOLOR)
    
    if savepdf:
        plt.savefig('%s/fig-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()
    else:
        plt.savefig('%s/CH%s.jpg' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)

def plot_special_site(**pdata):

    # unpack passed data
    tag        = pdata['tag']
    dir        = pdata['dir']
    HIV_DIR    = pdata['HIV_DIR']
    FIG_DIR    = pdata['FIG_DIR']
    xtick      = pdata['xtick']
    xminortick = pdata['xminortick']
    ytick      = pdata['ytick_sp']
    yminortick = pdata['yminor_sp']
    savepdf    = pdata['savepdf']

    # information for special sites
    data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    muVec    = data_pro['muVec']

    sample_times = data_pro['sample_times']
    times        = data_pro['times']
    time_step    = data_pro['time_step']
    special_sites    = data_pro['special_sites']
    if len(special_sites) == 0:
        print(f'CH{tag[6:]} has no special site')
        return

    # import data with extended time
    data_sc = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,dir,tag,time_step), allow_pickle="True")
    sc_all_ex   = data_sc['selection']# time range:times
    sc_extended = data_sc['all']      # time range:ExTimes

    # get ExTimes (extended time after interpolation)
    TLeft = int(round(times[-1]*0.5/10)*10)
    TRight = int(round(times[-1]*0.5/10)*10)
    etleft  = np.linspace(-TLeft,-10,int(TLeft/10))
    etright = np.linspace(times[-1]+10,times[-1]+TRight,int(TRight/10))
    ExTimes = np.concatenate((etleft, times, etright))

    df_sc   = pd.read_csv('%s/analysis/%s-analyze.csv'%(HIV_DIR,tag), comment='#', memory_map=True)
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
    sc_a    =  np.zeros((len(index_s),len(times)))    # time varying sc (time range:times)
    sc_a_ex =  np.zeros((len(index_s),len(times)))    # time varying sc (time range:times)
    sc_ex   =  np.zeros((len(index_s),len(ExTimes)))  # time varying sc (time range:ExTimes)
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
            print('error, %d'%(site))
        else:
            sc_a_ex[i] = sc_all_ex[int(index_mu)] - sc_all_ex[int(index_TF)]
            sc_ex[i] = sc_extended[int(index_mu)] - sc_extended[int(index_TF)]

    time_index = []
    for t in range(len(sample_times)):
        index = list(times).index(sample_times[t])
        time_index.append(index)

    sc_s    = np.zeros((len(index_s),len(sample_times)))#time range:sample_times
    sc_s_ex = np.zeros_like(sc_s)
    for i in range(len(sample_times)):
        index        = time_index[i]
        sc_s[:,i]    = sc_a[:,index]
        sc_s_ex[:,i] = sc_a_ex[:,index]

    # PLOT FIGURE
    # set up figure grid
    fig = plt.figure(figsize=(6,2),dpi=500)
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
               'xlabel':      'Times (days)',
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
               'xlabel':      'Times (days)',
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
        mp.line(ax=ax[2], x=[times], y=[sc_a_ex[i]], colors=[C_group[i]],plotprops=lprops, **pprops)

        # sc-time(time range:sample_times)
        if i == len(index_s) - 1:
            sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
            mp.plot(type='scatter', ax=ax[2], x=[sample_times], y=[sc_s_ex[i]], colors=[C_group[i]],plotprops=sprops, **pprops)
        else:
            sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}
            mp.scatter(ax=ax[2], x=[sample_times], y=[sc_s_ex[i]], colors=[C_group[i]],plotprops=sprops, **pprops)

    if savepdf:
        plt.savefig('%s/fig-sp-CH%s.pdf' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
        plt.show()
    else:
        plt.savefig('%s/HIV/sp-CH%s.jpg' % (FIG_DIR,tag[-5:]), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)


@dataclass
class FData:
    ne:0
    sample_times:[]
    traj_var:[]
    traj_group:[]
    sc_sample_ex:[]
    times:[]
    sc_all_ex:[]
    var_ec:[]
    var_tag:[]

def GetFigureData(tag,HIV_DIR,name):
    data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    muVec    = data_pro['muVec']
    escape_group = data_pro['escape_group']
    sample_times = data_pro['sample_times']
    ne           = len(escape_group)
    sample_times = data_pro['sample_times']
    times        = data_pro['times']
    time_step    = data_pro['time_step']

    # import data with extended time
    data_tc     = np.load('%s/output/c_%s_%d%s.npz'%(HIV_DIR,tag,time_step,name), allow_pickle="True")
    sc_all_ex   = data_tc['selection']# time range:times

    df_escape   = pd.read_csv('%s/epitopes/escape_group-%s.csv'%(HIV_DIR,tag), memory_map=True)
    epitopes    = df_escape['epitope'].unique()

    # get ExTimes (extended time after interpolation)
    TLeft = int(round(times[-1]*0.5/10)*10)
    TRight = int(round(times[-1]*0.5/10)*10)
    etleft  = np.linspace(-TLeft,-10,int(TLeft/10))
    etright = np.linspace(times[-1]+10,times[-1]+TRight,int(TRight/10))
    ExTimes = np.concatenate((etleft, times, etright))

    time_index = []
    for t in range(len(sample_times)):
        index = list(times).index(sample_times[t])
        time_index.append(index)

    sc_sample_ex = np.zeros((len(sc_all_ex),len(sample_times)))
    for i in range(len(sample_times)):
        index          = time_index[i]
        sc_sample_ex[:,i] = sc_all_ex[:,index]

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

    return FData(ne,sample_times,traj_var,traj_group,sc_sample_ex,times,sc_all_ex,var_ec,var_tag)

def plot_all_epitopes(**pdata):

    # unpack passed data
    tags      = pdata['tags']
    name     = pdata['name']
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
        data_pro = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
        escape_group = data_pro['escape_group']
        ne           = len(escape_group)
        if ne == 0:
            print(f'CH{tag[6:]} does not contain any trait group')
        else:
            FData = GetFigureData(tag,HIV_DIR,name)
            ne_all.append(ne)
            sample_times_all.append(FData.sample_times)
            traj_group_all.append(FData.traj_group)
            sc_sample_ex_all.append(FData.sc_sample_ex)
            times_all.append(FData.times)
            sc_all_ex_all.append(FData.sc_all_ex)

    # PLOT FIGURE
    # set up figure grid
    fig   = plt.figure(figsize=(6, 3),dpi=1000)

    box_traj1 = dict(left=0.10, right=0.47, bottom=0.63, top=0.95)
    box_traj2 = dict(left=0.10, right=0.47, bottom=0.10, top=0.47)
    box_tc1   = dict(left=0.58, right=0.95, bottom=0.63, top=0.95)
    box_tc2   = dict(left=0.58, right=0.95, bottom=0.10, top=0.47)

    gs_traj1  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_traj1)
    gs_traj2  = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_traj2)
    gs_tc1    = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc1)
    gs_tc2    = gridspec.GridSpec(1, 1, width_ratios=[1.0], height_ratios=[1.0], **box_tc2)

    ax_traj1  = plt.subplot(gs_traj1[0, 0])
    ax_traj2  = plt.subplot(gs_traj2[0, 0])
    ax_tc1    = plt.subplot(gs_tc1[0, 0])
    ax_tc2    = plt.subplot(gs_tc2[0, 0])

    dx = -0.04
    dy =  0.03

    # a, c escape group frequencies short time
    pprops1 = { 'xticks':      [0,100,200],
                'xminorticks': [50,150],
                'xticklabels': [0,100,200],
                'yticks':      [0, 1.0],
                'yminorticks': [0.25, 0.5, 0.75],
                'nudgey':      1.1,
                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.8 },
                'xlabel':      'Times (days)',
                'ylabel':      'Trait frequencies ',
                'axoffset':    0.1,
                'theme':       'open',
                'combine':     True}

    pprops2 = { 'xticks':      [0,100,200,300,400,500,600,700],
                'xminorticks': [50,150,250,350,450,550,650],
                'xticklabels': [0,100,200,300,400,500,600,700],
                'yticks':      [0, 1.0],
                'yminorticks': [0.25, 0.5, 0.75],
                'nudgey':      1.1,
                'plotprops':   {'lw': SIZELINE, 'ls': '-', 'alpha': 0.8 },
                'xlabel':      'Times (days)',
                'ylabel':      'Trait frequencies',
                'axoffset':    0.1,
                'theme':       'open',
                'combine':     True}

    for n in range(len(sample_times_all)):
        traj_group   = traj_group_all[n]
        sample_times = [sample_times_all[n] for k in range(len(traj_group))]
        color_n      = [C_NEU for k in range(len(traj_group))]
        if sample_times_all[n][-1] < 200:
            mp.plot(type='line', ax=ax_traj1, x=sample_times, y=traj_group, colors=color_n, **pprops1)
        else:
            mp.plot(type='line', ax=ax_traj2, x=sample_times, y=traj_group, colors=color_n, **pprops2)

    # ax_traj1.text( box_traj1['left']+dx,  box_traj1['top']+dy, 'a'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # ax_traj2.text( box_traj2['left']+dx,  box_traj2['top']+dy, 'c'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    # b,d escape coefficients with extended time (time range:times)
    pprops1 = { 'xticks':      [0,100,200],
                'xminorticks': [50,150],
                'xticklabels': [0,100,200],
                'ylim':        [0,0.2],
                'yticks':      [0,0.1,0.2],
                'yminorticks': [0.05,0.15],
                'yticklabels': [0,10,20],
                'xlabel':      'Times (days)',
                'ylabel':      'Inferred escape \ncoefficient, ' + r'$\hat{s}$' + ' (%)'}

    pprops2 = { 'xticks':      [0,100,200,300,400,500,600,700],
                'xminorticks': [50,150,250,350,450,550,650],
                'xticklabels': [0,100,200,300,400,500,600,700],
                'ylim':        [-0.02,0.42],
                'yticks':      [0,0.1,0.2,0.3,0.4],
                'yminorticks': [-0.02,0.05,0.15,0.25,0.35],
                'yticklabels': [0,10,20,30,40],
                'xlabel':      'Times (days)',
                'ylabel':      'Inferred escape \ncoefficient, ' + r'$\hat{s}$' + ' (%)'}

    lprops = {'lw': SIZELINE, 'ls': '-', 'alpha': 0.5 }
    sprops = { 'lw' : 0, 's' : 6, 'marker' : 'o','alpha':1}

    for n in range(len(sample_times_all)):
        ne           = ne_all[n]
        sc_all_ex    = sc_all_ex_all[n][-ne:,:]
        sc_sample_ex = sc_sample_ex_all[n][-ne:,:]
        sample_times = [sample_times_all[n] for k in range(len(sc_sample_ex))]
        times        = [times_all[n] for k in range(len(sc_all_ex))]
        color_n      = [C_NEU for k in range(len(sc_sample_ex))]

        if sample_times_all[n][-1] < 200:
            mp.line(ax=ax_tc1, x=times, y=sc_all_ex, colors=color_n,plotprops=lprops, **pprops1)
            mp.scatter(ax=ax_tc1, x=sample_times, y=sc_sample_ex, colors=color_n,plotprops=sprops, **pprops1)
        else:
            mp.line(ax=ax_tc2, x=times, y=sc_all_ex, colors=color_n,plotprops=lprops, **pprops2)
            mp.scatter(ax=ax_tc2, x=sample_times, y=sc_sample_ex, colors=color_n,plotprops=sprops, **pprops2)

    lprops['ls'] = ':'
    lprops['alpha'] = 1.0

    mp.plot(type='line',ax=ax_tc1, x=[[0,200]],y=[[0,0]],colors=[BKCOLOR],plotprops=lprops, **pprops1)
    mp.plot(type='line',ax=ax_tc2, x=[[0,700]],y=[[0,0]],colors=[BKCOLOR],plotprops=lprops, **pprops2)

    # ax_tc1.text( box_tc1['left']+dx,  box_tc1['top']+dy, 'b'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)
    # ax_tc2.text( box_tc2['left']+dx,  box_tc2['top']+dy, 'd'.lower(), transform=fig.transFigure, **DEF_SUBLABELPROPS)

    plt.savefig('%s/fig-tc.pdf' % (FIG_DIR), facecolor = fig.get_facecolor(), edgecolor=None, **FIGPROPS)
