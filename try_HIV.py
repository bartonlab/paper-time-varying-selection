#!/usr/bin/env python
# coding: utf-8

import sys,os
import argparse
from typing import List
import numpy as np
import pandas as pd
import scipy as sp
from scipy import integrate
from scipy.interpolate import interp1d
import scipy.io as sc_io
from dataclasses import dataclass
import time as time_module
from itertools import product


## nucleotide parameter
NUC = ['-', 'A', 'C', 'G', 'T']
q = len(NUC)

# get information about special sites, escape group and regularization value
@dataclass
class Result:
    seq_length: int
    special_sites: List[int]
    uniq_t: List[int]
    r_rates: List[float]
    escape_group: List[List[int]]
    escape_TF: List[List[int]]
    trait_dis: List[List[int]]

def AnalyzeData(tag,HIV_DIR):

    df_info = pd.read_csv('%s/constant/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    seq     = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

    """get sequence length"""
    seq_length = len(seq[0])-2

    """get raw time points"""
    times = []
    for i in range(len(seq)):
        times.append(seq[i][0])
    uniq_t = np.unique(times)

    '''get recombinant rate'''
    r_rates = np.loadtxt('%s/input/r_rates/r-%s.dat'%(HIV_DIR, tag))
    if len(r_rates) != len(uniq_t):
        print('Error: the length of r_rates is not equal to the length of time')
        sys.exit

    """Get binary sites"""
    escape_group  = [] # escape group (each group should have more than 2 escape sites)

    try:
        df_trait = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        
        # get all binary traits for one tag
        df_rows = df_trait[df_trait['epitope'].notna()]
        unique_traits = df_rows['epitope'].unique()

        for epi in unique_traits:
            # collect all escape sites for one binary trait
            df_e = df_rows[(df_rows['epitope'] == epi)] # find all escape mutation for this epitope
            unique_sites = df_e['polymorphic_index'].unique()
            unique_sites = [int(site) for site in unique_sites]
            escape_group.append(list(unique_sites))

    except FileNotFoundError:
        print(f"CH{tag[-5:]} has no binary trait")
        
    """Get special sites and TF sequence"""
    escape_TF     = [] # corresponding wild type nucleotide
    df_epi = df_info[(df_info['epitope'].notna()) & (df_info['escape'] == True)]
    nonsy_sites = df_epi['polymorphic_index'].unique() # all sites can contribute to epitope

    for n in range(len(escape_group)):
        escape_TF_n = []
        for site in escape_group[n]:
            # remove escape sites to find special sites
            index = np.where(nonsy_sites == site)
            nonsy_sites = np.delete(nonsy_sites, index)

            # find the corresponding TF
            escape_TF_site = []
            df_TF = df_info[(df_info['polymorphic_index'] == site) & (df_info['escape'] == False)]
            for i in range(len(df_TF)):
                TF = df_TF.iloc[i].nucleotide
                escape_TF_site.append(int(NUC.index(TF)))
            escape_TF_n.append(escape_TF_site)
        escape_TF.append(escape_TF_n)
    
    # After removing all escape sites, the rest nonsynonymous sites are special sites
    special_sites = nonsy_sites 

    """trait distance"""
    trait_dis = []
    if len(escape_group) > 0:
        for i in range(len(escape_group)):
            i_dis = []
            for j in range(len(escape_group[i])-1):
                index0 = df_info[df_info['polymorphic_index']==escape_group[i][j]].iloc[0].alignment
                index1 = df_info[df_info['polymorphic_index']==escape_group[i][j+1]].iloc[0].alignment
                i_dis.append(int(index1-index0))
            trait_dis.append(i_dis)

    return Result(seq_length, special_sites, uniq_t, r_rates, escape_group, escape_TF, trait_dis)


def infer_epitope(tag,epitope,gamma_2tv,begin,end,xdotpre):
    """Infer time-varying selection coefficients from HIV data"""

    HIV_DIR    = 'data/HIV'
    
    beta       = 4.0
    gamma_1    = 10.0
    gamma_2c   = 100000.0
    
    ############################################################################
    ################################# function #################################    
    # calculate recombination flux term at time t
    def get_rec_flux_at_t(r_rates, x_trait, p_mut_k, trait_dis):

        fluxIn  = 0
        fluxOut = 0

        for nn in range(len(escape_group[n_index])-1):
            fluxIn  += trait_dis[n_index][nn] * (1 - x_trait) *p_mut_k[nn][0]
            fluxOut += trait_dis[n_index][nn] * p_mut_k[nn][1]*p_mut_k[nn][2]
        
        flux = r_rates * (fluxIn - fluxOut)

        return flux

    # calculate diffusion matrix C at time t
    def diffusion_matrix_at_t(x,xx):
        x_length = len(x)
        C = np.zeros([x_length,x_length])
        for i in range(x_length):
            C[i,i] = x[i] - x[i] * x[i]
            for j in range(int(i+1) ,x_length):
                C[i,j] = xx[i,j] - x[i] * x[j]
                C[j,i] = xx[i,j] - x[i] * x[j]
        return C

    # calculate mutation flux term at sampled time
    def cal_mut_flux(x,ex,muVec):
        flux = np.zeros((len(x),x_length))
        for t in range(len(x)):
            for i in range(seq_length):
                for a in range(q):
                    index = seq_index[i]
                    if muVec[index][a] != -1:
                        aa = np.where(poly_index == int(muVec[index][a]))[0]
                        for b in range(q):
                            if b != a:
                                if muVec[index][b] != -1:
                                    bb = np.where(poly_index == int(muVec[index][b]))[0]
                                    flux[t,aa] +=  muMatrix[b][a] * x[t,bb] - muMatrix[a][b] * x[t,aa]
                                else:
                                    flux[t,aa] += -muMatrix[a][b] * x[t,aa]

            for nn in range(len(escape_group[n_index])):
                for a in range(q):
                    WT = escape_TF[n_index][nn]

                    if a not in WT:
                        for b in WT:
                            flux[t, -1] += muMatrix[b][a] * (1 - x[t,-1]) - muMatrix[a][b] * ex[t,nn,a]

        return flux

    # calculate the frequency change at all times
    def cal_delta_x(single_freq,times):

        delta_x  = np.zeros((len(single_freq),x_length))   # difference between the frequency at time t and time t-1s
        
        # calculate manually
        if xdotpre:
            for t in range(1,len(single_freq)):
                delta_x[t] = (single_freq[t] - single_freq[t-1])/(times[t]-times[t-1])
            delta_x[0] = delta_x[1]

        else:
            for t in range(len(single_freq)-1):
                delta_x[t] = (single_freq[t+1] - single_freq[t])/(times[t+1]-times[t])

            # dt for the last time point, make sure the expected x[t+1] is less than 1
            for ii in range(x_length):
                if single_freq[-1,ii] == 1:
                    delta_x[-1,ii] = 0
                else:
                    delta_x[-1,ii] = delta_x[-2,ii]

        return delta_x

    # regularization value gamma_1 and gamma_2
    # gamma_1: time-independent, gamma_2: time-dependent
    def get_gamma1(last_time):
        # individual site: gamma_1s, escape group: gamma_1p
        gamma_1s = np.round(gamma_1/last_time,3) # constant MPL gamma value / max time
        gamma_1p = gamma_1s/10
        
        gamma1   = np.ones(x_length)*gamma_1s
        for n in range(ne):
            gamma1[x_length-ne+n] = gamma_1p
        
        return gamma1

    def get_gamma2(last_time, beta, begin, end):

        gamma2 = np.ones((x_length,len(ExTimes))) * gamma_2c

        # Use a time-varying gamma_prime, gamma_2tv is the middle value, 
        # boundary value is 4 times larger, decrese/increase exponentially within 10% generation.
        gamma_t = np.ones(len(ExTimes))
        tv_range = max(int(round(last_time*0.1/10)*10),1)
        alpha  = np.log(beta) / tv_range
        for ti, t in enumerate(ExTimes): # loop over all time points, ti: index, t: time
            # vary at the beginning part
            if begin:
                if t <= 0:
                    gamma_t[ti] = beta
                elif 0 < t and t <= tv_range:
                    gamma_t[ti] = beta * np.exp(-alpha * t)
            # vary at the end part
            if end:
                if t >= last_time:
                    gamma_t[ti] = beta
                elif last_time - tv_range < t and t <= last_time:
                    gamma_t[ti] = 1 * np.exp(alpha * (t - last_time + tv_range))
        
        gamma2[-1] = gamma_t * gamma_2tv

        return gamma2.T

    def insert_time(arr, allowed_gaps=(7, 8, 9, 10, 11, 12, 13)):
        """
        Insert values into an array, ensuring the difference between adjacent values 
        is within the allowed_gaps range as evenly distributed as possible.
        """
        result = []

        for i in range(len(arr) - 1):
            result.append(arr[i])  # add current value
            diff = arr[i+1] - arr[i]
            
            if diff < max(allowed_gaps):
                continue

            while diff > max(allowed_gaps):
                # choose the gap that is closest to 10
                if diff % 10 == 0:
                    step = diff/10
                else:
                    step = (diff // 10) + 1
                gap = min(allowed_gaps, key=lambda x: abs(x - diff / step))
                next_value = result[-1] + gap
                result.append(next_value)
                diff = arr[i+1] - next_value  # update the remaining difference
            
            # check if the last gap is in the allowed_gaps
            if diff not in allowed_gaps:
                print(f"Warning: the gap between {result[-1]} and {arr[i+1]} is not in the allowed_gaps range.")
            
        # Add the last value
        if result[-1] != arr[-1]:
            result.append(arr[-1])

        return np.array(result)

    def get_ExTimes(times):
        t_extend = int(round(times[-1]*0.5/10)*10)
        if t_extend <= 10:
            time_step = 5
        elif t_extend <= 30:
            time_step = 10
        elif t_extend <= 100:
            time_step = 20
        elif t_extend <= 300:
            time_step = 50
        else:
            time_step = 100

        etleft  = np.arange(-t_extend, 0, time_step)
        etright = np.arange(times[-1]+time_step,times[-1]+t_extend,time_step)
        if times[-1]+t_extend - etright[-1]  < time_step/2:
            etright[-1] = times[-1]+t_extend
        else:
            etright = np.append(etright, times[-1]+t_extend)
        ExTimes = np.concatenate((etleft, times, etright))
        
        return ExTimes

    def get_cut_index(epitope):
        df_trait = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        # get all binary traits for one tag
        df_rows = df_trait[df_trait['epitope'].notna()]
        unique_traits = df_rows['epitope'].unique()
        
        n_index = np.where(unique_traits == epitope)[0][0]
        poly_sites = escape_group[n_index]

        selected_rows = muVec[poly_sites[0]:poly_sites[-1], :]
        poly_index = selected_rows[selected_rows != -1]

        return poly_sites,poly_index,n_index

    ################################################################################
    ######################### time varying inference ###############################
    # load the data
    muMatrix = np.loadtxt("%s/input/Zanini-extended.dat"%HIV_DIR)
    
    # load processed data from rawdata file
    rawdata  = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
    
    # information 
    muVec        = rawdata['muVec']
    escape_group = rawdata['escape_group'].tolist()
    escape_TF    = rawdata['escape_TF'].tolist()
    trait_dis    = rawdata['trait_dis'].tolist()
    sample_times = rawdata['sample_times']
    seq_index,poly_index,n_index = get_cut_index(epitope)
    r_rates      = rawdata['r_rates']
    
    xx           = rawdata['double_freq']
    new_index = np.append(poly_index,len(xx[0])-len(escape_group)+n_index)
    new_index = [int(i) for i in new_index]
    
    seq_length   = len(seq_index)
    ne           = 1
    x_length     = len(new_index)

    # data for calculation
    x            = rawdata['single_freq'][:,new_index]
    xx           = rawdata['double_freq'][np.ix_(range(len(xx)), new_index, new_index)]
    ex           = rawdata['escape_freq'][:,seq_index,:]
    p_mut_k      = rawdata['p_mut_k_freq'][:,n_index,:,:]
    
    # extend the time range
    interp_times = insert_time(sample_times)
    ExTimes  = get_ExTimes(interp_times)

    # get gamma_1 and gamma_2
    gamma_1 = get_gamma1(sample_times[-1])
    gamma_2 = get_gamma2(sample_times[-1],beta, begin, end)

    # get dx
    delta_x_raw = cal_delta_x(x, sample_times)
    flux_mu_raw = cal_mut_flux(x, ex, muVec)
    
    # Use linear interpolates to get the input arrays at any given time point
    interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_mut = interp1d(sample_times, p_mut_k, axis=0, kind='linear', bounds_error=False, fill_value=0) if ne > 0 else 0
    interp_dx  = interp1d(sample_times, delta_x_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_mu  = interp1d(sample_times, flux_mu_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_r   = interp1d(sample_times, r_rates, kind='linear', bounds_error=False, fill_value=0)
    interp_g2  = interp1d(ExTimes, gamma_2, axis=0, kind='linear', bounds_error=False, fill_value=0)

    # solve the bounadry condition ODE to infer selections
    def fun(time,s):
        """ Function defining the right-hand side of the system of ODE's"""
        # s1                 = s[:x_length,:]   # the actual selection coefficients s1 = s
        # s2                 = s[x_length:,:]   # the derivatives of the selection coefficients, s2 = s'
        dsdt               = np.zeros_like(s)  # the RHS of the system of ODE's

        # s' = s2, s2:the derivatives of the selection coefficients
        dsdt[:x_length, :] = s[x_length:,:]

        single_freq = interp_x(time)
        double_freq = interp_xx(time)
        flux_mut    = interp_mu(time)
        p_mut_k     = interp_mut(time)
        r_rate      = interp_r(time)
        delta_x     = interp_dx(time)
        gamma2      = interp_g2(time)

        # s2'(t) = A(t)s1(t) + b(t), s1: the actual selection coefficients
        for ti, t in enumerate(time): # loop over all time points, ti: index, t: time

            if t < 0 or t > sample_times[-1]:
                # outside the range, only gamma
                # s'' = gamma1* s(t)/gamma1(t)
                dsdt[x_length:, ti] = gamma_1 * s[:x_length, ti] / gamma2[ti]

            else:
                # calculate the frequency at time t
                C_t = diffusion_matrix_at_t(single_freq[ti], double_freq[ti]) # covariance matrix
                flux_rec = get_rec_flux_at_t(r_rate[ti], single_freq[ti,x_length-ne:], p_mut_k[ti], trait_dis) if ne > 0 else 0
                # calculate A(t) = C(t) + gamma_1 * I
                A_t = C_t + np.diag(gamma_1) # add gamma_1 to the diagonal
                
                # calculate b(t)
                b_t      = flux_mut[ti] - delta_x[ti]
                for n in range(ne): # recombination only for binary trait part
                    b_t[x_length-ne+n] += flux_rec[n]
        
                # s'' = A(t)s(t) + b(t)
                dsdt[x_length:, ti] = (A_t @ s[:x_length, ti] + b_t) / gamma2[ti]

        return dsdt

    # Boundary conditions
    # solution to the system of differential equation with the derivative of the selection coefficients zero at the endpoints
    def bc(b1,b2):
        # Neumann boundary condition
        return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints

    # initial guess for the selection coefficients
    ss_extend = np.zeros((2*x_length,len(ExTimes)))
    
    solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=100000, tol=1e-3)

    # Get the solution for sample times
    # removes the superfluous part of the array and only save the sampled times points
    # not include the extended time points
    time_sample       = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
    sc_sample         = solution.sol(time_sample)
    desired_sc_sample = sc_sample[:x_length,:]

    return x,desired_sc_sample,sample_times
