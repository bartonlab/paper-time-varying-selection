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


def main(args):
    """Infer time-varying selection coefficients from HIV data"""

    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Time Varying Selection coefficients inference')
    parser.add_argument('-tag',          type=str,    default='700010058-3',        help='input HIV data tag')
    parser.add_argument('-name',         type=str,    default='',                   help='suffix for output data')
    parser.add_argument('-dir',          type=str,    default='data/HIV',           help='directory for HIV data')
    parser.add_argument('-output',       type=str,    default='output',             help='directory for HIV data')
    parser.add_argument('-beta',         type=float,  default=4.0,                  help='magnification of extended gamma_2 at the ends')
    parser.add_argument('-g1',           type=float,  default=10,                   help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('-g2c',          type=float,  default=100000,               help='regularization restricting the time derivative of the selection coefficients,constant')
    parser.add_argument('-g2tv',         type=float,  default=50,                   help='regularization restricting the time derivative of the selection coefficients,time varying')
    parser.add_argument('-o',            type=float,  default=10,                   help='magnetitude of negative guassian distribution')
    parser.add_argument('--raw',         action='store_true',  default=False,       help='whether or not to save the raw data')
    parser.add_argument('--TV',          action='store_false', default=True,        help='whether or not to infer')
    parser.add_argument('--cr',          action='store_true', default=False,        help='whether or not to use a constant recombination rate')
    parser.add_argument('--pt',          action='store_false', default=True,        help='whether or not to print the execution time')
    
    arg_list  = parser.parse_args(args)

    tag        = arg_list.tag
    name       = arg_list.name
    HIV_DIR    = arg_list.dir
    output_dir = arg_list.output
    beta       = arg_list.beta
    gamma_1    = arg_list.g1     # regularization parameter, which will be change according to the time points
    gamma_2c   = arg_list.g2c
    gamma_2tv  = arg_list.g2tv
    raw_save   = arg_list.raw
    infer_tv   = arg_list.TV
    cr         = arg_list.cr
    print_time = arg_list.pt
    omega      = arg_list.o
    
    ############################################################################
    ################################# function #################################
    # loading data from dat file
    def getSequence(history,escape_TF,escape_group):
        sVec      = []
        nVec      = []
        eVec      = []

        temp_sVec   = []
        temp_nVec   = []
        temp_eVec   = []

        times       = []
        time        = 0
        times.append(time)

        ne          = len(escape_group)

        for t in range(len(history)):
            if history[t][0] != time:
                time = history[t][0]
                times.append(int(time))
                sVec.append(temp_sVec)
                nVec.append(temp_nVec)
                eVec.append(temp_eVec)
                temp_sVec   = []
                temp_nVec   = []
                temp_eVec   = []

            temp_nVec.append(history[t][1])
            temp_sVec.append(history[t][2:])

            if ne > 0: # the patient contains escape group
                temp_escape = np.zeros(ne, dtype=int)
                for n in range(ne):
                    for nn in range(len(escape_group[n])):
                        index = escape_group[n][nn] + 2
                        if history[t][index] not in escape_TF[n][nn]:
                            temp_escape[n] = 1
                            break
                temp_eVec.append(temp_escape)

            if t == len(history)-1:
                sVec.append(temp_sVec)
                nVec.append(temp_nVec)
                eVec.append(temp_eVec)

        return sVec,nVec,eVec

    # get muVec
    def getMutantS(sVec):
        # use muVec matrix to record the index of time-varying sites
        muVec = -np.ones((seq_length, q)) # default value is -1, positive number means the index
        x_length  = 0

        for i in range(seq_length):            
            # find all possible alleles in site i
            alleles     = [int(sVec[t][k][i]) for t in range(len(sVec)) for k in range(len(sVec[t]))]
            allele_uniq = np.unique(alleles)
            for allele in allele_uniq:
                muVec[i][int(allele)] = x_length
                x_length += 1

        return x_length,muVec

    # calculate single and pair allele frequency (multiple case)
    def get_allele_frequency(sVec,nVec,eVec,muVec):

        x  = np.zeros((len(nVec),x_length))           # single allele frequency
        xx = np.zeros((len(nVec),x_length,x_length))  # pair allele frequency
        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            for k in range(len(nVec[t])):
                # individual locus part
                for i in range(seq_length):
                    qq = int(sVec[t][k][i])
                    aa = int(muVec[i][qq])
                    if aa != -1: # if aa = -1, it means the allele does not exist
                        x[t,aa] += nVec[t][k]
                        for j in range(int(i+1), seq_length):
                            qq = int(sVec[t][k][j])
                            bb = int(muVec[j][qq])
                            if bb != -1:
                                xx[t,aa,bb] += nVec[t][k]
                                xx[t,bb,aa] += nVec[t][k]
                # escape part
                for n in range(ne):
                    aa = int(x_length-ne+n)
                    x[t,aa] += eVec[t][k][n] * nVec[t][k]
                    for m in range(int(n+1), ne):
                        bb = int(x_length-ne+m)
                        xx[t,aa,bb] += eVec[t][k][n] * eVec[t][k][m] * nVec[t][k]
                        xx[t,bb,aa] += eVec[t][k][n] * eVec[t][k][m] * nVec[t][k]
                    for j in range(seq_length):
                        qq = int(sVec[t][k][j])
                        bb = int(muVec[j][qq])
                        if bb != -1:
                            xx[t,bb,aa] += eVec[t][k][n] * nVec[t][k]
                            xx[t,aa,bb] += eVec[t][k][n] * nVec[t][k]
            x[t,:]    = x[t,:]/pop_size_t
            xx[t,:,:] = xx[t,:,:]/pop_size_t
        return x,xx

    # calculate escape frequency
    def get_escape_fre_term(sVec,nVec):
        ex  = np.zeros((len(nVec),seq_length,q))
        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            for k in range(len(sVec[t])):
                for n in range(ne):
                    site_mutation = []
                    for nn in escape_group[n]:
                        index = escape_group[n].index(nn)
                        WT = escape_TF[n][index]
                        if sVec[t][k][nn] not in WT:
                            site_mutation.append(nn)
                    if len(site_mutation) == 1:
                        site = site_mutation[0]
                        qq = int(sVec[t][k][site])
                        ex[t,site,qq] += nVec[t][k]
            ex[t,:,:] = ex[t,:,:] / pop_size_t
        return ex

    def compareElements(k_bp, sVec_n, sWT_n_all, compare_end=False):
        same = False
        for k in range(len(sWT_n_all)):
            sWT_n = sWT_n_all[k]
            if not compare_end: # compare the sequence before k point
                if sVec_n[:k_bp] == sWT_n[:k_bp]:
                    same = True
                    break
            else: # compare the sequence after k point
                if sVec_n[k_bp:] == sWT_n[k_bp:]:
                    same = True
                    break
        return same
    
    # calculate frequencies for recombination part (binary case)  
    def get_p_k(sVec,nVec,escape_group,escape_TF):

        ne        = len(escape_group)
        n_k       = np.max([len(escape_group[n]) for n in range(ne)]) - 1

        p_mut_k   = np.zeros((len(nVec),ne, n_k, 3)) # 0: time, 1: all k point, 2: p_k, p_k-, p_k+

        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            
            for n in range(len(escape_group)):
                escape_group_n = escape_group[n]

                sWT_n_all = list(product(*escape_TF[n]))
                sWT_n_all = [list(combination) for combination in sWT_n_all]
                
                for k in range(len(sVec[t])): # different sequences at time t
                    sVec_n = [int(sVec[t][k][i]) for i in escape_group_n]

                    for nn in range(len(escape_group_n)-1):
                        k_bp = nn + 1

                        # compare sequence with all possible WT sequence
                        # if the sequence is different from all WT sequence, result is True
                        head = compareElements(k_bp, sVec_n, sWT_n_all, compare_end=False)
                        tail = compareElements(k_bp, sVec_n, sWT_n_all, compare_end=True)
                        
                        # containing mutation before and after break point k,p_k
                        if not head and not tail:
                            p_mut_k[t][n][nn][0] += nVec[t][k]
                        
                        # MT before break point k and WT after break point k,p_k-
                        if not head and tail:
                            p_mut_k[t][n][nn][1] += nVec[t][k]
                        
                        # WT before break point k and MT after break point k,p_k+
                        if head and not tail:
                            p_mut_k[t][n][nn][2] += nVec[t][k]

            p_mut_k[t] = p_mut_k[t] / pop_size_t

        return p_mut_k

    # calculate recombination flux term at time t
    def get_rec_flux_at_t(r_rates, x_trait, p_mut_k, trait_dis):
        flux = np.zeros(ne)
        for n in range(ne):
            fluxIn  = 0
            fluxOut = 0

            for nn in range(len(escape_group[n])-1):
                fluxIn  += trait_dis[n][nn] * (1 - x_trait[n]) *p_mut_k[n][nn][0]
                fluxOut += trait_dis[n][nn] * p_mut_k[n][nn][1]*p_mut_k[n][nn][2]
            
            flux[n] = r_rates * (fluxIn - fluxOut)

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
                    aa = int(muVec[i][a])
                    if aa != -1:
                        for b in range(q):
                            bb = int(muVec[i][b])
                            if b != a:
                                if bb != -1:
                                    flux[t,aa] +=  muMatrix[b][a] * x[t,bb] - muMatrix[a][b] * x[t,aa]
                                else:
                                    flux[t,aa] += -muMatrix[a][b] * x[t,aa]
            for n in range(ne):
                for nn in range(len(escape_group[n])):
                    for a in range(q):
                        WT = escape_TF[n][nn]
                        index = escape_group[n][nn]
                        if a not in WT:
                            for b in WT:
                                flux[t, x_length-ne+n] += muMatrix[b][a] * (1 - x[t,x_length-ne+n]) - muMatrix[a][b] * ex[t,index,a]
        return flux

    # calculate the frequency change at all times
    def cal_delta_x(single_freq,times):

        delta_x  = np.zeros((len(single_freq),x_length))   # difference between the frequency at time t and time t-1s
        # calculate manually
        for t in range(len(single_freq)-1):
            delta_x[t] = (single_freq[t+1] - single_freq[t])/(times[t+1]-times[t])

        # dt for the last time point, make sure the expected x[t+1] is less than 1
        for ii in range(x_length):
            if single_freq[-1,ii] == 1 and delta_x[-2,ii] > 0:
                delta_x[-1,ii] = 0
            elif single_freq[-1,ii] == 0 and delta_x[-2,ii] < 0:
                delta_x[-1,ii] = 0
            else:
                delta_x[-1,ii] = delta_x[-2,ii]

        return delta_x

    # regularization value gamma_1 and gamma_2
    # gamma_1: time-independent, gamma_2: time-dependent
    def get_gamma1(last_time):
        # individual site: gamma_1s, escape group: gamma_1e
        gamma_1s = round(gamma_1/last_time,3) # constant MPL gamma value / max time
        gamma_1e = gamma_1s/10
        
        gamma1   = np.ones(x_length)*gamma_1s
        for n in range(ne):
            gamma1[x_length-ne+n] = gamma_1e
        
        return gamma1
    
    def get_gamma2(times, beta):
        # Use a time-varying gamma_prime, gamma_2tv is the middle value, 
        # boundary value is 4 times larger, decrese/increase exponentially within 10% generation.
        gamma_t = np.ones(len(times))
        tv_range = max(int(round(times[-1]*0.1/10)*10),1)
        alpha  = np.log(beta) / tv_range
        for ti, t in enumerate(times): # loop over all time points, ti: index, t: time
            if t <= tv_range:
                gamma_t[ti] = beta * np.exp(-alpha * t)
            elif t > times[-1] - tv_range:
                gamma_t[ti] = 1 * np.exp(alpha * (t - times[-1] + tv_range))
        
        gamma2 = np.ones((x_length,len(times)))* gamma_2c
        # special site
        for p_site in p_sites:
            for qq in range(len(NUC)):
                index = int (muVec[p_site][qq]) 
                if index != -1:
                    gamma2[index] = gamma_t * gamma_2tv
        # binary trait
        for n in range(ne):
            gamma2[x_length-ne+n] = gamma_t * gamma_2tv

        # # Use a constant gamma_2tv
        # gamma2 = np.ones(x_length)*gamma_2c
        # for n in range(ne): # binary trait
        #     gamma2[x_length-ne+n] = gamma_2tv
        # for p_site in p_sites: # special site
        #     for qq in range(len(NUC)):
        #         index = int (muVec[p_site][qq]) 
        #         if index != -1:
        #             gamma2[index] = gamma_2tv

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

    ################################################################################
    ############################# HIV data process #################################

    if raw_save:
        # obtain raw sequence data

        data     = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

        # information for escape group
        result       = AnalyzeData(tag,HIV_DIR)
        escape_group = result.escape_group
        escape_TF    = result.escape_TF
        trait_dis    = result.trait_dis
        seq_length   = result.seq_length
        sample_times = result.uniq_t
        r_rates      = result.r_rates
        ne           = len(escape_group)

        ## regularization parameter
        p_sites      = result.special_sites

        # obtain sequence data and frequencies
        sVec,nVec,eVec = getSequence(data,escape_TF,escape_group)
        x_length,muVec = getMutantS(sVec)
        x_length      += ne

        # get all frequencies, 
        # x: single allele frequency, xx: pair allele frequency
        x,xx         = get_allele_frequency(sVec,nVec,eVec,muVec)
        
        # ex: escape frequency
        if ne != 0:
            ex      = get_escape_fre_term(sVec,nVec)
            p_mut_k = get_p_k(sVec,nVec,escape_group,escape_TF)
        else:
            ex      = 0
            p_mut_k = 0

        #record all input information before interpolation
        f = open('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), mode='w+b')
        escape_group = np.array(escape_group, dtype=object)
        escape_TF    = np.array(escape_TF , dtype=object)
        trait_dis    = np.array(trait_dis , dtype=object)
        np.savez_compressed(f, muVec=muVec, single_freq=x, double_freq=xx, escape_freq=ex,\
                            r_rates=r_rates, p_mut_k_freq=p_mut_k,special_sites=p_sites, \
                            escape_group=escape_group, escape_TF=escape_TF,trait_dis=trait_dis,\
                            seq_length=seq_length, sample_times=sample_times)
        f.close()

        # rawdata_tag = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
        # sc_io.savemat('%s/rawdata/rawdata_%s.mat'%(HIV_DIR,tag), rawdata_tag)

    ################################################################################
    ######################### time varying inference ###############################
    
    if not infer_tv:
        sys.exit(0)

    muMatrix = np.loadtxt("%s/input/Zanini-extended.dat"%HIV_DIR)
    
    # load processed data from rawdata file
    try:
        rawdata  = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
        # information for individual sites
        x            = rawdata['single_freq']
        xx           = rawdata['double_freq']
        ex           = rawdata['escape_freq']
        muVec        = rawdata['muVec']
        sample_times = rawdata['sample_times']
        seq_length   = rawdata['seq_length']

        if cr:
            r_rates = np.ones(len(sample_times)) * 1.4e-5
        else:
            r_rates = rawdata['r_rates']

        # information for escape group
        p_mut_k      = rawdata['p_mut_k_freq']
        p_sites      = rawdata['special_sites']
        escape_group = rawdata['escape_group'].tolist()
        escape_TF    = rawdata['escape_TF'].tolist()
        trait_dis    = rawdata['trait_dis'].tolist()

        ne           = len(escape_group)
        x_length     = len(x[0])

    except FileNotFoundError:
        print("error, rawdata file does not exist, please process the data first")
        sys.exit(1)

    # extend the time range
    if sample_times[-1] > 100:
        interp_times = insert_time(sample_times)
    else:
        interp_times = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
    ExTimes  = get_ExTimes(interp_times)
    time_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
    # print(f"\nCH{tag[6:]} Raw sample time: {sample_times}\nExTimes: {ExTimes}")

    # get dx
    delta_x_raw = cal_delta_x(x, sample_times)
    flux_mu_raw = cal_mut_flux(x, ex, muVec)
    
    # get gamma_1 and gamma_2
    gamma_1 = get_gamma1(sample_times[-1])
    gamma_2 = get_gamma2(time_all,beta)

    # Use linear interpolates to get the input arrays at any integer time point
    interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_mut = interp1d(sample_times, p_mut_k, axis=0, kind='linear', bounds_error=False, fill_value=0) if ne > 0 else 0
    interp_dx  = interp1d(sample_times, delta_x_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_mu  = interp1d(sample_times, flux_mu_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
    interp_r   = interp1d(sample_times, r_rates, kind='linear', bounds_error=False, fill_value=0)
    
    single_freq = interp_x(time_all)
    double_freq = interp_xx(time_all)
    flux_mut    = interp_mu(time_all)
    p_mut_k     = interp_mut(time_all) if ne > 0 else 0
    r_rate      = interp_r(time_all) if not cr else np.ones(len(time_all))*1.4e-5
    delta_x     = interp_dx(time_all)
    
    # Get matrix A and vector b
    A_all = np.zeros((len(time_all),x_length,x_length))
    b_all = np.zeros((len(time_all),x_length))

    for ti in range(len(time_all)):
        # calculate A(t) = C(t), add regularization term at ODE part
        A_all[ti] = diffusion_matrix_at_t(single_freq[ti], double_freq[ti]) # covariance matrix
        
        # calculate b(t)
        b_all[ti]   = flux_mut[ti] - delta_x[ti]
        flux_rec = get_rec_flux_at_t(r_rate[ti], single_freq[ti,x_length-ne:], p_mut_k[ti], trait_dis) if ne > 0 else 0
        for n in range(ne): # recombination only for binary trait part
            b_all[ti, x_length-ne+n] += flux_rec[n]

    if print_time:
        start_time = time_module.time()

    # solve the bounadry condition ODE to infer selections
    def fun(time,s):
        """ Function defining the right-hand side of the system of ODE's"""
        # s1                 = s[:x_length,:]   # the actual selection coefficients s1 = s
        # s2                 = s[x_length:,:]   # the derivatives of the selection coefficients, s2 = s'
        dsdt               = np.zeros_like(s)  # the RHS of the system of ODE's

        # s' = s2, s2:the derivatives of the selection coefficients
        dsdt[:x_length, :] = s[x_length:,:]

        # s2'(t) = A(t)s1(t) + b(t), s1: the actual selection coefficients
        for ti, t in enumerate(time): # loop over all time points, ti: index, t: time

            # set value for gamma_1 of traits part
            # high covariance with positive part and low covariance with negative part
            for n in range(ne):
                if s[x_length-ne+n, ti] < 0:
                    gamma_1[x_length-ne+n] = gamma_1[0]*omega
                else:
                    gamma_1[x_length-ne+n] = gamma_1[0]/10

            if t < 0 or t > sample_times[-1]:
                # get gamma2 for extrapolated time points
                if t < 0:
                    gamma2_t = gamma_2[0]
                else:
                    gamma2_t = gamma_2[-1]

                # s'' = gamma1* s(t)/gamma1(t)
                dsdt[x_length:, ti] = gamma_1 * s[:x_length, ti] / gamma2_t
                # for i in range(x_length): 
                #     if i < x_length-ne:# individual sites
                #         dsdt[x_length+i, ti] = gamma_1[i] * s[i, ti] / gamma2_t[i]
                #     else:
                #         dsdt[x_length+i, ti] = (gamma_1[i]/2 + 0.25 / s[i, ti]) / gamma2_t[i]

            else:
                # get A(t), b(t) and gamma2(t)
                time_index = round(t)
                A_t      = A_all[time_index]                
                b_t      = b_all[time_index]
                gamma2_t = gamma_2[time_index]
        
                # s'' = A(t)s(t) + b(t)
                dsdt[x_length:, ti] = ((A_t+np.diag(gamma_1)) @ s[:x_length, ti] + b_t) / gamma2_t
                # for i in range(x_length): 
                #     if i < x_length-ne:# individual sites
                #         # gamma2 s''(t) = C @ s(t) + gamma1 s + b(t)
                #         dsdt[x_length+i, ti] = (A_t[i] @ s[:x_length, ti] + gamma_1[i] * s[i, ti] + b_t[i]) / gamma2_t[i]
                #     else: # binary trait part
                #         # gamma2 s''(t) = C @ s(t) + 1/4s - gamma1/2 + b(t)
                #         dsdt[x_length+i, ti] = (A_t[i] @ s[:x_length, ti] + gamma_1[i]/2 + 0.25 / s[i, ti] + b_t[i]) / gamma2_t[i]
                #         # dsdt[x_length+i, ti] = (A_t[i] @ s[:x_length, ti] + gamma_1[i]/2 + b_t[i]) / gamma2_t[i]
                #         # dsdt[x_length+i, ti] = (A_t[i] @ s[:x_length, ti] + gamma_1[i] * s[i, ti] + b_t[i]) / gamma2_t[i]

        return dsdt

    # Boundary conditions
    # solution to the system of differential equation with the derivative of the selection coefficients zero at the endpoints
    def bc(b1,b2):
        # Neumann boundary condition
        return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints

    # initial guess for the selection coefficients
    ss_extend = np.zeros((2*x_length,len(ExTimes)))
    
    solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=100000, tol=1e-3)

    # Get the solution and remove the superfluous part of the array
    # including the extended time points
    sc_all         = solution.sol(ExTimes)
    desired_sc_all = sc_all[:x_length,:] 
            
    # not include the extended time points
    time_sample       = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
    sc_sample         = solution.sol(time_sample)
    desired_sc_sample = sc_sample[:x_length,:]

    if print_time:
        end_time = time_module.time()
        print(f"CH{tag[6:]}---------{end_time - start_time} seconds")

    # save the solution with constant_time-varying selection coefficient
    name = str(int(omega))+'_'
    g = open('%s/%s/%ssc_%s.npz'%(HIV_DIR, output_dir, name, tag), mode='w+b')
    np.savez_compressed(g, all = desired_sc_all, selection=desired_sc_sample, time=sample_times, ExTimes=ExTimes)
    g.close()
    
if __name__ == '__main__':
    main(sys.argv[1:])
