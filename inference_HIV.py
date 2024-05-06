#!/usr/bin/env python
# coding: utf-8

import sys,os
import argparse
import numpy as np
import scipy as sp
import pandas as pd
from scipy import integrate
import scipy.interpolate as sp_interpolate
import statistics
import pickle
from dataclasses import dataclass
import time as time_module
from itertools import product

## nucleotide parameter
NUC = ['-', 'A', 'C', 'G', 'T']
q = len(NUC)

# get information about special sites, escape group and regularization value
@dataclass
class Result:
    variants: 0
    seq_length: 0
    special_sites: []
    uniq_t:[]
    time_step:0
    escape_group:[]
    escape_TF:[]
    trait_dis:[]
    IntTime:[]

def AnalyzeData(tag,HIV_DIR):
    if tag == '704010042-3' or '703010131-3':
        df_info = pd.read_csv('%s/constant/analysis/%s-analyze-cut.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        seq     = np.loadtxt('%s/sequence/%s-cut.dat'%(HIV_DIR,tag))
    else:
        df_info = pd.read_csv('%s/constant/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        seq     = np.loadtxt('%s/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

    """get raw time points"""
    times = []
    for i in range(len(seq)):
        times.append(seq[i][0])
    uniq_t = np.unique(times)

    """get variants number and sequence length"""
    df_poly  = df_info[df_info['nucleotide']!=df_info['TF']]
    variants = len(df_poly)
    seq_length = df_info.iloc[-1].polymorphic_index + 1

    """get special sites and escape sites"""
    # get all epitopes for one tag
    df_rows = df_info[df_info['epitope'].notna()]
    unique_epitopes = df_rows['epitope'].unique()

    min_n = 2 # the least escape sites a trait group should have (more than min_n)
    special_sites = [] # special site considered as time-varying site but not escape site
    escape_group  = [] # escape group (each group should have more than 2 escape sites)
    escape_TF     = [] # corresponding wild type nucleotide
    for epi in unique_epitopes:
        df_e = df_rows[(df_rows['epitope'] == epi) & (df_rows['escape'] == True)] # find all escape mutation for one epitope
        unique_sites = df_e['polymorphic_index'].unique()

        if len(unique_sites) <= min_n:
            special_sites.append(unique_sites)
        else:
            escape_group.append(list(unique_sites))
            escape_TF_epi = []
            for site in unique_sites:
                tf_values = []
                df_site = df_info[df_info['polymorphic_index'] == site]
                for i in range(len(df_site)):
                    if df_site.iloc[i].escape != True:
                        tf_values.append(int(NUC.index(df_site.iloc[i].nucleotide)))
                escape_TF_epi.append(tf_values)
            escape_TF.append(escape_TF_epi)

    special_sites = [item for sublist in special_sites for item in sublist]

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

    """find proper time step"""
    if uniq_t[-1] < 100:
        time_step = 1
    else:
        if seq_length > 300:
            time_step = 20
        elif seq_length < 100:
            time_step = 1
        else:
            time_step = 5

    """find proper gamma value"""
    # interpolation time according to the time step get above, all the inserted time points are integer
    times = [0]
    for t in range(1,len(uniq_t)):
        tp_i   = round((uniq_t[t]-uniq_t[t-1])/time_step) # number of insertion points
        if tp_i > 0:
            ts_i   = round((uniq_t[t]-uniq_t[t-1])/tp_i) # modified time step
            for i in range(tp_i):
                time_i = uniq_t[t-1] + (i+1) * ts_i
                time_s = uniq_t[t]
                # if the inserted time is close to the bounadry time point, throw out the inserted point
                if abs(time_i-time_s) <= ts_i/2:
                    times.append(int(time_s))
                else:
                    times.append(int(time_i))
        else:
            times.append(uniq_t[t])
    IntTime = list(times)

    return Result(variants, seq_length, special_sites, uniq_t, time_step, escape_group, escape_TF, trait_dis, IntTime)

def main(args):
    """Infer time-varying selection coefficients from HIV data"""

    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Time Varying Selection coefficients inference')
    parser.add_argument('-tag',          type=str,    default='700010077-5',        help='input HIV data tag')
    parser.add_argument('-name',         type=str,    default='',                   help='suffix for output data')
    parser.add_argument('-dir',          type=str,    default='data/HIV',           help='directory for HIV data')
    parser.add_argument('-r',            type=float,  default=1.4e-5,               help='recombination rate')
    parser.add_argument('-theta',        type=float,  default=0.5,                  help='the extension of time range')
    parser.add_argument('-g1',           type=float,  default=10,                   help='regularization restricting the magnitude of the selection coefficients for constant MPL')
    parser.add_argument('-g2c',          type=float,  default=100000,               help='regularization restricting the time derivative of the selection coefficients,constant')
    parser.add_argument('-g2tv',         type=float,  default=200,                  help='regularization restricting the time derivative of the selection coefficients,time varying')
    parser.add_argument('--raw',         action='store_true',  default=False,       help='whether or not to save the raw data')
    parser.add_argument('--TV',          action='store_false', default=True,        help='whether or not to infer')
    parser.add_argument('--pt',          action='store_false', default=True,        help='whether or not to print the execution time')

    arg_list  = parser.parse_args(args)

    tag        = arg_list.tag
    name       = arg_list.name
    HIV_DIR    = arg_list.dir
    r_rate     = arg_list.r
    theta      = arg_list.theta
    gamma_1    = arg_list.g1  # regularization parameter, which will be change according to the time points
    gamma_2c   = arg_list.g2c
    gamma_2tv  = arg_list.g2tv
    raw_save   = arg_list.raw
    infer      = arg_list.TV
    print_time = arg_list.pt
    
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
        ex  = np.zeros((len(nVec),ne,seq_length,q))
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
                        ex[t,n,site,qq] += nVec[t][k]
            ex[t,:,:,:] = ex[t,:,:,:] / pop_size_t
        return ex

    def compareElements(k_bp, sVec_n, sWT_n_all, compare_end=False):
        different = False
        for k in range(len(sWT_n_all)):
            sWT_n = sWT_n_all[k]
            if not compare_end: # compare the sequence before k point
                if sVec_n[:k_bp] != sWT_n[:k_bp]:
                    different = True
                    break
            else: # compare the sequence after k point
                if sVec_n[k_bp:] != sWT_n[k_bp:]:
                    different = True
                    break
        return different
    
    # calculate frequencies for recombination part (binary case)  
    def get_p_k(sVec,nVec,seq_length,escape_group,escape_TF):
        p_mut_k   = np.zeros((len(nVec),seq_length,3)) # 0: time, 1: all k point, 2: p_k, p_k-, p_k+
        p_wt      = np.zeros((len(nVec),len(escape_group))) # 0: time, 1: escape group

        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            
            for n in range(len(escape_group)):
                escape_group_n = escape_group[n]

                sWT_n_all = list(product(*escape_TF[n]))
                sWT_n_all = [list(combination) for combination in sWT_n_all]
                
                for k in range(len(sVec[t])): # different sequences at time t
                    sVec_n = [int(sVec[t][k][i]) for i in escape_group_n]
                    
                    # no mutation within the trait group
                    if sVec_n in sWT_n_all:
                        p_wt[t][n] += nVec[t][k]

                    for nn in range(len(escape_group_n)-1):
                        k_bp = nn + 1

                        # compare sequence with all possible WT sequence
                        # if the sequence is different from all WT sequence, result is True
                        head = compareElements(k_bp, sVec_n, sWT_n_all, compare_end=False)
                        tail = compareElements(k_bp, sVec_n, sWT_n_all, compare_end=True)
                        
                        # containing mutation before and after break point k,p_k
                        if head and tail:
                            p_mut_k[t][escape_group_n[0]+nn][0] += nVec[t][k]
                        
                        # MT before break point k and WT after break point k,p_k-
                        if head and not tail:
                            p_mut_k[t][escape_group_n[0]+nn][1] += nVec[t][k]
                        
                        # WT before break point k and MT after break point k,p_k+
                        if not head and tail:
                            p_mut_k[t][escape_group_n[0]+nn][2] += nVec[t][k]

            p_wt[t]    = p_wt[t] / pop_size_t
            p_mut_k[t] = p_mut_k[t] / pop_size_t

        return p_wt,p_mut_k

    # calculate mutation flux term 
    def get_mutation_flux(x,ex,muVec):
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
                                flux[t, x_length-ne+n] += muMatrix[b][a] * (1 - x[t,x_length-ne+n]) - muMatrix[a][b] * ex[t,n,index,a]
        return flux

    # calculate recombination flux term
    def get_recombination_flux(x,p_wt,p_mut_k,trait_dis):
        flux = np.zeros((len(x),x_length))
        for n in range(ne):
            for t in range(len(x)):
                fluxIn  = 0
                fluxOut = 0

                for nn in range(len(escape_group[n])-1):
                    k_index = escape_group[n][0]+nn
                    fluxIn  += trait_dis[n][nn] * p_wt[t][n]*p_mut_k[t][k_index][0]
                    fluxOut += trait_dis[n][nn] * p_mut_k[t][k_index][1]*p_mut_k[t][k_index][2]
                
                flux[t,x_length-ne+n] = r_rate * (fluxIn - fluxOut)

        return flux

    # diffusion matrix C
    def diffusion_matrix_at_t(x,xx):
        x_length = len(x[0])
        C = np.zeros([len(x),x_length,x_length])
        for t in range(len(x)):
            for i in range(x_length):
                C[t,i,i] = x[t,i] - x[t,i] * x[t,i]
                for j in range(int(i+1) ,x_length):
                    C[t,i,j] = xx[t,i,j] - x[t,i] * x[t,j]
                    C[t,j,i] = xx[t,i,j] - x[t,i] * x[t,j]
        return C

    # calculate the difference between the frequency at time t and time t-1
    def cal_delta_x(single_freq,times):
        delta_x = np.zeros((len(single_freq),x_length))   # difference between the frequency at time t and time t-1s
    #     calculate by np.gradient function
    #         for ii in range(x_length):
    #             delta_x[:,ii] = np.gradient(single_freq.T[ii],times)
    #     calculate manually
        for t in range(len(single_freq)-1):
            delta_x[t] = (single_freq[t+1] - single_freq[t])/(times[t+1]-times[t])

        # dt for the last time point, make sure the expected x[t+1] is less than 1
        dt_last = times[-1] - times[-2]
        for ii in range(x_length):
            if single_freq[-1,ii] + delta_x[-1,ii]*dt_last> 1:
                delta_x[-1,ii] = (1 - single_freq[-1,ii])/dt_last
            else:
                delta_x[-1,ii] = delta_x[-2,ii]

        return delta_x

    # Interpolation function definition
    # Interpolates the input arrays so that they will have the same number of generations as the original population.
    interpolation = lambda a,b: sp_interpolate.interp1d(a,b,kind='linear')
    
    def interpolator_x(single_freq, double_freq, current_times, result_times):
        single_freq_temp = np.zeros((len(result_times),x_length))
        double_freq_temp = np.zeros((len(result_times),x_length,x_length))
        
        for i in range(x_length):
            single_freq_temp[:,i] = interpolation(current_times, single_freq[:,i])(result_times)
            for j in range(x_length):
                double_freq_temp[:,i,j] = interpolation(current_times, double_freq[:,i,j])(result_times)
        
        return single_freq_temp, double_freq_temp
    
    def interpolator_ex(escape_freq, current_times, result_times):
        if ne != 0:
            escape_freq_temp  = np.zeros((len(result_times),ne,seq_length,q))
        else: # if there is no escape group, return 0
            escape_freq_temp  = []
        
        for n in range(ne):
            for i in range(seq_length):
                for a in range(q):
                    escape_freq_temp[:,n,i,a] = interpolation(current_times, escape_freq[:,n,i,a])(result_times)
        
        return escape_freq_temp

    def interpolator_p(p_wt, p_mut_k, current_times, result_times):
        wt_temp    = np.zeros((len(result_times),ne))
        mut_k_temp = np.zeros((len(result_times),seq_length,3))
        # interpolation for wild type frequency
        for n in range(ne):
            wt_temp[:,n] = interpolation(current_times, p_wt[:,n])(result_times)

        # interpolation for frequency related to recombination part
        for i in range(seq_length):
            for j in range(3):
                mut_k_temp[:,i,j] = interpolation(current_times, p_mut_k[:,i,j])(result_times)

        wt_temp    = wt_temp[:len(result_times)]
        mut_k_temp = mut_k_temp[:len(result_times)]

        return wt_temp, mut_k_temp

    # functions for determining whether the selection coefficients are constant or not
    # computes the autoconvolution of each coefficient
    autoconvolution = lambda a: np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(a)))

    def mean_deviation(a):
        """Calculates the average absolute deviation of the inferred coefficient around the average."""
        return np.average(np.absolute(np.swapaxes(a,0,1) - np.average(a,axis=1)), axis=0)

    def standard_deviation(a):
        """Computes the standard deviation for each selection coefficient."""
        result = np.zeros(len(a))
        for i in range(len(a)):
            result[i] = statistics.stdev(a[i])
        return result

    def max_min(a):
        """Computes max-min for each selection coefficient."""
        maximum, minimum = np.zeros(len(a)), np.zeros(len(a))
        for i in range(len(a)):
            maximum[i] = max(a[i])
            minimum[i] = min(a[i])
        return (maximum - minimum)

    ################################################################################
    ############################# HIV data process #################################

    if raw_save:
        # obtain raw sequence data
        if tag == '704010042-3' or '703010131-3':
            data     = np.loadtxt('%s/sequence/%s-cut.dat'%(HIV_DIR,tag))
        else:
            data     = np.loadtxt('%s/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

        # information for escape group
        result       = AnalyzeData(tag,HIV_DIR)
        escape_group = result.escape_group
        escape_TF    = result.escape_TF
        trait_dis    = result.trait_dis
        seq_length   = result.seq_length
        sample_times = result.uniq_t
        times        = result.IntTime
        ne           = len(escape_group)

        ## regularization parameter
        p_sites      = result.special_sites
        time_step    = result.time_step

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
        else:
            ex      = 0
        
        # ex: escape frequency, p_wt,p_mut_k: frequency related to recombination part
        p_wt,p_mut_k = get_p_k(sVec,nVec,seq_length,escape_group,escape_TF)

        #record all input information before interpolation
        f = open('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), mode='w+b')
        escape_group = np.array(escape_group, dtype=object)
        escape_TF    = np.array(escape_TF , dtype=object)
        trait_dis    = np.array(trait_dis , dtype=object)
        np.savez_compressed(f, muVec=muVec, single_freq=x, double_freq=xx, escape_freq=ex, p_wt_freq=p_wt, p_mut_k_freq=p_mut_k,\
                            special_sites=p_sites, escape_group=escape_group, escape_TF=escape_TF,trait_dis=trait_dis,\
                            seq_length=seq_length, time_step=time_step, sample_times=sample_times, times=times)
        f.close()

    ################################################################################
    ######################### time varying inference ###############################
    if infer:
        muMatrix = np.loadtxt("%s/Zanini-extended.dat"%HIV_DIR)
        
        # load processed data from rawdata file
        try:
            rawdata  = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
            # information for individual sites
            x            = rawdata['single_freq']
            xx           = rawdata['double_freq']
            ex           = rawdata['escape_freq']
            muVec        = rawdata['muVec']
            sample_times = rawdata['sample_times']
            times        = rawdata['times']
            time_step    = rawdata['time_step']
            seq_length   = rawdata['seq_length']

            # information for escape group
            p_wt         = rawdata['p_wt_freq']
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

        # after interpolation, calculate all the required data
        single_freq, double_freq = interpolator_x(x, xx, sample_times, times)
        escape_freq              = interpolator_ex(ex, sample_times, times)
        p_wt_freq, p_mut_k_freq  = interpolator_p(p_wt, p_mut_k,sample_times, times)

        # covariance matrix, flux term and delta_x
        covariance_n = diffusion_matrix_at_t(single_freq, double_freq)
        covariance   = np.swapaxes(covariance_n, 0, 2)
        flux_mu      = get_mutation_flux(single_freq,escape_freq,muVec)         # mutation part
        flux_rec     = get_recombination_flux(single_freq,p_wt_freq,p_mut_k_freq,trait_dis) # recombination part
        delta_x      = cal_delta_x(single_freq,times)

        # extend the time range
        TLeft   = int(round(times[-1]*theta/10)*10)
        TRight  = int(round(times[-1]*theta/10)*10)
        ex_gap  = int(theta*20)
        etleft  = np.linspace(-TLeft,-ex_gap,int(TLeft/ex_gap))
        etright = np.linspace(times[-1]+ex_gap,times[-1]+TRight,int(TRight/ex_gap))
        ExTimes = np.concatenate((etleft, times, etright))

        # regularization value gamma_1 and gamma_2
        # individual site: gamma_1s, escape group: gamma_1p
        gamma_1s = round(gamma_1/sample_times[-1],3) # constant MPL gamma value / max time
        gamma_1p = gamma_1s/10
        gamma1   = np.ones(x_length)*gamma_1s
        for n in range(ne):
            gamma1[x_length-ne+n] = gamma_1p

        # gamma 2 is also time varying, it is larger at the boundary
        gamma_t = np.ones(len(ExTimes))
        # tv_range = max(int(round(times[-1]*0.1/10)*10),1)
        # alpha1  = np.log(4) / tv_range
        # alpha2  = np.log(4) / tv_range
        # for t in range(len(ExTimes)):
        #     if ExTimes[t] <= 0:
        #         gamma_t[t] = 4
        #     elif ExTimes[t] >= times[-1]:
        #         gamma_t[t] = 4
        #     elif 0 < ExTimes[t] and ExTimes[t] <= tv_range:
        #         gamma_t[t] = 4 * np.exp(-alpha1 * ExTimes[t])
        #     elif times[-1]-tv_range <= ExTimes[t] and ExTimes[t] < times[-1]:
        #         gamma_t[t] = 1 * np.exp(alpha2 * (ExTimes[t]-times[-1]+tv_range))
        #     else:
        #         gamma_t[t] = 1

        # individual site: gamma_2c, escape group and special site: gamma_2tv
        gamma2 = np.ones((x_length,len(ExTimes)))*gamma_2c
        for n in range(ne):
            gamma2[x_length-ne+n] = gamma_t * gamma_2tv
        for p_site in p_sites: # special site - time varying
            for qq in range(len(NUC)):
                index = int (muVec[p_site][qq]) 
                if index != -1:
                    gamma2[index] = gamma_t * gamma_2tv

        start_time = time_module.time()

        # solve the bounadry condition ODE to infer selections
        def fun(a,b):
            """ Function defining the right-hand side of the system of ODE's"""
            b_1                 = b[:x_length,:]   # the actual selection coefficients
            b_2                 = b[x_length:,:]   # the derivatives of the selection coefficients, s'
            result              = np.zeros((2*x_length,len(a))) # The RHS of the system of ODE's
            result[:x_length]   = b_2       # sets the derivatives of the selection coefficients 'b_1', equal to s'
            mat_prod            = np.sum(covariance[:,:,:len(a)] * b_1[:,len(etleft):len(etleft)+len(times)], 1)
            
            for t in range(len(a)): # right hand side of second half of the ODE system
                # within the time range
                if len(etleft) <= t < len(etleft)+len(times):
                    tt = t - len(etleft)
                    for i in range(x_length):
                        result[x_length+i,t] = (mat_prod[i,tt] + gamma1[i] * b_1[i,t] + flux_mu[tt,i] + flux_rec[tt,i] - delta_x[tt,i]) / gamma2[i,t]
                
                # outside the time range, no selection strength
                else:
                    for i in range(x_length):
                        result[x_length+i,t] = gamma1[i] * b_1[i,t] / gamma2[i,t]

            return result
        
        def fun_advanced(a,b):
            """ The function that will be used if it is necessary for the BVP solver to add more nodes.
            Note that the inference may be much slower if this has to be used."""

            b_1                 = b[:x_length,:]   # the actual selection coefficients
            b_2                 = b[x_length:,:]   # the derivatives of the selection coefficients, s'
            result              = np.zeros((2*x_length,len(a))) # The RHS of the system of ODE's
            result[:x_length]   = b_2       # sets the derivatives of the selection coefficients 'b_1', equal to s'

            # create new interpolated single and double site frequencies
            single_freq_int, double_freq_int = interpolator_x(single_freq, double_freq, times, a)
            escape_freq_int                  = interpolator_ex(escape_freq, times, a)
            p_wt_int, p_mut_k_int            = interpolator_p(p_wt_freq,p_mut_k_freq,times,a,seq_length,ne)

            # use the interpolations from above to get the values of delta_x and the covariance matrix at the nodes
            flux_mu_int  = get_mutation_flux(single_freq_int, escape_freq_int, muVec) # mutation part
            flux_rec_int = get_recombination_flux(single_freq_int,p_wt_int, p_mut_k_int, trait_dis)   # recombination part
            delta_x_int  = cal_delta_x(single_freq_int,a)
            covar_int    = diffusion_matrix_at_t(single_freq_int, double_freq_int)
            covar_int    = np.swapaxes(covar_int,0,2)

            # calculate the other half of the RHS of the ODE system
            mat_prod_int  = np.sum(covar_int[:,:,:len(a)] * b_1[:,len(etleft):len(etleft)+len(times)], 1)

            for t in range(len(a)): # right hand side of second half of the ODE system
                # within the time range
                if len(etleft) <= t < len(etleft)+len(times):
                    tt = t - len(etleft)
                    for i in range(x_length):
                        result[x_length+i,t] = (mat_prod_int[i,tt] + gamma1[i] * b_1[i,t] + flux_mu_int[tt,i] + flux_rec_int[tt,i] - delta_x_int[tt,i]) / gamma2[i,t]
                
                # outside the time range, no selection strength
                else:
                    for i in range(x_length):
                        result[x_length+i,t] = gamma1[i] * b_1[i,t] / gamma2[i,t]

            return result

        # Boundary conditions
        # solution to the system of differential equation with the derivative of the selection coefficients zero at the endpoints
        def bc(b1,b2):
            # Neumann boundary condition
            return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints
            # Dirichlet boundary condition
            # return np.ravel(np.array([b1[:x_length],b2[:x_length]])) # s = 0 at the extended endpoints

        ss_extend = np.zeros((2*x_length,len(ExTimes)))
        
        try:
            solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)
        except ValueError:
            print("BVP solver has to add new nodes")
            solution = sp.integrate.solve_bvp(fun_advanced, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)

        selection_coefficients = solution.sol(ExTimes)
        # removes the superfluous part of the array and only save the real time points
        desired_coefficients   = selection_coefficients[:x_length,len(etleft):len(etleft)+len(times)]

        # calculating statistics to be used in determining which coefficients are likely to be time-varying
        mean_dev = mean_deviation(desired_coefficients)
        std_dev  = standard_deviation(desired_coefficients)
        max_var  = max_min(desired_coefficients)
        mean_dev_auto = mean_deviation(autoconvolution(desired_coefficients))
        std_dev_auto  = standard_deviation(autoconvolution(desired_coefficients))
        max_var_auto  = max_min(autoconvolution(desired_coefficients))

        # save the solution with constant_time-varying selection coefficient
        g = open('%s/output-1/c_%s_%d%s.npz'%(HIV_DIR,tag,time_step,name), mode='w+b')
        np.savez_compressed(g, selection=desired_coefficients, all = selection_coefficients, time=times, \
                            mean_dev=mean_dev, std_dev=std_dev, max_var=max_var, mean_dev_auto=mean_dev_auto, \
                            std_dev_auto=std_dev_auto, max_var_auto=max_var_auto)
        g.close()

        end_time = time_module.time()
        if print_time:
            print(f"Execution time for CH{tag[6:]} : {end_time - start_time} seconds")

if __name__ == '__main__':
    main(sys.argv[1:])
