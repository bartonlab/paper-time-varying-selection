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
import math
from dataclasses import dataclass
import time as time_module

# GitHub
HIV_DIR = '../data/HIV200'
SIM_DIR = '../data/simulation'
FIG_DIR = 'figures'

## simulation parameter
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
    gamma:0
    escape_group:[]
    escape_TF:[]
    IntTime:[]

def AnalyzeData(tag):
    df_info = pd.read_csv('%s/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    seq     = np.loadtxt('%s/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))

    """get raw time points"""
    times = [];
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
            tf_values = []
            for site in unique_sites:
                tf_value = df_e[df_e['polymorphic_index'] == site]['TF'].values
                tf_values.append(NUC.index(tf_value[0]))
            escape_TF.append(tf_values)

    special_sites = [item for sublist in special_sites for item in sublist]

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
    time_points = math.ceil(uniq_t[-1]/time_step)+1
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
    gamma = round(10/uniq_t[-1],3)

    return Result(variants,seq_length,special_sites,uniq_t,time_step,gamma, escape_group,escape_TF,IntTime)


def main(args):
    """Infer time-varying selection coefficients from HIV data"""

    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Time Varying Selection coefficients inference')
    parser.add_argument('-tag',           type=str,    default='700010077-5',        help='input HIV data tag')
    parser.add_argument('-name',          type=str,    default='',                   help='suffix for output data')
    parser.add_argument('--raw',          action='store_true', default=False,        help='whether or not to save the raw data')
    parser.add_argument('--throw',        action='store_true', default=False,        help='whether or not to throw out some weak linkage variants')
    parser.add_argument('--TV',           action='store_false', default=True,        help='whether or not to infer without extend')
    parser.add_argument('--TV_extend',    action='store_false', default=True,        help='whether or not to infer with extend')

    arg_list  = parser.parse_args(args)

    tag            = arg_list.tag
    name           = arg_list.name
    raw_save       = arg_list.raw
    throw          = arg_list.throw
    InferTV        = arg_list.TV
    InferTV_extend = arg_list.TV_extend

    ############################################################################
    ################################# function #################################

    # loading data from dat file
    def getSequence(history,escape_TF,escape_group):
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
                        if history[t][index] != escape_TF[n][nn]:
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
        # use muVec matrix to record the index of time-varying sites(after throwing out weak linkage sites)
        muVec = -np.ones((seq_length, q)) # default value is -1, positive number means the index
        x_length  = 0
        for i in range(seq_length):
            alleles = []
            for t in range(len(sVec)):
                for k in range(len(sVec[t])):
                    alleles.append(int(sVec[t][k][i]))
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
                # individual part
                for i in range(seq_length):
                    qq = int(sVec[t][k][i])
                    aa = int(muVec[i][qq])
                    if aa != -1:
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
                    n_mutations = 0
                    for nn in escape_group[n]:
                        index = escape_group[n].index(nn)
                        WT = escape_TF[n][index]
                        if sVec[t][k][nn] != WT:
                            n_mutations += 1
                            site = nn
                    if n_mutations == 1:
                        qq = int(sVec[t][k][site])
                        ex[t,n,site,qq] += nVec[t][k]
            ex[t,:,:,:] = ex[t,:,:,:] / pop_size_t
        return ex

    # diffusion matrix C
    def diffusion_matrix_at_t(x,xx):
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
        delta_x[-1] = delta_x[-2]
        return delta_x


    # Interpolation function definition
    interpolation = lambda a,b: sp_interpolate.interp1d(a,b,kind='linear')
    def interpolator(single_freq, double_freq, escape_freq, current_times, result_times):
        """ Interpolates the input arrays so that they will have the same number of generations as the original population. """

        single_freq_temp = np.zeros((len(result_times),x_length))
        double_freq_temp = np.zeros((len(result_times),x_length,x_length))
        if ne != 0:
            escape_freq_temp  = np.zeros((len(result_times),ne,seq_length,q))
        else:
            escape_freq_temp  = []
        for i in range(x_length):
            single_freq_temp[:,i] = interpolation(current_times, single_freq[:,i])(result_times)
            for j in range(x_length):
                double_freq_temp[:,i,j] = interpolation(current_times, double_freq[:,i,j])(result_times)
        for n in range(ne):
            for i in range(seq_length):
                for a in range(q):
                    escape_freq_temp[:,n,i,a] = interpolation(current_times, escape_freq[:,n,i,a])(result_times)
        return single_freq_temp, double_freq_temp, escape_freq_temp

    # flux term with escape term
    def get_flux(x,ex,muVec):
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
                        if a != WT:
                            flux[t, x_length-ne+n] += muMatrix[WT][a] * (1 - x[t,x_length-ne+n]) - muMatrix[a][WT] * ex[t,n,index,a]
        return flux

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
        data     = np.loadtxt("%s/sequence/%s-poly-seq2state.dat" %(HIV_DIR,tag))
        muMatrix = np.loadtxt("%s/Zanini-extended.dat"%HIV_DIR)

        # information for escape group
        result = AnalyzeData(tag)
        escape_group = result.escape_group
        escape_TF    = result.escape_TF
        seq_length   = result.seq_length
        sample_times = result.uniq_t
        times        = result.IntTime
        ne           = len(escape_group)

        ## regularization parameter
        p_sites        = result.special_sites
        time_step      = result.time_step
        gamma_1s       = result.gamma

        sVec      = []
        nVec      = []
        eVec      = []
        sVec,nVec,eVec = getSequence(data,escape_TF,escape_group)

        x_length,muVec = getMutantS(sVec)
        x_length   += ne

        x,xx         = get_allele_frequency(sVec,nVec,eVec,muVec)

        if ne != 0:
            ex      = get_escape_fre_term(sVec,nVec)
        else:
            ex      = 0

        #record all input information before interpolation
        f = open('%s/rawdata/rawdata_new_%s.npz'%(HIV_DIR,tag), mode='w+b')
        escape_group = np.array(escape_group, dtype=object)
        escape_TF  = np.array(escape_TF , dtype=object)
        np.savez_compressed(f, single_freq=x, double_freq=xx, escape_freq=ex,\
                            muVec=muVec,special_sites=p_sites,seq_length=seq_length,\
                            escape_group=escape_group, escape_TF=escape_TF,\
                            gamma = gamma_1s, time_step=time_step, sample_times=sample_times, times=times)
        f.close()

    ################################################################################
    ########################### HIV data inference #################################

    muMatrix = np.loadtxt("%s/Zanini-extended.dat"%HIV_DIR)
    rawdata  = np.load('%s/rawdata/rawdata_new_%s.npz'%(HIV_DIR,tag), allow_pickle=True)

    # information for individual sites
    x            = rawdata['single_freq']
    xx           = rawdata['double_freq']
    ex           = rawdata['escape_freq']
    muVec        = rawdata['muVec']
    # inde_site    = rawdata['inde_site']
    sample_times = rawdata['sample_times']
    times        = rawdata['times']
    seq_length   = rawdata['seq_length']

    # information for escape group
    escape_group = rawdata['escape_group'].tolist()
    escape_TF    = rawdata['escape_TF'].tolist()

    ne           = len(escape_group)
    x_length     = len(x[0])

    ## regularization parameter
    p_sites      = rawdata['special_sites']
    time_step    = rawdata['time_step']
    gamma_1s     = rawdata['gamma']
    gamma_1p     = gamma_1s/10
    gamma_2c     = 100000
    gamma_2tv    = 200

    ################################################################################
    ######################### time varying inference ###############################

#     after interpolation, calculate all the required data
    single_freq, double_freq, escape_freq = interpolator(x,xx,ex, sample_times, times)
    covariance_n = diffusion_matrix_at_t(single_freq, double_freq)
    covariance   = np.swapaxes(covariance_n, 0, 2)
    flux         = get_flux(single_freq,escape_freq,muVec)
    delta_x      = cal_delta_x(single_freq,times)

    start_time = time_module.time()

    TLeft = int(round(times[-1]*0.5/10)*10)
    TRight = int(round(times[-1]*0.5/10)*10)

    etleft  = np.linspace(-TLeft,-10,int(TLeft/10))
    etright = np.linspace(times[-1]+10,times[-1]+TRight,int(TRight/10))
    ExTimes = np.concatenate((etleft, times, etright))

    def fun(a,b):
        """ Function defining the right-hand side of the system of ODE's"""
        b_1                 = b[:x_length,:]   # the actual selection coefficients
        b_2                 = b[x_length:,:]   # the derivatives of the selection coefficients, s'
        result              = np.zeros((2*x_length,len(a))) # The RHS of the system of ODE's
        result[:x_length]   = b_2       # sets the derivatives of the selection coefficients 'b_1', equal to s'
        mat_prod            = np.sum(covariance[:,:,:len(a)] * b_1[:,len(etleft):len(etleft)+len(times)], 1)
        for t in range(len(a)): # right hand side of second half of the ODE system
            if len(etleft) <= t < len(etleft)+len(times):
                tt = t - len(etleft)
                result[x_length:2*x_length-ne,t] = (mat_prod[:x_length-ne,tt] + gamma_1s * b_1[:x_length-ne,t] + flux[tt,:x_length-ne] - delta_x[tt,:x_length-ne]) / gamma_2c
                if ne != 0:
                    result[2*x_length-ne:,t]     = (mat_prod[x_length-ne:,tt] + gamma_1p * b_1[x_length-ne:,t] + flux[tt,x_length-ne:] - delta_x[tt,x_length-ne:]) / gamma_2tv

                for i in range(len(p_sites)):#consider the individual site as time-varying site, use gamma and gamma_' for time varying part
                    for qq in range(q):
                        index = int (muVec[p_sites[i],qq]) # convert the index from all sequences to modified sequences
                        if index != -1:
                            result[x_length + index,t] = (mat_prod[index,tt] + gamma_1s * b_1[index,t] + flux[tt,index] - delta_x[tt,index]) / gamma_2tv

            else:
                result[x_length:2*x_length-ne,t] = gamma_1s * b_1[:x_length-ne,t] / gamma_2c
                if ne != 0:
                    result[2*x_length-ne:,t]     = gamma_1p * b_1[x_length-ne:,t] / gamma_2tv

        return result

    def fun_advanced(a,b):
        """ The function that will be used if it is necessary for the BVP solver to add more nodes.
        Note that the inference may be much slower if this has to be used."""

        b_1                 = b[:x_length,:]   # the actual selection coefficients
        b_2                 = b[x_length:,:]   # the derivatives of the selection coefficients, s'
        result              = np.zeros((2*x_length,len(a))) # The RHS of the system of ODE's
        result[:x_length]   = b_2       # sets the derivatives of the selection coefficients 'b_1', equal to s'

        # create a covariance matrix and delta_x arrays with the proper amount of nodes
        covar_int   = np.zeros((x_length,x_length,len(a)))
        delta_x_int = np.zeros((len(a),x_length))
        flux_int    = np.zeros((len(a),x_length))

        # create new interpolated single and double site frequencies
        single_freq_int, double_freq_int, escape_freq_int = interpolator(single_freq, double_freq, escape_freq, times, a)

        # use the interpolations from above to get the values of delta_x and the covariance matrix at the nodes
        flux_int    = get_flux(single_freq_int, escape_freq_int,muVec)
        delta_x_int = cal_delta_x(single_freq_int,a)
        covar_int   = diffusion_matrix_at_t(single_freq_int, double_freq_int)
        covar_int   = np.swapaxes(covar_int,0,2)

        # calculate the other half of the RHS of the ODE system
        mat_prod_int  = np.sum(covar_int[:,:,:len(a)] * b_1[:,len(etleft):len(etleft)+len(times)], 1)

        for t in range(len(a)): # right hand side of second half of the ODE system
            if len(etleft) <= t < len(etleft)+len(times):
                tt = t - len(etleft)
                result[x_length:2*x_length-ne,t] = (mat_prod_int[:x_length-ne,tt] + gamma_1s * b_1[:x_length-ne,t] + flux_int[tt,:x_length-ne] - delta_x_int[tt,:x_length-ne]) / gamma_2c
                if ne != 0:
                    result[2*x_length-ne:,t]     = (mat_prod_int[x_length-ne:,tt] + gamma_1p * b_1[x_length-ne:,t] + flux_int[tt,x_length-ne:] - delta_x_int[tt,x_length-ne:]) / gamma_2tv

                for i in range(len(p_sites)):#consider the individual site as time-varying site, use gamma and gamma_' for time varying part
                    for qq in range(q):
                        index = int (muVec[p_sites[i],qq]) # convert the index from all sequences to modified sequences
                        if index != -1:
                            result[x_length + index,t] = (mat_prod_int[index,tt] + gamma_1s * b_1[index,t] + flux_int[tt,index] - delta_x_int[tt,index]) / gamma_2tv

            else:
                result[x_length:2*x_length-ne,t] = gamma_1s * b_1[:x_length-ne,t] / gamma_2c
                if ne != 0:
                    result[2*x_length-ne:,t]     = gamma_1p * b_1[x_length-ne:,t] / gamma_2tv

        return result

    # Boundary conditions
    # solution to the system of differential equation with the derivative of the selection coefficients zero at the endpoints

    def bc(b1,b2):
        return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints

    ss = np.zeros((2*x_length,len(ExTimes)))
    try:
        solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss, max_nodes=10000, tol=1e-3)
    except ValueError:
        print("BVP solver has to add new nodes")
        solution = sp.integrate.solve_bvp(fun_advanced, bc, ExTimes, ss, max_nodes=10000, tol=1e-3)

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
    g = open('%s/output/c_%s_%d%s.npz'%(HIV_DIR,tag,time_step,name), mode='w+b')
    np.savez_compressed(g, selection=desired_coefficients, all = selection_coefficients, time=times, \
                        mean_dev=mean_dev, std_dev=std_dev, max_var=max_var, mean_dev_auto=mean_dev_auto, \
                        std_dev_auto=std_dev_auto, max_var_auto=max_var_auto)
    g.close()

    end_time = time_module.time()
    print(f"Execution time for shorter time for CH{tag[6:]} : {end_time - start_time} seconds")

if __name__ == '__main__':
    main(sys.argv[1:])
