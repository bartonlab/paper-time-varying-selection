#!/usr/bin/env python
# coding: utf-8

import sys,os
import argparse
import numpy as np
from scipy.interpolate import interp1d

## nucleotide parameter
NUC = ['-', 'A', 'C', 'G', 'T']
q = len(NUC)
CALLS = 0

def main(args):
    """Infer time-varying selection coefficients from HIV data"""

    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Time Varying Selection coefficients inference')
    parser.add_argument('-tag',          type=str,    default='700010058-3',        help='input HIV data tag')
    parser.add_argument('-dir',          type=str,    default='data/HIV',           help='directory for HIV data')
    parser.add_argument('--add',         action='store_true', default=False,        help='whether or not to add time to the input data')

    arg_list  = parser.parse_args(args)

    tag        = arg_list.tag
    HIV_DIR    = arg_list.dir
    add_time   = arg_list.add

    ############################################################################
    ################################# function #################################
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

    # calculate dxdt
    def cal_dx_all(x, sample_times):
        # mid point method to calculate dxdt, get the velocity at the mid points
        dt = np.diff(sample_times)[:, None]
        v = np.diff(x, axis=0) / dt
        # get the dxdt at the original time points by averaging the neighboring values
        v_node = np.empty_like(x)
        v_node[0] = v[0]
        v_node[-1] = v[-1]
        v_node[1:-1] = 0.5 * (v[:-1] + v[1:])

        return v, v_node

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

        etleft  = np.arange(times[0]-t_extend, times[0], time_step)
        etright = np.arange(times[-1]+time_step,times[-1]+t_extend,time_step)
        if times[-1]+t_extend - etright[-1]  < time_step/2:
            etright[-1] = times[-1]+t_extend
        else:
            etright = np.append(etright, times[-1]+t_extend)
        ExTimes = np.concatenate((etleft, times, etright))
        
        return ExTimes

    ################################################################################
    ######################### time varying inference ###############################

    muMatrix = np.loadtxt("%s/input/Zanini-extended.dat"%HIV_DIR)
    sc_const = np.loadtxt("%s/constant/output/sc-%s.dat"%(HIV_DIR,tag))

    if add_time:
        tag_name = tag + '-add'
    else:
        tag_name = tag

    # load processed data from rawdata file
    try:
        rawdata  = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag_name), allow_pickle=True)
        
        # information for individual sites
        x_raw        = rawdata['single_freq']
        xx_raw       = rawdata['double_freq']
        ex_raw       = rawdata['escape_freq']
        muVec        = rawdata['muVec']
        sample_times = rawdata['sample_times']
        seq_length   = rawdata['seq_length']
        r_rates      = rawdata['r_rates']

        # information for escape group
        p_mut_k_raw  = rawdata['p_mut_k_freq']
        p_sites      = rawdata['special_sites']
        escape_group = rawdata['escape_group'].tolist()
        escape_TF    = rawdata['escape_TF'].tolist()
        trait_dis    = rawdata['trait_dis'].tolist()

        ne           = len(escape_group)
        x_length     = len(x_raw[0])

    except FileNotFoundError:
        print("error, rawdata file does not exist, please process the data first")
        sys.exit(1)
    
    # get index for special sites
    tv_index = []
    for p_site in p_sites:
        for qq in range(len(NUC)):
            index = int (muVec[p_site][qq]) 
            if index != -1:
                tv_index.append(index)

    # get mutation flux and dxdt
    flux_mu_raw = cal_mut_flux(x_raw, ex_raw, muVec)
    dxdt, dxdt_node = cal_dx_all(x_raw, sample_times)

    # extend the time range
    if sample_times[-1] > 100:
        interp_times = insert_time(sample_times)
    else:
        interp_times = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
    ExTimes  = get_ExTimes(interp_times)
    time_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))

    # Check if mutant epitope fixed and find the fixation time
    fixation_time = np.ones(ne) * (sample_times[-1] + 100) # set a default fixation later than the last sample time
    for n in range(ne):
        x_epitope = x_raw.T[-ne+n]

        # check if the mutant epitope is fixed
        if np.max(x_epitope[:-1]) == 1 and x_epitope[-1] == 1:
            # Find the fixation time
            for ti in range(len(x_epitope)-1):
                if x_epitope[ti] == 1 and x_epitope[ti+1] == 1:
                    fixation_time[n] = sample_times[ti]
                    break

    # # Use linear interpolates to get the input arrays at any integer time point
    interp_x   = interp1d(sample_times, x_raw, axis=0, kind='linear', bounds_error=False, fill_value=(x_raw[0], x_raw[-1]))
    interp_xx  = interp1d(sample_times, xx_raw, axis=0, kind='linear', bounds_error=False, fill_value=(xx_raw[0], xx_raw[-1]))
    interp_mut = interp1d(sample_times, p_mut_k_raw, axis=0, kind='linear', bounds_error=False, fill_value=(p_mut_k_raw[0], p_mut_k_raw[-1])) if ne > 0 else 0
    interp_mu  = interp1d(sample_times, flux_mu_raw, axis=0, kind='linear', bounds_error=False, fill_value=(flux_mu_raw[0], flux_mu_raw[-1]))
    interp_r   = interp1d(sample_times, r_rates, kind='linear', bounds_error=False, fill_value=(r_rates[0], r_rates[-1]))
    
    x_all = interp_x(time_all)
    xx_all = interp_xx(time_all)
    flux_mu_all    = interp_mu(time_all)
    p_mut_k_all    = interp_mut(time_all) if ne > 0 else 0
    r_rates_all    = interp_r(time_all)
    
    # Get matrix C and vector b
    C_all = np.zeros((len(time_all),x_length,x_length))
    flux_all = np.zeros((len(time_all),x_length))

    for ti in range(len(time_all)):
        # calculate covariance matrix C(t), add regularization term at ODE part
        C_all[ti] = diffusion_matrix_at_t(x_all[ti], xx_all[ti]) # covariance matrix
        
        # calculate b(t)
        flux_all[ti] = flux_mu_all[ti]
        if ne > 0:
            flux_r_t = get_rec_flux_at_t(r_rates_all[ti], x_all[ti,x_length-ne:], p_mut_k_all[ti], trait_dis) if ne > 0 else 0
            flux_all[ti, x_length-ne:] += flux_r_t

    sc_initial = np.zeros(x_length)
    for i in range(seq_length):
        for a in range(q):
            aa = int(muVec[i][a]) # aa is the index for time-varying, i*q+a is the index for constant
            if aa != -1:
                sc_initial[aa] = sc_const[i*q+a] 
    for n in range(ne):
        sc_initial[x_length-ne+n] = sc_const[seq_length*q+n]

    'Save the interpolated data (C and b) to npz file.'
    f = open('%s/rawdata/interdata_%s.npz'%(HIV_DIR,tag_name), mode='w+b')
    np.savez_compressed(f, C_all=C_all, flux_all=flux_all, dxdt=dxdt, dxdt_node = dxdt_node,
                        fixation_time=fixation_time, sample_times=sample_times, ExTimes=ExTimes, 
                        tv_index=tv_index, sc_initial=sc_initial)
    f.close()
    
if __name__ == '__main__':
    main(sys.argv[1:])
