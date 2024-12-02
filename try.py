#!/usr/bin/env python
# coding: utf-8

import sys,os
import argparse
import numpy as np
import scipy as sp
from scipy import integrate
from scipy.interpolate import interp1d
from dataclasses import dataclass
import time as time_module

## nucleotide parameter
NUC = ['-', 'A', 'C', 'G', 'T']
q = len(NUC)
eval_count = 0

def main(args):
    """Infer time-varying selection coefficients from HIV data"""

    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Time Varying Selection coefficients inference')
    parser.add_argument('-tag',          type=str,    default='700010077-5',        help='input HIV data tag')
    parser.add_argument('-name',         type=str,    default='',                   help='suffix for output data')
    parser.add_argument('-dir',          type=str,    default='data/HIV',           help='directory for HIV data')
    parser.add_argument('-output',       type=str,    default='output',             help='directory for HIV data')
    parser.add_argument('-beta',         type=float,  default=4.0,                  help='magnification of extended gamma_2 at the ends')
    parser.add_argument('-g1',           type=float,  default=10,                   help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('-g2c',          type=float,  default=100000,               help='regularization restricting the time derivative of the selection coefficients,constant')
    parser.add_argument('-g2tv',         type=float,  default=50,                   help='regularization restricting the time derivative of the selection coefficients,time varying')
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

    # calculate the frequency change at all times
    def cal_delta_x(single_freq,times):

        delta_x  = np.zeros((len(single_freq),x_length))   # difference between the frequency at time t and time t-1s
        # calculate by np.gradient function
        # for ii in range(x_length):
        #     delta_x[:,ii] = np.gradient(single_freq.T[ii],times)

        # calculate manually
        for t in range(len(single_freq)-1):
            delta_x[t] = (single_freq[t+1] - single_freq[t])/(times[t+1]-times[t])

        # dt for the last time point, make sure the expected x[t+1] is less than 1
        dt_last = times[-1] - times[-2]
        for ii in range(x_length):
            if single_freq[-1,ii] + delta_x[-2,ii]*dt_last> 1:
                delta_x[-1,ii] = (1 - single_freq[-1,ii])/dt_last
            else:
                delta_x[-1,ii] = delta_x[-2,ii]

        return delta_x

    # regularization value gamma_1 and gamma_2
    # gamma_1: time-independent, gamma_2: time-dependent
    def get_gamma1(last_time):
        # individual site: gamma_1s, escape group: gamma_1p
        gamma_1s = round(gamma_1/last_time,3) # constant MPL gamma value / max time
        gamma_1p = gamma_1s/10
        
        gamma1   = np.ones(x_length)*gamma_1s
        for n in range(ne):
            gamma1[x_length-ne+n] = gamma_1p
        
        return gamma1
    
    def get_gamma2(last_time, beta):
        # Use a time-varying gamma_prime, gamma_2tv is the middle value, 
        # boundary value is 4 times larger, decrese/increase exponentially within 10% generation.
        gamma_t = np.ones(len(ExTimes))
        tv_range = max(int(round(last_time*0.1/10)*10),1)
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
        
        gamma2 = np.ones((x_length,len(ExTimes)))* gamma_2c
        for i in range(x_length):
            if i in p_sites:
                for qq in range(len(NUC)):
                    index = int (muVec[i][qq]) 
                    if index != -1:
                        gamma2[index] = gamma_t * gamma_2tv
            
            if i >= x_length-ne:
                gamma2[i] = gamma_t * gamma_2tv

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
    ######################### time varying inference ###############################
    
    if not infer_tv:
        sys.exit(0)

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
    interp_times = insert_time(sample_times)
    ExTimes  = get_ExTimes(interp_times)
    print(f"CH{tag[6:]} \nExTimes: {ExTimes}")

    # get gamma_1 and gamma_2
    gamma_1 = get_gamma1(sample_times[-1])
    gamma_2 = get_gamma2(sample_times[-1],beta)

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

    if print_time:
        start_time = time_module.time()

    # solve the bounadry condition ODE to infer selections
    def fun(time,s):
        global eval_count
        eval_count += 1
        
        """ Function defining the right-hand side of the system of ODE's"""
        # s1                 = s[:x_length,:]   # the actual selection coefficients s1 = s
        # s2                 = s[x_length:,:]   # the derivatives of the selection coefficients, s2 = s'
        dsdt               = np.zeros_like(s)  # the RHS of the system of ODE's

        # s' = s2, s2:the derivatives of the selection coefficients
        dsdt[:x_length, :] = s[x_length:,:]

        single_freq = interp_x(time)
        double_freq = interp_xx(time)
        flux_mut    = interp_mu(time)
        p_mut_k     = interp_mut(time) if ne > 0 else 0
        r_rate      = interp_r(time) if not cr else np.ones(len(time))*1.4e-5
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
    print(solution.message)
    print(eval_count)

    # Get the solution for sample times
    # removes the superfluous part of the array and only save the sampled times points
    # including the extended time points
    sc_all         = solution.sol(ExTimes)
    desired_sc_all = sc_all[:x_length,:] 
            
    # not include the extended time points
    time_sample       = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
    sc_sample         = solution.sol(time_sample)
    desired_sc_sample = sc_sample[:x_length,:]

    print(f'Solver_time : {solution.x}')

    if print_time:
        end_time = time_module.time()
        print(f"Execution time : {end_time - start_time} seconds")

    # save the solution with constant_time-varying selection coefficient
    g = open('%s/%s/sc_%s%s.npz'%(HIV_DIR, output_dir, tag, name), mode='w+b')
    np.savez_compressed(g, all = desired_sc_all, selection=desired_sc_sample, time=sample_times, ExTimes=ExTimes)
    g.close()
    
if __name__ == '__main__':
    main(sys.argv[1:])
