#!/usr/bin/env python
# coding: utf-8

import sys,os
import argparse
import numpy as np
import scipy as sp
import time as time_module

## nucleotide parameter
NUC = ['-', 'A', 'C', 'G', 'T']
q = len(NUC)
CALLS = 0

def main(args):
    """Infer time-varying selection coefficients from HIV data"""

    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Time Varying Selection coefficients inference')
    parser.add_argument('-tag',          type=str,    default='700010058-3',        help='input HIV data tag')
    parser.add_argument('-name',         type=str,    default='',                   help='suffix for output data')
    parser.add_argument('-dir',          type=str,    default='data/HIV',           help='directory for HIV data')
    parser.add_argument('-output',       type=str,    default='output',             help='directory for HIV data')
    parser.add_argument('-g1',           type=float,  default=10,                   help='regularization restricting the magnitude of the selection coefficients')
    parser.add_argument('-g2c',          type=float,  default=100000,               help='regularization restricting the time derivative of the selection coefficients,constant')
    parser.add_argument('-g2tv',         type=float,  default=50,                   help='regularization restricting the time derivative of the selection coefficients,time varying')
    parser.add_argument('-theta',        type=float,  default=10,                   help='magnification of fixation gamma')
    parser.add_argument('--pt',          action='store_false', default=True,        help='whether or not to print the execution time')
    parser.add_argument('--linear',      action='store_true', default=False,        help='whether or not to use linear interpolation data')
    parser.add_argument('--fixation',    action='store_false', default=True,        help='whether or not to add time-varying regularization for fixation')
    parser.add_argument('--sp',          action='store_true', default=False,        help='whether or not to add time-varying regularization for special sites')

    arg_list  = parser.parse_args(args)

    tag        = arg_list.tag
    name       = arg_list.name
    HIV_DIR    = arg_list.dir
    output_dir = arg_list.output
    theta      = arg_list.theta
    gamma_1    = arg_list.g1     # regularization parameter, which will be change according to the time points
    gamma_2c   = arg_list.g2c
    gamma_2tv  = arg_list.g2tv
    if_linear   = arg_list.linear
    print_time = arg_list.pt
    if_fixation = arg_list.fixation
    if_sp      = arg_list.sp

    if if_linear: # add linear suffix to name if using linear interpolation data
        if name == '':
            name = '-linear'
        else:
            name = '-%s-linear'%name

    ############################################################################
    ################################# function #################################
    # regularization value gamma_1 and gamma_2
    # gamma_1: time-independent, gamma_2: time-dependent  
    def get_g2_weight(time, tv_range):
        # Use a time-varying gamma_prime, gamma_2tv is the middle value, 
        # boundary value is 4 times larger, decrese/increase exponentially within 10% generation.
        beta   = 4
        alpha  = np.log(beta) / tv_range
        if time <= tv_range:
            weight_t = np.exp(-alpha * time) * beta
        elif time > sample_times[-1] - tv_range:
            weight_t = np.exp(alpha * (time - sample_times[-1] + tv_range))
        else:
            weight_t = 1
        return weight_t

    # solve the bounadry condition ODE to infer selections
    def fun(time,s):
        """ Function defining the right-hand side of the system of ODE's"""
        # global CALLS
        # CALLS += 1
        # if CALLS % 500 == 0:
        #     print(f"fun calls = {CALLS}, time points in this call = {len(time)}")

        dsdt               = np.zeros_like(s)  # the RHS of the system of ODE's
        s1                 = s[:x_length, :] # (L, n_time)
        dsdt[:x_length, :] = s[x_length:,:]  # s' = s2, s2:the derivatives of the selection coefficients

        # mask: inside sample range
        tmin = sample_times[0]
        tmax = sample_times[-1]
        inside = (time >= tmin) & (time <= tmax)
        outside = ~inside

        L = x_length
        n_time = time.size

        # 1) baseline: (L, n_time)
        g1_mat = np.broadcast_to(g1_base[:, None], (L, n_time)).copy()

        # 2) trait part
        if ne > 0:
            s_trait = s[:L, :][trait_idx, :]              # (ne, n_time)

            # gamma_positive: default gamma_1e, but after fixation_time -> gamma_1e/theta
            gamma_pos = np.full((ne, n_time), gamma_1e, dtype=float)
            if if_fixation:
                # time[None, :] broadcasts to (ne, n_time)
                gamma_pos[time[None, :] > fixation_time[:, None]] = gamma_1e / theta

            # negative uses fixed gamma_1e*100 (matches your old code)
            g1_mat[trait_idx, :] = np.where(s_trait < 0.0, gamma_1e * 100.0, gamma_pos)

        # g2 s2'(t) = A(t)*s1(t) + b(t), s1: the actual selection coefficients
        # A(t) = C(t) + g1, b(t) = - dx(t) + F(t) + R(t)
        if outside.any(): # s'' = gamma1* s(t)/gamma2(t)
            # , g2 in outside is 4 times larger than g2 in inside
            dsdt[x_length:, outside] = (g1_mat[:, outside] * s1[:, outside]) / (g2_vec[:, None] * 4.0)
        
        if inside.any(): # s2'(t) = (C(t)s(t) + gamma1 s(t) + b(t)) / gamma2(t)
            # get C(t) and b(t) for inside time points
            t_in = time[inside]
            n_in = t_in.size

            ts = time_all
            
            # interval indices for interpolation
            k = np.searchsorted(ts, t_in, side="right") - 1
            k = np.clip(k, 0, len(ts) - 2)

            t0 = ts[k]
            t1 = ts[k + 1]
            a = (t_in - t0) / (t1 - t0)       # (n_in,)

            # Use linear interpolation to get C(t) and flux(t)
            C0, C1 = C_all[k], C_all[k + 1]
            f0, f1 = flux_all[k], flux_all[k + 1]
            C_t_all    = C0 + (C1 - C0) * a[:, None, None]
            flux_t_all = f0 + (f1 - f0) * a[:, None]

            # get dxdt(t) (step function)
            m = np.searchsorted(sample_times, t_in, side="left")
            ks = np.clip(m - 1, 0, len(sample_times) - 2)
            dxdt_t_all = dxdt[ks].copy()
            on_node = (m < len(sample_times)) & (sample_times[m] == t_in)
            if on_node.any(): # optional: node override when t_in equals sample_times exactly
                dxdt_t_all[on_node] = dxdt_node[m[on_node]]

            b_t_all = flux_t_all - dxdt_t_all  # b(t) = flux(t) - dxdt(t)

            # g2 weight per inside time point
            g2w = np.array([get_g2_weight(t, tv_range) for t in t_in], dtype=float)
            denom = g2_vec[:, None] * g2w

            # compute (C @ s1 + g1*s1 + b) / denom for each inside column
            s1_in = s1[:, inside]                  # (L, n_in)

            # right hand side: C(t)*s1(t) + g1*s1 + b(t)
            rhs = np.empty_like(s1_in)             # (L, n_in)
            for j in range(n_in):
                rhs[:, j] = C_t_all[j] @ s1_in[:, j]
            rhs += g1_mat[:, inside] * s1_in
            rhs += b_t_all.T                       # (L, n_in)

            dsdt[x_length:, inside] = rhs / denom

        return dsdt

    # Boundary conditions
    # solution to the system of differential equation 
    # with the derivative of the selection coefficients zero at the endpoints
    def bc(b1,b2):
        # Neumann boundary condition
        return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints

    ################################################################################
    ######################### time varying inference ###############################
    # load processed data from rawdata file
    try:
        if if_linear:
            interp_data  = np.load('%s/rawdata/interdata_%s_linear.npz'%(HIV_DIR,tag), allow_pickle=True)
        else:
            interp_data  = np.load('%s/rawdata/interdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
        
        C_all = interp_data['C_all']
        flux_all = interp_data['flux_all']
        dxdt     = interp_data['dxdt']
        dxdt_node = interp_data['dxdt_node']
        fixation_time = interp_data['fixation_time']
        sample_times = interp_data['sample_times']
        ExTimes = interp_data['ExTimes']
        tv_index = interp_data['tv_index']
        sc_initial = interp_data['sc_initial']
        
    except FileNotFoundError:
        print("error, rawdata file does not exist, please process the data first")
        sys.exit(1)
    
    x_length = flux_all.shape[1]
    ne       = len(fixation_time)
    trait_idx    = np.arange(x_length - ne, x_length)

    time_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))

    # get gamma_1 and gamma_2
    full_time = sample_times[-1] - sample_times[0]
    gamma_1s = round(gamma_1/full_time,3) # constant MPL gamma value / max time
    gamma_1e = gamma_1s/10
    g1_vec  = np.ones(x_length)*gamma_1s # set gamma_1 for traits at ODE part
    g1_base = g1_vec.copy()

    tv_range = max(int(round(sample_times[-1]*0.1/10)*10),1)
    g2_vec = np.ones(x_length)* gamma_2c
    # special site - smaller gamma 2 - time varying
    if if_sp:
        for idx in tv_index:
            g2_vec[idx] = gamma_2tv
    # binary trait - smaller gamma 2 - time varying
    for n in range(ne):
        g2_vec[x_length-ne+n] = gamma_2tv

    if print_time:
        start_time = time_module.time()

    # initial guess for the selection coefficients
    ss_extend = np.zeros((2*x_length,len(ExTimes)))
    ss_extend[:x_length, :] = sc_initial[:, None]

    solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=100000, tol=1e-3)

    sc_all = solution.sol(time_all)[:x_length,:]
    
    g = open('%s/%s/sc-CH%s%s.npz'%(HIV_DIR, output_dir, tag[6:], name), mode='w+b')
    np.savez_compressed(g, sc_all=sc_all, time=sample_times, ExTimes=ExTimes)
    g.close()

    if print_time:
        end_time = time_module.time()
        print(f"CH{tag[6:]}, finished in {end_time - start_time} seconds, if_fixation={if_fixation}, if_linear={if_linear}, if_sp={if_sp}")
    
if __name__ == '__main__':
    main(sys.argv[1:])
