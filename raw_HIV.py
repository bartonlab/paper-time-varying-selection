#!/usr/bin/env python
# coding: utf-8

from email import parser
from dataclasses import dataclass
from typing import List
import sys,os
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from itertools import product

## nucleotide parameter
NUC = ['-', 'A', 'C', 'G', 'T']
nuc_to_idx = {n: i for i, n in enumerate(NUC)}

q = len(NUC)
CALLS = 0

@dataclass
class Result:
    seq_length: int
    special_sites: List[int]
    uniq_t: List[int]
    r_rates: List[float]
    escape_group: List[List[int]]
    escape_TF: List[List[List[int]]]
    trait_dis: List[List[int]]

def AnalyzeData(tag,HIV_DIR,add_time=False):

    # ---------- paths ----------
    suffix = "-add" if add_time else ""
    df_info_path = f"{HIV_DIR}/constant/analysis/{tag}-analyze.csv"
    seq_path     = f"{HIV_DIR}/input/sequence/{tag}-poly-seq2state{suffix}.dat"
    r_path       = f"{HIV_DIR}/input/r_rates/r-{tag}{suffix}.dat"
    trait_path   = f"{HIV_DIR}/constant/epitopes/escape_group-{tag}.csv"

    # ---------- read seq_length + uniq_t efficiently ----------
    # 1) seq_length: read only first row
    first_row = np.loadtxt(seq_path, max_rows=1)
    seq_length = int(first_row.shape[0] - 2)

    # 2) uniq_t: read only first column
    times = np.loadtxt(seq_path, usecols=0)
    uniq_t = np.unique(times).astype(int).tolist()

    df_info = pd.read_csv(df_info_path, comment='#', memory_map=True)
    
    # ---------- recombinant rates ----------
    r_rates = np.loadtxt(r_path)
    r_rates = np.atleast_1d(r_rates).astype(float).tolist()

    if len(r_rates) != len(uniq_t):
        raise ValueError(f"len(r_rates)={len(r_rates)} != len(uniq_t)={len(uniq_t)} for tag={tag}")

    # ---------- escape groups (binary traits) ----------
    escape_group: List[List[int]] = []
    try:
        df_trait = pd.read_csv(trait_path, comment="#", memory_map=True)
        df_rows = df_trait[df_trait["epitope"].notna()]

        for epi, sub in df_rows.groupby("epitope", sort=False):
            sites = sub["polymorphic_index"].dropna().astype(int).unique().tolist()
            escape_group.append(sites)
    except FileNotFoundError:
        print(f"CH{tag[-5:]} did not find escape group file.")

    # ---------- nonsynonymous sites (candidate special sites) ----------
    df_info = pd.read_csv(df_info_path, comment="#", memory_map=True)
    epi_mask = df_info["epitope"].notna()
    df_epi = df_info.loc[epi_mask & df_info["escape"]]
    nonsy_sites = df_epi["polymorphic_index"].dropna().astype(int).unique()

    # ---------- precompute TF map: site -> [nuc_index, nuc_index, ...] ----------
    df_tf = df_info.loc[epi_mask & (~df_info["escape"]), ["polymorphic_index", "nucleotide"]].copy()
    df_tf["polymorphic_index"] = df_tf["polymorphic_index"].astype(int) # ensure polymorphic_index is int for mapping
    df_tf["nuc_idx"] = df_tf["nucleotide"].map(nuc_to_idx).astype(int) # convert nucleotide (str) to index (int)
    tf_map = df_tf.groupby("polymorphic_index")["nuc_idx"].agg(list).to_dict()

    # ---------- build escape_TF and special_sites ----------
    escape_sites_flat = [s for group in escape_group for s in group]
    escape_sites_set = set(escape_sites_flat)

    # special sites = nonsy_sites - all escape sites
    special_sites = nonsy_sites[~np.isin(nonsy_sites, np.fromiter(escape_sites_set, dtype=int))].tolist()

    # escape_TF: list over groups -> list over sites -> list of TF nuc indices
    escape_TF: List[List[List[int]]] = []
    for group in escape_group:
        escape_TF.append([tf_map.get(int(site), []) for site in group])

    # ---------- trait distance (use alignment lookup dict) ----------
    trait_dis: List[List[int]] = []
    if escape_group:
        # create a mapping from polymorphic_index to alignment position for sites in escape groups
        align_map = df_info.dropna(subset=["polymorphic_index"]).assign(
            polymorphic_index=lambda d: d["polymorphic_index"].astype(int)
        ).drop_duplicates("polymorphic_index").set_index("polymorphic_index")["alignment"].to_dict()

        for group in escape_group:
            aligns = [align_map[int(site)] for site in group] # get alignment positions for sites in the group
            trait_dis.append(np.diff(aligns).astype(int).tolist()) # diff of alignment positions gives trait distance

    return Result(
        seq_length=seq_length,
        special_sites=special_sites,
        uniq_t=uniq_t,
        r_rates=r_rates,
        escape_group=escape_group,
        escape_TF=escape_TF,
        trait_dis=trait_dis,
    )

# load sequence data and get sVec, nVec, eVec
def getSequence(history, escape_TF, escape_group):
    hist = np.asarray(history)  # shape: (N, 2 + seq_length)
    if hist.size == 0:
        raise ValueError("Input sequence is empty")

    # ---------- 1) Split by time ----------
    tcol = hist[:, 0]
    # find indices where time changes, split into blocks of rows with the same time
    cut = np.nonzero(np.diff(tcol) != 0)[0] + 1
    idx_blocks = np.split(np.arange(hist.shape[0]), cut)

    # ---------- 2) Get sVec and nVec ----------
    # each block corresponds to a time point
    sVec = [hist[idx, 2:] for idx in idx_blocks]
    nVec = [hist[idx, 1].tolist() for idx in idx_blocks]

    # ---------- 3)  Get eVec ----------
    ne = len(escape_group)
    if ne == 0: # no escape group, return empty for eVec
        eVec = [[] for _ in idx_blocks]
        return sVec, nVec, eVec

    # link escape_site with escape_TF
    # group_site2wt[g] = [(site, wild type), ...]
    group_site2wt = []
    for g, group in enumerate(escape_group):
        specs_g = []
        tf_g = escape_TF[g]
        for k, site in enumerate(group):
            specs_g.append((int(site), set(tf_g[k])))
        group_site2wt.append(specs_g)

    eVec = []
    for sVec_t in sVec:
        eVec_t = []
        for seq in sVec_t:
            # if any escape site does not have allowed TF, mark as 1 (escape), otherwise 0 (non-escape)
            esc = np.fromiter(
                (1 if any(seq[site] not in wild_types for site, wild_types in site2wt) else 0
                for site2wt in group_site2wt),
                dtype=int,
                count=ne,
            )
            eVec_t.append(esc)
        eVec.append(eVec_t)

    return sVec, nVec, eVec

def getMutantS(sVec, seq_length, q):
    # Concatenate sequences at all time points into a single numpy array for processing
    S = np.concatenate([np.asarray(block) for block in sVec], axis=0).astype(np.int64)

    muVec = np.full((seq_length, q), -1, dtype=np.int64)
    x_length = 0

    for i in range(seq_length):
        allele_uniq = np.unique(S[:, i]) # get all possible alleles in site i across all time points
        m = allele_uniq.size
        muVec[i, allele_uniq] = np.arange(x_length, x_length + m, dtype=np.int64)
        x_length += m

    return x_length, muVec

def main(args):
    """Infer time-varying selection coefficients from HIV data"""

    # Read in parameters from command line
    parser = argparse.ArgumentParser(description='Time Varying Selection coefficients inference')
    parser.add_argument('-tag',          type=str,    default='700010058-3',       help='input HIV data tag')
    parser.add_argument('-dir',          type=str,    default='data/HIV',          help='directory for HIV data')
    parser.add_argument('--add',         action='store_true',  default=False,      help='whether or not to add time to the input data')
    parser.add_argument('--raw',         action='store_false', default=True,       help='whether or not to save the raw data')
    parser.add_argument('--linear',      action='store_true', default=False,       help='whether or not to use linear interpolation')

    arg_list  = parser.parse_args(args)

    HIV_DIR = arg_list.dir
    tag     = arg_list.tag
    if_add  = arg_list.add
    if_raw  = arg_list.raw
    if_linear = arg_list.linear

    ############################################################################
    ################################# function #################################
    # calculate single and pair allele frequency (multiple case)
    def get_allele_frequency(sVec,nVec,eVec,muVec):
        T = len(nVec)
        x  = np.zeros((T, x_length), dtype=float)
        xx = np.zeros((T, x_length, x_length), dtype=float)
        base = x_length - ne
        for t in range(T):
            n_t = nVec[t]
            pop_size_t = float(np.sum(n_t))

            for k in range(len(n_t)):
                n_tk = float(n_t[k])
                seq_tk = sVec[t][k] # get the sequence for time t and sequence index k
                new_idx = muVec[np.arange(seq_length), np.asarray(seq_tk, dtype=int)]

                # === individual locus part ===
                for i in range(seq_length):
                    aa = new_idx[i]
                    if aa == -1:
                        continue
                    # --- single allele frequency (mutation) ---
                    x[t, aa] += n_tk
                    # --- pair allele frequency (mutation-mutation) ---
                    for j in range(i+1, seq_length):
                        bb = new_idx[j]
                        if bb == -1:
                            continue
                        xx[t, aa, bb] += n_tk
                        xx[t, bb, aa] += n_tk

                # === escape part ===
                if ne == 0:
                    continue
                escape_tk = eVec[t][k] # if escape for time t and sequence index k
                for n in range(ne):
                    escape_n = escape_tk[n] # binary variable for whether escape happens in epitope n
                    if escape_n == 0: # no escape, no contribution to x and xx
                        continue
                    aa = base + n
                    # --- single allele frequency (epitope) ---
                    x[t,aa] += n_tk
                    # --- pair allele frequency (epitope-mutation) ---
                    for j in range(seq_length):
                        bb = new_idx[j]
                        if bb == -1:
                            continue
                        xx[t, aa, bb] += n_tk
                        xx[t, bb, aa] += n_tk
                    # --- pair allele frequency (epitope-epitope) ---
                    for m in range(int(n+1), ne):
                        escape_m = escape_tk[m] # binary variable for whether escape happens in epitope m
                        if escape_m == 0: # no escape, no contribution to x and xx
                            continue
                        bb = base + m
                        xx[t, aa, bb] += n_tk
                        xx[t, bb, aa] += n_tk
            # normalize  
            x[t,:]    = x[t,:]/pop_size_t
            xx[t,:,:] = xx[t,:,:]/pop_size_t
        return x,xx

    # calculate frequency for sequences only have one escape mutation at one epitope
    def get_escape_fre_term(sVec, nVec):

        ex = np.zeros((len(nVec), seq_length, q), dtype=float)

        # precompute the wild type sets for each escape site in each group for O(1) membership check
        wt_sets = [
            [set(escape_TF[n][idx]) for idx in range(len(escape_group[n]))]
            for n in range(ne)
        ]

        for t in range(len(nVec)):
            pop_size_t = float(np.sum(nVec[t]))

            for k in range(len(sVec[t])):
                w = float(nVec[t][k])
                row = sVec[t][k]

                for n in range(ne):
                    mut_count = 0
                    mut_site = -1

                    for idx, nn in enumerate(escape_group[n]):
                        if row[nn] not in wt_sets[n][idx]: # if escape mutation
                            mut_count += 1
                            mut_site = nn
                            if mut_count > 1: # break if more than 1 escape mutation at the same epitope
                                break

                    if mut_count == 1:
                        qq = int(row[mut_site])
                        ex[t, mut_site, qq] += w

            ex[t, :, :] /= pop_size_t

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

    # ---------- paths ----------
    suffix = "-add" if if_add else ""
    seq_path = f"{HIV_DIR}/input/sequence/{tag}-poly-seq2state{suffix}.dat"
    raw_path = f"{HIV_DIR}/rawdata/rawdata_{tag}{suffix}.npz"
    int_path = f"{HIV_DIR}/rawdata/interdata_{tag}_linear{suffix}.npz"

    ################################################################################
    ############################# HIV data process #################################
    if if_raw:
        # obtain raw sequence data
        data = np.loadtxt(seq_path)

        # information for escape group
        result       = AnalyzeData(tag,HIV_DIR,add_time=if_add)

        escape_group = result.escape_group
        escape_TF    = result.escape_TF
        trait_dis    = result.trait_dis
        seq_length   = result.seq_length
        sample_times = result.uniq_t
        r_rates      = result.r_rates
        p_sites      = result.special_sites

        ne           = len(escape_group)

        # obtain sequence data and frequencies
        sVec,nVec,eVec = getSequence(data,escape_TF,escape_group)
        x_length,muVec = getMutantS(sVec, seq_length, q)
        x_length      += ne

        # get index for special sites
        tv_index = []
        for p_site in p_sites:
            for qq in range(len(NUC)):
                index = int (muVec[p_site][qq]) 
                if index != -1:
                    tv_index.append(index)

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
        f = open(raw_path, mode='w+b')
        escape_group = np.array(escape_group, dtype=object)
        escape_TF    = np.array(escape_TF , dtype=object)
        trait_dis    = np.array(trait_dis , dtype=object)
        np.savez_compressed(f, muVec=muVec, single_freq=x, double_freq=xx, escape_freq=ex,\
                            r_rates=r_rates, p_mut_k_freq=p_mut_k,special_sites=p_sites, tv_index=tv_index,\
                            escape_group=escape_group, escape_TF=escape_TF,trait_dis=trait_dis,\
                            seq_length=seq_length, sample_times=sample_times)
        f.close()

    ################################################################################
    ##################### interpolation for C, flux and dxdt #######################

    if not if_linear:
        return
    
    muMatrix = np.loadtxt("%s/input/Zanini-extended.dat"%HIV_DIR)
    sc_const = np.loadtxt("%s/constant/output/sc-%s.dat"%(HIV_DIR,tag))

    # load processed data from rawdata file
    try:
        rawdata  = np.load(raw_path, allow_pickle=True)
        
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
        tv_index     = rawdata['tv_index']
        escape_group = rawdata['escape_group'].tolist()
        escape_TF    = rawdata['escape_TF'].tolist()
        trait_dis    = rawdata['trait_dis'].tolist()

        ne           = len(escape_group)
        x_length     = len(x_raw[0])

    except FileNotFoundError:
        print("error, rawdata file does not exist, please process the data first")
        sys.exit(1)

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
            flux_r_t = get_rec_flux_at_t(r_rates_all[ti], x_all[ti,x_length-ne:], p_mut_k_all[ti], trait_dis)
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
    f = open(int_path, mode='w+b')
    np.savez_compressed(f, x_all=x_all, C_all=C_all, flux_all=flux_all, dxdt=dxdt, dxdt_node = dxdt_node,
                        fixation_time=fixation_time, sample_times=sample_times, ExTimes=ExTimes, 
                        tv_index=tv_index, sc_initial=sc_initial)
    f.close()

if __name__ == '__main__':
    main(sys.argv[1:])
