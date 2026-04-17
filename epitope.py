# LIBRARIES
from curses import raw
import json
import os
import sys
from time import time
import scipy as sp
from scipy import integrate
import scipy.linalg as la
from scipy.interpolate import interp1d
from collections import defaultdict
import numpy as np
import pandas as pd
import re
import urllib.request
from math import e, isnan
import statistics
import glob
import subprocess
import copy

# GLOBAL VARIABLES

NUC = ['-', 'A', 'C', 'G', 'T']
PRO = ['-', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
       'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
REF = NUC[0]
CONS_TAG = 'CONSENSUS'
HXB2_TAG = 'B.FR.1983.HXB2-LAI-IIIB-BRU.K03455.19535'
TIME_INDEX = 3
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+++++++++++++++++++++++++++'

# FUNCTIONS
def index2frame(i):
    """ Return the open reading frames corresponding to a given HXB2 index. """

    frames = []

    if ( 790<=i<=2292) or (5041<=i<=5619) or (8379<=i<=8469) or (8797<=i<=9417):
        frames.append(1)
    if (5831<=i<=6045) or (6062<=i<=6310) or (8379<=i<=8653):
        frames.append(2)
    if (2253<=i<=5096) or (5559<=i<=5850) or (5970<=i<=6045) or (6225<=i<=8795):
        frames.append(3)

    return frames

def codon2aa(c):
    """ Return the amino acid character corresponding to the input codon. """

    # If all nucleotides are missing, return gap
    if c[0]=='-' and c[1]=='-' and c[2]=='-': return '-'

    # Else if some nucleotides are missing, return '?'
    elif c[0]=='-' or c[1]=='-' or c[2]=='-': return '?'

    # If the first or second nucleotide is ambiguous, AA cannot be determined, return 'X'
    elif c[0] in ['W', 'S', 'M', 'K', 'R', 'Y'] or c[1] in ['W', 'S', 'M', 'K', 'R', 'Y']: return 'X'

    # Else go to tree
    elif c[0]=='T':
        if c[1]=='T':
            if    c[2] in ['T', 'C', 'Y']: return 'F'
            elif  c[2] in ['A', 'G', 'R']: return 'L'
            else:                          return 'X'
        elif c[1]=='C':                    return 'S'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'Y'
            elif  c[2] in ['A', 'G', 'R']: return '*'
            else:                          return 'X'
        elif c[1]=='G':
            if    c[2] in ['T', 'C', 'Y']: return 'C'
            elif  c[2]=='A':               return '*'
            elif  c[2]=='G':               return 'W'
            else:                          return 'X'
        else:                              return 'X'

    elif c[0]=='C':
        if   c[1]=='T':                    return 'L'
        elif c[1]=='C':                    return 'P'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'H'
            elif  c[2] in ['A', 'G', 'R']: return 'Q'
            else:                          return 'X'
        elif c[1]=='G':                    return 'R'
        else:                              return 'X'

    elif c[0]=='A':
        if c[1]=='T':
            if    c[2] in ['T', 'C', 'Y']: return 'I'
            elif  c[2] in ['A', 'M', 'W']: return 'I'
            elif  c[2]=='G':               return 'M'
            else:                          return 'X'
        elif c[1]=='C':                    return 'T'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'N'
            elif  c[2] in ['A', 'G', 'R']: return 'K'
            else:                          return 'X'
        elif c[1]=='G':
            if    c[2] in ['T', 'C', 'Y']: return 'S'
            elif  c[2] in ['A', 'G', 'R']: return 'R'
            else:                          return 'X'
        else:                              return 'X'

    elif c[0]=='G':
        if   c[1]=='T':                    return 'V'
        elif c[1]=='C':                    return 'A'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'D'
            elif  c[2] in ['A', 'G', 'R']: return 'E'
            else:                          return 'X'
        elif c[1]=='G':                    return 'G'
        else:                              return 'X'

    else:                                  return 'X'

def read_file(HIV_DIR,name):
    result = []  # initialize the result list
    p = open(HIV_DIR+'/input/'+name,'r')
    for line in p:
        # split the line into items and convert them to integers
        line_data = [int(item) for item in line.split()]
        result.append(line_data)
    p.close()
    return result

def read_file_s(HIV_DIR,name):
    result = [] # initialize the result list
    p = open(HIV_DIR+'/input/'+name,'r')
    for line in p:
        line_data = []  # store the data for each line
        for item in line.split():  # split the line into items
            # if the item contains a '/', split it into two integers and add them to the list
            if '/' in item:  
                line_data.append(list(map(int, item.split('/'))))
            # if the item does not contain a '/', add it to the list
            else:  
                line_data.append([int(item)])
        result.append(line_data)
    p.close()
    return result

def get_frame(tag, poly, nuc, i_alig, i_HXB2, shift, TF_sequence,polymorphic_sites,poly_states):
    """ Return number of reading frames in which the input nucleotide is nonsynonymous in context, compared to T/F. """

    ns = []

    frames = index2frame(i_HXB2)
    
    match_states = poly_states[poly_states.T[polymorphic_sites.index(i_alig)]==NUC.index(nuc)]
    
    for fr in frames:

        pos = int((i_HXB2+shift-fr)%3) # position of the nucleotide in the reading frame
        TF_codon = [temp_nuc for temp_nuc in TF_sequence[i_alig-pos:i_alig-pos+3]]

        if len(TF_codon)<3:
            print('\tmutant at site %d in codon for CH%s that does not terminate in alignment' % (i_alig,tag[-5:]))

        else:
            mut_codon       = [a for a in TF_codon]
            mut_codon[pos]  = nuc
            replace_indices = [k for k in range(3) if (k+i_alig-pos) in polymorphic_sites and k!=pos]

            # If any other sites in the codon are polymorphic, consider mutation in context
            if len(replace_indices)>0:
                is_ns = False
                for s in match_states:
                    TF_codon = [temp_nuc for temp_nuc in TF_sequence[i_alig-pos:i_alig-pos+3]]
                    for k in replace_indices:
                        mut_codon[k] = NUC[s[polymorphic_sites.index(k+i_alig-pos)]]
                        TF_codon[k]  = NUC[s[polymorphic_sites.index(k+i_alig-pos)]]
                    if codon2aa(mut_codon) != codon2aa(TF_codon):
                        is_ns = True
                if is_ns:
                    ns.append(fr)

            elif codon2aa(mut_codon) != codon2aa(TF_codon):
                ns.append(fr)

    return ns

def get_cut_sequences(HIV_DIR,tag,cut_s,cut_p):
    '''modify the sequence by removing the sites that has a weak linakge with the epitopes'''
    
    '''get special sites and escape sites'''
    df_info  = pd.read_csv('%s/constant/analysis/%s-analyze-old.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    df_epi = df_info[(df_info['epitope'].notna()) & (df_info['escape'] == True)] # epitopes containing sysnonymous mutation sites
    nonsy_sites = df_epi['polymorphic_index'].unique() # all sites can contribute to epitope
    
    df_trait = pd.read_csv('%s/constant/epitopes/escape_group-%s-old.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    indep_epitopes = df_trait['epitope'].unique()

    escape_group  = [] # escape group (each group should have more than 2 escape sites)
    for epi in indep_epitopes:
        df_e = df_trait[df_trait['epitope'] == epi] # find all escape mutation for one epitope
        unique_sites = df_e['polymorphic_index'].unique()
        unique_sites = [int(site) for site in unique_sites]
        escape_group.append(list(unique_sites))
        for site in unique_sites:
            # remove escape sites to find special sites
            index = np.where(nonsy_sites == site)
            nonsy_sites = np.delete(nonsy_sites, index)
    
    # After removing all escape sites, the rest nonsynonymous sites are special sites
    special_sites = nonsy_sites 
    escape_sites = [item for sublist in escape_group for item in sublist]
    
    '''get the raw sequence'''
    sequence    = np.loadtxt("%s/input/sequence/%s-poly-seq2state-old.dat" %(HIV_DIR,tag))
    seq_length  = len(sequence[0])-2
    ne          = len(escape_group)
   
    '''load the sij data to get the likage between sites'''
    df_sij   = pd.read_csv('%s/constant/sij/%s-sij.csv' %(HIV_DIR,tag), comment='#', memory_map=True,low_memory=False)
    df_sij[['effect']] = df_sij[['effect']].astype(float)

    # get the sites that are not escape sites and special sites as the initial cut sites
    cut_sites = {i for i in range(seq_length) if i not in escape_sites and i not in special_sites}

    # weak linkage with trait mutations
    for n in range(ne):
        for nn in escape_group[n]:
            # get all possible mutant nucleotides
            q_nn = []
            df_nn = df_info[(df_info['polymorphic_index'] == nn) & (df_info['escape'] == 'True')]
            for ii in range(len(df_nn)):
                q_nn.append(NUC.index(df_nn.iloc[ii].nucleotide))

            for j in range(len(q_nn)):
                # get the selection coefficient for the variant
                s_nq = df_sij[(df_sij['mask_polymorphic_index'] == str(nn)) &(df_sij['mask_nucleotide'] == NUC[q_nn[j]])]
                df_nq = df_info[(df_info['polymorphic_index'] == nn) & (df_info['nucleotide'] == NUC[q_nn[j]])]
                sc_nq = df_nq.iloc[0].sc_MPL
                
                # get the sites that have weak linkage with the variant
                site_nq = []
                for i in cut_sites:
                    s_nq_i = s_nq[s_nq['target_polymorphic_index'] == str(i)]
                    effect_i = []
                    for ii in range(len(s_nq_i)):
                        effect_i.append(abs(s_nq_i.iloc[ii].effect))

                    if len(effect_i)!= 0 and max(effect_i) < abs(sc_nq*cut_s):
                        site_nq.append(i)

                cut_sites = cut_sites & set(site_nq)

    # weak linkage with traits
    for n in range(ne):
        # get the selection coefficient for the binary trait
        s_nq = df_sij[df_sij['mask_polymorphic_index'] == 'epi'+str(n)]
        
        df_nq = df_trait[df_trait['epitope'] == indep_epitopes[n]]
        pc_nq = df_nq.iloc[0].tc_MPL

        # get the sites that have weak linkage with the binary trait        
        site_nq = []
        for i in cut_sites:
            s_nq_i = s_nq[s_nq['target_polymorphic_index'] == str(i)]
            effect_i = []
            for ii in range(len(s_nq_i)):
                if s_nq_i.iloc[ii].effect != 0:
                    effect_i.append(abs(float(s_nq_i.iloc[ii].effect)))

            if len(effect_i)!= 0 and max(effect_i)  < abs(pc_nq*cut_p):
                site_nq.append(i)
        cut_sites = cut_sites & set(site_nq)

    # weak linkage with special sites
    for ss in special_sites:
        # get all possible mutant nucleotides
        q_ss = []
        df_ss = df_info[(df_info['polymorphic_index'] == ss) & (df_info['escape'] == 'True')]
        for ii in range(len(df_ss)):
            q_ss.append(NUC.index(df_ss.iloc[ii].nucleotide))

        for qq in q_ss:
            # get the selection coefficient for the variant
            s_nq = df_sij[(df_sij['mask_polymorphic_index'] == str(ss)) &(df_sij['mask_nucleotide'] == NUC[qq])]
            df_nq = df_info[(df_info['polymorphic_index'] == ss) & (df_info['nucleotide'] == NUC[qq])]
            sc_nq = df_nq.iloc[0].sc_MPL

            # get the sites that have weak linkage with the variant
            site_sq = []
            for i in cut_sites:
                s_nq_i = s_nq[s_nq['target_polymorphic_index'] == str(i)]
                effect_i = []
                for ii in range(len(s_nq_i)):
                    effect_i.append(abs(s_nq_i.iloc[ii].effect))

                if len(effect_i)!= 0 and max(effect_i) < abs(sc_nq*cut_s):
                    site_sq.append(i)

            cut_sites = cut_sites & set(site_sq)

    print('CH%s raw sequence length is %d, remove %d sites'%(tag[-5:],seq_length,len(cut_sites)))

    cut_sites = list(cut_sites)
    np.savetxt('%s/input/sequence/cutsites-%s.dat'%(HIV_DIR,tag),cut_sites,fmt='%d')

    # write the cut sequence into a new file
    f = open('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag), 'w')
    for i in range(len(sequence)):
        sequence_i = [sequence[i][j+2] for j in range(len(sequence[i])-2) if j not in cut_sites]
        f.write('%d\t%d\t' % (sequence[i][0],sequence[i][1]))
        f.write(' %s' % (' '.join(str(int(num)) for num in sequence_i)))
        f.write('\n')
    f.close()

    escape_new = copy.deepcopy(escape_group)
    remaining_sites = [i for i in range(seq_length) if i not in cut_sites]
    g = open('%s/input/traitsite/traitsite-%s-cut.dat'%(HIV_DIR,tag), 'w')
    for n in range(ne):
        for nn in range(len(escape_group[n])):
            escape_new[n][nn] = remaining_sites.index(escape_group[n][nn])
        g.write('%s\n'%'\t'.join([str(i) for i in escape_new[n]]))
    g.close()

def get_cut_analysis(HIV_DIR,tag):

    # read the information from the old analysis file
    df_info  = pd.read_csv('%s/constant/analysis/%s-analyze-old.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    df_trait = pd.read_csv('%s/constant/epitopes/escape_group-%s-old.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    indep_epitopes = df_trait['epitope'].unique()
    cols_1 = ['alignment', 'HXB2_index', 'nucleotide', 'TF', 'consensus', 'epitope', 'escape', 'sc_old']
    cols_2 = [i for i in list(df_trait) if 'f_at' in i]
    cols_3 = [i for i in list(df_trait) if 'xp_at' in i]

    sequence    = np.loadtxt("%s/input/sequence/%s-poly-seq2state-old.dat" %(HIV_DIR,tag))
    seq_length  = len(sequence[0])-2

    # Get constant MPL results after cutting sites
    sc      = np.loadtxt('%s/constant/output/sc-%s.dat'%(HIV_DIR,tag))
    cut_sites = np.loadtxt('%s/input/sequence/cutsites-%s.dat'%(HIV_DIR,tag))
    
    # write the information for remainning sites into a new analysis file
    g = open('%s/constant/analysis/%s-analyze.csv'%(HIV_DIR,tag),'w')
    g.write('polymorphic_index,')
    g.write('%s,' % (','.join(cols_1)))
    g.write('sc_MPL,tc_MPL,')
    g.write('%s\n' % (','.join(cols_2)))
    polymorphic_index = 0
    for ii in range(seq_length):
        if ii not in cut_sites:
            df_ii = df_info[df_info['polymorphic_index'] == ii]
            for jj in range(len(df_ii)):
                nucleotide = df_ii.iloc[jj].nucleotide
                TF         = df_ii.iloc[jj].TF

                nuc_index  = NUC.index(nucleotide)+polymorphic_index*5
                TF_index   = NUC.index(TF)+polymorphic_index*5
                sc_MPL     = sc[nuc_index]-sc[TF_index]

                if pd.notna(df_ii.iloc[jj].tc_MPL):
                    epitope = df_ii.iloc[jj].epitope
                    n_epitope = indep_epitopes.tolist().index(epitope)
                    tc_MPL = sc[len(sc)-len(indep_epitopes)+n_epitope]
                else:
                    tc_MPL = 'nan'

                g.write('%d, '%polymorphic_index)
                g.write('%s, ' % (','.join([str(df_ii.iloc[jj][c]) for c in cols_1])))
                g.write('%s,%s,' % (sc_MPL, tc_MPL))
                g.write('%s\n' % (','.join([str(df_ii.iloc[jj][c]) for c in cols_2])))

            polymorphic_index += 1
    g.close()

    # write the information for remaining sites into a new epitope file
    df_info_cut = pd.read_csv('%s/constant/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    cols_1 = ['HXB2_index', 'nucleotide', 'TF', 'consensus', 'epitope', 'escape', 'sc_old']
    f = open('%s/constant/epitopes/escape_group-%s.csv'%(HIV_DIR,tag),'w')
    f.write('polymorphic_index,')
    f.write('%s,' % (','.join(cols_1)))
    f.write('sc_MPL,tc_MPL,')
    f.write('%s,' % (','.join(cols_2)))
    f.write('%s\n' % (','.join(cols_3)))

    for ii in range((len(df_trait))):
        HXB2_index = str(df_trait.iloc[ii].HXB2_index)
        nucleotide = df_trait.iloc[ii].nucleotide
        df_i_cut   = df_info_cut[(df_info_cut['HXB2_index'] == HXB2_index) & (df_info_cut['nucleotide'] == nucleotide)]
        index_ii   = df_i_cut.iloc[0].polymorphic_index
        sc_MPL     = df_i_cut.iloc[0].sc_MPL
        tc_MPL     = df_i_cut.iloc[0].tc_MPL
        f.write('%d, '%index_ii)
        f.write('%s, ' % (','.join([str(df_trait.iloc[ii][c]) for c in cols_1])))
        f.write('%s, %s,' % (sc_MPL, tc_MPL))
        f.write('%s, ' % (','.join([str(df_trait.iloc[ii][c]) for c in cols_2])))
        f.write('%s\n' % (','.join([str(df_trait.iloc[ii][c]) for c in cols_3])))

    f.close()


def get_MSA(ref, noArrow=True):
    """Take an input FASTA file and return the multiple sequence alignment, along with corresponding tags. """

    temp_msa = [i.split() for i in open(ref).readlines()]
    temp_msa = [i for i in temp_msa if len(i)>0]

    msa = []
    tag = []

    for i in temp_msa:
        if i[0][0]=='>':
            msa.append('')
            if noArrow: tag.append(i[0][1:])
            else: tag.append(i[0])
        else: msa[-1]+=i[0]

    msa = np.array(msa)

    return msa, tag

def clip_MSA(HXB2_start, HXB2_end, msa, tag):
    """ Clip the input MSA to the specified range of HXB2 indices and return. """

    align_start = 0
    align_end = 0
    HXB2_index = tag.index(HXB2_TAG)
    HXB2_seq = msa[HXB2_index]
    HXB2_count = 0
    for i in range(len(HXB2_seq)):
        if HXB2_seq[i]!='-':
            HXB2_count += 1
        if HXB2_count==HXB2_start:
            align_start = i
        if HXB2_count==HXB2_end+1:
            align_end = i
    return np.array([np.array(list(s[align_start:align_end].upper())) for s in msa])

def filter_excess_gaps(msa, tag, sequence_max_gaps, site_max_gaps, verbose=True):
    """ Remove sequences and sites from the alignment which have excess gaps. """

    msa = list(msa)
    tag = list(tag)

    HXB2_idx = tag.index(HXB2_TAG)
    HXB2_seq = msa[HXB2_idx]
    del msa[HXB2_idx]
    del tag[HXB2_idx]

    cons_idx = tag.index(CONS_TAG)
    cons_seq = msa[cons_idx]
    del msa[cons_idx]
    del tag[cons_idx]

    # Remove sequences with too many gaps
    temp_msa = []
    temp_tag = []
    cons_gaps = np.sum(cons_seq=='-')
    for i in range(len(msa)):
        if np.sum(msa[i]=='-')-cons_gaps<sequence_max_gaps:
            temp_msa.append(msa[i])
            temp_tag.append(tag[i])
    temp_msa = np.array(temp_msa)
    if verbose:
        print('\tselected %d of %d sequences with <%d gaps in excess of consensus' %
              (len(temp_msa), len(msa), sequence_max_gaps))

    # Drop sites that have too many gaps
    kept_indices = []
    for i in range(len(HXB2_seq)):
        if HXB2_seq[i]!='-' or np.sum(temp_msa[:,i]=='-')/len(temp_msa)<site_max_gaps:
            kept_indices.append(i)
    temp_msa = np.array([HXB2_seq[kept_indices], cons_seq[kept_indices]] + [s[kept_indices] for s in temp_msa])
    temp_tag = [HXB2_TAG, CONS_TAG] + temp_tag
    if verbose:
        print('\tremoved %d of %d sites with >%d%% gaps' %
              (len(msa[0])-len(kept_indices), len(msa[0]), site_max_gaps*100))

    return temp_msa, temp_tag

def order_sequences(msa, tag):
    """ Put sequences in time order. """

    msa = list(msa)
    tag = list(tag)

    HXB2_idx = tag.index(HXB2_TAG)
    HXB2_seq = msa[HXB2_idx]
    del msa[HXB2_idx]
    del tag[HXB2_idx]

    cons_idx = tag.index(CONS_TAG)
    cons_seq = msa[cons_idx]
    del msa[cons_idx]
    del tag[cons_idx]

    temp_msa = [HXB2_seq, cons_seq]
    temp_tag = [HXB2_TAG, CONS_TAG]
    msa, tag, temp = get_times(msa, tag, sort=True)

    return np.array(temp_msa + list(msa)), np.array(temp_tag + list(tag))

def get_times(msa, tag, sort=False):
    """Return sequences and times collected from an input MSA and tags (optional: time order them)."""

    times = []
    for i in range(len(tag)):
        if tag[i] not in [HXB2_TAG, CONS_TAG]:
            tsplit = tag[i].split('.')
            times.append(int(tsplit[TIME_INDEX]))
        else:
            times.append(-1)

    if sort:
        t_sort = np.argsort(times)
        return np.array(msa)[t_sort], np.array(tag)[t_sort], np.array(times)[t_sort]

    else:
        return np.array(times)


def impute_ambiguous(msa, tag, start_index=0, impute_edge_gaps=False):
    """ Impute ambiguous nucleotides with the most frequently observed ones in the alignment. """

    # Impute ambiguous nucleotides
    for i in range(len(msa[0])):
        for j in range(start_index, len(msa)):
            orig = msa[j][i].upper()
            if orig not in NUC:
                avg = [np.sum([msa[k][i]==a for k in range(start_index, len(msa))]) for a in NUC]
                new = NUC[np.argmax(avg)]
                if orig=='R': # A or G
                    if avg[NUC.index('A')]>avg[NUC.index('G')]:
                        new = 'A'
                    else:
                        new = 'G'
                elif orig=='Y': # T or C
                    if avg[NUC.index('T')]>avg[NUC.index('C')]:
                        new = 'T'
                    else:
                        new = 'C'
                elif orig=='K': # G or T
                    if avg[NUC.index('G')]>avg[NUC.index('T')]:
                        new = 'G'
                    else:
                        new = 'T'
                elif orig=='M': # A or C
                    if avg[NUC.index('A')]>avg[NUC.index('C')]:
                        new = 'A'
                    else:
                        new = 'C'
                elif orig=='S': # G or C
                    if avg[NUC.index('G')]>avg[NUC.index('C')]:
                        new = 'G'
                    else:
                        new = 'C'
                elif orig=='W': # A or T
                    if avg[NUC.index('A')]>avg[NUC.index('T')]:
                        new = 'A'
                    else:
                        new = 'T'
                msa[j][i] = new
                
    # Impute leading and trailing gaps
    if impute_edge_gaps:
        for j in range(start_index, len(msa)):
            gap_lead = 0
            gap_trail = 0

            gap_idx = 0
            while msa[j][gap_idx]=='-':
                gap_lead += 1
                gap_idx += 1

            gap_idx = -1
            while msa[j][gap_idx]=='-':
                gap_trail -= 1
                gap_idx -= 1

            for i in range(gap_lead):
                avg = [np.sum([msa[k][i]==a for k in range(start_index, len(msa))]) for a in NUC]
                new = NUC[np.argmax(avg)]
                msa[j][i] = new

            for i in range(gap_trail, 0):
                avg = [np.sum([msa[k][i]==a for k in range(start_index, len(msa))]) for a in NUC]
                new = NUC[np.argmax(avg)]
                msa[j][i] = new

            if (gap_lead>0) or (gap_trail<0):
                print('\timputed %d leading gaps and %d trailing gaps in sequence %d' % (gap_lead, -1*gap_trail, j))

    return msa


def get_TF(msa, tag, TF_accession, protein=False):
    """ Return the transmitted/founder sequence in an alignment. If there is no known TF sequence,
        return the most frequently observed nucleotide at each site from the earliest available sequences. """

    TF_sequence = []

    if TF_accession=='avg':
        idxs = [i for i in range(len(msa)) if tag[i]!=HXB2_TAG and tag[i]!=CONS_TAG]
        temp_msa = np.array(msa)[idxs]
        temp_tag = np.array(tag)[idxs]

        times = get_times(temp_msa, temp_tag, sort=False)
        first_time = np.min(times)
        first_seqs = [temp_msa[i] for i in range(len(temp_msa)) if times[i]==first_time]
        for i in range(len(first_seqs[0])):
            if protein:
                avg = [np.sum([s[i]==a for s in first_seqs]) for a in PRO]
                TF_sequence.append(PRO[np.argmax(avg)])
            else:
                avg = [np.sum([s[i]==a for s in first_seqs]) for a in NUC]
                TF_sequence.append(NUC[np.argmax(avg)])

    else:
        accs = [i.split('.')[-1] for i in tag]
        TF_sequence = msa[accs.index(TF_accession)]

    return TF_sequence

def create_index(msa, tag, TF_seq, cons_seq, HXB2_seq, HXB2_start, min_seqs, max_dt, df_epitope, df_exposed, out_file,
                 return_polymorphic=True, return_truncated=True):
    """ Create a reference to map between site indices for the whole alignment, polymorphic sites only, and HXB2.
        To preserve quality, identify last time point such that all earlier time points have at least min_seqs
        sequences (except time 0, which is allowed to have 1=TF) and maximum time gap of max_dt between samples.
        Include location of known epitopes, flanking residues for those epitopes, and exposed regions of Env.
        Also record the TF and consensus nucleotides at each site. Return the list of polymorphic sites. """

    msa = list(msa)
    tag = list(tag)

    HXB2_idx = tag.index(HXB2_TAG)
    HXB2_seq = msa[HXB2_idx]
    del msa[HXB2_idx]
    del tag[HXB2_idx]

    cons_idx = tag.index(CONS_TAG)
    cons_seq = msa[cons_idx]
    del msa[cons_idx]
    del tag[cons_idx]

    f = open('%s' % out_file, 'w')
    f.write('alignment,polymorphic,HXB2,TF,consensus,epitope,exposed,edge_gap,flanking\n')

    # Check for minimum number of sequences/maximum dt to truncate alignment
    temp_msa, temp_tag, times = get_times(msa, tag, sort=True)
    u_times = np.unique(times)
    t_count = [np.sum(times==t) for t in u_times]

    t_allowed = [u_times[0]]
    t_last    = u_times[0]
    for i in range(1, len(t_count)):
        if t_count[i]<min_seqs:
            continue
        elif u_times[i]-t_last>max_dt:
            break
        else:
            t_allowed.append(u_times[i])
            t_last = u_times[i]
    t_max    = t_allowed[-1]
    temp_msa = temp_msa[np.isin(times, t_allowed)]
    temp_tag = temp_tag[np.isin(times, t_allowed)]

    HXB2_index = HXB2_start
    polymorphic_index = 0
    polymorphic_sites = []
    for i in range(len(temp_msa[0])):

        # Index polymorphic sites
        poly_str = 'NA'
        if np.sum([s[i]==temp_msa[0][i] for s in temp_msa])<len(temp_msa):
            poly_str = '%d' % polymorphic_index
            polymorphic_index += 1
            polymorphic_sites.append(i)

        # Index HXB2
        HXB2_str = 'NA'
        if HXB2_seq[i]!='-':
            HXB2_str = '%d' % HXB2_index
            HXB2_index += 1
            HXB2_alpha  = 0
        else:
            HXB2_str = '%d%s' % (HXB2_index-1, ALPHABET[HXB2_alpha])
            HXB2_alpha += 1

        # Flag epitope regions
        epitope_str = ''
        flanking = 0
        for epitope_iter, epitope_entry in df_epitope.iterrows():
            if (HXB2_index-1>=epitope_entry.start-15 and HXB2_index-1<epitope_entry.start
                and epitope_entry.detected<=t_max):
                flanking += 1
            elif (HXB2_index-1<=epitope_entry.end+15 and HXB2_index-1>epitope_entry.end
                  and epitope_entry.detected<=t_max):
                flanking += 1
            if (HXB2_index-1>=epitope_entry.start and HXB2_index-1<=epitope_entry.end
                and epitope_entry.detected<=t_max):
                epitope_str = epitope_entry.epitope
            # special case: first 3 AA inserted wrt HXB2
            elif epitope_entry.epitope=='DEPAAVGVG':
                if (i>=3870 and HXB2_index-1<=epitope_entry.end
                    and epitope_entry.detected<=t_max and '-3' in out_file):
                    epitope_str = epitope_entry.epitope

        # Flag exposed sites on Env
        exposed = False
        if np.sum((HXB2_index-1>=df_exposed.start) & (HXB2_index-1<=df_exposed.end))>0:
            exposed = True

        # Flag edge gaps
        edge_def = 200
        edge_gap = False
        if np.sum(temp_msa[:, i]=='-')>0 and ((i<edge_def) or (len(temp_msa[0])-i<edge_def)):
            gap_seqs = [j for j in range(len(temp_msa)) if temp_msa[j][i]=='-']
            gap_msa = temp_msa[gap_seqs]
            edge_gap = True
            if i<edge_def:
                for s in gap_msa:
                    if np.sum(s[:i]=='-')<i:
                        edge_gap = False
                        break
            else:
                for s in gap_msa:
                    if np.sum(s[i:]=='-')<len(temp_msa[0])-i:
                        edge_gap = False
                        break

        # Save to file
        f.write('%d,%s,%s,%s,%s,%s,%s,%s,%d\n' % (i, poly_str, HXB2_str, TF_seq[i], cons_seq[i], epitope_str, exposed, edge_gap, flanking))
    f.close()

    temp_msa = [HXB2_seq, cons_seq] + list(temp_msa)
    temp_tag = [HXB2_TAG, CONS_TAG] + list(temp_tag)

    if return_polymorphic and return_truncated:
        return polymorphic_sites, temp_msa, temp_tag
    elif return_polymorphic:
        return polymorphic_sites
    elif return_truncated:
        return temp_msa, temp_tag


def save_MPL_alignment(msa, tag, out_file, polymorphic_sites=[], return_states=True, protein=False):
    """ Save a nucleotide alignment into MPL-readable form. Optionally return converted states and times. """

    idxs = [i for i in range(len(msa)) if tag[i]!=HXB2_TAG and tag[i]!=CONS_TAG]
    temp_msa = np.array(msa)[idxs]
    temp_tag = np.array(tag)[idxs]

    if polymorphic_sites==[]:
        polymorphic_sites = range(len(temp_msa[0]))

    poly_times = get_times(temp_msa, temp_tag, sort=False)

    poly_states = []
    if protein:
        for s in temp_msa:
            poly_states.append([str(PRO.index(a)) for a in s[polymorphic_sites]])
    else:
        for s in temp_msa:
            poly_states.append([str(NUC.index(a)) for a in s[polymorphic_sites]])

    f = open(out_file, 'w')
    for i in range(len(poly_states)):
        f.write('%d\t1\t%s\n' % (poly_times[i], ' '.join(poly_states[i])))
    f.close()

    if return_states:
        return np.array(poly_states, int), np.array(poly_times)

def save_trajectories(sites, states, times, TF_sequence, df_index, out_file):
    """ Save allele frequency trajectories and supplementary information. """

    index_cols = ['alignment', 'polymorphic', 'HXB2']
    cols = [i for i in list(df_index) if i not in index_cols]
    
    f = open(out_file, 'w')
    f.write('polymorphic_index,alignment_index,HXB2_index,nonsynonymous,nucleotide')
    f.write(',%s' % (','.join(cols)))
    f.write(',%s\n' % (','.join(['f_at_%d' % t for t in np.unique(times)])))

    for i in sites:
        for j in range(len(NUC)):
            traj = []
            for t in np.unique(times):
                tid = times==t
                num = np.sum(states[tid].T[sites.index(i)]==j)
                denom = np.sum(tid)
                traj.append(num/denom)

            if np.sum(traj)!=0:
                ii = df_index.iloc[i]
                match_states = states[states.T[sites.index(i)]==j]

                # Get effective HXB2 index to determine open reading frames
                eff_HXB2_index = 0
                shift = 0
                frames = []
                try:
                    eff_HXB2_index = int(ii.HXB2)
                    frames = index2frame(eff_HXB2_index)
                except:
                    eff_HXB2_index = int(ii.HXB2[:-1])
                    shift = ALPHABET.index(ii.HXB2[-1]) + 1
                    frames = index2frame(eff_HXB2_index)

                # Check whether mutation is nonsynonymous by inserting TF nucleotide in context
                nonsyn = get_nonsynonymous(sites, NUC[j], i, eff_HXB2_index, shift, frames, TF_sequence, match_states)

                # Flag whether variant is an edge gap
                edge_gap = False
                if NUC[j]=='-' and ii.edge_gap==True:
                    edge_gap = True

                # # If mutation is in Env, check for modification of N-linked glycosylation site (N-X-S/T motif)
                # glycan = get_glycosylation(sites, NUC[j], i, eff_HXB2_index, shift, TF_sequence, match_states)
                
                f.write('%d,%d,%s,%d,%s' % (sites.index(i), i, str(ii.HXB2), nonsyn, NUC[j]))
                f.write(',%s' % (','.join([str(ii[c]) if c!='edge_gap' else str(edge_gap) for c in cols])))
                f.write(',%s\n' % (','.join(['%.4e' % freq for freq in traj])))

    f.close()

def get_nonsynonymous(polymorphic_sites, nuc, i, i_HXB2, shift, frames, TF_sequence, match_states, verbose=True):
    """ Return number of reading frames in which the input nucleotide is nonsynonymous in context, compared to T/F. """

    ns = 0
    for fr in frames:

        pos = int((i_HXB2+shift-fr)%3) # position of the nucleotide in the reading frame
        TF_codon = [temp_nuc for temp_nuc in TF_sequence[i-pos:i-pos+3]]

        if len(TF_codon)<3 and verbose:
            print('\tmutant at site %d in codon that does not terminate in alignment, assuming syn' % i)

        else:
            mut_codon = [a for a in TF_codon]
            mut_codon[pos] = nuc
            replace_indices = [k for k in range(3) if (k+i-pos) in polymorphic_sites and k!=pos]

            # If any other sites in the codon are polymorphic, consider mutation in context
            if len(replace_indices)>0:
                is_ns = False
                for s in match_states:
                    TF_codon = [temp_nuc for temp_nuc in TF_sequence[i-pos:i-pos+3]]
                    for k in replace_indices:
                        mut_codon[k] = NUC[s[polymorphic_sites.index(k+i-pos)]]
                        TF_codon[k] = NUC[s[polymorphic_sites.index(k+i-pos)]]
                    if codon2aa(mut_codon)!=codon2aa(TF_codon):
                        is_ns = True
                if is_ns:
                    ns += 1

            elif codon2aa(mut_codon)!=codon2aa(TF_codon):
                ns += 1

    return ns

def process_sequence(MAX_GAP_NUM, MAX_GAP_FREQ, MIN_SEQS, MAX_DT, HIV_DIR, change_tags):
    # Load the sequence coverage for each patient
    df_range = pd.read_csv('%s/raw/range.csv' % (HIV_DIR), comment='#', memory_map=True)

    # Load the list of transmitted/founder (TF) sequences
    df_TF = pd.read_csv('%s/raw/TF.csv' % (HIV_DIR), comment='#', memory_map=True)

    # Load the list of exposed residues on Env
    df_exposed = pd.read_csv('%s/raw/exposed.csv' % (HIV_DIR), comment='#', memory_map=True)

    # Load the set of epitopes targeted by patients
    df_epitope = pd.read_csv('%s/raw/epitopes.csv' % (HIV_DIR), comment='#', dtype={'ppt': str}, memory_map=True)

    for tag_name in change_tags:
        # Read in MSA
        df_tag    = df_range[df_range['tag'] == tag_name]
        ppt       = df_tag.iloc[0]['ppt']
        tag_start = df_tag.iloc[0]['start']
        tag_end   = df_tag.iloc[0]['end']
        
        msa, tag = get_MSA('%s/raw/interim/%s-HIValign-filtered.fasta' % (HIV_DIR, ppt), noArrow=True)
        print('%s' % tag_name)
        
        # Clip MSA to specified range
        msa = clip_MSA(tag_start, tag_end, msa, tag)
        
        # Filter sequences for quality (maximum #excess gaps)
        current_msa, current_tag = filter_excess_gaps(msa, tag, MAX_GAP_NUM, MAX_GAP_FREQ, verbose=True)
        
        # Put sequences in time order (first entries are HXB2, consensus)
        current_msa, current_tag = order_sequences(current_msa, current_tag)
        
        # Locate or compute TF sequence
        TF_accession = df_TF[df_TF.ppt==ppt].iloc[0].TF
        TF_sequence = get_TF(current_msa, current_tag, TF_accession)
        
        temp_msa = [current_msa[list(current_tag).index(HXB2_TAG)], TF_sequence]
        temp_tag = [HXB2_TAG, '%s-TF' % ppt]
        
        # Impute ambiguous nucleotides and gaps at start/end of alignment
        current_msa = impute_ambiguous(current_msa, current_tag, start_index=2, impute_edge_gaps=False)
        
        # Map between local (alignment) index, index of polymorphic sites, and HXB2
        # Track location of known T cell epitopes, TF sequence, and exposed residues of Env
        poly_sites, current_msa, current_tag = create_index(current_msa, current_tag, TF_sequence, 
                                            current_msa[list(current_tag).index(CONS_TAG)], 
                                            current_msa[list(current_tag).index(HXB2_TAG)],
                                            tag_start, MIN_SEQS, MAX_DT, 
                                            df_epitope[df_epitope.ppt==ppt], df_exposed, 
                                            '%s/raw/processed/%s-index.csv' % (HIV_DIR, tag_name), 
                                            return_polymorphic=True, return_truncated=True)
        
        # Save alignment to format readable by MPL
        poly_states, poly_times = save_MPL_alignment(current_msa, current_tag, 
                                                        '%s/input/sequence/%s-poly-seq2state.dat' % (HIV_DIR, tag_name), 
                                                        polymorphic_sites=poly_sites, return_states=True)
        
        # Save allele frequency trajectories for polymorphic sites
        # Later this file will be edited to include inferred selection coefficients
        df_index = pd.read_csv('%s/raw/processed/%s-index.csv' % (HIV_DIR, tag_name), comment='#', memory_map=True)
        save_trajectories(poly_sites, poly_states, poly_times, TF_sequence, df_index,
                            '%s/raw/interim/%s-poly.csv' % (HIV_DIR, tag_name))

def find_nons_mutations(HIV_DIR, tag):
    '''
    find trait sites and corresponding TF sequence and save the information into new csv file
    '''

    """Load the set of epitopes targeted by patients"""
    df_poly  = pd.read_csv('%s/raw/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    df_epi   = pd.read_csv('%s/raw/epitopes.csv'%HIV_DIR, comment='#', memory_map=True)
    df_index = pd.read_csv('%s/raw/processed/%s-index.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    
    # TF sequence
    TF_sequence = []
    for i in range(len(df_index)):
        TF_sequence.append(df_index.iloc[i].TF)

    # alignment for polymorphic sites
    df_index_p  = df_index[df_index['polymorphic'].notna()]
    polymorphic_sites  = []
    for i in range(len(df_index_p)):
        polymorphic_sites.append(int(df_index_p.iloc[i].alignment))

    # sequence for polymorphic sites
    seq = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    poly_times = np.zeros(len(seq))
    poly_states = np.zeros((len(seq),len(seq[0])-2),dtype=int)
    for i in range(len(seq)):
        poly_times[i] = int(seq[i][0])
        for j in range(len(seq[0])-2):
            poly_states[i][j] = int(seq[i][j+2])

    escape_values = ['False'] * len(df_poly)  # define the inserted column "escape"

    df_poly['epitope'].notna()

    for i in range(len(df_poly)):
        if pd.notna(df_poly.at[i, 'epitope']) and df_poly.iloc[i].nonsynonymous > 0 and df_poly.iloc[i].nucleotide != df_poly.iloc[i].TF:
            poly = int(df_poly.iloc[i].polymorphic_index)
            nons = df_poly.iloc[i].nonsynonymous
            i_alig = df_poly.iloc[i].alignment_index
            HXB2_s = df_poly.iloc[i].HXB2_index

            # get HXB2 index and shift
            try:
                i_HXB2 = int(HXB2_s)
                shift = 0
            except:
                i_HXB2 = int(HXB2_s[:-1])
                shift = ALPHABET.index(HXB2_s[-1]) + 1
            
            frames = index2frame(i_HXB2)

            """ judge if this site is trait site """
            if len(frames) == nons : #the mutation in this site is nonsynonymous in all reading frame
                escape_values[i] = 'True'
            else:
                """get the reading frame of the mutant site"""
                nuc = df_poly.iloc[i].nucleotide
                nonsfram = get_frame(tag, poly, nuc, i_alig, i_HXB2, shift, TF_sequence, polymorphic_sites, poly_states)

                """get the reading frame of the epitope"""
                df       = df_epi[(df_epi.epitope == df_poly.iloc[i].epitope)]
                epiframe = df.iloc[0].readingframe

                ''' decide whether this polymorphic site is nonsynonymous in its epitope reading frame '''
                if epiframe in nonsfram:
                    escape_values[i] = 'True'

    """modify the csv file and save it"""
    columns_to_remove = ['exposed', 'edge_gap','flanking']
    df_poly = df_poly.drop(columns=columns_to_remove)

    df_poly.insert(4, 'escape', escape_values)
    df_poly.to_csv('%s/constant/interim/%s-poly.csv' %(HIV_DIR,tag), index=False,na_rep='nan')

    escape_group, escape_TF,epinames = get_trait(HIV_DIR, tag)

    if len(escape_group)!= 0:
        print(f'CH{tag[-5:]} has {len(escape_group)} binary traits,', end = ' ')
        for n in range(len(escape_group)):
            print(f'epitope {epinames[n]} : {escape_group[n]},', end = ' ')
        print()
    else:
        print('%s has no bianry trait'%tag)

def get_trait(HIV_DIR, tag):
    
    df_info = pd.read_csv('%s/constant/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True)

    """Get escape sites"""
    # get all epitopes for one tag
    df_rows = df_info[df_info['epitope'].notna()]
    unique_epitopes = df_rows['epitope'].unique()

    escape_group  = [] # escape group (each group should have more than 2 escape sites)
    escape_TF     = [] # corresponding wild type nucleotide
    epinames      = [] # name for binary trait

    for epi in unique_epitopes:
        df_e = df_rows[(df_rows['epitope'] == epi) & (df_rows['escape'] == True)] # find all escape mutation for one epitope
        unique_sites = df_e['polymorphic_index'].unique()
        unique_sites = [int(site) for site in unique_sites]

        if len(df_e) > 1:# if there are more than escape mutation instead of escape site for this epitope
            epi_name = epi[0]+epi[-1]+str(len(epi))
            epinames.append(epi_name)
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

    return escape_group, escape_TF, epinames

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
def getMutantS(seq_length, sVec):
    q = len(NUC)
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
def get_allele_frequency(sVec,nVec,eVec,muVec,x_length):

    seq_length = len(muVec)
    ne         = len(eVec[0][0])

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

# diffusion matrix C
def diffusion_matrix_at_t(x_0, x_1, xx_0, xx_1, dt, C_int):
    x_length = len(x_0)
    for i in range(x_length):
        C_int[i, i] += dt * (((3 - (2 * x_1[i])) * (x_0[i] + x_1[i])) - 2 * x_0[i] * x_0[i]) / 6
        for j in range(int(i+1) ,x_length):
            dCov1 = -dt * (2 * x_0[i] * x_0[j] + 2 * x_1[i] * x_1[j] + x_0[i] * x_1[j] + x_1[i] * x_0[j]) / 6
            dCov2 =  dt * (xx_0[i,j] + xx_1[i,j]) / 2

            C_int[i, j] += dCov1 + dCov2
            C_int[j, i] += dCov1 + dCov2

    return C_int

def determine_dependence(HIV_DIR, tag):

    # obtain raw sequence data
    seq     = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    escape_group, escape_TF, epinames = get_trait(HIV_DIR,tag)

    f = open('%s/input/traitsite/traitsite-%s.dat'%(HIV_DIR,tag), 'w')
    g = open('%s/input/traitseq/traitseq-%s.dat'%(HIV_DIR,tag), 'w')
    ff = open('%s/input/traitdis/traitdis-%s.dat'%(HIV_DIR,tag), 'w')

    # information for escape group
    seq_length   = len(seq[0])-2
    times = []
    for i in range(len(seq)):
        times.append(seq[i][0])
    sample_times = np.unique(times)

    ne           = len(escape_group)
    
    if ne == 0:
        f.close()
        g.close()
        ff.close()
        return
    
    # obtain sequence data and frequencies
    sVec,nVec,eVec = getSequence(seq,escape_TF,escape_group)
    x_length,muVec = getMutantS(seq_length, sVec)
    x_length      += ne

    # get all frequencies, 
    # x: single allele frequency, xx: pair allele frequency
    x,xx  = get_allele_frequency(sVec,nVec,eVec,muVec,x_length)
    C_int = np.zeros((x_length,x_length))
        
    for t in range(1, len(sample_times)):
        dt = sample_times[t]-sample_times[t-1]
        diffusion_matrix_at_t(x[t-1], x[t], xx[t-1], xx[t], dt, C_int)

    for i in range(x_length):
        for j in range(x_length):
            if abs(C_int[i][j]) < 1e-10:
                C_int[i][j] = 0
    
    # save the covariance matrix into a temporary file with 6 significant figures
    np.savetxt('temp_cov.np.dat',C_int,fmt='%.6e')

    # run the c++ code to get the reduced row echelon form
    status = subprocess.run('./rref.out', shell=True)

    # load the reduced row echelon form
    co_rr = np.loadtxt('temp_rref.np.dat')

    # delete the temporary files
    status = subprocess.run('rm temp_*.dat', shell=True)

    ll = len(co_rr)
    df_poly   = pd.read_csv('%s/constant/interim/%s-poly.csv'%(HIV_DIR,tag), memory_map=True)

    Independent = [True] * ne

    for n in range(ne):
        co_rr_c = list(co_rr.T[ll-ne+n])
        
        pivot = co_rr_c.index(np.sum(co_rr_c))
    
        if np.sum(abs(co_rr[pivot])) > 1:
            print(f'CH{tag[-5:]} : trait {epinames[n]}, linked variants:', end=' ')
            Independent[n] = False
            for i in range(len(co_rr[pivot])):
                if co_rr[pivot][i] != 0:
                    if i < ll-ne:
                        result = np.where(muVec == i)
                        variant = str(result[0][0]) + NUC[result[1][0]]
                        df_i = df_poly[df_poly['polymorphic_index'] == result[0][0]]
                        
                        if pd.notna(df_i.iloc[0]['epitope']):
                            epi = df_i.iloc[0]['epitope']
                            print(f'{variant}({epi[0]}{epi[-1]}{len(epi)}', end='')
                        else:
                            print(f'{variant}(', end='')
                            
                        if df_i.iloc[0]['TF'] == NUC[result[1][0]]:
                            print(f', WT)', end=', ')
                        else:
                            print(f')', end=', ')
                            
                    else:
                        nn = i - ll
                        print(f'{epinames[nn]}', end=', ')
            print()

    "store the information for trait sites into files (trait site and TF trait sequences)"
    for n in range(ne):
        if Independent[n]:
            f.write('%s\n'%'\t'.join([str(i) for i in escape_group[n]]))
            for m in range(len(escape_group[n])):
                if m != len(escape_group[n]) - 1:
                    g.write('%s\t'%'/'.join([str(i) for i in escape_TF[n][m]]))
                else:
                    g.write('%s'%'/'.join([str(i) for i in escape_TF[n][m]]))
            g.write('\n')
    f.close()
    g.close()

    "store the information for trait sites into files (the number of normal sites between 2 trait sites)"
    df_sequence = pd.read_csv('%s/raw/processed/%s-index.csv' %(HIV_DIR,tag), comment='#', memory_map=True,usecols=['alignment','polymorphic'])
    for i in range(len(escape_group)):
        if Independent[i]:
            i_dis = []
            for j in range(len(escape_group[i])-1):
                index0 = df_sequence[df_sequence['polymorphic']==escape_group[i][j]].iloc[0].alignment
                index1 = df_sequence[df_sequence['polymorphic']==escape_group[i][j+1]].iloc[0].alignment
                i_dis.append(int(index1-index0))
            ff.write('%s\n'%'\t'.join([str(i) for i in i_dis]))
    ff.close()


def get_independent(HIV_DIR):
    
    COV_DIR = HIV_DIR + '/constant/output'
    
    flist = glob.glob('%s/c-*.dat'%COV_DIR)

    for f in flist:
        name = f.split('/')[-1]
        tag = name.split('.')[0]

        temp_cov = 'temp_cov.np.dat'
        temp_rr  = 'temp_rref.np.dat'
        out_rr   = '%s/rr-%s.dat' % (COV_DIR, tag)

        status = subprocess.run('cp %s %s' % (f, temp_cov), shell=True)
        status = subprocess.run('./rref.out', shell=True)
        status = subprocess.run('mv %s %s' % (temp_rr, out_rr), shell=True)

    status = subprocess.run('rm %s' % (temp_cov), shell=True)
    print('Done!')

def analyze_result(HIV_DIR,tag):
    '''
    collect data and then write into csv file
    '''
    def get_xp_s(seq,traitsite,traitallele):
        times = []

        for i in range(len(seq)):
            times.append(seq[i][0])
        uniq_t = np.unique(times)
        xp    = np.zeros([len(traitsite),len(uniq_t)])

        for t in range(len(uniq_t)):
            tid = times==uniq_t[t]
            counts = np.sum(tid)
            seq_t = seq[tid][:,2:]
            for i in range(len(traitsite)):
                num = 0
                for n in range(len(seq_t)):
                    
                    nonsyn = True
                    for j in range(len(traitsite[i])):
                        if seq_t[n][int(traitsite[i][j])] not in traitallele[i][j]:
                            nonsyn = False
                    if nonsyn == False:
                        num += 1
                
                xp[i,t] = num/counts

        return xp

    seq     = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    L       = len(seq[0])-2    #the number of polymorphic sites

    sc      = np.loadtxt('%s/constant/output/sc-%s.dat'%(HIV_DIR,tag))

    try:
        traitsite = read_file(HIV_DIR,'traitsite/traitsite-%s.dat'%(tag))
    except FileNotFoundError:
        traitsite = []
        print(f'Error: {tag} does not have traitsite file')

    trait_sites = []
    for i in range(len(traitsite)):
        for j in range(len(traitsite[i])):
            trait_sites.append(traitsite[i][j])

    df_poly = pd.read_csv('%s/constant/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True)

    cols = [i for i in df_poly.columns if 'f_at_' in i]
    times = [int(cols[i].split('_')[-1]) for i in range(len(cols))]

    f = open(HIV_DIR+'/constant/analysis/'+tag+'-analyze.csv','w')
    f.write('polymorphic_index,alignment,HXB2_index,nucleotide,TF,consensus,epitope,escape,sc_MPL,tc_MPL')
    f.write(',%s' % (','.join(cols)))
    f.write('\n')

    for ii in range(len(df_poly)):
        polymorphic_index = df_poly.iloc[ii].polymorphic_index
        alignment         = df_poly.iloc[ii].alignment_index
        HXB2_index        = df_poly.iloc[ii].HXB2_index
        nucleotide        = df_poly.iloc[ii].nucleotide
        TF                = df_poly.iloc[ii].TF
        consensus         = df_poly.iloc[ii].consensus
        epitope           = df_poly.iloc[ii].epitope
        escape            = df_poly.iloc[ii].escape

        # get selection coefficient
        nuc_index  = NUC.index(nucleotide)+polymorphic_index*5
        TF_index   = NUC.index(TF)+polymorphic_index*5
        sc_MPL     = sc[nuc_index]-sc[TF_index]
        tc_MPL     = 'nan'
        df_i       = df_poly.iloc[ii]
        if sc_MPL != 0:
            for i in range(len(traitsite)):
                if polymorphic_index in traitsite[i]:
                    tc_MPL = sc[i+L*5]
        f.write('%d,%d,%s,%s,' % (polymorphic_index, alignment, HXB2_index, nucleotide))
        f.write('%s,%s,%s,%s,%s,%s' % (TF, consensus, epitope, escape, sc_MPL, tc_MPL))
        f.write(',%s' % (','.join([str(df_i[c]) for c in cols])))
        f.write('\n')
    f.close()

    if len(traitsite) != 0:
        df = pd.read_csv('%s/constant/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        index_cols = ['polymorphic_index', 'alignment']
        cols = [i for i in list(df) if i not in index_cols]

        polyseq  = read_file_s(HIV_DIR,'traitseq/traitseq-'+tag+'.dat')
        xp = get_xp_s(seq,traitsite,polyseq)

        g = open('%s/constant/epitopes/escape_group-%s.csv'%(HIV_DIR,tag),'w')
        g.write('polymorphic_index')
        g.write(',%s' % (','.join(cols)))
        for t in range(len(times)):
            g.write(',xp_at_%s'%times[t])
        g.write('\n')

        for i in range(len(traitsite)):
            for j in range(len(traitsite[i])):
                df_poly = df[(df.polymorphic_index == traitsite[i][j]) & (df.nucleotide != df.TF) & (df.escape == True)]
                for n in range(len(df_poly)):
                    g.write('%d' %traitsite[i][j])
                    g.write(',%s' % (','.join([str(df_poly.iloc[n][c]) for c in cols])))
                    for t in range(len(times)):
                        g.write(',%f'%xp[i,t])
                    g.write('\n')

def prepare_figure_info(tag, if_linear=False, HIV_DIR='data/HIV'):
    ''' 
    Output the information for plotting into npz file
    '''
    try:
        rawdata = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    except FileNotFoundError:
        print(f'No rawdata found for CH{tag[6:]}')
        return

    # ----- index for mutation and wild type -----
    try:
        base_cols = ["polymorphic_index", "HXB2_index", "nucleotide", "TF", "epitope", "escape"]
        df_index   = pd.read_csv('%s/constant/analysis/%s-analyze.csv' % (HIV_DIR, tag), usecols=base_cols, memory_map=True)
    except FileNotFoundError:
        print(f'No constant escape data found for CH{tag[6:]}')
        return 

    # load raw csv file 
    df_mut = df_index[df_index['TF'] != df_index['nucleotide']].copy() # only keep mutation row
    muVec  = rawdata['muVec']

    # precompute indices for each row
    nuc_to_idx = {n:i for i, n in enumerate(NUC)}
    site = df_mut["polymorphic_index"].to_numpy()
    nuc_mut = df_mut["nucleotide"].astype(str).to_numpy() # get nucleotide
    nuc_tf  = df_mut["TF"].astype(str).to_numpy()
    mut_q = np.vectorize(nuc_to_idx.get)(nuc_mut) # get nucleotide index
    tf_q  = np.vectorize(nuc_to_idx.get)(nuc_tf)
    
    # constant indices, idx = 5 * l + q
    idx_mut_const = site * 5 + mut_q
    idx_tf_const  = site * 5 + tf_q
    idx_mut_const = idx_mut_const.astype(int)
    idx_tf_const  = idx_tf_const.astype(int)
    
    # time-varying indices, getting from muVec
    idx_mut = muVec[site, mut_q]
    idx_tf  = muVec[site, tf_q]
    idx_mut = idx_mut.astype(int)
    idx_tf  = idx_tf.astype(int)

    # file name
    if if_linear:
        info_file = '%s/rawdata/plotinfo_%s_linear.json'%(HIV_DIR,tag)
        interp_file = '%s/rawdata/interdata_%s_linear.npz'%(HIV_DIR,tag)
    else:
        info_file = '%s/rawdata/plotinfo_%s.json'%(HIV_DIR,tag)
        interp_file = '%s/rawdata/interdata_%s.npz'%(HIV_DIR,tag)

    # --- basics information ---
    escape_group = rawdata['escape_group']
    sample_times = rawdata['sample_times']
    ne           = len(escape_group)
    if ne == 0:
        print(f'No escape group found for CH{tag[6:]}')
        data = dict(
            sample_times = sample_times.tolist(), ne = ne,
            idx_mut = idx_mut.tolist(),idx_tf = idx_tf.tolist(),
            idx_mut_const = idx_mut_const.tolist(),idx_tf_const = idx_tf_const.tolist()
        )
        
        with open(info_file, "w") as f:
            json.dump(data, f)

        return
    
    # ----- epitope frequency information ----- 
    try:
        df_escape   = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv'%(HIV_DIR,tag), memory_map=True)
    except FileNotFoundError:
        print(f'No constant escape data found for CH{tag[6:]}')
        return 

    try:
        x_sigmoid = np.load(interp_file, allow_pickle="True")['x_all'].T
    except FileNotFoundError:
        print(f'No interpolated data found for CH{tag[6:]}')
        return
    
    epitopes    = df_escape['epitope'].unique()
    if len(epitopes) != ne:
        raise ValueError(f'The number of epitopes in escape group file does not match the number of escape groups for CH{tag[6:]}')
    
    epitope_name = [] # name for epitope
    x_escape_mut = [] # frequencies for individual escape sites
    x_epitope    = [] # frequencies for escape groups
    for n in range(ne):
        df_esc  = df_escape[(df_escape.epitope==epitopes[n])]

        # get the name for epitopes
        epi_nuc = ''.join(epitopes[n])
        epitope_name.append(epi_nuc[0]+epi_nuc[-1]+str(len(epi_nuc)))

        # get frequencies for escape sites and groups
        x_epitope.append(x_sigmoid[-ne+n].tolist())
        x_escape_n = []
        for df_iter, df_entry in df_esc.iterrows():
            escape_idx = int(muVec[df_entry['polymorphic_index'], nuc_to_idx[df_entry['nucleotide']]])
            x_escape_n.append(x_sigmoid[escape_idx].tolist())
        x_escape_mut.append(x_escape_n)

    # ----- T cell response intensity -----
    try:
        patient = tag.split('-')[0]
        intensity_x = [[] for _ in range(ne)]
        intensity_y = [[] for _ in range(ne)]
        df_intensity = pd.read_csv('%s/T-cell-intensity/%s.csv'%(HIV_DIR,patient), memory_map=True)
        time_cols = df_intensity.filter(like='f_at_').columns
        df_long = df_intensity.melt(id_vars=['epitope'],value_vars=time_cols,var_name='time_point',value_name='date_value')

        df_long['time'] = df_long['time_point'].str.split('_').str[2]
        valid_data = df_long[df_long['epitope'].isin(epitopes) & df_long['date_value'].notna()]

        result = valid_data.groupby('epitope', group_keys=False).apply(lambda x: (x['time'].tolist(), x['date_value'].tolist()))
        if len(result) > 0:
            for epi, (times, dates) in result.items():
                if epi in epitopes:
                    epi_index = list(epitopes).index(epi)
                    intensity_x[epi_index] = [int(t) for t in times]
                    intensity_y[epi_index] = dates
    except FileNotFoundError:
        print(f'No T cell intensity data found for CH{tag[6:]}')
        intensity_x = []
        intensity_y = []
    
    # ----- Save the information to npz file -----
    data = dict(
        sample_times = sample_times.tolist(), ne = ne,
        x_escape_mut = x_escape_mut, x_epitope = x_epitope, epitope_name = list(epitope_name),
        idx_mut = idx_mut.tolist(),idx_tf = idx_tf.tolist(),
        idx_mut_const = idx_mut_const.tolist(),idx_tf_const = idx_tf_const.tolist(),
        intensity_x = intensity_x,intensity_y = intensity_y,
    )

    with open(info_file, "w") as f:
        json.dump(data, f)

# def cal_freq_change(tag, HIV_DIR = 'data/HIV'):
#     """
#     Calculate diff between time-varying and constant pattern
#     """

#     # calculate recombination flux term at time t
#     def get_rec_flux_at_t(r_rates, x_trait, p_mut_k, trait_dis):
#         flux = np.zeros(ne)
#         for n in range(ne):
#             fluxIn  = 0
#             fluxOut = 0

#             for nn in range(len(escape_group[n])-1):
#                 fluxIn  += trait_dis[n][nn] * (1 - x_trait[n]) *p_mut_k[n][nn][0]
#                 fluxOut += trait_dis[n][nn] * p_mut_k[n][nn][1]*p_mut_k[n][nn][2]
            
#             flux[n] = r_rates * (fluxIn - fluxOut)

#         return flux

#     def diffusion_matrix_at_t(x,xx):
#         x_length = len(x)
#         C = np.zeros([x_length,x_length])
#         for i in range(x_length):
#             C[i,i] = x[i] - x[i] * x[i]
#             for j in range(int(i+1) ,x_length):
#                 C[i,j] = xx[i,j] - x[i] * x[j]
#                 C[j,i] = xx[i,j] - x[i] * x[j]
#         return C
    
#     # calculate mutation flux term at sampled time
#     def cal_mut_flux(x,ex,muVec):
#         flux = np.zeros((len(x),x_length))
#         for t in range(len(x)):
#             for i in range(seq_length):
#                 for a in range(q):
#                     aa = int(muVec[i][a])
#                     if aa != -1:
#                         for b in range(q):
#                             bb = int(muVec[i][b])
#                             if b != a:
#                                 if bb != -1:
#                                     flux[t,aa] +=  muMatrix[b][a] * x[t,bb] - muMatrix[a][b] * x[t,aa]
#                                 else:
#                                     flux[t,aa] += -muMatrix[a][b] * x[t,aa]
#             for n in range(ne):
#                 for nn in range(len(escape_group[n])):
#                     for a in range(q):
#                         WT = escape_TF[n][nn]
#                         index = escape_group[n][nn]
#                         if a not in WT:
#                             for b in WT:
#                                 flux[t, x_length-ne+n] += muMatrix[b][a] * (1 - x[t,x_length-ne+n]) - muMatrix[a][b] * ex[t,index,a]
#         return flux

#     def get_sc_common(sc_all, muVec):
#         sc_const = np.zeros(x_length)
#         for i in range(seq_length):
#             for a in range(q):
#                 aa = int(muVec[i][a])
#                 if aa != -1:
#                     sc_const[aa] = sc_all[i*q + a]
#         for n in range(ne):
#             sc_const[x_length-ne+n] = sc_all[seq_length*q + n]
#         return sc_const

#     def insert_time(arr, allowed_gaps=(7, 8, 9, 10, 11, 12, 13)):
#         """
#         Insert values into an array, ensuring the difference between adjacent values 
#         is within the allowed_gaps range as evenly distributed as possible.
#         """
#         result = []

#         for i in range(len(arr) - 1):
#             result.append(arr[i])  # add current value
#             diff = arr[i+1] - arr[i]
            
#             if diff < max(allowed_gaps):
#                 continue

#             while diff > max(allowed_gaps):
#                 # choose the gap that is closest to 10
#                 if diff % 10 == 0:
#                     step = diff/10
#                 else:
#                     step = (diff // 10) + 1
#                 gap = min(allowed_gaps, key=lambda x: abs(x - diff / step))
#                 next_value = result[-1] + gap
#                 result.append(next_value)
#                 diff = arr[i+1] - next_value  # update the remaining difference
            
#             # check if the last gap is in the allowed_gaps
#             if diff not in allowed_gaps:
#                 print(f"Warning: the gap between {result[-1]} and {arr[i+1]} is not in the allowed_gaps range.")
            
#         # Add the last value
#         if result[-1] != arr[-1]:
#             result.append(arr[-1])

#         return np.array(result)

#     # Load data
#     rawdata  = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
#     muMatrix = np.loadtxt("%s/input/Zanini-extended.dat"%HIV_DIR)

#     # information for individual sites
#     x            = rawdata['single_freq']
#     xx           = rawdata['double_freq']
#     ex           = rawdata['escape_freq']
#     muVec        = rawdata['muVec']
#     sample_times = rawdata['sample_times']
#     seq_length   = rawdata['seq_length']
#     r_rates      = rawdata['r_rates']

#     # information for escape group
#     p_mut_k      = rawdata['p_mut_k_freq']
#     escape_group = rawdata['escape_group'].tolist()
#     escape_TF    = rawdata['escape_TF'].tolist()
#     trait_dis    = rawdata['trait_dis'].tolist()
#     ne           = len(escape_group)
#     x_length     = len(x[0])

#     # Get sc for common mutation and wild-type
#     q = len(NUC)
#     sc_const_all = np.loadtxt('%s/constant/output/sc-%s.dat'%(HIV_DIR,tag))
#     sc_const = get_sc_common(sc_const_all, muVec)

#     # check if the length for constant selection coefficient matches frequency data
#     if len(sc_const) != x_length:
#         print(f'Error: length of selection coefficient does not match frequency data for {tag}')
#         print(f'       length of sc: {len(sc_const)}, length of x: {x_length}')
#         return None
    
#     # extend the time range
#     if sample_times[-1] > 100:
#         interp_times = insert_time(sample_times)
#     else:
#         interp_times = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))

#     times_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
#     # sc_all = np.load('%s/output/sc_%s.npz'%(HIV_DIR,tag), allow_pickle=True)['selection'].T
#     sc_all = np.load('%s/output_nosp/sc_%s_try.npz'%(HIV_DIR,tag), allow_pickle=True)['selection'].T

#     # selection coefficients for interpolated time points
#     interp_index = {t: i for i, t in enumerate(interp_times)}
#     sc_tv = np.zeros((len(interp_times), x_length))
#     for ti, t in enumerate(times_all):
#         if t in interp_index:
#             sc_tv[interp_index[t]] = sc_all[ti]

#     # Use linear interpolates to get the input arrays at any integer time point
#     interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
#     interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
#     interp_ex  = interp1d(sample_times, ex, axis=0, kind='linear', bounds_error=False, fill_value=0) if ne > 0 else 0
#     interp_mut = interp1d(sample_times, p_mut_k, axis=0, kind='linear', bounds_error=False, fill_value=0) if ne > 0 else 0
#     interp_r   = interp1d(sample_times, r_rates, kind='linear', bounds_error=False, fill_value=0)
    
#     single_freq = interp_x(interp_times)
#     double_freq = interp_xx(interp_times)
#     epitope_freq = interp_ex(interp_times) if ne > 0 else 0
#     p_mut_k     = interp_mut(interp_times) if ne > 0 else 0
#     r_rate      = interp_r(interp_times)

#     # get mutation flux at sampled time points
#     flux_mut = cal_mut_flux(single_freq, epitope_freq, muVec)

#     change_tv = 0
#     change_const = 0
#     change_null = 0
#     for ti in range(len(interp_times)-1):

#         x_t, xx_t = single_freq[ti], double_freq[ti]
#         sc_t = sc_tv[ti] # Time-varying
        
#         dx = single_freq[ti+1] - single_freq[ti]
#         dt = interp_times[ti+1] - interp_times[ti]

#         # calculate C(t)
#         C_raw = diffusion_matrix_at_t(x_t, xx_t) # covariance matrix

#         # calculate flux(t) = flux_mut(t) + flux_rec(t)
#         flux_total = flux_mut[ti]
#         flux_rec = get_rec_flux_at_t(r_rate[ti], x_t[x_length-ne:], p_mut_k[ti], trait_dis) if ne > 0 else 0
#         for n in range(ne): # recombination only for binary trait part
#             flux_total[x_length-ne+n] += flux_rec[n]

#         d_const = C_raw @ sc_const + flux_total - dx/dt
#         d_tv    = C_raw @ sc_t + flux_total - dx/dt
#         d_null  = flux_total - dx/dt

#         change_const += np.sqrt(np.sum(d_const * d_const))* dt     # d_const^T d_const * dt
#         change_tv    += np.sqrt(np.sum(d_tv * d_tv))* dt           # d_tv^T C^-1 d_tv * dt
#         change_null  += np.sqrt(np.sum(d_null * d_null))* dt       # d_null^T C^-1 d_null * dt

#     actions = [change_const, change_tv, change_null]
#     min_id  = np.argmin(actions)
#     action_str = ['constant', 'time-varying', 'null'] 
#     print(f'CH{tag[-5:]}|{change_const:.4f}|{change_tv:.4f}|{change_null:.4f}|{action_str[min_id]}')

# def cal_freq_change_epitope(tag, HIV_DIR = 'data/HIV'):
#     """
#     Calculate diff between time-varying and constant pattern (only epitope)
#     """
#     # calculate recombination flux term at time t
#     def get_rec_flux_at_t(r_rates, x_trait, p_mut_k, trait_dis):
#         flux = np.zeros(ne)
#         for n in range(ne):
#             fluxIn  = 0
#             fluxOut = 0

#             for nn in range(len(escape_group[n])-1):
#                 fluxIn  += trait_dis[n][nn] * (1 - x_trait[n]) *p_mut_k[n][nn][0]
#                 fluxOut += trait_dis[n][nn] * p_mut_k[n][nn][1]*p_mut_k[n][nn][2]
            
#             flux[n] = r_rates * (fluxIn - fluxOut)

#         return flux

#     def diffusion_matrix_at_t(x,xx):
#         x_length = len(x)
#         C = np.zeros([x_length,x_length])
#         for i in range(x_length):
#             C[i,i] = x[i] - x[i] * x[i]
#             for j in range(int(i+1) ,x_length):
#                 C[i,j] = xx[i,j] - x[i] * x[j]
#                 C[j,i] = xx[i,j] - x[i] * x[j]
#         return C

#     # calculate mutation flux term at sampled time
#     def cal_mut_flux_epitope(x,ex):
#         flux = np.zeros((len(x),ne))
#         for t in range(len(x)):
#             for n in range(ne):
#                 for nn in range(len(escape_group[n])):
#                     for a in range(q):
#                         WT = escape_TF[n][nn]
#                         index = escape_group[n][nn]
#                         if a not in WT:
#                             for b in WT:
#                                 flux[t, n] += muMatrix[b][a] * (1 - x[t,n]) - muMatrix[a][b] * ex[t,index,a]
#         return flux

#     def insert_time(arr, allowed_gaps=(7, 8, 9, 10, 11, 12, 13)):
#         """
#         Insert values into an array, ensuring the difference between adjacent values 
#         is within the allowed_gaps range as evenly distributed as possible.
#         """
#         result = []

#         for i in range(len(arr) - 1):
#             result.append(arr[i])  # add current value
#             diff = arr[i+1] - arr[i]
            
#             if diff < max(allowed_gaps):
#                 continue

#             while diff > max(allowed_gaps):
#                 # choose the gap that is closest to 10
#                 if diff % 10 == 0:
#                     step = diff/10
#                 else:
#                     step = (diff // 10) + 1
#                 gap = min(allowed_gaps, key=lambda x: abs(x - diff / step))
#                 next_value = result[-1] + gap
#                 result.append(next_value)
#                 diff = arr[i+1] - next_value  # update the remaining difference
            
#             # check if the last gap is in the allowed_gaps
#             if diff not in allowed_gaps:
#                 print(f"Warning: the gap between {result[-1]} and {arr[i+1]} is not in the allowed_gaps range.")
            
#         # Add the last value
#         if result[-1] != arr[-1]:
#             result.append(arr[-1])

#         return np.array(result)
    
#     # Load data
#     rawdata  = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
#     muMatrix = np.loadtxt("%s/input/Zanini-extended.dat"%HIV_DIR)

#     # information for epitope
#     p_mut_raw    = rawdata['p_mut_k_freq']
#     escape_group = rawdata['escape_group'].tolist()
#     escape_TF    = rawdata['escape_TF'].tolist()
#     trait_dis    = rawdata['trait_dis'].tolist()
#     ne           = len(escape_group)
#     x_length     = len(rawdata['single_freq'][0])
#     if ne == 0:
#         print(f'CH{tag[-5:]}|No epitope sites, skipped')
#         return

#     # information for individual sites
#     x            = rawdata['single_freq'][:,x_length-ne:]
#     xx           = rawdata['double_freq'][:,x_length-ne:,x_length-ne:]
#     ex           = rawdata['escape_freq']
#     sample_times = rawdata['sample_times']
#     r_rates      = rawdata['r_rates']

#     # Get sc for common mutation and wild-type
#     q = len(NUC)
#     sc_const_all = np.loadtxt('%s/constant/output/sc-%s.dat'%(HIV_DIR,tag))
#     ec_const = sc_const_all[x_length - ne : x_length]

#     # check if the length for constant selection coefficient matches frequency data
#     if len(ec_const) != ne:
#         print(f'Error: length of selection coefficient does not match frequency data for {tag}')
#         print(f'       length of sc: {len(ec_const)}, length of x: {x_length}')
    
#     # extend the time range
#     if sample_times[-1] > 100:
#         interp_times = insert_time(sample_times)
#     else:
#         interp_times = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))

#     # extend the time range
#     times_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
#     sc_tv_all = np.load('%s/output/sc_%s.npz'%(HIV_DIR,tag), allow_pickle=True)['selection'].T
#     ec_tv_all = sc_tv_all[:, x_length - ne : x_length]

#     # selection coefficients for interpolated time points
#     interp_index = {t: i for i, t in enumerate(interp_times)}
#     ec_tv = np.zeros((len(interp_times), ne))
#     for ti, t in enumerate(times_all):
#         if t in interp_index:
#             ec_tv[interp_index[t]] = ec_tv_all[ti]

#     # Use linear interpolates to get the input arrays at any integer time point
#     interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
#     interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
#     interp_ex  = interp1d(sample_times, ex, axis=0, kind='linear', bounds_error=False, fill_value=0) if ne > 0 else 0
#     interp_mut = interp1d(sample_times, p_mut_raw, axis=0, kind='linear', bounds_error=False, fill_value=0) if ne > 0 else 0
#     interp_r   = interp1d(sample_times, r_rates, kind='linear', bounds_error=False, fill_value=0)

#     single_freq = interp_x(interp_times)
#     double_freq = interp_xx(interp_times)
#     epitope_freq = interp_ex(interp_times) if ne > 0 else 0
#     p_mut_k     = interp_mut(interp_times) if ne > 0 else 0
#     r_rate      = interp_r(interp_times)

#     # get mutation flux at sampled time points
#     flux_mut = cal_mut_flux_epitope(single_freq, epitope_freq)

#     # Calculate changes for different patterns
#     change_tv = 0
#     change_const = 0
#     change_null = 0

#     for ti in range(len(interp_times)-1):

#         x_t, xx_t = single_freq[ti], double_freq[ti]
        
#         dx = single_freq[ti+1] - single_freq[ti]
#         dt = times_all[ti+1] - times_all[ti]

#         # calculate C(t)
#         C_raw = diffusion_matrix_at_t(x_t, xx_t) # covariance matrix

#         # calculate flux(t) = flux_mut(t) + flux_rec(t)
#         flux_rec   = get_rec_flux_at_t(r_rate[ti], x_t, p_mut_k[ti], trait_dis) if ne > 0 else 0
#         flux_total = flux_mut[ti] + flux_rec

#         # use constant sc for individual mutation
#         # print(ec_const, ec_tv[ti])
#         d_const = C_raw @ ec_const + flux_total - dx/dt
#         d_tv    = C_raw @ ec_tv[ti] + flux_total - dx/dt
#         d_null  = flux_total - dx/dt

#         dx2_const = np.sqrt(np.sum(d_const * d_const))      # d_const^T d_const * dt
#         dx2_tv    = np.sqrt(np.sum(d_tv * d_tv))            # d_tv^T C^-1 d_tv * dt
#         dx2_null  = np.sqrt(np.sum(d_null * d_null))        # d_null^T C^-1 d_null * dt

#         change_const += dx2_const * dt
#         change_tv    += dx2_tv * dt
#         change_null  += dx2_null * dt

#     actions = [change_const, change_tv, change_null]
#     min_id  = np.argmin(actions)
#     action_str = ['constant', 'time-varying', 'null']
#     print(f'CH{tag[-5:]}|{change_const:.4f}|{change_tv:.4f}|{change_null:.4f}|{action_str[min_id]}')

# def cal_diff_and_action(tag, sc_file, if_const=False, alpha=1e-3, HIV_DIR = 'data/HIV'):
    
#     # Get sc for common mutation and wild-type
#     q = len(NUC)
#     # Load data
#     rawdata  = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
#     interp_data  = np.load('%s/rawdata/interdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)

#     # information for individual sites
#     seq_length = rawdata['seq_length']
#     muVec      = rawdata['muVec']
#     ne         = len(rawdata['escape_group'])

#     C_all = interp_data['C_all']
#     b_all = interp_data['b_all']
#     sample_times = interp_data['sample_times']
#     ExTimes      = interp_data['ExTimes']

#     x_length = C_all.shape[1]

#     times_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
#     times_used = [int(t) for t in ExTimes if t >= sample_times[0] and t <= sample_times[-1]]

#     time_index = {t: i for i, t in enumerate(times_all)}

#     if if_const:
#         if 'dat' in sc_file:
#             sc_const_raw = np.loadtxt('%s/%s'%(HIV_DIR, sc_file))
#             sc_const = np.zeros(x_length)
#             for i in range(seq_length):
#                 for a in range(q):
#                     idx = int(muVec[i, a])
#                     if idx != -1:
#                         sc_const[idx] = sc_const_raw[i*q+a]
#             for n in range(ne):
#                 sc_const[x_length - ne + n] = sc_const_raw[seq_length*q + n]
#         else:
#             sc_const_all = np.load('%s/%s'%(HIV_DIR, sc_file), allow_pickle=True)['selection']
#             sc_const = np.mean(sc_const_all, axis=1)
#     else:
#         if 'CH' in sc_file:
#             sc_all = np.load('%s/%s'%(HIV_DIR, sc_file), allow_pickle=True)['sc_all'].T
#         else:
#             sc_all = np.load('%s/%s'%(HIV_DIR, sc_file), allow_pickle=True)['selection'].T
        
#     diff_int = 0
#     action_int = 0
#     diff_all = np.zeros(len(times_used)-1)
#     action_all = np.zeros(len(times_used)-1)
#     for ti in range(len(times_used)-1):
        
#         t_index = time_index[times_used[ti]]
#         if if_const:
#             sc_t = sc_const
#         else:
#             sc_t = sc_all[t_index] # Time-varying selection coefficient at time ti
        
#         dt = times_used[ti+1] - times_used[ti]
#         C_t  = C_all[t_index]
        
#         eps_t = alpha * (np.trace(C_t)/x_length)
#         C_eps = C_t + eps_t * np.eye(x_length)

#         diff_vector = C_t @ sc_t + b_all[t_index] 
#         diff = np.sqrt(np.sum(diff_vector * diff_vector))
#         diff_all[ti] = diff
#         diff_int += diff * dt     # d^T d * dt

#         w, Q = np.linalg.eigh(C_t)  # ascending
#         a = Q.T @ diff_vector
#         tau = alpha * w[-1]

#         keep = w > tau
#         action_keep = np.sum((a[keep]**2) / w[keep])
#         action_all[ti] = action_keep
#         action_int += action_keep * dt

#         # cho = la.cho_factor(C_eps, lower=True)
#         # Cr = la.cho_solve(cho, diff_vector)
#         # action_t = diff_vector @ Cr 
#         # action_int += action_t * dt

#         # print(f'time {times_used[ti]:.1f}, diff={diff:.4f}, action_keep={action_keep:.6f}')
        
#     print(f"Diff_int: {diff_int}, Action_int: {action_int}")

#     return diff_all, action_all, times_used[:-1]

def cal_diff(tag, sc_file, if_const=False, if_linear = False, HIV_DIR = 'data/HIV'):
    
    # Get sc for common mutation and wild-type
    q = len(NUC)
    # Load data
    rawdata  = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
    if if_linear:
        interp_data  = np.load('%s/rawdata/interdata_%s_linear.npz'%(HIV_DIR,tag), allow_pickle=True)
    else:
        interp_data  = np.load('%s/rawdata/interdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)

    # information for individual sites
    seq_length = rawdata['seq_length']
    muVec      = rawdata['muVec']
    ne         = len(rawdata['escape_group'])

    C_all = interp_data['C_all']
    flux_all = interp_data['flux_all']
    dxdt     = interp_data['dxdt']
    dxdt_node = interp_data['dxdt_node']
    sample_times = interp_data['sample_times']
    ExTimes      = interp_data['ExTimes']

    x_length = C_all.shape[1]
    times_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
    times_used = [int(t) for t in ExTimes if t >= sample_times[0] and t <= sample_times[-1]]
    time_index = {t: i for i, t in enumerate(times_all)} # sc used all time points

    # get selection coefficients
    if if_const:
        if 'dat' in sc_file:
            sc_const_raw = np.loadtxt('%s/%s'%(HIV_DIR, sc_file))
            sc_const = np.zeros(x_length)
            for i in range(seq_length):
                for a in range(q):
                    idx = int(muVec[i, a])
                    if idx != -1:
                        sc_const[idx] = sc_const_raw[i*q+a]
            for n in range(ne):
                sc_const[x_length - ne + n] = sc_const_raw[seq_length*q + n]
        else:
            sc_const_all = np.load('%s/%s'%(HIV_DIR, sc_file), allow_pickle=True)['selection']
            sc_const = np.mean(sc_const_all, axis=1)
    else:
        sc_all = np.load('%s/%s'%(HIV_DIR, sc_file), allow_pickle=True)['sc_all'].T
    
    # get dxdt_used at times_used
    m = np.searchsorted(sample_times, times_used, side="left")
    ks = np.clip(m - 1, 0, len(sample_times) - 2)
    dxdt_used = dxdt[ks].copy()
    on_node = (m < len(sample_times)) & (sample_times[m] == times_used)
    if on_node.any(): # optional: node override when t_in equals sample_times exactly
        dxdt_used[on_node] = dxdt_node[m[on_node]]

    diff_int = 0
    # diff_all = np.zeros(len(times_used)-1)
    for ti in range(len(times_used)-1):
        
        time = times_used[ti]
        t_index = time_index[time]
        if if_const:
            sc_t = sc_const
        else:
            sc_t = sc_all[t_index] # Time-varying selection coefficient at time ti
        
        dt = times_used[ti+1] - time
        C_t  = C_all[t_index]

        diff_vector = C_t @ sc_t + flux_all[t_index] - dxdt_used[ti]
        diff = np.sum(diff_vector * diff_vector)
        diff_int += diff * dt     # d^T d * dt
        # diff_all[ti] = diff
        # print(f'time {times_used[ti]:.1f}, diff={diff:.4f}')

    return diff_int#, diff_all

def cal_action(tag, sc_file, if_const=False, if_linear = False, eps=1e-3, HIV_DIR = 'data/HIV'):
    
    # Get sc for common mutation and wild-type
    q = len(NUC)
    # Load data
    rawdata  = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)
    if if_linear:
        interp_data  = np.load('%s/rawdata/interdata_%s_linear.npz'%(HIV_DIR,tag), allow_pickle=True)
    else:
        interp_data  = np.load('%s/rawdata/interdata_%s.npz'%(HIV_DIR,tag), allow_pickle=True)

    # information for individual sites
    seq_length = rawdata['seq_length']
    muVec      = rawdata['muVec']
    ne         = len(rawdata['escape_group'])

    C_all = interp_data['C_all']
    flux_all = interp_data['flux_all']
    dxdt     = interp_data['dxdt']
    dxdt_node = interp_data['dxdt_node']
    sample_times = interp_data['sample_times']
    ExTimes      = interp_data['ExTimes']

    x_length = C_all.shape[1]
    times_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
    times_used = times_all
    # times_used = [int(t) for t in ExTimes if t >= sample_times[0] and t <= sample_times[-1]]
    # time_index = {t: i for i, t in enumerate(times_all)} # sc used all time points

    alpha = 1e-3
    # get selection coefficients
    if if_const:
        if 'dat' in sc_file:
            sc_const_raw = np.loadtxt('%s/%s'%(HIV_DIR, sc_file))
            sc_const = np.zeros(x_length)
            for i in range(seq_length):
                for a in range(q):
                    idx = int(muVec[i, a])
                    if idx != -1:
                        sc_const[idx] = sc_const_raw[i*q+a]
            for n in range(ne):
                sc_const[x_length - ne + n] = sc_const_raw[seq_length*q + n]
        else:
            sc_const_all = np.load('%s/%s'%(HIV_DIR, sc_file), allow_pickle=True)['selection']
            sc_const = np.mean(sc_const_all, axis=1)
    else:
        sc_all = np.load('%s/%s'%(HIV_DIR, sc_file), allow_pickle=True)['sc_all'].T
    
    # get dxdt_used at times_used
    m = np.searchsorted(sample_times, times_used, side="left")
    ks = np.clip(m - 1, 0, len(sample_times) - 2)
    dxdt_used = dxdt[ks].copy()
    on_node = (m < len(sample_times)) & (sample_times[m] == times_used)
    if on_node.any(): # optional: node override when t_in equals sample_times exactly
        dxdt_used[on_node] = dxdt_node[m[on_node]]

    action_int = 0
    # diff_all = np.zeros(len(times_used)-1)
    for ti in range(len(times_used)-1):
        
        time = times_used[ti]
        t_index = ti # time_index[time]
        if if_const:
            sc_t = sc_const
        else:
            sc_t = sc_all[t_index] # Time-varying selection coefficient at time ti
        
        dt = times_used[ti+1] - time
        C_t  = C_all[t_index]
        C_eps   = C_t + eps * np.eye(C_t.shape[0])

        diff = C_t @ sc_t + flux_all[t_index] - dxdt_used[ti]

        action_int += (diff @ np.linalg.solve(C_eps, diff)) * dt # d_const^T C^-1 d_const * dt

    return action_int

def modify_seq(tag, HIV_DIR='data/HIV'):
    '''
    Remove one time pooint for each patient to calculate LOO-CV
    '''

    seq = np.loadtxt('%s/input/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    L   = len(seq[0])-2
    time_points = np.unique(seq[:,0])

    for time in time_points:
        with open('%s/input/sequence/cut/%s-%d-poly-seq2state.dat'%(HIV_DIR,tag,time), "w") as f:
            for k in range(len(seq)):
                if seq[k][0] != time:
                    f.write('%s\n'%'\t'.join([str(int(i)) for i in seq[k]]))

    return time_points

def modify_r(tag, time_points):
    '''
    Modify recombination rates file for LOO-CV
    '''

    r_rates = np.loadtxt('data/HIV/input/r_rates/r-%s.dat'%tag)

    for i in range(len(time_points)):
        r_cut = [r_rates[j] for j in range(len(r_rates)) if j != i]
        np.savetxt('data/HIV/input/r_rates/cut/r-%s-%d.dat'%(tag,time_points[i]), r_cut)
        
    return

def output_const_vs_tv(tag, base_csv, sc_const_file, sc_tv_file, out_csv, HIV_DIR='data/HIV'):
    ''' 
    output the average value for time-varying selection coefficient into csv file
    '''
    def make_epitope_tv(epitope):
        if pd.isna(epitope):
            return ""

        epitope = str(epitope)
        epi_name = epitope[0] + epitope[-1] + str(len(epitope))

        return epi_name if epi_name in epitope_infer else ""
    
    # load raw csv file
    base_cols = ["polymorphic_index", "HXB2_index", "nucleotide", "TF", "epitope", "escape"]
    df = pd.read_csv(base_csv, usecols=base_cols)
    df_mut = df[df['TF'] != df['nucleotide']].copy()
    
   # ----- read data from json file ----- 
    info_file = '%s/rawdata/plotinfo_%s.json'%(HIV_DIR,tag)
    try:
        with open(info_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'CH{tag[-5:]} has no plot info json file')
        return

    idx_mut = data['idx_mut']
    idx_tf  = data['idx_tf']
    idx_mut_const = data['idx_mut_const']
    idx_tf_const  = data['idx_tf_const']
    epitope_infer =  set(data['epitope_name']) 

    # add epitope_tv
    df_mut["epitope_tv"] = df_mut["epitope"].apply(make_epitope_tv)
    
    # load constant selection coefficients
    data_const = np.loadtxt(sc_const_file)
    sc_const = data_const[idx_mut_const] - data_const[idx_tf_const]
    df_mut["sc_const"] = sc_const

    # load selection coefficients
    data_tc  = np.load(sc_tv_file, allow_pickle="True")
    sc_all   = data_tc['sc_all']# time range:times

    diff = sc_all[idx_mut, :] - sc_all[idx_tf, :]   # shape: (n_rows, T)
    avg = diff.mean(axis=1)
    std = diff.std(axis=1)

    # add to dataframe
    df_mut["sc_avg"] = avg
    df_mut["sc_std"] = std

    df_mut.to_csv(out_csv, index=False, float_format="%.6e")

def output_tv_results(tag, base_csv, sc_const_file, sc_tv_file, out_csv, HIV_DIR='data/HIV'):
    ''' 
    output the average value for time-varying selection coefficient into csv file
    '''
    def to_epi_name(epi):
        if epi is None or (isinstance(epi, float) and np.isnan(epi)):
            raise ValueError("Found NaN/None in column 'epitope'.")
        epi = str(epi)
        return f"{epi[0]}{epi[-1]}{len(epi)}"

    # load csv file including special sites information
    base_cols = ["polymorphic_index", "HXB2_index", "nucleotide", "TF", "epitope"]
    df_epitope = pd.read_csv(base_csv, usecols=base_cols)

    # add 2 columns 
    df_epitope['tc_const'] = np.nan
    df_epitope['tc_avg'] = np.nan

    info_file = '%s/rawdata/plotinfo_%s.json'%(HIV_DIR,tag)
    try:
        with open(info_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'CH{tag[-5:]} has no plot info json file')

    ne = data['ne']
    epitope_name = data['epitope_name']
    epi2idx = {name: i for i, name in enumerate(epitope_name)}

    # load constant selection coefficients
    data_const = np.loadtxt(sc_const_file)
    ec_const = data_const[-ne:] # only escape sites

    # load selection coefficients
    data_tc  = np.load(sc_tv_file, allow_pickle=True)
    sc_all = data_tc["sc_all"]
    ec_all = sc_all[-ne:, :]
    ec_all_avg = ec_all.mean(axis=1)

    epi_series = df_epitope["epitope"].map(to_epi_name)
    epi_idx = epi_series.map(epi2idx)
    if epi_idx.isna().any(): # if there is any epitope not in epitope_name, raise error
        missing_names = epi_series[epi_idx.isna()].unique().tolist()
        raise ValueError(
            f"epi_name not found in epitope_name list: {missing_names} for CH{tag[-5:]}. "
        )

    epi_idx = epi_idx.to_numpy(dtype=int)
    out_index = df_epitope.index

    df_epitope.loc[out_index, "tc_const"] = ec_const[epi_idx]
    df_epitope.loc[out_index, "tc_avg"] = ec_all_avg[epi_idx]

    df_epitope.to_csv(out_csv, index=False, float_format="%.6e")

def read_dat_loadtxt(path: str):
    """
    each row: t count seq...
    return: dict[t][seq]=count
    """
    arr = np.loadtxt(path, dtype=int)

    t = arr[:,0]
    c = arr[:,1]
    seq = arr[:,2:]

    data = defaultdict(lambda: defaultdict(int))

    for ti, ci, si in zip(t, c, seq):
        data[int(ti)][tuple(si.tolist())] += int(ci)
    return data

def write_dat(path: str, data):
    """
    data: dict[t][seq]=count
    """
    with open(path, "w") as f:
        for t in sorted(data.keys()):
            for seq_key, val in data[t].items():
                if val <= 0:
                    continue
                seq_str = " ".join(map(str, seq_key))  # tuple -> "0 1 0 ..."
                if isinstance(val, float):
                    f.write(f"{t} {val:.6f} {seq_str}\n")
                else:
                    f.write(f"{t} {val} {seq_str}\n")
               

def counts_dat_to_freq_dat(raw_dat: str, out_dat: str,
                           seq_DIR: str = 'data/HIV/input/sequence'):

    raw_path = '%s/%s.dat' % (seq_DIR, raw_dat)
    out_path = '%s/%s.dat' % (seq_DIR, out_dat)
    # read raw dat file and store counts in a dict
    data = read_dat_loadtxt(raw_path)

    data_freq = defaultdict(dict)
    for t in sorted(data.keys()):
        total_count = sum(data[t].values())
        for seq, c in data[t].items():
            if c > 0:
                data_freq[t][seq] = c / total_count
            else:
                raise ValueError(f"Count for time {t} and sequence {seq} is {c} <= 0, cannot write to file")
            
    # output dat files, using frequency instead of count
    with open(out_path, "w") as f:
        for t in sorted(data.keys()):
            total_count = sum(data[t].values())
            for seq_key, val in data[t].items():
                if val <= 0:
                    continue
                seq_str = " ".join(map(str, seq_key))  # tuple -> "0 1 0 ..."
                f.write(f"{t} {val/total_count:.6f} {seq_str}\n")

def sigmoid_interp_weights(steps: int, k: float = 10.0) -> np.ndarray:
    """
    Return s_j for j=0..steps, where s maps [0,1] -> [0,1] with sigmoid shape.
    """
    u = np.linspace(0.0, 1.0, steps + 1)
    s = 1.0 / (1.0 + np.exp(-k * (u - 0.5)))
    s = (s - s.min()) / (s.max() - s.min() + 1e-15)
    return s  # length steps+1

def sigmoid_freq_dat(seq_dat, out_dat, seq_DIR = 'data/HIV/input/sequence'):
    def write_time_slice(fh, t: int, slice_dict: dict, renormalized=False):
        if renormalized:
            ssum = sum(slice_dict.values())
            if ssum > 0:
                inv = 1.0 / ssum
                for key in list(slice_dict.keys()):
                    slice_dict[key] *= inv

        for seq_key, val in slice_dict.items():
            if val <= 0:
                continue
            seq_str = " ".join(map(str, seq_key))
            fh.write(f"{t} {val:.6f} {seq_str}\n")

    input_path  = '%s/%s.dat' % (seq_DIR, seq_dat)
    output_path = '%s/%s.dat' % (seq_DIR, out_dat)

    data = read_dat_loadtxt(input_path)
    times = sorted(data.keys())
    data_freq = defaultdict(dict)
    for t in times:
        total_count = sum(data[t].values())
        for seq, c in data[t].items():
            if c > 0:
                freq = c / total_count
                data_freq[t][seq] = freq
            else:
                raise ValueError(f"Count for time {t} and sequence {seq} is {c} <= 0, cannot write to file")

    with open(output_path, "w") as fh:
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            steps = t1 - t0

            seqs0 = data_freq[t0].keys()
            seqs1 = data_freq[t1].keys()
            union_seqs = set(seqs0) | set(seqs1)

            s = sigmoid_interp_weights(steps)

            for j in range(steps): # last time points are included
                tt = t0 + j
                sj = float(s[j])
                seq_freqs = {}
                # interpolate frequency for each sequence
                for seq in union_seqs:
                    y0 = data_freq[t0].get(seq, 0.0)
                    y1 = data_freq[t1].get(seq, 0.0)
                    freq = y0 + (y1 - y0) * sj
                    if freq > 0:
                        seq_freqs[seq] = freq
                # output the interpolated results
                write_time_slice(fh, tt, seq_freqs)

        # output for the last time point
        write_time_slice(fh, times[-1], dict(data_freq[times[-1]]))