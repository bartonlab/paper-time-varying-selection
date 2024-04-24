# LIBRARIES
import os
import sys
import numpy as np
import pandas as pd
import re
import urllib.request
from math import isnan
import statistics

NUC = ['-', 'A', 'C', 'G', 'T']
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

def find_trait_site(tag,min_n,HIV_DIR):
    '''
    find trait sites and corresponding TF sequence and save the information into new csv file
    '''

    """Load the set of epitopes targeted by patients"""
    df_poly  = pd.read_csv('%s/constant/interim/%s-poly.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    df_epi   = pd.read_csv('%s/constant/epitopes.csv'%HIV_DIR, comment='#', memory_map=True)

    df_index = pd.read_csv('%s/constant/processed/%s-index.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
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
    seq = np.loadtxt('%s/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
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
    columns_to_remove = ['exposed', 'edge_gap','flanking','glycan']
    df_poly = df_poly.drop(columns=columns_to_remove)

    df_poly.insert(4, 'escape', escape_values)
    df_poly.to_csv('%s/constant/interim/%s-escape.csv' %(HIV_DIR,tag), index=False,na_rep='nan')
    
    """get all epitopes for one tag"""
    df_poly = pd.read_csv('%s/constant/interim/%s-escape.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    df_rows = df_poly[df_poly['epitope'].notna()]
    unique_epitopes = df_rows['epitope'].unique()

    "store the information for trait sites into files (trait site and TF trait sequences)"
    f = open('%s/input/traitsite/traitsite-%s.dat'%(HIV_DIR,tag), 'w')
    g = open('%s/input/traitseq/traitseq-%s.dat'%(HIV_DIR,tag), 'w')

    trait_all   = []
    trait_all_i = []

    for epi in unique_epitopes:

        trait_sites = []
        df_e = df_rows[(df_rows['epitope'] == epi) & (df_rows['escape'] == True)] # find all escape mutation for one epitope
        trait_sites = df_e['polymorphic_index'].unique()

        if len(trait_sites) > min_n:
            f.write('%s\n'%'\t'.join([str(i) for i in trait_sites]))
            trait_all.append(trait_sites)
            TF_seq  = []
            for j in range(len(trait_sites)):
                n_poly  = df_e[df_e['polymorphic_index'] == trait_sites[j]]
                TF      = n_poly.iloc[0].TF
                TF_seq.append(NUC.index(TF))
            g.write('%s\n'%'\t'.join([str(i) for i in TF_seq]))
        elif len(trait_sites) > 0 :
            trait_all_i.append(trait_sites)
    f.close()
    g.close()


def analyze_result_short(tag,HIV_DIR):
    '''
    collect data and then write into csv file
    '''

    def get_xp(seq,traitsite,polyseq):
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
                    poly_value = sum([abs(seq_t[n][int(traitsite[i][j])]-polyseq[i][j]) for j in range(len(traitsite[i]))])
                    if poly_value > 0:
                        num += 1
                xp[i,t] = num/counts

        return xp

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

    seq     = np.loadtxt('%s/sequence/%s-poly-seq2state.dat'%(HIV_DIR,tag))
    L       = len(seq[0])-2    #the number of polymorphic sites

    # sc      = np.loadtxt('%s/constant/output/sc-%s.dat'%(HIV_DIR,tag))
    # sc_old  = np.loadtxt('%s/constant/notrait/sc-%s.dat'%(HIV_DIR,tag))
    
    try:
        traitsite = read_file(HIV_DIR,'traitsite/traitsite-%s.dat'%(tag))
    except:
        traitsite = []
        print(f'{tag} does not have traitsite file')

    trait_sites = []
    for i in range(len(traitsite)):
        for j in range(len(traitsite[i])):
            trait_sites.append(traitsite[i][j])

    df_poly = pd.read_csv('%s/constant/interim/%s-escape.csv' %(HIV_DIR,tag), comment='#', memory_map=True)

    index_cols  = ['polymorphic_index', 'alignment_index', 'HXB2_index','nonsynonymous','escape','nucleotide',]
    index_cols += ['TF','consensus','epitope','exposed','edge_gap','flanking','s_MPL','s_SL']
    cols = [i for i in list(df_poly) if i not in index_cols]
    times = [int(cols[i].split('_')[-1]) for i in range(len(cols))]

    f = open(HIV_DIR+'/constant/analysis/'+tag+'-analyze.csv','w')
    # f.write('polymorphic_index,alignment,HXB2_index,nucleotide,TF,consensus,epitope,escape,sc_old,sc_MPL,tc_MPL')
    f.write('polymorphic_index,alignment,HXB2_index,nucleotide,TF,consensus,epitope,escape')
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
        # nuc_index  = NUC.index(nucleotide)+polymorphic_index*5
        # TF_index   = NUC.index(TF)+polymorphic_index*5
        # sc_MPL     = sc[nuc_index]-sc[TF_index]
        # sc_mpl_old = sc_old[nuc_index]-sc_old[TF_index]
        # tc_MPL     = 'nan'
        df_i       = df_poly.iloc[ii]
        # if sc_MPL != 0:
        #     for i in range(len(traitsite)):
        #         if polymorphic_index in traitsite[i]:
        #             tc_MPL = sc[i+L*5]
        f.write('%d,%d,%s,%s,' % (polymorphic_index, alignment, HXB2_index, nucleotide))
        f.write('%s,%s,%s,%s' % (TF, consensus, epitope, escape))
        # f.write('%s,%s,%s,%s,%f,%f,%s' % (TF, consensus, epitope, escape, sc_mpl_old, sc_MPL,tc_MPL))
        f.write(',%s' % (','.join([str(df_i[c]) for c in cols])))
        f.write('\n')
    f.close()

    if len(traitsite) != 0:
        df = pd.read_csv('%s/constant/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
        
        index_cols = ['polymorphic_index', 'alignment']
        cols = [i for i in list(df) if i not in index_cols]

        if tag != '700010077-3' and tag != '705010162-3':
            polyseq  = read_file(HIV_DIR,'traitseq/traitseq-'+tag+'.dat')
            xp = get_xp(seq,traitsite,polyseq)
        else:
            polyseq  = read_file_s(HIV_DIR,'traitseq/traitallele-'+tag+'.dat')
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

def analyze_result(HIV_DIR,output,tag):
    df_poly = pd.read_csv('%s/constant/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    cols = [i for i in df_poly.columns if 'f_at_' in i]
    
    # get selection coefficient
    data_pro     = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    muVec        = data_pro['muVec']
    time_step    = data_pro['time_step']
    p_sites      = data_pro['special_sites']

    data_tc      = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,output,tag,time_step), allow_pickle="True")
    sc_tv_all    = data_tc['selection']# time range:times

    f = open('%s/%s/%s-tv.csv'%(HIV_DIR,output,tag),'w')
    f.write('polymorphic_index,HXB2_index,nucleotide,TF,consensus,epitope,escape,sc_c,sc_tv,sc_sigma')
    f.write(',%s' % (','.join(cols)))
    f.write('\n')

    for ii in range(len(df_poly)):
        site_index = df_poly.iloc[ii].polymorphic_index
        HXB2_index = df_poly.iloc[ii].HXB2_index
        nucleotide = df_poly.iloc[ii].nucleotide
        TF         = df_poly.iloc[ii].TF
        consensus  = df_poly.iloc[ii].consensus
        epitope    = df_poly.iloc[ii].epitope
        escape     = df_poly.iloc[ii].escape
        sc_c       = df_poly.iloc[ii].sc_MPL
        sc_sigma   = 'nan' 

        if nucleotide != TF:
            if site_index not in p_sites:
                index_mu   = muVec[site_index,NUC.index(nucleotide)]
                index_TF   = muVec[site_index,NUC.index(TF)]

                sc_tv      = sc_tv_all[int(index_mu)] - sc_tv_all[int(index_TF)]
                sc_mean    = np.average(sc_tv)
                sc_sigma   = statistics.stdev(sc_tv)
            else:
                sc_mean    = np.average(sc_tv)
                sc_sigma   = 'time varying'
        else:
            sc_mean    = 0
            sc_sigma   = 0

        f.write('%d,%s,%s,%s,' % (site_index, HXB2_index, nucleotide, TF))
        f.write('%s,%s,%s,%f,%s,%s,' % (consensus, epitope, escape, sc_c, sc_mean, sc_sigma))
        f.write('%s' % (','.join([str(df_poly.iloc[ii][c]) for c in cols])))
        f.write('\n')

    f.close()

def analyze_epitope(HIV_DIR,output,tag):
    df_epi = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    cols = [i for i in list(df_epi)]
    cols_time = [i for i in cols if 'sc_at_' in i]
    times = [int(cols[i].split('_')[-1]) for i in range(len(cols_time))]

    # get selection coefficient
    data_pro     = np.load('%s/rawdata/rawdata_%s.npz'%(HIV_DIR,tag), allow_pickle="True")
    time_step    = data_pro['time_step']

    data_tc      = np.load('%s/%s/c_%s_%d.npz'%(HIV_DIR,output,tag,time_step), allow_pickle="True")
    sc_tv_all    = data_tc['selection']# time range:times

    g = open('%s/%s/escape_group-%s.csv'%(HIV_DIR,output,tag),'w')
    g.write('%s' % (','.join(cols)))
    for t in range(len(times)):
        g.write(',sc_at_%s'%times[t])
    g.write('\n')

    epitopes = df_epi['epitope'].unique()
    for n in range(len(epitopes)):
        sc_tv_n  = sc_tv_all[-(len(epitopes)-n)]
        df_epi_n = df_epi[df_epi['epitope'] == epitopes[n]]
        for ii in range(len(df_epi_n)):
            g.write('%s' % (','.join([str(df_epi.iloc[ii][c]) for c in cols])))
            for t in range(len(times)):
                g.write(',%f'%sc_tv_n[t])
            g.write('\n')

def get_cut_sequences(HIV_DIR,tag,cut_s,cut_p):
    '''modify the sequence by removing the sites that has a weak linakge with the epitopes'''
    
    # get special sites and escape sites
    df_info = pd.read_csv('%s/constant/analysis/%s-analyze.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    df_rows = df_info[df_info['epitope'].notna()]
    unique_epitopes = df_rows['epitope'].unique()
    min_n = 2 # the least escape sites a trait group should have (more than min_n)

    special_sites = [] # special site considered as time-varying site but not escape site
    escape_group  = [] # escape group (each group should have more than 2 escape sites)
    for epi in unique_epitopes:
        df_e = df_rows[(df_rows['epitope'] == epi) & (df_rows['escape'] == True)] # find all escape mutation for one epitope
        unique_sites = df_e['polymorphic_index'].unique()

        if len(unique_sites) <= min_n:
            special_sites.append(unique_sites)
        else:
            escape_group.append(list(unique_sites))

    escape_sites = [item for sublist in escape_group for item in sublist]
    special_sites = [item for sublist in special_sites for item in sublist]
    
    # get the raw sequence
    sequence = np.loadtxt("%s/sequence/%s-poly-seq2state.dat" %(HIV_DIR,tag))
    seq_length = len(sequence[0])-2
    ne          = len(escape_group)
   
    # load the sij data to get the likage between sites
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
        df_nq = df_info[(df_info['polymorphic_index'] == escape_group[n][0]) & (df_info['nucleotide'] != df_info['TF'])]
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

    print('raw sequence length is %d, remove %d sites'%(seq_length,len(cut_sites)),end='')
    
    cut_sites = list(cut_sites)

    # write the cut sequence into a new file
    f = open('%s/sequence/%s-cut.dat'%(HIV_DIR,tag), 'w')
    for i in range(len(sequence)):
        sequence_i = [sequence[i][j+2] for j in range(len(sequence[i])-2) if j not in cut_sites]
        f.write('%d\t%d\t' % (sequence[i][0],sequence[i][1]))
        f.write(' %s' % (' '.join(str(int(num)) for num in sequence_i)))
        f.write('\n')
    f.close()

    # write the information for remainning sites into a new analysis file
    remove_cols = ['polymorphic_index','sc_old','sc_MPL','tc_MPL']
    cols = [i for i in list(df_info) if i not in remove_cols]
    g = open('%s/constant/analysis/%s-analyze-cut.csv'%(HIV_DIR,tag),'w')
    g.write('polymorphic_index,')
    g.write('%s\n' % (','.join(cols)))
    index = 0
    for ii in range(seq_length):
        if ii not in cut_sites:
            df_ii = df_info[df_info['polymorphic_index'] == ii]
            for jj in range(len(df_ii)):
                g.write('%d,'%index)
                g.write('%s\n' % (','.join([str(df_ii.iloc[jj][c]) for c in cols])))
            index += 1
    g.close()

    # write the information for remainning sites into a new epitope file
    df_info_cut = pd.read_csv('%s/constant/analysis/%s-analyze-cut.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    df_epitope = pd.read_csv('%s/constant/epitopes/escape_group-%s.csv' %(HIV_DIR,tag), comment='#', memory_map=True)
    cols = [i for i in list(df_epitope) if i not in remove_cols]
    g = open('%s/constant/epitopes/escape_group-%s-cut.csv'%(HIV_DIR,tag),'w')
    g.write('polymorphic_index,')
    g.write('%s\n' % (','.join(cols)))
    for ii in range((len(df_epitope))):
        HXB2_index = str(df_epitope.iloc[ii].HXB2_index)
        df_i_cut   = df_info_cut[df_info_cut['HXB2_index'] == HXB2_index]
        index      = df_i_cut.iloc[0].polymorphic_index
        g.write('%d,'%index)
        g.write('%s\n' % (','.join([str(df_epitope.iloc[ii][c]) for c in cols])))
    g.close()

    print(f', %d variants'%len(df_info_cut))