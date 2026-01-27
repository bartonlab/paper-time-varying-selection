import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import re
import os
import scipy as sp
from scipy import integrate
import scipy.interpolate as sp_interpolate
from scipy.interpolate import interp1d
import statistics
import time as time_module

# GitHub
SIM_DIR = 'data/simulation'
HIV_DIR = 'data/HIV'
FIG_DIR = 'figures'

# global variables
binary_nuc = ['A','T']  # binary case

def read_file(name):
    result = []
    with open('%s/%s'%(SIM_DIR,name), 'r') as file:
        for line in file:
            line_data = []
            for item in line.split():
                line_data.append(int(item))
            result.append(line_data)
    return result

def get_recombination(genotype1,genotype2,seq_length):
    #choose one possible mutation site
    recombination_point = np.random.randint(seq_length-1) + 1
    # get two offspring genotypes
    genotype_off = genotype1[:recombination_point] + genotype2[recombination_point:]
    return genotype_off

def recombination_event(genotype,genotype_ran,pop,seq_length):
    if pop[genotype] > 1:
        pop[genotype] -= 1
        if pop[genotype] == 0:
            del pop[genotype]

        new_genotype = get_recombination(genotype,genotype_ran,seq_length)
        if new_genotype in pop:
            pop[new_genotype] += 1
        else:
            pop[new_genotype] = 1
    return pop

# create all recombinations that occur in a single generation
def recombination_step(pop,rec_rate,seq_length):
    genotypes = list(pop.keys())
    numbers = list(pop.values())
    weights = [float(n) / sum(numbers) for n in numbers]
    for genotype in genotypes:
        n = pop[genotype]
        # calculate the likelihood to recombine
        # recombination rate per locus r,  P = (1 - (1-r)^(L - 1)) = r(L-1)
        total_rec = rec_rate*(seq_length - 1)
        nflux_rec = np.random.binomial(n, total_rec)
        for j in range(nflux_rec):
            genotype_ran = np.random.choice(genotypes, p=weights)
            recombination_event(genotype,genotype_ran,pop,seq_length)
    return pop

# for different genotypes, they have different mutation probablity
# this total mutation value represents the likelihood for one genotype to mutate
# if there is only 2 alleles and only one mutation rate, total_mu = mutation rate * sequence length
# take a supplied genotype and mutate a site at random.
def get_mutant(genotype,seq_length): #binary case
    #choose one possible mutation site
    site = np.random.randint(seq_length)
    # mutate (binary case, from WT to mutant or vice)
    mutation = list(binary_nuc)
    mutation.remove(genotype[site])
    # get new mutation sequence
    new_genotype = genotype[:site] + mutation[0] + genotype[site+1:]
    return new_genotype

# check if the mutant already exists in the population.
#If it does, increment this mutant genotype
#If it doesn't create a new genotype of count 1.
# If a mutation event creates a new genotype, calculate its fitness.
def mutation_event(genotype,pop,seq_length):
    if pop[genotype] > 1:
        pop[genotype] -= 1
        if pop[genotype] == 0:
            del pop[genotype]

        new_genotype = get_mutant(genotype,seq_length)
        if new_genotype in pop:
            pop[new_genotype] += 1
        else:
            pop[new_genotype] = 1
    return pop

# create all the mutations that occur in a single generation
def mutation_step(pop,mut_rate,seq_length):
    genotypes = list(pop.keys())
    for genotype in genotypes:
        n = pop[genotype]
        # calculate the likelihood to mutate
        total_mu = seq_length * mut_rate # for binary case
        nMut = np.random.binomial(n, total_mu)
        for j in range(nMut):
            mutation_event(genotype,pop,seq_length)
    return pop

# transfer output from alphabet to number
def get_sequence(genotype, q):
    escape_states = []
    for i in range(len(genotype)):
        for k in range(q):
            if genotype[i] == binary_nuc[k]:
                escape_states.append(str(k))
    return escape_states

def initial_dis(pop,inital_state,pop_size,seq_length,q,p_mut=0.2):
    n_seqs  = int(pop_size/inital_state)
    for ss in range(inital_state):
        sequences = ''
        for i in range(seq_length):
            temp_seq   = np.random.choice(np.arange(0, q), p=[1-p_mut, p_mut])
            allele_i   = binary_nuc[temp_seq]
            sequences += allele_i
        if ss != inital_state-1:
            if sequences in pop:
                pop[sequences] += n_seqs
            else:
                pop[sequences]  = n_seqs
        else:
            if sequences in pop:
                pop[sequences] += pop_size - (inital_state-1)*n_seqs
            else:
                pop[sequences]  = pop_size - (inital_state-1)*n_seqs

def simulate_simple(**pdata):
    """
    Example evolutionary trajectory for a 20-site system
    """

    # unpack passed data
    sim_dir       = pdata['dir']            # 'simple'
    input_dir     = pdata['input_dir']      # 'sequences'
    xfile         = pdata['xfile']          # 'sample-0'
    seq_length    = pdata['seq_length']     # 10
    pop_size      = pdata['pop_size']       # 1000
    generations   = pdata['generations']    # 500
    mut_rate      = pdata['mut_rate']       # 1e-3
    rec_rate      = pdata['rec_rate']       # 1e-3
    inital_state  = pdata['inital_state']   # 4

    bene          = pdata['bene']           # [0,1]
    dele          = pdata['dele']           # [4,5]
    p_1           = pdata['p_1']            # [6,7] , special sites 1
    p_2           = pdata['p_2']            # [8,9] , special sites 2

    fB            = pdata['s_ben']          # 0.02
    fD            = pdata['s_del']          # -0.02
    fi_1          = pdata['fi_1']           # time-varying selection coefficient for special sites 1
    fi_2          = pdata['fi_2']           # time-varying selection coefficient for special sites 2

    q  = len(binary_nuc)
    ############################################################################
    ############################## function ####################################
    # get fitness of new genotype
    def get_fitness_simple(genotype,time):
        fitness = 1.0
        
        # individual locus
        for i in range(seq_length):
            if genotype[i] != "A": # mutant type
                if i in p_1: # special sites 1
                    fitness += fi_1[time]
                elif i in p_2: # special sites 2
                    fitness += fi_2[time]
                elif i in bene: # beneficial mutation
                        fitness += fB
                elif i in dele: # deleterious mutation
                        fitness += fD
        
        return fitness

    # genetic drift
    def offspring_step_simple(pop,time):
        genotypes = list(pop.keys())
        r = []
        for genotype in genotypes:
            numbers = pop[genotype]
            fitness = get_fitness_simple(genotype,time)
            r.append(numbers * fitness)
        weights = [x / sum(r) for x in r]
        pop_size_t = np.sum([pop[i] for i in genotypes])
        counts = list(np.random.multinomial(pop_size_t, weights)) # genetic drift
        for (genotype, count) in zip(genotypes, counts):
            if (count > 0):
                pop[genotype] = count
            else:
                del pop[genotype]
        return pop

    ############################################################################
    ############################## Simulate ####################################
    # output file
    out_file = '%s/%s/%s/%s.dat'%(SIM_DIR,sim_dir,input_dir,xfile)
    if os.path.exists(out_file): # skip if the file already exists
        return

    # initialize population
    pop = {}
    initial_dis(pop,inital_state,pop_size,seq_length,q)

    # in every generation, it will mutate and then the genetic drift
    # calculate several times to get the evolution trajectory
    # At each step in the simulation, we append to a history object.
    history = []
    clone_pop = dict(pop)
    history.append(clone_pop)
    for t in range(generations):
        recombination_step(pop,rec_rate,seq_length)
        mutation_step(pop,mut_rate,seq_length)
        offspring_step_simple(pop,t)
        clone_pop = dict(pop)
        history.append(clone_pop)

    # write the output file - dat format
    f = open(out_file,'w')
    for i in range(len(history)):
        pop_at_t = history[i]
        genotypes = pop_at_t.keys()
        for genotype in genotypes:
            time = i
            counts = pop_at_t[genotype]
            sequence = get_sequence(genotype, q)
            f.write('%d\t%d\t' % (time,counts))
            for j in range(len(sequence)):
                f.write(' %s' % (' '.join(sequence[j])))
            f.write('\n')
    f.close()

def simulate_trait(**pdata):
    """
    Example evolutionary trajectory for a 20-site system
    """

    # unpack passed data
    sim_dir       = pdata['dir']            # 'trait'
    input_dir     = pdata['input_dir']      # 'sequences'
    xfile         = pdata['xfile']          #'1-con'
    seq_length    = pdata['seq_length']     # 20
    pop_size      = pdata['pop_size']       # 1000
    generations   = pdata['generations']    # 500
    mut_rate      = pdata['mut_rate']       # 1e-3
    rec_rate      = pdata['rec_rate']       # 1e-3
    inital_state  = pdata['inital_state']   # 4

    bene          = pdata['bene']           # [0,1]
    dele          = pdata['dele'] 
    escape_group  = pdata['escape_group']   # random choose 3 sites to consist of a binary trait
    p_sites       = pdata['p_sites']        # [9,10] , special sites
    
    fB            = pdata['s_ben']          # 4
    fD            = pdata['s_del']          # 0.02
    fi            = pdata['fi']             # time-varying selection coefficient
    fn            = pdata['fn']             # time-varying escape coefficient

    q  = len(binary_nuc)
    ne = len(escape_group)

    ############################################################################
    ############################## function ####################################
    # get fitness of new genotype
    def get_fitness_trait(genotype,time):
        fitness = 1
    
        # individual locus
        for i in range(seq_length):
            if genotype[i] != "A": # mutant type
                if i in p_sites: # special site
                    fitness += fi[time]
                elif i in bene: # beneficial mutation
                    fitness += fB
                elif i in dele: # deleterious mutation
                    fitness += fD
        
        # binary trait
        for n in range(ne):
            for nn in escape_group[n]:
                if genotype[nn] != "A":
                    fitness += fn[time]
                    break
        return fitness

    # genetic drift
    def offspring_step_trait(pop,time):
        genotypes = list(pop.keys())
        r = []
        for genotype in genotypes:
            numbers = pop[genotype]
            fitness = get_fitness_trait(genotype,time)
            r.append(numbers * fitness)
        weights = [x / sum(r) for x in r]
        pop_size_t = np.sum([pop[i] for i in genotypes])
        counts = list(np.random.multinomial(pop_size_t, weights)) # genetic drift
        for (genotype, count) in zip(genotypes, counts):
            if (count > 0):
                pop[genotype] = count
            else:
                del pop[genotype]
        return pop

    ############################################################################
    ############################## Simulate ####################################
    # output file
    out_file = '%s/%s/%s/%s.dat'%(SIM_DIR,sim_dir,input_dir,xfile)
    if os.path.exists(out_file): # skip if the file already exists
        return
    
    # Initialize population
    pop = {}
    initial_dis(pop,inital_state,pop_size,seq_length,q)

    # in every generation, it will mutate and then the genetic drift
    # calculate several times to get the evolution trajectory
    # At each step in the simulation, we append to a history object.
    history = []
    clone_pop = dict(pop)
    history.append(clone_pop)
    for t in range(generations):
        recombination_step(pop,rec_rate,seq_length)
        mutation_step(pop,mut_rate,seq_length)
        offspring_step_trait(pop,t)
        clone_pop = dict(pop)
        history.append(clone_pop)

    # write the output file - dat format
    f = open(out_file,'w')

    for i in range(len(history)):
        pop_at_t = history[i]
        genotypes = pop_at_t.keys()
        for genotype in genotypes:
            time = i
            counts = pop_at_t[genotype]
            sequence = get_sequence(genotype, q)
            f.write('%d\t%d\t' % (time,counts))
            for j in range(len(sequence)):
                f.write(' %s' % (' '.join(sequence[j])))
            f.write('\n')
    f.close()

# loading data from dat file (simple)
def getSequence_simple(history,sample_times):
    sVec      = []
    nVec      = []

    for time in sample_times:
        idx = history.T[0] == time
        data_t = history[idx]
        temp_nVec   = []
        temp_sVec   = []

        for t in range(len(data_t)):
            temp_nVec.append(data_t[t][1])
            temp_sVec.append(data_t[t][2:])

        nVec.append(temp_nVec)
        sVec.append(temp_sVec)

    return sVec,nVec
    
# loading data from dat file (trait)
def getSequence_trait(history,escape_group,sample_times):
    sVec      = []
    nVec      = []
    eVec      = []

    ne          = len(escape_group)

    for time in sample_times:
        idx = history.T[0] == time
        data_t = history[idx]
        temp_nVec   = []
        temp_sVec   = []
        temp_eVec   = []
        for t in range(len(data_t)):
            temp_nVec.append(data_t[t][1])
            temp_sVec.append(data_t[t][2:])

            if ne > 0:
                temp_escape = np.zeros(ne, dtype=int)
                for n in range(ne):
                    for nn in range(len(escape_group[n])):
                        index = escape_group[n][nn] + 2
                        if data_t[t][index] != 0:
                            temp_escape[n] = 1
                            break
                temp_eVec.append(temp_escape)
        nVec.append(temp_nVec)
        sVec.append(temp_sVec)
        eVec.append(temp_eVec)

    return sVec,nVec,eVec

# calculate frequencies for recombination part
def get_p_k(sVec,nVec,seq_length,escape_group,escape_TF):
    p_mut_k   = np.zeros((len(nVec),seq_length,3)) # 0: time, 1: all k point, 2: p_k, p_k-, p_k+
    for t in range(len(nVec)):
        pop_size_t = np.sum([nVec[t]])
        
        for n in range(len(escape_group)):
            escape_group_n = escape_group[n]
            sWT_n     = [int(i) for i in escape_TF[n]]

            for k in range(len(sVec[t])): # different sequences at time t
                sVec_n = [int(sVec[t][k][i]) for i in escape_group_n]

                for nn in range(len(escape_group_n)-1):
                    k_bp = nn + 1
                    
                    # containing mutation before and after break point k,p_k
                    if sWT_n[:k_bp] != sVec_n[:k_bp] and sWT_n[k_bp:] != sVec_n[k_bp:]:
                        p_mut_k[t][escape_group_n[0]+nn][0] += nVec[t][k]
                    
                    # MT before break point k and WT after break point k,p_k-
                    if sWT_n[:k_bp] != sVec_n[:k_bp] and sWT_n[k_bp:] == sVec_n[k_bp:]:
                        p_mut_k[t][escape_group_n[0]+nn][1] += nVec[t][k]
                    
                    # WT before break point k and MT after break point k,p_k+
                    if sWT_n[:k_bp] == sVec_n[:k_bp] and sWT_n[k_bp:] != sVec_n[k_bp:]:
                        p_mut_k[t][escape_group_n[0]+nn][2] += nVec[t][k]

        p_mut_k[t] = p_mut_k[t] / pop_size_t

    return p_mut_k
 
# calculate diffusion matrix C at any t
def diffusion_matrix_at_t(x,xx):
    x_length = len(x)
    C = np.zeros([x_length,x_length])
    for i in range(x_length):
        C[i,i] = x[i] - x[i] * x[i]
        for j in range(int(i+1) ,x_length):
            C[i,j] = xx[i,j] - x[i] * x[j]
            C[j,i] = xx[i,j] - x[i] * x[j]
    return C
    
# calculate the difference between the frequency at time t and time t-1
def cal_dx_all(x_all,times,x_length):
    dx_all = np.zeros((len(x_all),x_length))   # difference between the frequency at time t and time t-1s
    # Calculate manually
    for tt in range(len(x_all)-1):
        dx_all[tt] = (x_all[tt+1] - x_all[tt])/(times[tt+1]-times[tt])
    
    # dt for the last time point, make sure the expected x[t+1] is less than 1 and larger than 0
    for ii in range(x_length):
        if x_all[-1,ii] == 1 and dx_all[-2,ii] > 0:
            dx_all[-1,ii] = 0
        elif x_all[-1,ii] == 0 and dx_all[-2,ii] < 0:
            dx_all[-1,ii] = 0
        else:
            dx_all[-1,ii] = dx_all[-2,ii]

    return dx_all

# get muVec for binary case without threshold
def getMutantS(seq_length):
    muVec    = -np.ones(seq_length)
    x_length = 0
    for i in range(seq_length):
        muVec[i] = x_length
        x_length += 1
    return x_length,muVec
    
def infer_simple(**pdata):
    """
    Infer time-varying example (binary case)
    """

    # unpack passed data
    sim_dir       = pdata['dir']            # 'simple'
    input_dir     = pdata['input_dir']      # 'sequences'
    xfile         = pdata['xfile']          # index of the simulation
    output_dir    = pdata['output_dir']     # 'output'

    seq_length    = pdata['seq_length']     # 10
    totalT        = pdata['generations']    # 1000
    mut_rate      = pdata['mut_rate']       # 1e-3

    p_1           = pdata['p_1']            # [6,7]
    p_2           = pdata['p_2']            # [8,9]

    gamma_1s      = pdata['gamma_s']/totalT # gamma_s/time points
    gamma_2c      = pdata['gamma_2c']       # 1000000
    gamma_2tv     = pdata['gamma_2tv']      # 200
    theta         = pdata['theta']          # 0.5
    beta          = pdata['beta']           # 4

    p_sites       = p_1+p_2                 # [6,7,8,9] , special sites
    ############################################################################
    ############################## Function ####################################

    # calculate single and pair allele frequency (binary case)
    def get_allele_frequency(sVec,nVec,muVec):

        x  = np.zeros((len(nVec),x_length))           # single allele frequency
        xx = np.zeros((len(nVec),x_length,x_length))  # pair allele frequency
        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            # individual locus part
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    x[t,aa] = np.sum([sVec[t][k][i] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for j in range(int(i+1), seq_length):
                    bb = int(muVec[j])
                    if bb != -1:
                        xx[t,aa,bb] = np.sum([sVec[t][k][i] * sVec[t][k][j] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                        xx[t,aa,bb] = np.sum([sVec[t][k][i] * sVec[t][k][j] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
        return x,xx
    
    # calculate mutation flux term (binary_case)
    def get_mut_flux(x,muVec):
        flux = np.zeros((len(x),x_length))
        for t in range(len(x)):
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    flux[t,aa] = mut_rate * ( 1 - 2 * x[t,aa])
        return flux

    def get_gamma2(times, beta):
        # Use a time-varying gamma_prime, gamma_2tv is the middle value, 
        # boundary value is beta times larger, decrese/increase exponentially within 10% generation.
        gamma_t = np.ones(len(times))
        if beta != 1:
            tv_range = max(int(round(times[-1]*0.1/10)*10),1)
            alpha  = np.log(beta) / tv_range
            for ti, t in enumerate(times): # loop over all time points, ti: index, t: time
                if t <= tv_range:
                    gamma_t[ti] = beta * np.exp(-alpha * t)
                elif t > times[-1] - tv_range:
                    gamma_t[ti] = 1 * np.exp(alpha * (t - times[-1] + tv_range))

        # individual site: gamma_2c, escape group and special site: gamma_2tv
        gamma_2 = np.ones((x_length,len(times))) * gamma_2c
        for p_site in p_sites: # special site - time varying
            index = int (muVec[p_site]) 
            if index != -1:
                gamma_2[index] = gamma_t * gamma_2tv
        
        return gamma_2.T

    # solve the bounadry condition ODE to infer selections
    def fun_simple(time,s):
        """ Function defining the right-hand side of the system of ODE's"""

        t = np.asarray(time, dtype=float)

        s1 = s[:x_length,:]   # the actual selection coefficients s1 = s
        s2 = s[x_length:,:]   # the derivatives of the selection coefficients, s2 = s'

        # s' = s2
        dsdt = np.empty_like(s)  # the RHS of the system of ODE's
        dsdt[:x_length, :] = s2

        # s2'(t) = A(t)s1(t) + b(t)
        ds2 = np.empty_like(s1)  # s2'
        mask_left = t < 0
        mask_right = t > sample_times[-1]
        mask_mid = (~mask_left) & (~mask_right)

        # left : s'' = gamma1 * s / gamma2_left
        gamma2_left = gamma_2[0, :]
        if np.any(mask_left):
            idx = np.where(mask_left)[0]
            ds2[:, idx] = (gamma_1[:, None] * s1[:, idx]) / gamma2_left[:, None]

        # right : s'' = gamma1 * s / gamma2_right
        gamma2_right = gamma_2[-1, :]
        if np.any(mask_right):
            idx = np.where(mask_right)[0]
            ds2[:, idx] = (gamma_1[:, None] * s1[:, idx]) / gamma2_right[:, None]

        # Middle : s''(t) = (A(t) s(t) + b(t)) / gamma2(t)
        if np.any(mask_mid):
            idx_mid = np.where(mask_mid)[0]
            t_mid = t[idx_mid]
            # round to get integer time index
            k_mid = np.rint(t_mid).astype(int)
            # make sure the index is within range [0, n_time]
            k_mid = np.clip(k_mid, 0, sample_times[-1])

            for pos in range(len(k_mid)):
                j = idx_mid[pos]    # index in the original time array
                k = int(k_mid[pos]) # integer time, also the index for A_all, b_all and gamma_2
                # A_all[k] shape (L, L), s1[:, j] shape (L,)
                num = A_all[k] @ s1[:, j] + b_all[k]      # (L,)
                ds2[:, j] = num / gamma_2[k]

        dsdt[x_length:, :] = ds2

        return dsdt

    # boundary condition
    def bc(b1,b2):
        # if using Neumann boundary condition
        return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints

    ############################################################################
    ####################### Inference (binary case) ############################
    # get the name of the output file and check if it exists
    name_dir = xfile.split('-')[0]
    name     = xfile.replace(name_dir+'-','')
    out_file = '%s/%s/%s/c_%s.npz'%(SIM_DIR,sim_dir,output_dir,name)
    if os.path.exists(out_file):
        return
    
    # obtain raw data and information of traits
    data         = np.loadtxt("%s/%s/%s/%s.dat"%(SIM_DIR,sim_dir,input_dir,xfile))

    # get raw time points
    times = []
    for i in range(len(data)):
        times.append(data[i][0])
    sample_times = np.unique(times)
    time_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))

    # obtain sequence data and frequencies
    sVec,nVec      = getSequence_simple(data,sample_times)
    x_length,muVec = getMutantS(seq_length)

    # get all frequencies, x_raw: single allele frequency, xx_raw: pair allele frequency
    x_raw,xx_raw   = get_allele_frequency(sVec,nVec,muVec) 
    
    # get dx
    dx_raw = cal_dx_all(x_raw, sample_times, x_length)
    mu_raw = get_mut_flux(x_raw, muVec)

    # get gamma_1 and gamma_2
    gamma_1 = np.ones(x_length)*gamma_1s
    gamma_2 = get_gamma2(time_all, beta)

    # get the input arrays at any integer time point
    if len(sample_times) == len(time_all):
        # no interpolation is needed
        x_all  = x_raw
        xx_all = xx_raw
        dx_all = dx_raw
        mu_all = mu_raw

    else:
        # Use linear interpolates to get data
        interp_x   = interp1d(sample_times,  x_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_xx  = interp1d(sample_times, xx_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_dx  = interp1d(sample_times, dx_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_mu  = interp1d(sample_times, mu_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        
        x_all = interp_x(time_all)
        xx_all = interp_xx(time_all)
        dx_all = interp_dx(time_all)
        mu_all = interp_mu(time_all)

    t_extend = int(round(time_all[-1]*theta/10.0)*10)
    etleft   = np.linspace(-t_extend,-10,int(t_extend/10)) # time added before the beginning time (dt=10)
    etright  = np.linspace(time_all[-1]+10,time_all[-1]+t_extend,int(t_extend/10))
    ExTimes  = np.concatenate((etleft, time_all, etright))

    # Get matrix A and vector b at all time points
    n_time = len(time_all)

    # Create C at all time points first
    x_outer = x_all[:, :, None] * x_all[:, None, :]

    # off-diagonal part
    C_all = xx_all - x_outer

    # use only upper triangle (j > i)
    iu = np.triu_indices(x_length, k=1)
    il = (iu[1], iu[0])   # (j,i)
    C_all[:, il[0], il[1]] = C_all[:, iu[0], iu[1]] # force: C[j,i] = C[i,j]

    # fill diagonal
    diag_vals = x_all - x_all**2
    for t_idx in range(n_time):
        np.fill_diagonal(C_all[t_idx], diag_vals[t_idx])
    
    # A(t) = C(t) + gamma_1 * I
    A_all = C_all + gamma_1.reshape(1, x_length) * np.eye(x_length)

    # # b(t) = flux_mu(t) - dx(t)
    b_all = mu_all - dx_all

    # initial guess for the selection coefficients
    ss_extend = np.zeros((2*x_length,len(ExTimes)))
    
    # solve the boundary value problem
    solution = sp.integrate.solve_bvp(fun_simple, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)
    
    # Check if the solution converged
    if solution.status != 0:
        print("Error: The BVP solver did not converge for file %s.dat"%(xfile))
        # return
    
    if np.isnan(solution.y).any():
        print("Error: solution contains NaN for file %s.dat"%(xfile))
        # return

    # Get the solution and remove the superfluous part of the array
    # including the extended time points
    sc_all         = solution.sol(ExTimes)
    desired_sc_all = sc_all[:x_length,:]

    # not include the extended time points
    sc_sample         = solution.sol(sample_times)
    desired_sc_sample = sc_sample[:x_length,:]

    # save the solution with constant_time-varying selection coefficients
    g = open(out_file, mode='w+b')
    np.savez_compressed(g, all = desired_sc_all, selection=desired_sc_sample, time=sample_times, ExTimes=ExTimes)
    g.close()

def infer_trait(**pdata):
    """
    Infer time-varying example (binary case) 
        - add binary trait part
        - gamma_1 become smaller then s < 0 
    """
    # unpack passed data
    sim_dir       = pdata['dir']            # 'trait'
    xfile         = pdata['xfile']          # index of the simulation
    input_dir     = pdata['input_dir']      # 'sequences'
    output_dir    = pdata['output_dir']     # 'output'

    seq_length    = pdata['seq_length']     # 20
    totalT        = pdata['generations']    # 1000
    mut_rate      = pdata['mut_rate']       # 1e-3
    rec_rate      = pdata['rec_rate']
    p_sites       = pdata['p_sites']        # [13,18] , special sites
    theta         = pdata['theta']          # 0.5
    beta          = pdata['beta']           # 4

    gamma_1s      = pdata['gamma_s']/totalT # gamma_s/time points
    gamma_1t      = gamma_1s/10
    gamma_2c      = pdata['gamma_2c']       # 1000000
    gamma_2tv     = pdata['gamma_2tv']      # 200 

    ############################################################################
    ############################## Function ####################################

    # calculate single and pair allele frequency (binary case)
    def get_allele_frequency(sVec,nVec,eVec,muVec):

        x  = np.zeros((len(nVec),x_length))           # single allele frequency
        xx = np.zeros((len(nVec),x_length,x_length))  # pair allele frequency
        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            # individual locus part
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    x[t,aa] = np.sum([sVec[t][k][i] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for j in range(int(i+1), seq_length):
                    bb = int(muVec[j])
                    if bb != -1:
                        xx[t,aa,bb] = np.sum([sVec[t][k][i] * sVec[t][k][j] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                        xx[t,aa,bb] = np.sum([sVec[t][k][i] * sVec[t][k][j] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
            # escape part
            for n in range(ne):
                aa      = x_length-ne+n
                x[t,aa] = np.sum([eVec[t][k][n] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for m in range(int(n+1), ne):
                    bb          = x_length-ne+m
                    xx[t,aa,bb] = np.sum([eVec[t][k][n] * eVec[t][k][m] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                    xx[t,bb,aa] = np.sum([eVec[t][k][n] * eVec[t][k][m] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for j in range(seq_length):
                    bb = int(muVec[j])
                    if bb != -1:
                        xx[t,bb,aa] = np.sum([sVec[t][k][j] * eVec[t][k][n] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                        xx[t,aa,bb] = np.sum([sVec[t][k][j] * eVec[t][k][n] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
        return x,xx

    # calculate escape frequency (binary case)
    def get_escape_fre_term(sVec,nVec):
        ex  = np.zeros((len(nVec),ne,seq_length))
        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            for k in range(len(sVec[t])):
                for n in range(ne):
                    n_mutations = 0
                    for nn in escape_group[n]:
                        if sVec[t][k][nn] != 0:
                            n_mutations += 1
                            site = nn
                    if n_mutations == 1:
                        ex[t,n,site] += nVec[t][k]
            ex[t,:,:] = ex[t,:,:] / pop_size_t
        return ex
    
    # calculate mutation flux term (binary_case)
    def get_mut_flux(x,ex,muVec):
        flux = np.zeros((len(x),x_length))
        for t in range(len(x)):
            # individual locus part
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    flux[t,aa] = mut_rate * ( 1 - 2 * x[t,aa])
            # binary trait part
            for n in range(ne):
                for nn in escape_group[n]:
                    flux[t,x_length-ne+n] += mut_rate * (1 - x[t,x_length-ne+n] - ex[t,n,nn] )
        return flux

    # calculate recombination flux term (binary_case)
    def get_rec_flux_at_t(x_trait,p_mut_k,trait_dis):
        flux = np.zeros(x_length)

        for n in range(ne):
            fluxIn  = 0
            fluxOut = 0

            for nn in range(len(escape_group[n])-1):
                k_index = escape_group[n][0]+nn
                fluxIn  += trait_dis[n][nn] * (1-x_trait[n])*p_mut_k[k_index][0]
                fluxOut += trait_dis[n][nn] * p_mut_k[k_index][1]*p_mut_k[k_index][2]
            
            flux[x_length-ne+n] = rec_rate * (fluxIn - fluxOut)

        return flux
    
    # regularization value gamma_1 and gamma_2
    # gamma_1: time-independent, gamma_2: time-dependent
    def get_gamma1():
        # individual site: gamma_1s, escape group: gamma_1t
        gamma_1   = np.ones(x_length)*gamma_1s
        for n in range(ne):
            gamma_1[x_length-ne+n] = gamma_1t
        
        return gamma_1

    def get_gamma2(times, beta):
        # Use a time-varying gamma_prime, gamma_2tv is the middle value, 
        # boundary value is beta times larger, decrese/increase exponentially within 10% generation.
        gamma_t = np.ones(len(times))
        tv_range = max(int(round(times[-1]*0.1/10)*10),1)
        alpha  = np.log(beta) / tv_range
        for ti, t in enumerate(times): # loop over all time points, ti: index, t: time
            if t <= tv_range:
                gamma_t[ti] = beta * np.exp(-alpha * t)
            elif t > times[-1] - tv_range:
                gamma_t[ti] = 1 * np.exp(alpha * (t - times[-1] + tv_range))

        # individual site: gamma_2c, escape group and special site: gamma_2tv
        gamma_2 = np.ones((x_length,len(times)))*gamma_2c
        # special site
        for p_site in p_sites:
            index = int (muVec[p_site])  
            if index != -1:
                gamma_2[index] = gamma_t * gamma_2tv
        # binary trait
        for n in range(ne):
            gamma_2[x_length-ne+n] = gamma_t * gamma_2tv

        return gamma_2.T

    ############################################################################
    ####################### Inference (binary case) ############################
    name = xfile.split('-',1)[1]
    out_file = '%s/%s/%s/c_%s.npz'%(SIM_DIR,sim_dir,output_dir,name)
    # if os.path.exists(out_file):
    #     return
    
    # obtain raw data and information of traits
    data         = np.loadtxt('%s/%s/%s/%s.dat'%(SIM_DIR,sim_dir,input_dir,xfile))
    escape_group = read_file('%s/traitsite/traitsite-%s.dat'%(sim_dir,name))
    trait_dis    = read_file('%s/traitdis/traitdis-%s.dat'%(sim_dir,name))
    escape_TF    = read_file('%s/traitseq.dat'%(sim_dir))
    ne           = len(escape_group)

    # get raw time points
    times = []
    for i in range(len(data)):
        times.append(data[i][0])
    sample_times = np.unique(times)
    time_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))

    # obtain sequence data and frequencies
    sVec,nVec,eVec = getSequence_trait(data,escape_group,sample_times)
    x_length,muVec = getMutantS(seq_length)
    x_length      += ne

    # get all frequencies, x: single allele frequency, xx: pair allele frequency
    # ex: escape frequency, p_wt,p_mut_k: frequency related to recombination part
    x,xx        = get_allele_frequency(sVec,nVec,eVec,muVec) 
    ex          = get_escape_fre_term(sVec,nVec)
    p_mut_k_raw = get_p_k(sVec,nVec,seq_length,escape_group,escape_TF)

    # get dx
    dx_all_raw = cal_dx_all(x, sample_times, x_length)
    mu_all_raw = get_mut_flux(x,ex,muVec)
    
    # get gamma_1 and gamma_2
    gamma_1 = get_gamma1()
    gamma_2 = get_gamma2(time_all, beta)

    # get the input arrays at any integer time point
    if len(sample_times) == len(time_all):
        # no interpolation is needed
        x_all = x
        xx_all = xx
        p_mut_k     = p_mut_k_raw
        dx_all     = dx_all_raw
        mu_all     = mu_all_raw

    else:
        # Use linear interpolates to get the input arrays at any integer time point
        interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_mut = interp1d(sample_times, p_mut_k_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_dx  = interp1d(sample_times, dx_all_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_mu  = interp1d(sample_times, mu_all_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
                
        x_all = interp_x(time_all)
        xx_all = interp_xx(time_all)
        p_mut_k     = interp_mut(time_all)
        dx_all     = interp_dx(time_all)
        mu_all     = interp_mu(time_all)

    # extend the time range
    t_extend = int(round(time_all[-1]*theta/10)*10)
    etleft   = np.linspace(-t_extend,-10,int(t_extend/10)) # time added before the beginning time (dt=10)
    etright  = np.linspace(time_all[-1]+10,time_all[-1]+t_extend,int(t_extend/10))
    ExTimes  = np.concatenate((etleft, time_all, etright))

    # Get matrix A and vector b
    A_all = np.zeros((len(time_all),x_length,x_length))
    b_all = np.zeros((len(time_all),x_length))

    for ti in range(len(time_all)):
        # calculate A(t) = C(t), add regularization term at ODE part
        A_all[ti] = diffusion_matrix_at_t(x_all[ti], xx_all[ti]) # covariance matrix

        # calculate b(t)
        rec_t = get_rec_flux_at_t(x_all[ti,x_length-ne:], p_mut_k[ti], trait_dis)
        b_all[ti]   = mu_all[ti] + rec_t - dx_all[ti] 

    def fun_trait(time,s):
        """ Function defining the right-hand side of the system of ODE's"""
        s1                 = s[:x_length,:]   # the actual selection coefficients s1 = s
        s2                 = s[x_length:,:]   # the derivatives of the selection coefficients, s2 = s'
        dsdt               = np.zeros_like(s)  # the RHS of the system of ODE's

        # s' = s2
        dsdt[:x_length, :] = s2

        # s2'(t) = A(t)s1(t) + b(t)
        for ti, t in enumerate(time): # loop over all time points, ti: index, t: time
            # set value for gamma_1 of traits part
            # high covariance with positive part and low covariance with negative part
            for n in range(ne):
                if s[x_length-ne+n, ti] < 0:
                    gamma_1[x_length-ne+n] = gamma_1t*100 # keep a high penalty for negative selection 
                else:
                    gamma_1[x_length-ne+n] = gamma_1t

            if t < 0:
                # s'' = gamma1* s(t)/gamma1(t)
                gamma2_t = gamma_2[0]
                dsdt[x_length:, ti] = gamma_1 * s1[:, ti] / gamma2_t

            elif t > sample_times[-1]:
                # s'' = gamma1* s(t)/gamma1(t)
                gamma2_t = gamma_2[-1]
                dsdt[x_length:, ti] = gamma_1 * s1[:, ti] / gamma2_t

            else:
                # get A(t), b(t) and gamma2(t)
                time_index = round(t)
                A_t      = A_all[time_index]  + gamma_1.reshape(x_length,1) * np.eye(x_length)
                b_t      = b_all[time_index]
                gamma2_t = gamma_2[time_index]

                # s'' = A(t)s(t) + b(t)
                dsdt[x_length:, ti] = (A_t @ s1[:, ti] + b_t) / gamma2_t

        return dsdt

    # Boundary conditions
    def bc(b1,b2):
        # Neumann boundary condition
        return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints
        
    # initial guess for the selection coefficients
    ss_extend = np.zeros((2*x_length,len(ExTimes)))
    
    # solve the boundary value problem
    solution = sp.integrate.solve_bvp(fun_trait, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)
    
    # Get the solution and remove the superfluous part of the array
    # including the extended time points
    sc_all         = solution.sol(ExTimes)
    desired_sc_all = sc_all[:x_length,:]

    # not include the extended time points
    sc_sample         = solution.sol(time_all)
    desired_sc_sample = sc_sample[:x_length,:]

    # save the solution with constant_time-varying selection coefficient
    g = open(out_file, mode='w+b')
    np.savez_compressed(g, all = desired_sc_all, selection=desired_sc_sample, time=time_all, ExTimes=ExTimes)
    g.close()

def cut_seq(**pdata):

    # unpack passed data
    sim_dir       = pdata['dir']           # 'trait'
    xfile         = pdata['xfile']         # index of the simulation
    cut_dir       = pdata['cut_dir']            # 10
    observed_time = pdata['cut_time']      

    # output file 
    out_file = "%s/%s/cut/%s/sequences/%s.dat"%(SIM_DIR,sim_dir,cut_dir,xfile)
    if os.path.exists(out_file): # skip existing files
        return
    
    # obtain raw data and information of traits
    data         = np.loadtxt('%s/%s/sequences/%s.dat'%(SIM_DIR,sim_dir,xfile))

    # write the output file - dat format
    f = open(out_file,'w')

    for i in range(len(data)):
        if data[i][0] in observed_time:
            f.write('%d\t%d\t' % (data[i][0],data[i][1]))
            f.write(' %s' % (' '.join([str(int(j)) for j in data[i][2:]])))
            f.write('\n')
    
    f.close()

def sample_one_timepoint(tt, data, ns, rng):
    """
    Sample ns sequences at time point tt from data.
    Return a list of strings, each string is one line to write to file.
    """
    mask = (data[:,0] == tt)
    if not np.any(mask):
        return []

    nVec = data[mask, 1].astype(int)
    sVec = data[mask, 2:].astype(int)

    # Generate index pool [0,0,...,1,1,...,2,2,...] representing each sequence repeated n times
    iVec = np.repeat(np.arange(len(nVec)), nVec)

    # Sampling
    iSample = rng.choice(iVec, ns, replace=True)

    # Count the number of each sequence in the sample
    rows = []
    for k in range(len(nVec)):
        nSample = np.sum(iSample == k)
        if nSample > 0:
            seq_str = " ".join(map(str, sVec[k]))
            rows.append(f"{tt}\t{nSample}\t{seq_str}\n")
    return rows

def write_dt_from_base(tts, data, ns, dt, out_path, rng):
    """Write sampled sequences at intervals of dt from the base data."""
    buffer = []
    for tt in tts[::dt]:
        buffer.extend(sample_one_timepoint(tt, data, ns, rng))

    with open(out_path, "w") as f:
        f.writelines(buffer)

def write_dt_from_dt1(tts, dt, base_path, out_path):
    """Construct other dt files from the dt=1 file."""
    data = np.loadtxt(base_path)
    buffer = []

    for tt in tts[::dt]:
        mask = (data[:,0] == tt)
        if not np.any(mask):
            continue

        rows = data[mask]
        for r in rows:
            tt_val = int(r[0])
            n_val  = int(r[1])
            seq = " ".join(map(str, r[2:].astype(int)))
            buffer.append(f"{tt_val}\t{n_val}\t{seq}\n")

    with open(out_path, "w") as f:
        f.writelines(buffer)

def cut_sequence_nsdt(**pdata):

    """
    Convert the whole trajectory into different sub-trajectories with different ns and dt.
    """

    # unpack passed data
    t0, tk  = pdata['t0'], pdata['T']
    ns_list  = pdata['ns']
    dt_list  = pdata['dt']
    xfile    = pdata['xfile']
    folder   = pdata['folder']

    rng = np.random.default_rng()
    base_file = f"{SIM_DIR}/{folder}/sequences/{xfile}.dat"
    data = np.loadtxt(base_file)
    all_times = np.arange(t0, tk + 1)

    # write the results
    for ns in ns_list:
        for j, dt in enumerate(dt_list):
            out_file = f"{SIM_DIR}/{folder}/sequences/nsdt/{xfile}-ns{ns}-dt{dt}.dat"
            # if os.path.exists(out_file): # skip existing files
            #     continue

            if dt == 1:
                if ns == 1000:
                    # copy the base file directly
                    shutil.copy(base_file, out_file)
                else:
                    # sample directly from base data
                    write_dt_from_base(all_times, data, ns, dt, out_file, rng)
            else:
                # reuse the previous results to make sure the sequences at the same time point are identical
                base_dt1 = f"{SIM_DIR}/{folder}/sequences/nsdt/{xfile}-ns{ns}-dt1.dat"
                write_dt_from_dt1(all_times, dt, base_dt1, out_file)

def extract_random_times(SIM_DIR, folder, xfile, ns, target_times, out_file):
    """
    From the dt=1 sampled sequence file, extract sequences at custom time points.

    SIM_DIR  : root directory
    folder   : e.g., "BH"
    xfile    : e.g., "sample-1"
    ns       : number of sampled sequences
    target_times : list of time points to extract, e.g., [3, 7, 15, 100]
    out_file : path to the output file
    """
    # if os.path.exists(out_file): # skip existing files
    #     return

    dt1_path = f"{SIM_DIR}/{folder}/sequences/nsdt/{xfile}-ns{ns}-dt1.dat"
    if not os.path.exists(dt1_path):
        raise FileNotFoundError(f"dt1 file not found: {dt1_path}")

    data = np.loadtxt(dt1_path)
    t_col = data[:, 0].astype(int)

    target_set = set(target_times)

    # Select rows with time points in target_set
    mask = np.isin(t_col, list(target_set))
    selected = data[mask]

    # Write to new file
    with open(out_file, "w") as f:
        for r in selected:
            tt = int(r[0])
            n  = int(r[1])
            seq_str = " ".join(map(str, r[2:].astype(int)))
            f.write(f"{tt}\t{n}\t{seq_str}\n")

def simulate_sigmoid(**pdata):
    """
    Example evolutionary trajectory for a 20-site system
    """

    # unpack passed data
    sim_dir       = pdata['dir']            # 'trait'
    input_dir     = pdata['input_dir']      # 'sequences'
    xfile         = pdata['xfile']          #'1-con'
    seq_length    = pdata['seq_length']     # 16
    pop_size      = pdata['pop_size']       # 1000
    generations   = pdata['generations']    # 500
    mut_rate      = pdata['mut_rate']       # 1e-3
    rec_rate      = pdata['rec_rate']       # 1e-3
    inital_state  = pdata['inital_state']   # 4

    bene          = pdata['bene']           # [0,1]
    dele          = pdata['dele'] 
    escape_group  = pdata['escape_group']   # random choose 3 sites to consist of a binary trait
    
    fB            = pdata['s_ben']          # 0.02
    fD            = pdata['s_del']          # -0.02
    fn            = pdata['fn']             # time-varying escape coefficient

    q  = len(binary_nuc)
    ne = len(escape_group)

    ############################################################################
    ############################## function ####################################
    # get fitness of new genotype
    def get_fitness_trait(genotype,time):
        fitness = 1
    
        # individual locus
        for i in range(seq_length):
            if genotype[i] != "A": # mutant type
                if i in bene: # beneficial mutation
                    fitness += fB
                elif i in dele: # deleterious mutation
                    fitness += fD
        
        # binary trait
        for n in range(ne):
            for nn in escape_group[n]:
                if genotype[nn] != "A":
                    fitness += fn[time]
                    break
        return fitness

    # genetic drift
    def offspring_step_trait(pop,time):
        genotypes = list(pop.keys())
        r = []
        for genotype in genotypes:
            numbers = pop[genotype]
            fitness = get_fitness_trait(genotype,time)
            r.append(numbers * fitness)
        weights = [x / sum(r) for x in r]
        pop_size_t = np.sum([pop[i] for i in genotypes])
        counts = list(np.random.multinomial(pop_size_t, weights)) # genetic drift
        for (genotype, count) in zip(genotypes, counts):
            if (count > 0):
                pop[genotype] = count
            else:
                del pop[genotype]
        return pop

    ############################################################################
    ############################## Simulate ####################################
    # output file
    out_file = '%s/%s/%s/%s.dat'%(SIM_DIR,sim_dir,input_dir,xfile)
    # if os.path.exists(out_file): # skip if the file already exists
    #     return
    
    # Initialize population
    pop = {}
    initial_dis(pop,inital_state,pop_size,seq_length,q)

    # in every generation, it will mutate and then the genetic drift
    # calculate several times to get the evolution trajectory
    # At each step in the simulation, we append to a history object.
    history = []
    clone_pop = dict(pop)
    history.append(clone_pop)
    for t in range(generations):
        recombination_step(pop,rec_rate,seq_length)
        mutation_step(pop,mut_rate,seq_length)
        offspring_step_trait(pop,t)
        clone_pop = dict(pop)
        history.append(clone_pop)

    # write the output file - dat format
    f = open(out_file,'w')

    for i in range(len(history)):
        pop_at_t = history[i]
        genotypes = pop_at_t.keys()
        for genotype in genotypes:
            time = i
            counts = pop_at_t[genotype]
            sequence = get_sequence(genotype, q)
            f.write('%d\t%d\t' % (time,counts))
            for j in range(len(sequence)):
                f.write(' %s' % (' '.join(sequence[j])))
            f.write('\n')
    f.close()

def infer_sigmoid(**pdata):
    """
    Infer time-varying example (binary case) for sigmoid pattern simulation
        - used a different regularization for fixed mutaton
        - use different interpolated time points for different dt
    """
    # unpack passed data
    sim_dir       = pdata['dir']            # 'trait'
    xfile         = pdata['xfile']          # index of the simulation
    input_dir     = pdata['input_dir']      # 'sequences'
    output_dir    = pdata['output_dir']     # 'output'

    seq_length    = pdata['seq_length']     # 20
    totalT        = pdata['generations']    # 1000
    mut_rate      = pdata['mut_rate']       # 1e-3
    rec_rate      = pdata['rec_rate']
    theta         = pdata['theta']          # 0.5

    gamma_1       = pdata['gamma_s']
    gamma_2c      = pdata['gamma_2c']       # 1000000
    gamma_2tv     = pdata['gamma_2tv']      # 200 

    ############################################################################
    ############################## Function ####################################

    # calculate single and pair allele frequency (binary case)
    def get_allele_frequency(sVec,nVec,eVec,muVec):

        x  = np.zeros((len(nVec),x_length))           # single allele frequency
        xx = np.zeros((len(nVec),x_length,x_length))  # pair allele frequency
        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            # individual locus part
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    x[t,aa] = np.sum([sVec[t][k][i] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for j in range(int(i+1), seq_length):
                    bb = int(muVec[j])
                    if bb != -1:
                        xx[t,aa,bb] = np.sum([sVec[t][k][i] * sVec[t][k][j] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                        xx[t,aa,bb] = np.sum([sVec[t][k][i] * sVec[t][k][j] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
            # escape part
            for n in range(ne):
                aa      = x_length-ne+n
                x[t,aa] = np.sum([eVec[t][k][n] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for m in range(int(n+1), ne):
                    bb          = x_length-ne+m
                    xx[t,aa,bb] = np.sum([eVec[t][k][n] * eVec[t][k][m] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                    xx[t,bb,aa] = np.sum([eVec[t][k][n] * eVec[t][k][m] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for j in range(seq_length):
                    bb = int(muVec[j])
                    if bb != -1:
                        xx[t,bb,aa] = np.sum([sVec[t][k][j] * eVec[t][k][n] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                        xx[t,aa,bb] = np.sum([sVec[t][k][j] * eVec[t][k][n] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
        return x,xx

    # calculate escape frequency (binary case)
    def get_escape_fre_term(sVec,nVec):
        ex  = np.zeros((len(nVec),ne,seq_length))
        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            for k in range(len(sVec[t])):
                for n in range(ne):
                    n_mutations = 0
                    for nn in escape_group[n]:
                        if sVec[t][k][nn] != 0:
                            n_mutations += 1
                            site = nn
                    if n_mutations == 1:
                        ex[t,n,site] += nVec[t][k]
            ex[t,:,:] = ex[t,:,:] / pop_size_t
        return ex
    
    # calculate mutation flux term (binary_case)
    def get_mut_flux(x,ex,muVec):
        flux = np.zeros((len(x),x_length))
        for t in range(len(x)):
            # individual locus part
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    flux[t,aa] = mut_rate * ( 1 - 2 * x[t,aa])
            # binary trait part
            for n in range(ne):
                for nn in escape_group[n]:
                    flux[t,x_length-ne+n] += mut_rate * (1 - x[t,x_length-ne+n] - ex[t,n,nn] )
        return flux

    # calculate recombination flux term (binary_case)
    def get_rec_flux_at_t(x_trait,p_mut_k,trait_dis):
        flux = np.zeros(x_length)

        for n in range(ne):
            fluxIn  = 0
            fluxOut = 0

            for nn in range(len(escape_group[n])-1):
                k_index = escape_group[n][0]+nn
                fluxIn  += trait_dis[n][nn] * (1-x_trait[n])*p_mut_k[k_index][0]
                fluxOut += trait_dis[n][nn] * p_mut_k[k_index][1]*p_mut_k[k_index][2]
            
            flux[x_length-ne+n] = rec_rate * (fluxIn - fluxOut)

        return flux
    
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


    # regularization value gamma_2 : time-dependent
    def get_gamma2(times, beta=4):
        # Use a time-varying gamma_prime, gamma_2tv is the middle value, 
        # boundary value is beta times larger, decrese/increase exponentially within 10% generation.
        gamma_t = np.ones(len(times))
        tv_range = max(int(round(times[-1]*0.1/10)*10),1)
        alpha  = np.log(beta) / tv_range
        for ti, t in enumerate(times): # loop over all time points, ti: index, t: time
            if t <= tv_range:
                gamma_t[ti] = beta * np.exp(-alpha * t)
            elif t > times[-1] - tv_range:
                gamma_t[ti] = 1 * np.exp(alpha * (t - times[-1] + tv_range))

        # individual site: gamma_2c, escape group and special site: gamma_2tv
        gamma_2 = np.ones((x_length,len(times)))*gamma_2c
        # binary trait
        for n in range(ne):
            gamma_2[x_length-ne+n] = gamma_t * gamma_2tv

        return gamma_2.T


    def smooth_vector(x, half_window=3):
        x = np.asarray(x, dtype=float)
        T = x.shape[0]

        csum = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])

        t = np.arange(T)
        a = np.maximum(0, t - half_window)
        b = np.minimum(T, t + half_window + 1)

        return (csum[b] - csum[a]) / (b - a)[:, None]

    def smooth_matrix(x, half_window=3):
        """
        Smooth along time axis (axis=0) with a symmetric window.
        """

        x = np.asarray(x)
        T, M, N = x.shape
        # prefix sum along time
        csum = np.zeros((T + 1, M, N))
        csum[1:] = np.cumsum(x, axis=0)

        t = np.arange(T)
        t_start = np.maximum(0, t - half_window)
        t_end   = np.minimum(T, t + half_window + 1)  # exclusive

        sums = csum[t_end] - csum[t_start]            # (T, M, N)
        counts = (t_end - t_start)[:, None, None]     # (T, 1, 1)

        return sums / counts

    ############################################################################
    ####################### Inference (binary case) ############################
    name = xfile.split('-',1)[1]
    name_id = name.split('-')[0]
    out_file = '%s/%s/%s/c_%s.npz'%(SIM_DIR,sim_dir,output_dir,name)
    # if os.path.exists(out_file):
    #     return
    
    # obtain raw data and information of traits
    data         = np.loadtxt('%s/%s/%s/%s.dat'%(SIM_DIR,sim_dir,input_dir,xfile))
    if 'r' in name_id:
        escape_group = [[2,8,14]]
        trait_dis    = [[6, 6]]
    else:
        escape_group = read_file('%s/traitsite/traitsite-%s.dat'%(sim_dir,name_id))
        trait_dis    = read_file('%s/traitdis/traitdis-%s.dat'%(sim_dir,name_id))
    escape_TF    = read_file('%s/traitseq.dat'%(sim_dir))
    ne           = len(escape_group)

    # get raw time points
    times = []
    for i in range(len(data)):
        times.append(data[i][0])
    sample_times = np.unique(times)
    time_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
        
    # obtain sequence data and frequencies
    sVec,nVec,eVec = getSequence_trait(data,escape_group,sample_times)
    x_length,muVec = getMutantS(seq_length)
    x_length      += ne

    # get all frequencies, x: single allele frequency, xx: pair allele frequency
    # ex: escape frequency, p_wt,p_mut_k: frequency related to recombination part
    x,xx        = get_allele_frequency(sVec,nVec,eVec,muVec) 
    ex          = get_escape_fre_term(sVec,nVec)
    p_mut_k_raw = get_p_k(sVec,nVec,seq_length,escape_group,escape_TF)

    # get dx
    dx_raw = cal_dx_all(x, sample_times, x_length)
    mu_raw = get_mut_flux(x,ex,muVec)
    
    # get gamma_1 and gamma_2
    # individual site: gamma_1s, escape group: gamma_1t
    last_time = sample_times[-1]
    gamma_1s = round(gamma_1/last_time,3) # constant MPL gamma value / max time
    gamma_1e = gamma_1s/10
    gamma_1e_original = gamma_1e * np.ones(ne)
    gamma_1   = np.ones(x_length)*gamma_1s
    # get gamma_2
    gamma_2 = get_gamma2(time_all)

    # Check if mutant epitope fixed and find the fixation time
    fixation_time = np.ones(ne) * (-1)
    for n in range(ne):
        x_epitope = x.T[x_length-ne+n]

        is_fixed = x_epitope >= 1 # find the data points where the frequency = 1
        # find the indexes where all the following points are also fixed
        all_one_suffix = np.ones_like(is_fixed, dtype=bool)
        all_one_suffix[-1] = is_fixed[-1]
        for i in range(len(is_fixed) - 2, -1, -1):
            all_one_suffix[i] = is_fixed[i] and all_one_suffix[i + 1]

        fixation_start = np.where(all_one_suffix)[0]
        if len(fixation_start) > 0:
            fixation_time[n] = fixation_start[0]
        else:
            fixation_time[n] = -1  # not fixed

    # extend the time range
    if sample_times[1] - sample_times[0] == 1: # dt = 1, no need to interpolate
        interp_times = sample_times
    elif sample_times[1] - sample_times[0] >= 10: # dt = 50 or 10, use uniform 10 dt
        interp_times = np.linspace(sample_times[0], sample_times[-1], int((sample_times[-1]-sample_times[0])/10+1))
    else: # random dt, use insert_time function to get interpolated time points
        interp_times = insert_time(sample_times)
    t_extend = int(round(interp_times[-1]*theta/10)*10)
    etleft   = np.linspace(-t_extend,-10,int(t_extend/10)) # time added before the beginning time (dt=10)
    etright  = np.linspace(interp_times[-1]+10,interp_times[-1]+t_extend,int(t_extend/10))
    ExTimes  = np.concatenate((etleft, interp_times, etright))

    # get the input arrays at any integer time point
    if len(sample_times) == len(time_all):
        if sample_times[1] - sample_times[0] == 1:
            # use smooth
            x_all   = smooth_vector(x)
            xx_all  = smooth_matrix(xx)
            p_mut_k = smooth_matrix(p_mut_k_raw)
            dx_all  = smooth_vector(dx_raw)
            mu_all  = smooth_vector(mu_raw)

        else:
            # no interpolation is needed
            x_all   = x
            xx_all  = xx
            p_mut_k = p_mut_k_raw
            dx_all  = dx_raw
            mu_all  = mu_raw

    else:
        # Use linear interpolates to get the input arrays at any integer time point
        interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_mut = interp1d(sample_times, p_mut_k_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_dx  = interp1d(sample_times, dx_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_mu  = interp1d(sample_times, mu_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)

        x_all   = interp_x(time_all)
        xx_all  = interp_xx(time_all)
        p_mut_k = interp_mut(time_all)
        dx_all  = interp_dx(time_all)
        mu_all  = interp_mu(time_all)
        
    # Get matrix A and vector b
    A_all = np.zeros((len(time_all),x_length,x_length))
    b_all = np.zeros((len(time_all),x_length))
    for ti in range(len(time_all)):
        # calculate A(t) = C(t), do not add regularization term at this time
        A_all[ti] = diffusion_matrix_at_t(x_all[ti], xx_all[ti]) # covariance matrix
        
        # calculate b(t)
        rec_t = get_rec_flux_at_t(x_all[ti,x_length-ne:], p_mut_k[ti], trait_dis)
        b_all[ti]   = mu_all[ti] - dx_all[ti] + rec_t

    def fun_sigmoid(time,s):
        """ Function defining the right-hand side of the system of ODE's"""
        s1                 = s[:x_length,:]   # the actual selection coefficients s1 = s
        # s2                 = s[x_length:,:]   # the derivatives of the selection coefficients, s2 = s'
        dsdt               = np.zeros_like(s)  # the RHS of the system of ODE's

        # s' = s2
        dsdt[:x_length, :] = s[x_length:,:]

        # s2'(t) = A(t)s1(t) + b(t)
        for ti, t in enumerate(time): # loop over all time points, ti: index, t: time
            # set value for gamma_1 of traits part
            # high covariance with positive part and low covariance with negative part
            for n in range(ne):
                if fixation_time[n] != -1 and t > fixation_time[n]:
                    gamma_1e_original[n] = gamma_1e/theta 
                    fixation_time[n] = -1 # skip this judgment in next loops

                if s[x_length-ne+n, ti] < 0:
                    gamma_1[x_length-ne+n] = gamma_1e*100 # keep a high penalty for negative selection 
                else:
                    gamma_1[x_length-ne+n] = gamma_1e_original[n]

            if t < 0:
                # s'' = gamma1* s(t)/gamma1(t)
                gamma2_t = gamma_2[0]
                dsdt[x_length:, ti] = gamma_1 * s1[:, ti] / gamma2_t

            elif t > sample_times[-1]:
                # s'' = gamma1* s(t)/gamma1(t)
                gamma2_t = gamma_2[-1]
                dsdt[x_length:, ti] = gamma_1 * s1[:, ti] / gamma2_t

            else:
                # get A(t), b(t) and gamma2(t)
                time_index = round(t)
                A_t      = A_all[time_index]                
                b_t      = b_all[time_index]
                gamma2_t = gamma_2[time_index]

                # s'' = A(t)s(t) + b(t)
                dsdt[x_length:, ti] = ((A_t+np.diag(gamma_1)) @ s1[:, ti] + b_t) / gamma2_t
        
        return dsdt

    # Boundary conditions
    def bc(b1,b2):
        # Neumann boundary condition
        return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints
        
    # initial guess for the selection coefficients
    ss_extend = np.zeros((2*x_length,len(ExTimes)))
    
    # solve the boundary value problem
    solution = sp.integrate.solve_bvp(fun_sigmoid, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)
    
    # Get the solution and remove the superfluous part of the array
    # including the extended time points
    sc_all         = solution.sol(ExTimes)
    desired_sc_all = sc_all[:x_length,:]

    # not include the extended time points
    sc_sample         = solution.sol(time_all)
    desired_sc_sample = sc_sample[:x_length,:]

    # save the solution with constant_time-varying selection coefficient
    g = open(out_file, mode='w+b')
    np.savez_compressed(g, all = desired_sc_all, selection=desired_sc_sample, time=time_all, ExTimes=ExTimes)
    g.close()

def cal_sim_change(**pdata):
    # Load data
    
    """
    Calculate the change in a simulation
    """
    # unpack passed data
    sim_dir       = pdata['dir']            # 'trait'
    xfile         = pdata['xfile']          # index of the simulation
    input_dir     = pdata['input_dir']      # 'sequences'
    out_const     = pdata['out_const']     # 'output_const'
    out_tv        = pdata['out_tv']        # 'output_tv'

    eps           = pdata['eps']            # 1
    seq_length    = pdata['seq_length']     # 16
    mut_rate      = pdata['mut_rate']       # 1e-3
    rec_rate      = pdata['rec_rate']       # 1e-3
    real_const    = pdata['sc_const']       # real selection coefficient used in the simulation
    real_tv       = pdata['sc_tv']          # real selection coefficient of the binary trait used in the simulation   
    ############################################################################
    ############################## Function ####################################
    # calculate single and pair allele frequency (binary case)
    def get_allele_frequency(sVec,nVec,eVec,muVec):

        x  = np.zeros((len(nVec),x_length))           # single allele frequency
        xx = np.zeros((len(nVec),x_length,x_length))  # pair allele frequency
        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            # individual locus part
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    x[t,aa] = np.sum([sVec[t][k][i] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for j in range(int(i+1), seq_length):
                    bb = int(muVec[j])
                    if bb != -1:
                        xx[t,aa,bb] = np.sum([sVec[t][k][i] * sVec[t][k][j] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                        xx[t,aa,bb] = np.sum([sVec[t][k][i] * sVec[t][k][j] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
            # escape part
            for n in range(ne):
                aa      = x_length-ne+n
                x[t,aa] = np.sum([eVec[t][k][n] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for m in range(int(n+1), ne):
                    bb          = x_length-ne+m
                    xx[t,aa,bb] = np.sum([eVec[t][k][n] * eVec[t][k][m] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                    xx[t,bb,aa] = np.sum([eVec[t][k][n] * eVec[t][k][m] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                for j in range(seq_length):
                    bb = int(muVec[j])
                    if bb != -1:
                        xx[t,bb,aa] = np.sum([sVec[t][k][j] * eVec[t][k][n] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
                        xx[t,aa,bb] = np.sum([sVec[t][k][j] * eVec[t][k][n] * nVec[t][k] for k in range(len(sVec[t]))]) / pop_size_t
        return x,xx

    # calculate escape frequency (binary case)
    def get_escape_fre_term(sVec,nVec):
        ex  = np.zeros((len(nVec),ne,seq_length))
        for t in range(len(nVec)):
            pop_size_t = np.sum([nVec[t]])
            for k in range(len(sVec[t])):
                for n in range(ne):
                    n_mutations = 0
                    for nn in escape_group[n]:
                        if sVec[t][k][nn] != 0:
                            n_mutations += 1
                            site = nn
                    if n_mutations == 1:
                        ex[t,n,site] += nVec[t][k]
            ex[t,:,:] = ex[t,:,:] / pop_size_t
        return ex
    
    # calculate mutation flux term (binary_case)
    def get_mut_flux(x,ex,muVec):
        flux = np.zeros((len(x),x_length))
        for t in range(len(x)):
            # individual locus part
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    flux[t,aa] = mut_rate * ( 1 - 2 * x[t,aa])
            # binary trait part
            for n in range(ne):
                for nn in escape_group[n]:
                    flux[t,x_length-ne+n] += mut_rate * (1 - x[t,x_length-ne+n] - ex[t,n,nn] )
        return flux

    # calculate recombination flux term (binary_case)
    def get_rec_flux_at_t(x_trait,p_mut_k,trait_dis):
        flux = np.zeros(x_length)

        for n in range(ne):
            fluxIn  = 0
            fluxOut = 0

            for nn in range(len(escape_group[n])-1):
                k_index = escape_group[n][0]+nn
                fluxIn  += trait_dis[n][nn] * (1-x_trait[n])*p_mut_k[k_index][0]
                fluxOut += trait_dis[n][nn] * p_mut_k[k_index][1]*p_mut_k[k_index][2]
            
            flux[x_length-ne+n] = rec_rate * (fluxIn - fluxOut)

        return flux

    ############################################################################
    ####################### Action calculation ############################
    name = xfile.split('-',1)[1]

    # obtain raw data and information of traits
    data_seq     = np.loadtxt('%s/%s/%s/%s.dat'%(SIM_DIR,sim_dir,input_dir,xfile))
    sc_const     = np.loadtxt('%s/%s/%s/sc-%s.dat'%(SIM_DIR,sim_dir,out_const,name))
    sc_tv        = np.load('%s/%s/%s/c_%s.npz'%(SIM_DIR,sim_dir,out_tv,name), allow_pickle=True)['selection'].T

    escape_group = read_file('%s/traitsite/traitsite-%s.dat'%(sim_dir,name))
    trait_dis    = read_file('%s/traitdis/traitdis-%s.dat'%(sim_dir,name))
    escape_TF    = read_file('%s/traitseq.dat'%(sim_dir))
    ne           = len(escape_group)

    # get raw time points
    times = []
    for i in range(len(data_seq)):
        times.append(data_seq[i][0])
    sample_times = np.unique(times)

    # obtain sequence data and frequencies
    sVec,nVec,eVec = getSequence_trait(data_seq,escape_group,sample_times)
    x_length,muVec = getMutantS(seq_length)
    x_length      += ne

    # get all frequencies, x: single allele frequency, xx: pair allele frequency
    # ex: escape frequency, p_wt,p_mut_k: frequency related to recombination part
    x,xx    = get_allele_frequency(sVec,nVec,eVec,muVec) 
    ex      = get_escape_fre_term(sVec,nVec)
    p_mut_k_raw = get_p_k(sVec,nVec,seq_length,escape_group,escape_TF)
    
    # extend the time range
    interp_times = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
    if len(interp_times) > len(sample_times):
        
        # Use linear interpolates to get the input arrays at any integer time point
        interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_ex  = interp1d(sample_times, ex, axis=0, kind='linear', bounds_error=False, fill_value=0) if ne > 0 else 0
        interp_mut = interp1d(sample_times, p_mut_k_raw, axis=0, kind='linear', bounds_error=False, fill_value=0) if ne > 0 else 0
        
        single_freq = interp_x(interp_times)
        double_freq = interp_xx(interp_times)
        epitope_freq = interp_ex(interp_times) if ne > 0 else 0
        p_mut_k      = interp_mut(interp_times) if ne > 0 else 0

    else:
        single_freq   = x
        double_freq   = xx
        epitope_freq  = ex if ne > 0 else 0
        p_mut_k       = p_mut_k_raw if ne > 0 else 0

    # get mutation flux at sampled time points
    flux_mut = get_mut_flux(single_freq, epitope_freq, muVec)

    # Get matrix A and vector b
    change_tv = 0
    change_const = 0
    change_real = 0
    for ti in range(len(interp_times)-1):
        x_t, xx_t = single_freq[ti], double_freq[ti]
        sc_t = sc_tv[ti]
        sc_real = real_const + [real_tv[ti]]

        dt = interp_times[ti+1] - interp_times[ti]
        dx_t = (single_freq[ti+1] - single_freq[ti]) / dt

        # calculate C(t)
        C_raw = diffusion_matrix_at_t(x_t, xx_t) # covariance matrix

        # calculate flux(t) = flux_mut(t) + flux_rec(t)
        flux_total = flux_mut[ti]
        flux_rec = get_rec_flux_at_t(x_t[x_length-ne:], p_mut_k[ti], trait_dis) if ne > 0 else 0
        for n in range(ne): # recombination only for binary trait part
            flux_total[x_length-ne+n] += flux_rec[n]

        d_const = dx_t - C_raw @ sc_const - flux_total
        d_tv    = dx_t - C_raw @ sc_t - flux_total
        d_real  = dx_t - C_raw @ sc_real - flux_total

        dx2_const = np.sqrt(np.sum(d_const * d_const))      # d_const^T d_const * dt
        dx2_tv    = np.sqrt(np.sum(d_tv * d_tv))            # d_tv^T C^-1 d_tv * dt
        dx2_real  = np.sqrt(np.sum(d_real * d_real))        # d_real^T C^-1 d_real * dt

        change_const += dx2_const * dt
        change_tv    += dx2_tv * dt
        change_real  += dx2_real * dt

    changes = [change_const, change_tv, change_real]
    max_id  = np.argmax(changes)
    min_id  = np.argmin(changes)
    action_str = ['constant', 'time-varying', 'real']

    print(f'{name}|{change_const:.4f}|{change_tv:.4f}|{change_real:.4f}|{action_str[min_id]}|{action_str[max_id]}')

    # suffix = '|no epitope' if ne == 0 else ''
    # if action_const <= action_tv:
    #     print(f'CH{tag[-5:]}|{action_const:.4f}|{action_tv:.4f}|time-varying{suffix}')
    # else:
    #     print(f'CH{tag[-5:]}|{action_const:.4f}|{action_tv:.4f}|constant{suffix}')

    # return action_const, action_tv
