import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import re
import scipy as sp
from scipy import integrate
import scipy.interpolate as sp_interpolate
import statistics
import time as time_module

# GitHub
SIM_DIR = 'data/simulation'
HIV_DIR = 'data/HIV'
FIG_DIR = 'figures'

def read_file(name):
    result = []
    with open('%s/%s'%(SIM_DIR,name), 'r') as file:
        for line in file:
            line_data = []
            for item in line.split():
                line_data.append(int(item))
            result.append(line_data)
    return result

def simulate_simple(**pdata):
    """
    Example evolutionary trajectory for a 20-site system
    """

    # unpack passed data
    NUC           = pdata['NUC']            # ['A','T']
    dir           = pdata['dir']            # 'simple'
    xfile         = pdata['xfile']          #'0'
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

    q  = len(NUC)
    ############################################################################
    ############################## function ####################################
    # get fitness of new genotype
    def get_fitness_alpha(genotype,time):
        fitness = 1
        #alphabet format
        
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

    def get_recombination(genotype1,genotype2):
        #choose one possible mutation site
        recombination_point = np.random.randint(seq_length-1) + 1
        # get two offspring genotypes
        genotype_off = genotype1[:recombination_point] + genotype2[recombination_point:]
        return genotype_off

    def recombination_event(genotype,genotype_ran,pop):
        if pop[genotype] > 1:
            pop[genotype] -= 1
            if pop[genotype] == 0:
                del pop[genotype]

            new_genotype = get_recombination(genotype,genotype_ran)
            if new_genotype in pop:
                pop[new_genotype] += 1
            else:
                pop[new_genotype] = 1
        return pop
    
    # create all recombinations that occur in a single generation
    def recombination_step(pop):
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
                recombination_event(genotype,genotype_ran,pop)
        return pop
    
    # for different genotypes, they have different mutation probablity
    # this total mutation value represents the likelihood for one genotype to mutate
    # if there is only 2 alleles and only one mutation rate, total_mu = mutation rate * sequence length
    # take a supplied genotype and mutate a site at random.
    def get_mutant(genotype): #binary case
        #choose one possible mutation site
        site = np.random.randint(seq_length)
        # mutate (binary case, from WT to mutant or vice)
        mutation = list(NUC)
        mutation.remove(genotype[site])
        # get new mutation sequence
        new_genotype = genotype[:site] + mutation[0] + genotype[site+1:]
        return new_genotype

    # check if the mutant already exists in the population.
    #If it does, increment this mutant genotype
    #If it doesn't create a new genotype of count 1.
    # If a mutation event creates a new genotype, calculate its fitness.
    def mutation_event(genotype,pop):
        if pop[genotype] > 1:
            pop[genotype] -= 1
            if pop[genotype] == 0:
                del pop[genotype]

            new_genotype = get_mutant(genotype)
            if new_genotype in pop:
                pop[new_genotype] += 1
            else:
                pop[new_genotype] = 1
        return pop

    # create all the mutations that occur in a single generation
    def mutation_step(pop):
        genotypes = list(pop.keys())
        for genotype in genotypes:
            n = pop[genotype]
            # calculate the likelihood to mutate
            total_mu = seq_length * mut_rate # for binary case
            nMut = np.random.binomial(n, total_mu)
            for j in range(nMut):
                mutation_event(genotype,pop)
        return pop

    # genetic drift
    def offspring_step(pop,time):
        genotypes = list(pop.keys())
        r = []
        for genotype in genotypes:
            numbers = pop[genotype]
            fitness = get_fitness_alpha(genotype,time)
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

    # in every generation, it will mutate and then the genetic drift
    # calculate several times to get the evolution trajectory
    # At each step in the simulation, we append to a history object.
    def simulate(pop,history):
        clone_pop = dict(pop)
        history.append(clone_pop)
        for t in range(generations):
            recombination_step(pop)
            mutation_step(pop)
            offspring_step(pop,t)
            clone_pop = dict(pop)
            history.append(clone_pop)
        return history

    # transfer output from alphabet to number
    def get_sequence(genotype):
        escape_states = []
        for i in range(len(genotype)):
            for k in range(q):
                if genotype[i] == NUC[k]:
                    escape_states.append(str(k))
        return escape_states

    def initial_dis(pop,inital_state,pop_size):
        n_seqs  = int(pop_size/inital_state)
        for ss in range(inital_state):
            sequences = ''
            for i in range(seq_length):
                temp_seq   = np.random.choice(np.arange(0, q), p=[0.8, 0.2])
                allele_i   = NUC[temp_seq]
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

    ############################################################################
    ############################## Simulate ####################################
    pop = {}
    initial_dis(pop,inital_state,pop_size)
    history = []
    simulate(pop,history)

    # write the output file - dat format
    f = open("%s/%s/sequences/example-%s.dat"%(SIM_DIR,dir,xfile),'w')

    for i in range(len(history)):
        pop_at_t = history[i]
        genotypes = pop_at_t.keys()
        for genotype in genotypes:
            time = i
            counts = pop_at_t[genotype]
            sequence = get_sequence(genotype)
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
    NUC           = pdata['NUC']            # ['A','T']
    dir           = pdata['dir']            # 'trait'
    xfile         = pdata['xfile']          #'1-con'
    seq_length    = pdata['seq_length']     # 20
    pop_size      = pdata['pop_size']       # 1000
    generations   = pdata['generations']    # 500
    mut_rate      = pdata['mut_rate']       # 1e-3
    rec_rate      = pdata['rec_rate']       # 1e-3
    inital_state  = pdata['inital_state']   # 4

    nB            = pdata['n_ben']          # 4
    nD            = pdata['n_del']          # 0.02
    fB            = pdata['s_ben']          # 4
    fD            = pdata['s_del']          # 0.02
    escape_group  = pdata['escape_group']   # [[2,5,8]], escape group
    p_sites       = pdata['p_sites']        # [13,18] , special sites
    fi            = pdata['fi']             # time-varying selection coefficient
    fn            = pdata['fn']             # time-varying escape coefficient

    q  = len(NUC)
    ne = len(escape_group)

    ############################################################################
    ############################## function ####################################
    # get fitness of new genotype
    def get_fitness_alpha(genotype,time):
        fitness = 1
        #alphabet format
        
        # individual locus
        for i in range(seq_length):
            if genotype[i] != "A": # mutant type
                if i in p_sites: # special site
                    fitness += fi[time]
                else:
                    if i < nB: # beneficial mutation
                        fitness += fB
                    elif i >= seq_length-nD: # deleterious mutation
                        fitness += fD
        
        # binary trait
        for n in range(ne):
            for nn in escape_group[n]:
                if genotype[nn] != "A":
                    fitness += fn[time]
                    break
        return fitness

    def get_recombination(genotype1,genotype2):
        #choose one possible mutation site
        recombination_point = np.random.randint(seq_length-1) + 1
        # get two offspring genotypes
        genotype_off = genotype1[:recombination_point] + genotype2[recombination_point:]
        return genotype_off

    def recombination_event(genotype,genotype_ran,pop):
        if pop[genotype] > 1:
            pop[genotype] -= 1
            if pop[genotype] == 0:
                del pop[genotype]

            new_genotype = get_recombination(genotype,genotype_ran)
            if new_genotype in pop:
                pop[new_genotype] += 1
            else:
                pop[new_genotype] = 1
        return pop
    
    # create all recombinations that occur in a single generation
    def recombination_step(pop):
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
                recombination_event(genotype,genotype_ran,pop)
        return pop
    
    # for different genotypes, they have different mutation probablity
    # this total mutation value represents the likelihood for one genotype to mutate
    # if there is only 2 alleles and only one mutation rate, total_mu = mutation rate * sequence length
    # take a supplied genotype and mutate a site at random.
    def get_mutant(genotype): #binary case
        #choose one possible mutation site
        site = np.random.randint(seq_length)
        # mutate (binary case, from WT to mutant or vice)
        mutation = list(NUC)
        mutation.remove(genotype[site])
        # get new mutation sequence
        new_genotype = genotype[:site] + mutation[0] + genotype[site+1:]
        return new_genotype

    # check if the mutant already exists in the population.
    #If it does, increment this mutant genotype
    #If it doesn't create a new genotype of count 1.
    # If a mutation event creates a new genotype, calculate its fitness.
    def mutation_event(genotype,pop):
        if pop[genotype] > 1:
            pop[genotype] -= 1
            if pop[genotype] == 0:
                del pop[genotype]

            new_genotype = get_mutant(genotype)
            if new_genotype in pop:
                pop[new_genotype] += 1
            else:
                pop[new_genotype] = 1
        return pop

    # create all the mutations that occur in a single generation
    def mutation_step(pop):
        genotypes = list(pop.keys())
        for genotype in genotypes:
            n = pop[genotype]
            # calculate the likelihood to mutate
            total_mu = seq_length * mut_rate # for binary case
            nMut = np.random.binomial(n, total_mu)
            for j in range(nMut):
                mutation_event(genotype,pop)
        return pop

    # genetic drift
    def offspring_step(pop,time):
        genotypes = list(pop.keys())
        r = []
        for genotype in genotypes:
            numbers = pop[genotype]
            fitness = get_fitness_alpha(genotype,time)
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

    # in every generation, it will mutate and then the genetic drift
    # calculate several times to get the evolution trajectory
    # At each step in the simulation, we append to a history object.
    def simulate(pop,history):
        clone_pop = dict(pop)
        history.append(clone_pop)
        for t in range(generations):
            recombination_step(pop)
            mutation_step(pop)
            offspring_step(pop,t)
            clone_pop = dict(pop)
            history.append(clone_pop)
        return history

    # transfer output from alphabet to number
    def get_sequence(genotype):
        escape_states = []
        for i in range(len(genotype)):
            for k in range(q):
                if genotype[i] == NUC[k]:
                    escape_states.append(str(k))
        return escape_states

    def initial_dis(pop,inital_state,pop_size):
        n_seqs  = int(pop_size/inital_state)
        for ss in range(inital_state):
            sequences = ''
            for i in range(seq_length):
                temp_seq   = np.random.choice(np.arange(0, q), p=[0.8, 0.2])
                allele_i   = NUC[temp_seq]
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

    ############################################################################
    ############################## Simulate ####################################
    pop = {}
    initial_dis(pop,inital_state,pop_size)
    history = []
    simulate(pop,history)

    # write the output file - dat format
    f = open("%s/%s/sequences/example-%s.dat"%(SIM_DIR,dir,xfile),'w')

    for i in range(len(history)):
        pop_at_t = history[i]
        genotypes = pop_at_t.keys()
        for genotype in genotypes:
            time = i
            counts = pop_at_t[genotype]
            sequence = get_sequence(genotype)
            f.write('%d\t%d\t' % (time,counts))
            for j in range(len(sequence)):
                f.write(' %s' % (' '.join(sequence[j])))
            f.write('\n')
    f.close()

# loading data from dat file
def getSequence(history,escape_group):
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
                    if history[t][index] != 0:
                        temp_escape[n] = 1
                        break
            temp_eVec.append(temp_escape)

        if t == len(history)-1:
            sVec.append(temp_sVec)
            nVec.append(temp_nVec)
            eVec.append(temp_eVec)

    return sVec,nVec,eVec

# calculate frequencies for recombination part
def get_p_k(sVec,nVec,seq_length,escape_group,escape_TF):
    p_mut_k   = np.zeros((len(nVec),seq_length,3)) # 0: time, 1: all k point, 2: p_k, p_k-, p_k+
    p_wt      = np.zeros((len(nVec),len(escape_group))) # 0: time, 1: escape group

    for t in range(len(nVec)):
        pop_size_t = np.sum([nVec[t]])
        
        for n in range(len(escape_group)):
            escape_group_n = escape_group[n]
            sWT_n     = [int(i) for i in escape_TF[n]]

            for k in range(len(sVec[t])): # different sequences at time t
                sVec_n = [int(sVec[t][k][i]) for i in escape_group_n]
                
                # no mutation within the trait group
                if sWT_n == sVec_n:
                    p_wt[t][n] += nVec[t][k]

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

        p_wt[t]    = p_wt[t] / pop_size_t
        p_mut_k[t] = p_mut_k[t] / pop_size_t

    return p_wt,p_mut_k
 
# calculate diffusion matrix C
def diffusion_matrix_at_t(x,xx,x_length):
    C = np.zeros([len(x),x_length,x_length])
    for t in range(len(x)):
        for i in range(x_length):
            C[t,i,i] = x[t,i] - x[t,i] * x[t,i]
            for j in range(int(i+1) ,x_length):
                C[t,i,j] = xx[t,i,j] - x[t,i] * x[t,j]
                C[t,j,i] = xx[t,i,j] - x[t,i] * x[t,j]
    return C

# calculate the difference between the frequency at time t and time t-1
def cal_delta_x(single_freq,times,x_length):
    delta_x = np.zeros((len(single_freq),x_length))   # difference between the frequency at time t and time t-1s
#     calculate by np.gradient function
#         for ii in range(x_length):
#             delta_x[:,ii] = np.gradient(single_freq.T[ii],times)
#     calculate manually
    for tt in range(len(single_freq)-1):
        h = times[tt+1]-times[tt]
        delta_x[tt] = (single_freq[tt+1] - single_freq[tt])/h
    delta_x[-1] = delta_x[-2]
    return delta_x

# linear interpolation
interpolation = lambda a,b: sp_interpolate.interp1d(a,b,kind='linear')

def interpolator_p(p_wt, p_mut_k, current_times, result_times, seq_length, ne):
    """ Interpolates the input arrays so that they will have the same number of generations as the original population. """

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

def infer_simple(**pdata):
    """
    Infer time-varying example (binary case)
    """

    # unpack passed data
    NUC           = pdata['NUC']            # ['A','T']
    dir           = pdata['dir']            # 'trait'
    xfile         = pdata['xfile']          
    seq_length    = pdata['seq_length']     # 20
    totalT        = pdata['totalT']         # 500
    mut_rate      = pdata['mut_rate']       # 1e-3

    p_1           = pdata['p_1']            # [6,7]
    p_2           = pdata['p_2']            # [8,9]

    gamma_1s      = pdata['gamma_s']/totalT # gamma_s/time points
    gamma_2c      = pdata['gamma_2c']       # 1000000
    gamma_2tv     = pdata['gamma_2tv']      # 200

    bc_n          = pdata['bc_n']         # True

    p_sites       = p_1+p_2        # [6,7,8,9] , special sites
    ############################################################################
    ############################## Function ####################################

    def getSequence(history):
        sVec      = []
        nVec      = []
        temp_sVec   = []
        temp_nVec   = []

        times       = []
        time        = 0
        times.append(time)

        for t in range(len(history)):
            if history[t][0] != time:
                time = history[t][0]
                times.append(int(time))
                sVec.append(temp_sVec)
                nVec.append(temp_nVec)
                temp_sVec   = []
                temp_nVec   = []

            temp_nVec.append(history[t][1])
            temp_sVec.append(history[t][2:])

            if t == len(history)-1:
                sVec.append(temp_sVec)
                nVec.append(temp_nVec)

        return sVec,nVec

    # get muVec for binary case without threshold
    def getMutantS():
        muVec    = -np.ones(seq_length)
        x_length = 0
        for i in range(seq_length):
            muVec[i] = x_length
            x_length += 1
        return x_length,muVec

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
    def get_mutation_flux(x,muVec):
        flux = np.zeros((len(x),x_length))
        for t in range(len(x)):
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    flux[t, aa] = mut_rate * ( 1 - 2 * x[t, aa])
        return flux
    
    # Interpolation function for frequencies (binary case)
    def interpolator_x(single_freq, double_freq, current_times, result_times):
        """ Interpolates the input arrays so that they will have the same number of generations as the original population. """

        single_freq_temp = np.zeros((len(result_times),x_length))
        double_freq_temp = np.zeros((len(result_times),x_length,x_length))
        
        # interpolation for single frequency and double frequency
        for i in range(x_length):
            single_freq_temp[:,i] = interpolation(current_times, single_freq[:,i])(result_times)
            for j in range(x_length):
                double_freq_temp[:,i,j] = interpolation(current_times, double_freq[:,i,j])(result_times)

        single_freq_temp = single_freq_temp[:len(result_times)]
        double_freq_temp = double_freq_temp[:len(result_times)]

        return single_freq_temp, double_freq_temp

    ############################################################################
    ####################### Inference (binary case) ############################
    
    # obtain raw data and information of traits
    data         = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR,dir,xfile))

    # obtain sequence data and frequencies
    sVec,nVec      = getSequence(data)
    x_length,muVec = getMutantS()

    # get all frequencies, x: single allele frequency, xx: pair allele frequency
    # ex: escape frequency, p_wt,p_mut_k: frequency related to recombination part
    x,xx         = get_allele_frequency(sVec,nVec,muVec) 

    # infer the beginning part of the whole sequence
    times        = np.linspace(0,totalT,totalT+1)
    
    # use the data within the range
    single_freq  =  x[:totalT+1]
    double_freq  = xx[:totalT+1]

    # covariance matrix, flux term and delta_x
    covariance_n = diffusion_matrix_at_t(single_freq, double_freq,x_length)
    covariance   = np.swapaxes(covariance_n, 0, 2)
    flux_mu      = get_mutation_flux(single_freq,muVec)
    delta_x      = cal_delta_x(single_freq,times,x_length)

    # extend the time range
    TLeft   = int(round(times[-1]*2/10)*10) # time range added before the beginning time
    # TRight  = int(round(times[-1]*2/10)*10) # time range added after the ending time
    etleft  = np.linspace(-TLeft,-40,int(TLeft/40)) # time added before the beginning time (dt=10)
    # etright = np.linspace(times[-1]+40,times[-1]+TRight,int(TRight/40))
    # ExTimes = np.concatenate((etleft, times, etright))
    # TLeft   = int(round(times[-1]*0.5/10)*10) # time range added before the beginning time
    TRight  = int(round(times[-1]*0.5/10)*10) # time range added after the ending time
    # etleft  = np.linspace(-TLeft,-10,int(TLeft/10)) # time added before the beginning time (dt=10)
    etright = np.linspace(times[-1]+10,times[-1]+TRight,int(TRight/10))
    ExTimes = np.concatenate((etleft, times, etright))

    # regularization value gamma_1 and gamma_2
    # individual site: gamma_1s, escape group: gamma_1p
    gamma1 = np.ones(x_length)*gamma_1s

    # individual site: gamma_2c, escape group and special site: gamma_2tv
    # gamma 2 is also time varying, it is smaller at the boundary
    gamma_t = np.zeros(len(ExTimes))
    beta_1 = 4
    beta_2 = 4
    ratio_1 = 0.1
    ratio_2 = 0.1
    alpha1  = np.log(beta_1) / (ratio_1*1000)
    alpha2  = np.log(beta_2) / (ratio_2*1000)

    for t in range(len(ExTimes)):
        if t <= len(etleft):
            gamma_t[t] = beta_1
        elif t >= len(etleft)+len(times):
            gamma_t[t] = beta_2
        elif len(etleft) < t and t < len(etleft)+(len(times)-1)*ratio_1:
            tt = (t - len(etleft)) * 1
            gamma_t[t] = beta_1 * np.exp(-alpha1 * tt)
        elif len(etleft)+(len(times)-1)*(1 - ratio_2) < t and t < len(etleft)+len(times):
            tt = (t - len(etleft)) * 1 - (1 - ratio_2) * totalT
            gamma_t[t] = 1 * np.exp(alpha2 * tt)
        else:
            gamma_t[t] = 1

    gamma2 = np.ones((x_length,len(ExTimes))) * gamma_2c
    for i in range(len(p_sites)): # special site - time varying
        index = int (muVec[p_sites[i]]) 
        if index != -1:
            gamma2[index] = gamma_t * gamma_2tv

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
                    result[x_length+i,t] = (mat_prod[i,tt] + gamma1[i] * b_1[i,t] + flux_mu[tt,i] - delta_x[tt,i]) / gamma2[i,t]
            
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
        
        # use the interpolations from above to get the values of delta_x and the covariance matrix at the nodes
        flux_mu_int  = get_mutation_flux(single_freq_int,muVec)
        delta_x_int  = cal_delta_x(single_freq_int,a,x_length)
        covar_int    = diffusion_matrix_at_t(single_freq_int, double_freq_int,x_length)
        covar_int    = np.swapaxes(covar_int,0,2)

        # calculate the other half of the RHS of the ODE system
        mat_prod_int  = np.sum(covar_int[:,:,:len(a)] * b_1[:,len(etleft):len(etleft)+len(times)], 1)

        for t in range(len(a)): # right hand side of second half of the ODE system
            # within the time range
            if len(etleft) <= t < len(etleft)+len(times):
                tt = t - len(etleft)
                for i in range(x_length):
                    result[x_length+i,t] = (mat_prod_int[i,tt] + gamma1[i] * b_1[i,t] + flux_mu_int[tt,i] - delta_x_int[tt,i]) / gamma2[i]
            
            # outside the time range, no selection strength
            else:
                for i in range(x_length):
                    result[x_length+i,t] = gamma1[i] * b_1[i,t] / gamma2[i]

        return result

    def bc(b1,b2):
        if bc_n: # if using Neumann boundary condition
            return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints
        else: # using Dirichlet boundary condition
            return np.ravel(np.array([b1[:x_length],b2[:x_length]])) # s = 0 at the extended endpoints

    ss_extend = np.zeros((2*x_length,len(ExTimes)))
    try:
        solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)
    except ValueError:
        print("BVP solver has to add new nodes")
        solution = sp.integrate.solve_bvp(fun_advanced, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)

    selection_coefficients = solution.sol(ExTimes)
    # removes the superfluous part of the array and only save the real time points
    desired_coefficients   = selection_coefficients[:x_length,len(etleft):len(etleft)+len(times)]

    # save the solution with constant_time-varying selection coefficient
    g = open('%s/%s/output_ls_%d%d/c_%s.npz'%(SIM_DIR,dir,int(ratio_1*10),int(ratio_2*10),xfile), mode='w+b')
    np.savez_compressed(g, selection=desired_coefficients, all = selection_coefficients, time=times)
    g.close()

def infer_trait(**pdata):
    """
    Infer time-varying example (binary case)
    """

    # unpack passed data
    NUC           = pdata['NUC']            # ['A','T']
    dir           = pdata['dir']            # 'trait'
    xfile         = pdata['xfile']          
    seq_length    = pdata['seq_length']     # 20
    totalT        = pdata['totalT']         # 1000
    mut_rate      = pdata['mut_rate']       # 1e-3
    rec_rate      = pdata['rec_rate']

    p_sites       = pdata['p_sites']        # [13,18] , special sites

    gamma_1s      = pdata['gamma_s']/totalT # gamma_s/time points
    gamma_1p      = gamma_1s/10
    gamma_2c      = pdata['gamma_2c']       # 1000000
    gamma_2tv     = pdata['gamma_2tv']      # 500
    bc_n          = pdata['bc_n']           # True   

    ############################################################################
    ############################## Function ####################################
    # get muVec for binary case with threshold
    def getMutantS():
        muVec    = -np.ones(seq_length)
        x_length = 0
        for i in range(seq_length):
            muVec[i] = x_length
            x_length += 1
        return x_length,muVec

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
    def get_mutation_flux(x,ex,muVec):
        flux = np.zeros((len(x),x_length))
        for t in range(len(x)):
            for i in range(seq_length):
                aa = int(muVec[i])
                if aa != -1:
                    flux[t, aa] = mut_rate * ( 1 - 2 * x[t, aa])
            for n in range(ne):
                for nn in escape_group[n]:
                    flux[t, x_length-ne+n] += mut_rate * (1 - x[t, x_length-ne+n] - ex[t,n,nn] )
        return flux

    # calculate recombination flux term (binary_case)
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
                
                flux[t,x_length-ne+n] += rec_rate * (fluxIn - fluxOut)

        return flux
    
    # Interpolation function for frequencies (binary case)
    def interpolator_x(single_freq, double_freq, escape_freq, current_times, result_times):
        """ Interpolates the input arrays so that they will have the same number of generations as the original population. """

        single_freq_temp = np.zeros((len(result_times),x_length))
        double_freq_temp = np.zeros((len(result_times),x_length,x_length))
        escape_freq_temp = np.zeros((len(result_times),ne,seq_length))
        
        # interpolation for single frequency and double frequency
        for i in range(x_length):
            single_freq_temp[:,i] = interpolation(current_times, single_freq[:,i])(result_times)
            for j in range(x_length):
                double_freq_temp[:,i,j] = interpolation(current_times, double_freq[:,i,j])(result_times)
        # interpolation for escape frequency
        for n in range(ne):
            for i in range(seq_length):
                escape_freq_temp[:,n,i] = interpolation(current_times, escape_freq[:,n,i])(result_times)

        single_freq_temp = single_freq_temp[:len(result_times)]
        double_freq_temp = double_freq_temp[:len(result_times)]
        escape_freq_temp = escape_freq_temp[:len(result_times)]

        return single_freq_temp, double_freq_temp, escape_freq_temp

    ############################################################################
    ####################### Inference (binary case) ############################
    
    # obtain raw data and information of traits
    data         = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR,dir,xfile))
    escape_group = read_file('%s/traitsite/traitsite-%s.dat'%(dir,xfile))
    trait_dis    = read_file('%s/traitdis/traitdis-%s.dat'%(dir,xfile))
        
    escape_TF = [[0,0,0]]
    ne        = len(escape_group)

    # obtain sequence data and frequencies
    sVec,nVec,eVec = getSequence(data,escape_group)
    x_length,muVec = getMutantS()
    x_length      += ne

    # get all frequencies, x: single allele frequency, xx: pair allele frequency
    # ex: escape frequency, p_wt,p_mut_k: frequency related to recombination part
    x,xx         = get_allele_frequency(sVec,nVec,eVec,muVec) 
    ex           = get_escape_fre_term(sVec,nVec)
    p_wt,p_mut_k = get_p_k(sVec,nVec,seq_length,escape_group,escape_TF)

    # infer the beginning part of the whole sequence
    times        = np.linspace(0,totalT,totalT+1)

    # use the data within the range
    single_freq  =  x[:totalT+1]
    double_freq  = xx[:totalT+1]
    escape_freq  = ex[:totalT+1]
    p_wt_freq    = p_wt[:totalT+1]
    p_mut_k_freq = p_mut_k[:totalT+1]

    # covariance matrix, flux term and delta_x
    covariance_n = diffusion_matrix_at_t(single_freq, double_freq,x_length)
    covariance   = np.swapaxes(covariance_n, 0, 2)
    flux_mu      = get_mutation_flux(single_freq,escape_freq,muVec)
    flux_rec     = get_recombination_flux(single_freq,p_wt_freq,p_mut_k_freq,trait_dis)# recombination part
    delta_x      = cal_delta_x(single_freq,times,x_length)

    # extend the time range
    TLeft   = int(round(times[-1]*2/10)*10) # time range added before the beginning time
    etleft  = np.linspace(-TLeft,-40,int(TLeft/40)) # time added before the beginning time (dt=10)
    # TRight  = int(round(times[-1]*2/10)*10) # time range added after the ending time
    # etright = np.linspace(times[-1]+40,times[-1]+TRight,int(TRight/40))

    # TLeft   = int(round(times[-1]*0.5/10)*10) # time range added before the beginning time
    # etleft  = np.linspace(-TLeft,-10,int(TLeft/10)) # time added before the beginning time (dt=10)
    TRight  = int(round(times[-1]*0.5/10)*10) # time range added after the ending time
    etright = np.linspace(times[-1]+10,times[-1]+TRight,int(TRight/10))

    ExTimes = np.concatenate((etleft, times, etright))
    
    # regularization value gamma_1 and gamma_2
    # individual site: gamma_1s, escape group: gamma_1p
    gamma1 = np.ones(x_length)*gamma_1s
    for n in range(ne):
        gamma1[x_length-ne+n] = gamma_1p

    # gamma 2 is also time varying, it is larger at the boundary
    gamma_t = np.zeros(len(ExTimes))
    beta_1 = 4
    beta_2 = 4
    ratio_1 = 0.1
    ratio_2 = 0.1
    alpha1  = np.log(beta_1) / (ratio_1*1000)
    alpha2  = np.log(beta_2) / (ratio_2*1000)
    for t in range(len(ExTimes)):
        if t <= len(etleft):
            gamma_t[t] = beta_1
        elif t >= len(etleft)+len(times):
            gamma_t[t] = beta_2
        elif len(etleft) < t and t < len(etleft)+(len(times)-1)*ratio_1:
            tt = (t - len(etleft)) * 1
            gamma_t[t] = beta_1 * np.exp(-alpha1 * tt)
        elif len(etleft)+(len(times)-1)*(1 - ratio_2) < t and t < len(etleft)+len(times):
            tt = (t - len(etleft)) * 1 - (1 - ratio_2) * totalT
            gamma_t[t] = 1 * np.exp(alpha2 * tt)
        else:
            gamma_t[t] = 1

    # individual site: gamma_2c, escape group and special site: gamma_2tv
    gamma2 = np.ones((x_length,len(ExTimes)))*gamma_2c
    for n in range(ne):
        gamma2[x_length-ne+n] = gamma_t * gamma_2tv
    for i in range(len(p_sites)): # special site - time varying
        index = int (muVec[p_sites[i]]) 
        if index != -1:
            gamma2[index] = gamma_t * gamma_2tv

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
        single_freq_int, double_freq_int, escape_freq_int = interpolator_x(single_freq, double_freq, escape_freq, times, a)
        p_wt_int, p_mut_k_int                             = interpolator_p(p_wt_freq,p_mut_k_freq,times,a,seq_length,ne)
        
        # use the interpolations from above to get the values of delta_x and the covariance matrix at the nodes
        flux_mu_int  = get_mutation_flux(single_freq_int, escape_freq_int,muVec)
        flux_rec_int = get_recombination_flux(single_freq_int, p_wt_int,p_mut_k_int,trait_dis)# recombination part
        delta_x_int  = cal_delta_x(single_freq_int,a,x_length)
        covar_int    = diffusion_matrix_at_t(single_freq_int, double_freq_int,x_length)
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

    def bc(b1,b2):
        if bc_n: # Neumann boundary condition
            return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints
        else: # Dirichlet boundary condition
            return np.ravel(np.array([b1[:x_length],b2[:x_length]])) # s = 0 at the extended endpoints

    ss_extend = np.zeros((2*x_length,len(ExTimes)))
    try:
        solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)
    except ValueError:
        print("BVP solver has to add new nodes")
        solution = sp.integrate.solve_bvp(fun_advanced, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)

    selection_coefficients = solution.sol(ExTimes)
    # removes the superfluous part of the array and only save the real time points
    desired_coefficients   = selection_coefficients[:x_length,len(etleft):len(etleft)+len(times)]

    # save the solution with constant_time-varying selection coefficient
    if bc_n: # Neumann boundary condition
        g = open('%s/%s/output_ls_11/c_%s.npz'%(SIM_DIR,dir,xfile), mode='w+b')
    else: # Dirichlet boundary condition
        g = open('%s/%s/output_d/c_%s.npz'%(SIM_DIR,dir,xfile), mode='w+b')
    np.savez_compressed(g, selection=desired_coefficients, all = selection_coefficients, time=times)
    g.close()

# def infer_multiple(**pdata):
#     """
#     Infer time-varying example (multiple case)
#     """

#     # unpack passed data
#     NUC           = pdata['NUC']            # ['A','T']
#     dir           = pdata['dir']            # 'trait'
#     xfile         = pdata['xfile']          # '0'
#     seq_length    = pdata['seq_length']     # 20
#     totalT        = pdata['totalT']         # 500
#     mut_rate      = pdata['mut_rate']       # 1e-3
#     rec_rate      = pdata['rec_rate']

#     p_sites       = pdata['p_sites']        # [13,18] , special sites
#     # x_thresh      = pdata['x_thresh']

#     gamma_1s      = pdata['gamma_s']/totalT # gamma_s/time points
#     gamma_1p      = gamma_1s/10
#     gamma_2c      = pdata['gamma_2c']       # 1000000
#     gamma_2tv     = pdata['gamma_2tv']      # 500
#     IF_raw        = pdata['IF_raw']         # True  
    
#     ############################################################################
#     ############################## Function ####################################
#     # get muVec for multiple case
#     def getMutantS(sVec):
#         # use muVec matrix to record the index of time-varying sites
#         muVec = -np.ones((seq_length, q)) # default value is -1, positive number means the index
#         x_length  = 0
#         for i in range(seq_length):
#             # find all possible alleles in site i
#             alleles     = [int(sVec[t][k][i]) for t in range(len(sVec)) for k in range(len(sVec[t]))]
#             allele_uniq = np.unique(alleles)
#             for allele in allele_uniq:
#                 # do not throw out exsiting alleles
#                 muVec[i][int(allele)] = x_length
#                 x_length += 1
#         return x_length,muVec

#     # # get muVec for multiple case (use the frequency threshold)
#     # def getMutantS(sVec,nVec):
#     #     # use muVec matrix to record the index of time-varying sites(after throwing out weak linkage sites)
#     #     muVec = -np.ones((seq_length, q)) # default value is -1, positive number means the index
#     #     x_length  = 0
#     #     for i in range(seq_length):
#     #         # find all possible alleles in site i
#     #         alleles     = [int(sVec[t][k][i]) for t in range(len(sVec)) for k in range(len(sVec[t]))]
#     #         allele_uniq = np.unique(alleles)
#     #         for allele in allele_uniq:
#     #             # throw out the alleles with low frequency
#     #             allele_count = np.zeros(len(sVec))
#     #             allele_count = [np.sum([(sVec[t][k][i]==allele)*nVec[t][k] for k in range(len(sVec[t]))]) for t in range(len(sVec))]
#     #             if max(allele_count) / np.sum(nVec[0]) >= x_thresh:
#     #                 muVec[i][int(allele)] = x_length
#     #                 x_length += 1
#     #     return x_length,muVec
    
#     # calculate single and pair allele frequency (multiple case)
#     def get_allele_frequency(sVec,nVec,eVec,muVec):
#         x  = np.zeros((len(nVec),x_length))           # single allele frequency
#         xx = np.zeros((len(nVec),x_length,x_length))  # pair allele frequency
#         for t in range(len(nVec)):
#             pop_size_t = np.sum([nVec[t]])
#             for k in range(len(nVec[t])): # all frequencies at time t
#                 # individual locus part
#                 for i in range(seq_length):
#                     qq = int(sVec[t][k][i]) # allele at site i for sequence k at time t
#                     aa = int(muVec[i][qq])  # index of allele qq at site i
#                     if aa != -1: # if aa = -1, it means the allele does not exist
#                         x[t,aa] += nVec[t][k]
#                         for j in range(int(i+1), seq_length):
#                             qq = int(sVec[t][k][j])
#                             bb = int(muVec[j][qq])
#                             if bb != -1:
#                                 xx[t,aa,bb] += nVec[t][k]
#                                 xx[t,bb,aa] += nVec[t][k]
#                 # escape part
#                 for n in range(ne):
#                     aa = int(x_length-ne+n) # index of escape group n
#                     x[t,aa] += eVec[t][k][n] * nVec[t][k]
#                     for m in range(int(n+1), ne):
#                         bb = int(x_length-ne+m)
#                         xx[t,aa,bb] += eVec[t][k][n] * eVec[t][k][m] * nVec[t][k]
#                         xx[t,bb,aa] += eVec[t][k][n] * eVec[t][k][m] * nVec[t][k]
#                     for j in range(seq_length):
#                         qq = int(sVec[t][k][j])
#                         bb = int(muVec[j][qq])
#                         if bb != -1:
#                             xx[t,bb,aa] += eVec[t][k][n] * nVec[t][k]
#                             xx[t,aa,bb] += eVec[t][k][n] * nVec[t][k]

#             x[t,:]    = x[t,:]/pop_size_t
#             xx[t,:,:] = xx[t,:,:]/pop_size_t

#         return x,xx

#     # calculate escape frequency (multiple case)
#     def get_escape_fre_term(sVec,nVec):
#         ex  = np.zeros((len(nVec),ne,seq_length,q))
#         for t in range(len(nVec)):
#             pop_size_t = np.sum([nVec[t]])
#             for k in range(len(sVec[t])):
#                 for n in range(ne):
#                     site_mutation = []
#                     for nn in escape_group[n]:
#                         index = escape_group[n].index(nn)
#                         WT = escape_TF[n][index]
#                         if sVec[t][k][nn] != WT:
#                             site_mutation.append(nn)
#                     if len(site_mutation) == 1:
#                         site = site_mutation[0]
#                         qq = int(sVec[t][k][site])
#                         ex[t,n,site,qq] += nVec[t][k]
#             ex[t,:,:,:] = ex[t,:,:,:] / pop_size_t
#         return ex

#     # calculate mutation flux term (multiple case)
#     def get_mutation_flux(x,ex,muVec):
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
#                         if a != WT:
#                             flux[t, x_length-ne+n] += muMatrix[WT][a] * (1 - x[t,x_length-ne+n]) - muMatrix[a][WT] * ex[t,n,index,a]
#         return flux
    
#     # calculate recombination flux term (mutiple case)
#     def get_recombination_flux(x,p_wt,p_mut_k,trait_dis):
#         flux = np.zeros((len(x),x_length))
#         for n in range(ne):
#             for t in range(len(x)):
#                 fluxIn  = 0
#                 fluxOut = 0

#                 for nn in range(len(escape_group[n])-1):
#                     k_index = escape_group[n][0]+nn
#                     fluxIn  += trait_dis[n][nn] * p_wt[t][n]*p_mut_k[t][k_index][0]
#                     fluxOut += trait_dis[n][nn] * p_mut_k[t][k_index][1]*p_mut_k[t][k_index][2]
                
#                 flux[t,x_length-ne+n] = rec_rate * (fluxIn - fluxOut)

#         return flux
    
#     # Interpolation function for frequencies (multiple case)
#     def interpolator_x(single_freq, double_freq, escape_freq, current_times, result_times):
#         """ Interpolates the input arrays so that they will have the same number of generations as the original population. """

#         single_freq_temp = np.zeros((len(result_times),x_length))
#         double_freq_temp = np.zeros((len(result_times),x_length,x_length))
#         escape_freq_temp  = np.zeros((len(result_times),ne,seq_length,q))
        
#         # interpolation for single frequency and double frequency
#         for i in range(x_length):
#             single_freq_temp[:,i] = interpolation(current_times, single_freq[:,i])(result_times)
#             for j in range(x_length):
#                 double_freq_temp[:,i,j] = interpolation(current_times, double_freq[:,i,j])(result_times)
#         # interpolation for escape frequency
#         for n in range(ne):
#             for i in range(seq_length):
#                 for a in range(q):
#                     escape_freq_temp[:,n,i,a] = interpolation(current_times, escape_freq[:,n,i,a])(result_times)
        
#         single_freq_temp = single_freq_temp[:len(result_times)]
#         double_freq_temp = double_freq_temp[:len(result_times)]
#         escape_freq_temp = escape_freq_temp[:len(result_times)]

#         return single_freq_temp, double_freq_temp, escape_freq_temp

#     ############################################################################
#     ###################### Inference (multiple case) ###########################
#     # obtain raw data and information of traits
#     if IF_raw:
#         data         = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR,dir,xfile))
#         escape_group = read_file('%s/traitsite/traitsite-%s.dat'%(dir,xfile))
#         trait_dis    = read_file('%s/traitdis/traitdis-%s.dat'%(dir,xfile))
#     else: # read data with finite sampling noise 
#         data         = np.loadtxt("%s/%s/sequences/nsdt/example-%s.dat"%(SIM_DIR,dir,xfile))
#         file_number  = xfile.split('_')[0]
#         escape_group = read_file('traitsite/traitsite-%s.dat'%(file_number))
#         trait_dis    = read_file('traitdis/traitdis-%s.dat'%(file_number))

#     # information of mutation
#     q         = len(NUC)
#     muMatrix  = [[0,mut_rate],[mut_rate,0]]
#     escape_TF = [[0,0,0]]
#     ne        = len(escape_group)   

#     # obtain sequence data and frequencies
#     sVec,nVec,eVec = getSequence(data,escape_group)
#     x_length,muVec = getMutantS(sVec) #getMutantS(sVec,nVec)
#     x_length      += ne

#     # regularization value gamma_1 and gamma_2
#     # individual site: gamma_1s, escape group: gamma_1p
#     gamma1 = np.ones(x_length)*gamma_1s
#     for n in range(ne):
#         gamma1[x_length-ne+n] = gamma_1p

#     # individual site: gamma_2c, escape group and special site: gamma_2tv
#     gamma2 = np.ones(x_length)*gamma_2c
#     for n in range(ne):
#         gamma2[x_length-ne+n] = gamma_2tv
#     for p_site in p_sites: # special site - time varying
#         for qq in range(len(NUC)):
#             index = int (muVec[p_site][qq]) 
#             if index != -1:
#                 gamma2[index] = gamma_2tv

#     # get all frequencies, x: single allele frequency, xx: pair allele frequency
#     # ex: escape frequency, p_wt,p_mut_k: frequency related to recombination part
#     x,xx         = get_allele_frequency(sVec,nVec,eVec,muVec)
#     ex           = get_escape_fre_term(sVec,nVec)
#     p_wt,p_mut_k = get_p_k(sVec,nVec,seq_length,escape_group,escape_TF)

#     # infer the beginning part of the whole sequence
#     if IF_raw:
#         times        = np.linspace(0,totalT,totalT+1)
#     else:
#         t_step       = int(re.search(r'dt(\d+)', xfile)[1])
#         sample_times = np.linspace(0,totalT,int(totalT/t_step)+1)
#         times        = np.linspace(0,totalT,totalT+1)

#     # use the data within the range and interpolate if dt>1
#     if not IF_raw and len(sample_times) != len(times):
#         single_freq, double_freq, escape_freq = interpolator_x(x[:totalT+1], xx[:totalT+1], ex[:totalT+1], sample_times, times)
#         p_wt_freq, p_mut_k_freq = interpolator_p(p_wt[:totalT+1], p_mut_k[:totalT+1], sample_times, times, seq_length, ne)
#     else:
#         single_freq  =  x[:totalT+1]
#         double_freq  = xx[:totalT+1]
#         escape_freq  = ex[:totalT+1]
#         p_wt_freq    = p_wt[:totalT+1]
#         p_mut_k_freq = p_mut_k[:totalT+1]

#     # covariance matrix, flux term and delta_x
#     covariance_n = diffusion_matrix_at_t(single_freq, double_freq,x_length)
#     covariance   = np.swapaxes(covariance_n, 0, 2)
#     flux_mu      = get_mutation_flux(single_freq,escape_freq,muVec)         # mutation part
#     flux_rec     = get_recombination_flux(single_freq,p_wt_freq,p_mut_k_freq,trait_dis) # recombination part
#     delta_x      = cal_delta_x(single_freq,times,x_length)
    
#     # extend the time range
#     TLeft   = int(round(times[-1]*0.5/10)*10)
#     TRight  = int(round(times[-1]*0.5/10)*10)
#     etleft  = np.linspace(-TLeft,-10,int(TLeft/10))
#     etright = np.linspace(times[-1]+10,times[-1]+TRight,int(TRight/10))
#     ExTimes = np.concatenate((etleft, times, etright))

#     # solve the bounadry condition ODE to infer selections
#     def fun(a,b):
#         """ Function defining the right-hand side of the system of ODE's"""
#         b_1                 = b[:x_length,:]   # the actual selection coefficients
#         b_2                 = b[x_length:,:]   # the derivatives of the selection coefficients, s'
#         result              = np.zeros((2*x_length,len(a))) # The RHS of the system of ODE's
#         result[:x_length]   = b_2       # sets the derivatives of the selection coefficients 'b_1', equal to s'
#         mat_prod            = np.sum(covariance[:,:,:len(a)] * b_1[:,len(etleft):len(etleft)+len(times)], 1)

#         for t in range(len(a)): # right hand side of second half of the ODE system
#             # within the time range
#             if len(etleft) <= t < len(etleft)+len(times):
#                 tt = t - len(etleft)
#                 for i in range(x_length):
#                     result[x_length+i,t] = (mat_prod[i,tt] + gamma1[i] * b_1[i,t] + flux_mu[tt,i] + flux_rec[tt,i] - delta_x[tt,i]) / gamma2[i]
            
#             # outside the time range, no selection strength
#             else:
#                 for i in range(x_length):
#                     result[x_length+i,t] = gamma1[i] * b_1[i,t] / gamma2[i]

#         return result

#     def fun_advanced(a,b):
#         """ The function that will be used if it is necessary for the BVP solver to add more nodes.
#         Note that the inference may be much slower if this has to be used."""

#         b_1                 = b[:x_length,:]   # the actual selection coefficients
#         b_2                 = b[x_length:,:]   # the derivatives of the selection coefficients, s'
#         result              = np.zeros((2*x_length,len(a))) # The RHS of the system of ODE's
#         result[:x_length]   = b_2       # sets the derivatives of the selection coefficients 'b_1', equal to s'

#         # create new interpolated single and double site frequencies
#         single_freq_int, double_freq_int, escape_freq_int = interpolator_x(single_freq, double_freq, escape_freq, times, a)
#         p_wt_int, p_mut_k_int                             = interpolator_p(p_wt_freq,p_mut_k_freq,times,a,seq_length,ne)
        
#         # use the interpolations from above to get the values of delta_x and the covariance matrix at the nodes
#         flux_mu_int  = get_mutation_flux(single_freq_int, escape_freq_int, muVec)
#         flux_rec_int = get_recombination_flux(single_freq_int, p_wt_int, p_mut_k_int, trait_dis)# recombination part
#         delta_x_int  = cal_delta_x(single_freq_int,a,x_length)
#         covar_int    = diffusion_matrix_at_t(single_freq_int, double_freq_int,x_length)
#         covar_int    = np.swapaxes(covar_int,0,2)

#         # calculate the other half of the RHS of the ODE system
#         mat_prod_int  = np.sum(covar_int[:,:,:len(a)] * b_1[:,len(etleft):len(etleft)+len(times)], 1)

#         for t in range(len(a)): # right hand side of second half of the ODE system
#             # within the time range
#             if len(etleft) <= t < len(etleft)+len(times):
#                 tt = t - len(etleft)
#                 for i in range(x_length):
#                     result[x_length+i,t] = (mat_prod_int[i,tt] + gamma1[i] * b_1[i,t] + flux_mu_int[tt,i] + flux_rec_int[tt,i] - delta_x_int[tt,i]) / gamma2[i]
            
#             # outside the time range, no selection strength
#             else:
#                 for i in range(x_length):
#                     result[x_length+i,t] = gamma1[i] * b_1[i,t] / gamma2[i]

#         return result

#     def bc(b1,b2):
#         return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the endpoints

#     ss_extend = np.zeros((2*x_length,len(ExTimes)))

#     try:
#         solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)
#     except ValueError:
#         print("BVP solver has to add new nodes")
#         solution = sp.integrate.solve_bvp(fun_advanced, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)

#     selection_coefficients = solution.sol(ExTimes)
#     # removes the superfluous part of the array and only save the real time points
#     desired_coefficients   = selection_coefficients[:x_length,len(etleft):len(etleft)+len(times)] 

#     # save the solution with constant_time-varying selection coefficient
#     if IF_raw:
#         g = open('%s/%s/output_multiple/c_%s.npz'%(SIM_DIR,dir,xfile), mode='w+b')
#     else: # save the solution with finite sampling noise
#         g = open('%s/%s/output_multiple/nsdt/c_%s.npz'%(SIM_DIR,dir,xfile), mode='w+b')
#     np.savez_compressed(g, selection=desired_coefficients, all = selection_coefficients, time=times)
#     g.close()

# def py2c(**pdata):

#     """
#     Convert a trajectory into plain text to save the results.
#     """

#     # unpack passed data
#     T        = pdata['generations']
#     ns_vals  = pdata['ns_vals']
#     dt_vals  = pdata['dt_vals']
#     dir      = pdata['dir']
#     xfile    = pdata['xfile']

#     rng = np.random.RandomState()

#     # write the results
#     for i in range(len(ns_vals)):
#         ns      = ns_vals[i]
#         for j in range(len(dt_vals)):
#             dt = dt_vals[j]
#             f  = open('%s/%s/sequences/nsdt/example-%s_ns%d_dt%d.dat' % (SIM_DIR, dir, xfile, ns, dt), 'w')
#             if dt == 1:
#                 data  = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR, dir, xfile))
#                 for tt in range(0, T+1, dt):
#                     idx    = data.T[0]==tt
#                     nVec_t = data[idx].T[1]
#                     sVec_t = data[idx].T[2:].T

#                     iVec = np.zeros(int(np.sum(nVec_t)))
#                     ct   = 0
#                     for k in range(len(nVec_t)):
#                         iVec[ct:int(ct+nVec_t[k])] = k
#                         ct += int(nVec_t[k])
#                     iSample = rng.choice(iVec, ns, replace=False)
#                     for k in range(len(nVec_t)):
#                         nSample = np.sum(iSample==k)
#                         if nSample>0:
#                             f.write('%d\t%d\t%s\n' %(tt, nSample, ' '.join([str(int(kk)) for kk in sVec_t[k]])))

#             else:
#                 data = np.loadtxt('%s/%s/sequences/nsdt/example-%s_ns%d_dt1.dat'%(SIM_DIR, dir, xfile, ns))
#                 for tt in range(0, T+1, dt):
#                     idx    = data.T[0]==tt
#                     nVec_t = data[idx].T[1]
#                     sVec_t = data[idx].T[2:].T
#                     for k in range(len(nVec_t)):
#                         f.write('%d\t%d\t%s\n' %(tt, nVec_t[k], ' '.join([str(int(kk)) for kk in sVec_t[k]])))
#             f.close()

# def infer_const(**pdata):
#     """
#     Use constant method to infer time-varying example
#     """

#     # unpack passed data
#     NUC           = pdata['NUC']            # ['A','T']
#     xfile         = pdata['xfile']           #'1-con'

#     seq_length    = pdata['seq_length']     # 20
#     totalT        = pdata['totalT']         # 500
#     mut_rate = pdata['mut_rate']  # 1e-3

#     escape_group  = pdata['escape_group']   # [[2,5,8]], escape group

#     gamma_s      = pdata['gamma_s']
#     gamma_p      = gamma_s/10

#     ############################################################################
#     ############################## Function ####################################
#     # get muVec for multiple case
#     def getMutantS(sVec):
#         # use muVec matrix to record the index of time-varying sites(after throwing out weak linkage sites)
#         muVec = -np.ones((seq_length, q)) # default value is -1, positive number means the index
#         x_length  = 0
#         for i in range(seq_length):
#             # find all possible alleles in site i
#             alleles     = [int(sVec[t][k][i]) for t in range(len(sVec)) for k in range(len(sVec[t]))]
#             allele_uniq = np.unique(alleles)
#             for allele in allele_uniq:
#                 muVec[i][int(allele)] = x_length
#                 x_length += 1
#         return x_length,muVec

#     def get_allele_frequency(sVec,nVec,eVec,muVec):
#         x  = np.zeros((len(nVec),x_length))           # single allele frequency
#         xx = np.zeros((len(nVec),x_length,x_length))  # pair allele frequency
#         for t in range(len(nVec)):
#             pop_size_t = np.sum([nVec[t]])
#             for k in range(len(nVec[t])):
#                 # individual part
#                 for i in range(seq_length):
#                     qq = int(sVec[t][k][i])
#                     aa = int(muVec[i][qq])
#                     if aa != -1:
#                         x[t,aa] += nVec[t][k]
#                         for j in range(int(i+1), seq_length):
#                             qq = int(sVec[t][k][j])
#                             bb = int(muVec[j][qq])
#                             if bb != -1:
#                                 xx[t,aa,bb] += nVec[t][k]
#                                 xx[t,bb,aa] += nVec[t][k]
#                 # escape part
#                 for n in range(ne):
#                     aa = int(x_length-ne+n)
#                     x[t,aa] += eVec[t][k][n] * nVec[t][k]
#                     for m in range(int(n+1), ne):
#                         bb = int(x_length-ne+m)
#                         xx[t,aa,bb] += eVec[t][k][n] * eVec[t][k][m] * nVec[t][k]
#                         xx[t,bb,aa] += eVec[t][k][n] * eVec[t][k][m] * nVec[t][k]
#                     for j in range(seq_length):
#                         qq = int(sVec[t][k][j])
#                         bb = int(muVec[j][qq])
#                         if bb != -1:
#                             xx[t,bb,aa] += eVec[t][k][n] * nVec[t][k]
#                             xx[t,aa,bb] += eVec[t][k][n] * nVec[t][k]
#             x[t,:]    = x[t,:]/pop_size_t
#             xx[t,:,:] = xx[t,:,:]/pop_size_t
#         return x,xx

#     # calculate escape frequency (multiple case)
#     def get_escape_fre_term(sVec,nVec):
#         ex  = np.zeros((len(nVec),ne,seq_length,q))
#         for t in range(len(nVec)):
#             pop_size_t = np.sum([nVec[t]])
#             for k in range(len(sVec[t])):
#                 for n in range(ne):
#                     n_mutations = 0
#                     for nn in escape_group[n]:
#                         index = escape_group[n].index(nn)
#                         WT = escape_TF[n][index]
#                         if sVec[t][k][nn] != WT:
#                             n_mutations += 1
#                             site = nn
#                     if n_mutations == 1:
#                         qq = int(sVec[t][k][site])
#                         ex[t,n,site,qq] += nVec[t][k]
#             ex[t,:,:,:] = ex[t,:,:,:] / pop_size_t
#         return ex

#     # flux term with escape term
#     def get_mutation_flux(x,ex,muVec):
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
#                         if a != WT:
#                             flux[t, x_length-ne+n] += muMatrix[WT][a] * (1 - x[t,x_length-ne+n]) - muMatrix[a][WT] * ex[t,n,index,a]
#         return flux

#     ############################################################################
#     ####################### Inference (binary case) ############################
#     # obtain raw data
#     data      = np.loadtxt("%s/example/example-%s.dat"%(SIM_DIR,xfile))
#     q         = len(NUC)
#     ne        = len(escape_group)
#     escape_TF = [[0,0,0]]
#     muMatrix  = [[0,mut_rate],[mut_rate,0]]

#     sVec,nVec,eVec = getSequence(data,escape_group)

#     x_length,muVec = getMutantS(sVec)
#     x_length   += ne

#     single_freq,double_freq  = get_allele_frequency(sVec,nVec,eVec,muVec)
#     escape_freq  = get_escape_fre_term(sVec,nVec)

#     flux_all   = get_mutation_flux(single_freq,escape_freq,muVec)
#     totalCov  = np.zeros([x_length,x_length])
#     bayesian  = np.zeros([x_length,x_length])
#     totalflux = np.zeros(x_length)

#     # use the data within the range
#     x    = single_freq[:totalT+1]
#     xx   = double_freq[:totalT+1]
#     ex   = escape_freq[:totalT+1]
#     flux = flux_all[:totalT+1]

#     for i in range(x_length-ne):
#         bayesian[i, i] += gamma_s
#     for n in range(ne):
#         ii = x_length - ne + n
#         bayesian[ii, ii] += gamma_p

#     for t in range(len(x) - 1):
#         totalflux += (flux[t] + flux[t+1])/2
#         for i in range(x_length):
#             totalCov[i,i] += (((3-(2*x[t+1,i]))*(x[t+1,i]+x[t,i]))-(2*x[t,i]*x[t,i]))/6
#             for j in range(i+1,x_length):
#                 dCov1 = -((2*x[t,i]*x[t,j])+(2*x[t+1,i]*x[t+1,j])+(x[t,i]*x[t+1,j])+(x[t+1,i]*x[t,j]))/6
#                 dCov2 = (xx[t,i,j]+xx[t+1,i,j])/2

#                 totalCov[i,j] += dCov1 + dCov2
#                 totalCov[j,i] += dCov1 + dCov2

#     LHS_av = totalCov + bayesian
#     RHS_av = np.zeros((x_length,1))
#     for i in range(x_length):
#         RHS_av[i,0] = x[-1,i] - x[0,i] - totalflux[i]
#     solution_const_av = np.linalg.solve(LHS_av, RHS_av)

#     solution_const = solution_const_av.reshape(-1)

#     sc = np.zeros(seq_length+ne)
#     for i in range(seq_length):
#         sc[i] = solution_const[2*i+1]-solution_const[2*i]
#     for n in range(ne):
#         sc[seq_length+n] = solution_const[2*seq_length+n]

#     np.savetxt("%s/example/sc-%s-const.dat"%(SIM_DIR,xfile),sc)
