import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import re
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
    sim_dir       = pdata['dir']            # 'simple'
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
        fitness = 1.0
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
    f = open("%s/%s/sequences/example-%s.dat"%(SIM_DIR,sim_dir,xfile),'w')

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
    sim_dir       = pdata['dir']            # 'trait'
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

    q  = len(NUC)
    ne = len(escape_group)

    ############################################################################
    ############################## function ####################################
    # get fitness of new genotype
    def get_fitness_alpha(genotype,time):
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
    f = open("%s/%s/sequences/example-%s.dat"%(SIM_DIR,sim_dir,xfile),'w')

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
def cal_delta_x(single_freq,times,x_length):
    delta_x = np.zeros((len(single_freq),x_length))   # difference between the frequency at time t and time t-1s
    # Calculate manually
    for tt in range(len(single_freq)-1):
        delta_x[tt] = (single_freq[tt+1] - single_freq[tt])/(times[tt+1]-times[tt])
    
    # dt for the last time point, make sure the expected x[t+1] is less than 1 and larger than 0
    for ii in range(x_length):
        if single_freq[-1,ii] == 1 and delta_x[-2,ii] > 0:
            delta_x[-1,ii] = 0
        elif single_freq[-1,ii] == 0 and delta_x[-2,ii] < 0:
            delta_x[-1,ii] = 0
        else:
            delta_x[-1,ii] = delta_x[-2,ii]

    return delta_x

def infer_simple(**pdata):
    """
    Infer time-varying example (binary case)
    """

    # unpack passed data
    sim_dir       = pdata['dir']            # 'simple'
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

    ############################################################################
    ####################### Inference (binary case) ############################
    
    # obtain raw data and information of traits
    data         = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR,sim_dir,xfile))

    # get raw time points
    times = []
    for i in range(len(data)):
        times.append(data[i][0])
    sample_times = np.unique(times)
    time_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))

    # obtain sequence data and frequencies
    sVec,nVec      = getSequence(data)
    x_length,muVec = getMutantS()

    # get all frequencies, x: single allele frequency, xx: pair allele frequency
    # ex: escape frequency, p_wt,p_mut_k: frequency related to recombination part
    x,xx         = get_allele_frequency(sVec,nVec,muVec) 
    
    # get dx
    delta_x_raw = cal_delta_x(x, sample_times, x_length)
    flux_mu_raw = get_mut_flux(x, muVec)

    # get gamma_1 and gamma_2
    gamma_1 = np.ones(x_length)*gamma_1s
    gamma_2 = get_gamma2(time_all, beta)

    # get the input arrays at any integer time point
    if len(sample_times) == len(time_all):
        # no interpolation is needed
        single_freq = x
        double_freq = xx
        delta_x     = delta_x_raw
        flux_mu     = flux_mu_raw

    else:
        # Use linear interpolates to get data
        interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_dx  = interp1d(sample_times, delta_x_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_mu  = interp1d(sample_times, flux_mu_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        
        single_freq = interp_x(time_all)
        double_freq = interp_xx(time_all)
        delta_x     = interp_dx(time_all)
        flux_mu     = interp_mu(time_all)

    t_extend = int(round(time_all[-1]*theta/10)*10)
    etleft   = np.linspace(-t_extend,-10,int(t_extend/10)) # time added before the beginning time (dt=10)
    etright  = np.linspace(time_all[-1]+10,time_all[-1]+t_extend,int(t_extend/10))
    ExTimes  = np.concatenate((etleft, time_all, etright))

    # Get matrix A and vector b
    A_all = np.zeros((len(time_all),x_length,x_length))
    b_all = np.zeros((len(time_all),x_length))

    for ti in range(len(time_all)):
        # calculate A(t) = C(t)+ gamma_1 * I
        C_t = diffusion_matrix_at_t(single_freq[ti], double_freq[ti]) # covariance matrix
        A_all[ti] = C_t + np.diag(gamma_1)

        # calculate b(t)
        b_all[ti]   = flux_mu[ti] - delta_x[ti]
        
    # solve the bounadry condition ODE to infer selections
    def fun(time,s):
        """ Function defining the right-hand side of the system of ODE's"""
        s1                 = s[:x_length,:]   # the actual selection coefficients s1 = s
        s2                 = s[x_length:,:]   # the derivatives of the selection coefficients, s2 = s'
        dsdt               = np.zeros_like(s)  # the RHS of the system of ODE's

        # s' = s2
        dsdt[:x_length, :] = s2

        # s2'(t) = A(t)s1(t) + b(t)
        for ti, t in enumerate(time): # loop over all time points, ti: index, t: time

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
                dsdt[x_length:, ti] = (A_t @ s1[:, ti] + b_t) / gamma2_t

        return dsdt

    # boundary condition
    def bc(b1,b2):
        # if using Neumann boundary condition
        return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints
            
    # initial guess for the selection coefficients
    ss_extend = np.zeros((2*x_length,len(ExTimes)))
    
    # solve the boundary value problem
    solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)
    
    # Get the solution and remove the superfluous part of the array
    # including the extended time points
    sc_all         = solution.sol(ExTimes)
    desired_sc_all = sc_all[:x_length,:]

    # not include the extended time points
    sc_sample         = solution.sol(sample_times)
    desired_sc_sample = sc_sample[:x_length,:]

    # save the solution with constant_time-varying selection coefficient
    g = open('%s/%s/%s/c_%s.npz'%(SIM_DIR,sim_dir,output_dir,xfile), mode='w+b')
    np.savez_compressed(g, all = desired_sc_all, selection=desired_sc_sample, time=sample_times, ExTimes=ExTimes)
    g.close()

def infer_trait(**pdata):
    """
    Infer time-varying example (binary case)
    """
    # unpack passed data
    sim_dir       = pdata['dir']            # 'trait'
    xfile         = pdata['xfile']          # index of the simulation
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
    
    # obtain raw data and information of traits
    data         = np.loadtxt('%s/%s/sequences/example-%s.dat'%(SIM_DIR,sim_dir,xfile))
    escape_group = read_file('%s/traitsite/traitsite-%s.dat'%(sim_dir,xfile))
    trait_dis    = read_file('%s/traitdis/traitdis-%s.dat'%(sim_dir,xfile))
    escape_TF    = read_file('%s/traitseq.dat'%(sim_dir))
    ne           = len(escape_group)

    # get raw time points
    times = []
    for i in range(len(data)):
        times.append(data[i][0])
    sample_times = np.unique(times)
    time_all = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))

    # obtain sequence data and frequencies
    sVec,nVec,eVec = getSequence(data,escape_group)
    x_length,muVec = getMutantS()
    x_length      += ne

    # get all frequencies, x: single allele frequency, xx: pair allele frequency
    # ex: escape frequency, p_wt,p_mut_k: frequency related to recombination part
    x,xx        = get_allele_frequency(sVec,nVec,eVec,muVec) 
    ex          = get_escape_fre_term(sVec,nVec)
    p_mut_k_raw = get_p_k(sVec,nVec,seq_length,escape_group,escape_TF)

    # get dx
    delta_x_raw = cal_delta_x(x, sample_times, x_length)
    flux_mu_raw = get_mut_flux(x,ex,muVec)
    
    # get gamma_1 and gamma_2
    gamma_1 = get_gamma1()
    gamma_2 = get_gamma2(time_all, beta)

    # get the input arrays at any integer time point
    if len(sample_times) == len(time_all):
        # no interpolation is needed
        single_freq = x
        double_freq = xx
        p_mut_k     = p_mut_k_raw
        delta_x     = delta_x_raw
        flux_mu     = flux_mu_raw

    else:
        # Use linear interpolates to get the input arrays at any integer time point
        interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_mut = interp1d(sample_times, p_mut_k_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_dx  = interp1d(sample_times, delta_x_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
        interp_mu  = interp1d(sample_times, flux_mu_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
                
        single_freq = interp_x(time_all)
        double_freq = interp_xx(time_all)
        p_mut_k     = interp_mut(time_all)
        delta_x     = interp_dx(time_all)
        flux_mu     = interp_mu(time_all)

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
        A_all[ti] = diffusion_matrix_at_t(single_freq[ti], double_freq[ti]) # covariance matrix
        
        # calculate b(t)
        flux_rec = get_rec_flux_at_t(single_freq[ti,x_length-ne:], p_mut_k[ti], trait_dis)
        b_all[ti]   = flux_mu[ti] - delta_x[ti] + flux_rec
        
    def fun(time,s):
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
                    gamma_1[x_length-ne+n] = gamma_1t*100
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
                A_t      = A_all[time_index]                
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
    solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=10000, tol=1e-3)
    
    # Get the solution and remove the superfluous part of the array
    # including the extended time points
    sc_all         = solution.sol(ExTimes)
    desired_sc_all = sc_all[:x_length,:]

    # not include the extended time points
    sc_sample         = solution.sol(sample_times)
    desired_sc_sample = sc_sample[:x_length,:]

    # save the solution with constant_time-varying selection coefficient
    g = open('%s/%s/%s/c_%s.npz'%(SIM_DIR,sim_dir,output_dir,xfile), mode='w+b')
    np.savez_compressed(g, all = desired_sc_all, selection=desired_sc_sample, time=sample_times, ExTimes=ExTimes)
    g.close()
