#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# A simple Wright-Fisher simulation with an additive fitness model

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance

def main(args):
    """ Simulate Wright-Fisher evolution of a population and save the results. """

    # Read in simulation parameters from command line
    
    parser = argparse.ArgumentParser(description='Wright-Fisher evolutionary simulation.')
    parser.add_argument('-o',        type=str,    default='Wright-Fisher', help='output destination')
    parser.add_argument('-N',         type=int,   default=1000,            help='population size')
    parser.add_argument('--mu',       type=float, default=1.0e-3,          help='mutation rate')
    parser.add_argument('--sample',   type=int,   default=0,               help='number of sampled population for the calculation of covariance matrix')
    parser.add_argument('--record',   type=int,   default=1,               help='record sequence data every {record} generations and use them to calculate int-cov-mat')
    parser.add_argument('-s',         type=str,   default=None,            help='.npy or .npz file containing selection matrix of shape (T,L)')
    parser.add_argument('-i',         type=str,   default=None,            help='file containing initial distribution')
    parser.add_argument('--random',   type=int,   default=0,               help='number of random initial genotypes in the population')
    parser.add_argument('--timed',    type=int,   default=0,               help='set to 1 to time the simulation, and set to 2 to be informed of the progress every generation')
    parser.add_argument('--TVsample', type=str,   default=None,            help='.npy file containing the time-varying sample sizes, if applicable')
    parser.add_argument('--PopSize',  type=str,   default=None,            help='.npy file containing the time-varying population size, if applicable')
    parser.add_argument('--CovarianceOff',  action='store_true', default=False, help='whether or not compute covariance matrix after simulation')
    parser.add_argument('--CovarInt',       action='store_true', default=False, help='whether or not to compute integrated covariance matrix')
    parser.add_argument('--dynamics',       action='store_true', default=False, help='whether or not record int-cov-mat at each generation')
    parser.add_argument('--MultinomialOff', action='store_true', default=False, help='whether or not do a multinomial sampling when redistributing frequencies')
    parser.add_argument('--DetailOff',      action='store_true', default=False, help='whether or not to save addititional information at the end of the simulation')
    parser.add_argument('--FullTrajectory', action='store_true', default=False, help='whether or not to store trajectories at every generation')


    arg_list = parser.parse_args(args)

    out_str    = arg_list.o
    N          = arg_list.N       
    mu         = arg_list.mu
    sample     = arg_list.sample
    record     = arg_list.record
    timed      = arg_list.timed
    randomize  = arg_list.random
    s_string   = arg_list.s
    if s_string[-3:] == "npy":
        s_full     = np.load(arg_list.s)
    elif s_string[-3:] == "npz":
        s_full     = np.load(arg_list.s, allow_pickle="TRUE")
    else:
        print("Incorrect file type for the selection coefficients")
    T = len(s_full[:,0]) - 1
    L = len(s_full[0,:])             
    covariance_off = arg_list.CovarianceOff
    dynamics       = arg_list.dynamics
    covar_int      = arg_list.CovarInt
    MultinomialOff = arg_list.MultinomialOff
    DetailOff      = arg_list.DetailOff

    FullTrajectory = arg_list.FullTrajectory

    selection = s_full[0]
    if arg_list.i:
        initial = np.load(arg_list.i)
    if arg_list.PopSize != None and arg_list.PopSize != 'None':
        pop_size = np.load(arg_list.PopSize)
    else:
        pop_size = np.ones(T+1) * N
    if arg_list.TVsample:
        sample_size = np.load(arg_list.TVsample)
    else:
        sample_size = np.ones(T+1) * sample
    
    ##### FUNCTIONS #####
    
    def bin(a, L):
    # return binary seq of length L for the ath genotype
        return format(a, '0' + str(L) + 'b')

    def fitness(selection, seq, generations, epitope =[0, 0]):
        """ Calculate fitness for a binarized seq wrt. a given selection matrix at every time."""

        h = 1
        seq = [int(i) for i in seq]
        for i in range(len(seq)):
            h += seq[i] * selection[i]
        # if there is at least one mutation in the epitope region, then add a certain fitness.
        #for i in range(epitope[0]):
        #    if seq[i] == 1:
        #        h += epitope[1]
        #        break
        return h

    def usage():
        print("")
     
    
    def printUpdate(current, end, bar_length=20):
        """ Print an update of the simulation status. h/t Aravind Voggu on StackOverflow. """
        
        percent = float(current) / end
        dash    = ''.join(['-' for k in range(int(round(percent * bar_length)-1))]) + '>'
        space   = ''.join([' ' for k in range(bar_length - len(dash))])

        sys.stdout.write("\rSimulating: [{0}] {1}%".format(dash + space, int(round(percent * 100))))
        sys.stdout.write("\n")
        sys.stdout.flush()

        
    def No2Index(nVec_t, no):
        """ Find out what group a sampled individual is in (what sequence does it have) """
        
        tmp_No2Index = 0
        for i in range(len(nVec_t)):
            tmp_No2Index += nVec_t[i]
            if tmp_No2Index > no:
                return i

            
    def SampleSequences(sVec, nVec, TV_sample):
        """ Sample a certain number of sequences from the whole population. """

        sVec_sampled = []
        nVec_sampled = []
        for t in range(len(nVec)):
            nVec_tmp = []
            sVec_tmp = []
            if pop_size_record[t] > TV_sample[t] and TV_sample[t] > 0:
                nos_sampled = np.random.choice(int(pop_size_record[t]), int(TV_sample[t]), replace=False)
                indices_sampled = [No2Index(nVec[t], no) for no in nos_sampled]
                indices_sampled_unique = np.unique(indices_sampled)
                for i in range(len(indices_sampled_unique)):
                    nVec_tmp.append(np.sum([index == indices_sampled_unique[i] for index in indices_sampled]))
                    sVec_tmp.append(sVec[t][indices_sampled_unique[i]])
            else:
                for i in range(len(nVec[t])):
                    nVec_tmp.append(nVec[t][i])
                    sVec_tmp.append(sVec[t][i])
            nVec_sampled.append(np.array(nVec_tmp))
            sVec_sampled.append(np.array(sVec_tmp))
        return sVec_sampled, nVec_sampled
    
    
    def allele_counter(sVec, nVec):
        """ Counts the single-site and double-site frequencies given the subsampled sVec_s and nVec_s. """
        
        Q = np.array([np.sum(nVec[t]) for t in range(len(nVec))])   # array that contains the total number of sampled sequences at each time point
        single_freq_s = np.zeros((len(nVec),L))
        double_freq_s = np.zeros((len(nVec),L,L))
        for t in range(len(nVec)):
            for i in range(L):
                single_freq_s[t, i] = np.sum([sVec[t][k][i] * nVec[t][k] for k in range(len(nVec[t]))]) / Q[t]
                for j in range(L): 
                    if i != j:
                        double_freq_s[t,i,j] = np.sum([sVec[t][k][i] * sVec[t][k][j] * nVec[t][k] for k in range(len(nVec[t]))]) / Q[t]
        return single_freq_s, double_freq_s
    
    
    def allele_counter2(sVec_t, nVec_t):
        """ Counts the single-site and double-site frequencies at a given time given the subsampled sVec_t and normalized nVec_t."""
        
        single_freq_s = np.zeros((L))
        double_freq_s = np.zeros((L,L))
        for i in range(L):
            single_freq_s[i] = np.sum([sVec_t[k][i] * nVec_t[k] for k in range(len(sVec_t))])
            for j in range(L): 
                if i !=j:
                    double_freq_s[i,j] = np.sum([sVec_t[k][i] * sVec_t[k][j] * nVec_t[k] for k in range(len(sVec_t))])
        return single_freq_s, double_freq_s
    
    
    def covariance_calc(single_frequencies, double_frequencies):
        """ Calculate the covariance matrix at each generation """
        
        covar_temp = np.zeros((len(single_frequencies),L,L))
        for t in range(len(single_frequencies)):
            f1 = single_frequencies[t]
            f2 = double_frequencies[t]
            for i in range(L):
                for j in range(L):
                    if i == j:
                        covar_temp[t,i,i] = f1[i] * (1 - f1[i])
                    else:
                        covar_temp[t,i,j] = f2[i,j] - f1[i] * f1[j]
        return covar_temp

    
    def updateCovarianceIntegrate(dg, p1_0, p2_0, p1_1, p2_1, totalCov, pop_av):
        # The formula here comes from linearly interpolating the trajectories, and then integrating the covariance matrix
        # along the infinite number of points between every two time points. 
        N = len(p1_0)

        for a in range(N): 
            totalCov[a, a] += pop_av * dg * ( ((3 - 2 * p1_1[a]) * (p1_0[a] + p1_1[a])) - 2 * (p1_0[a] * p1_0[a]) ) / 6  
            if totalCov[a, a] < 0:
                print('error!')
            for b in range(a + 1, N):
                dCov1 = -dg * pop_av * ((2 * p1_0[a] * p1_0[b]) + (2 * p1_1[a] * p1_1[b]) + (p1_0[a] * p1_1[b]) + (p1_1[a] * p1_0[b])) / 6
                dCov2 = dg * pop_av * 0.5 * (p2_0[a, b] + p2_1[a, b])
                totalCov[a, b] += dCov1 + dCov2
                totalCov[b, a] += dCov1 + dCov2
    
    
    def processStandard(sequences, counts, times, q, totalCov, dynamics=False, out_str = None):

        N = len(totalCov)
        p1 = np.zeros(N, dtype = float)
        p2 = np.zeros((N, N), dtype = float)
        lastp1 = np.zeros(N, dtype = float)
        lastp2 = np.zeros((N, N), dtype = float)
        xp1 = np.zeros(N, dtype = float)
        xp2 = np.zeros((N, N), dtype = float)
        # store the total_cov at each time point when dynamics is enabled.
        cov_dynamics = np.zeros((len(sequences) - 1, len(totalCov), len(totalCov)), dtype = float)

        lastp1, lastp2 = allele_counter2(sequences[0], counts[0])
        lastpopulation = pop_size[0]

        for k in range(1, len(sequences)):

            p1, p2 = allele_counter2(sequences[k], counts[k])
            population = pop_size[k]
            pop_average = (lastpopulation + population) / 2
            updateCovarianceIntegrate(times[k] - times[k-1], lastp1, lastp2, p1, p2, totalCov, pop_average)
            if dynamics == True:
                cov_dynamics[k - 1] = totalCov
                #np.save(out_str + f'_t={k}', totalCov)

            if not k == len(sequences) - 1:
                lastp1 = p1
                lastp2 = p2
                lastpopulation = population

        if dynamics == True:
            return cov_dynamics
        else:
            return 0

    # _ SPECIES CLASS _ #

    class Species:

        def __init__(self, n = 1, f = 1, **kwargs):
            """ Initialize clone/provirus-specific variables. """
            self.n = n   # number of members

            if 'sequence' in kwargs:
                self.sequence = np.array(kwargs['sequence'])  # sequence identifier
                self.f        = fitness(selection, self.sequence, T+1)    # fitness

            else:
                self.sequence = np.zeros(L)
                self.f        = f

        @classmethod
        def clone(cls, s):
            return cls(n = 1, f = s.f, sequence = [k for k in s.sequence]) # Return a new copy of the input Species

        def mutate(self, t):
            """ Mutate and return self + new sequences. t parameter denotes time when mutation happens, which is used to enable/disable forbidden seq judgement""" 

            newSpecies = []
            if self.n>0:
                nMut    = np.random.binomial(self.n, mu * L) # get number of individuals that mutate
                self.n -= nMut # subtract number mutated from size of current clone

                # process mutations
                site = np.random.randint(L, size = nMut) # choose mutation sites at random
                for i in site:
                    s = Species.clone(self) # create a new copy sequence
                    s.sequence[i] = 1 - s.sequence[i] # mutate the randomly-selected site
                    s.f = fitness(selection, s.sequence, T+1)
                    newSpecies.append(s)

            # return the result
            if (self.n>0):
                newSpecies.append(self)
            return newSpecies

    # Trial length and recording frequency
    tStart = 1       # start generation
    tEnd   = T+1       # end generation
    if timed>0:
        start = timer() # track running time

    # Create species and begin recording
    pop, sVec, nVec = [], [], []

    if randomize>0:
        # Randomize the initial population (optional)
        if randomize>pop_size[0]: randomize = pop_size[0]
        n_seqs    = int(pop_size[0]/randomize)
        temp_sVec = []
        temp_nVec = []
        for i in range(randomize):
            temp_seq = np.random.randint(0, 2, size = L)
            if i==(randomize - 1):
                n_seqs = pop_size[0] - np.sum(temp_nVec)
            unique = True
            for k in range(len(pop)):
                if np.array_equal(temp_seq, pop[k].sequence):
                    unique = False
                    pop[k].n += n_seqs
                    temp_nVec[k] += n_seqs
                    break
            if unique:    
                pop.append(Species(n = n_seqs, sequence = temp_seq))
                temp_sVec.append(temp_seq)
                temp_nVec.append(n_seqs)
        sVec.append(np.array(temp_sVec))
        nVec.append(np.array(temp_nVec))    
     
    elif arg_list.i:
        # Use the given intial distribution
        temp_pop = np.load(arg_list.i)
        temp_sVec = []
        temp_nVec = []
        for i in range(len(temp_pop['counts'])):
            temp_seq = np.array([int(j) for j in temp_pop['sequences'][i]])
            n_seq = temp_pop['counts'][i]
            pop.append(Species(n = n_seq, f = fitness(selection, temp_seq, T+1), sequence = temp_seq))
            temp_sVec.append(temp_seq)
            temp_nVec.append(n_seq)
        sVec.append(np.array(temp_sVec))
        nVec.append(np.array(temp_nVec))

    else:
        # Start with all sequences being wild-type
        pop  = [Species(n = pop_size[0])]             # current population
        sVec = [np.array([np.zeros(L)])]              # array of sequences at each time point
        nVec = [np.array([pop_size[0]])]              # array of sequence counts at each time point
        
    if FullTrajectory:
        nVec_full, sVec_full = nVec.copy(), sVec.copy()

    # Evolve the population
    for t in range(tStart, tEnd):
        
        if timed==2:
            printUpdate(t, tEnd)    # status check

        # Select species to replicate
        selection = s_full[t]
        r = np.array([s.n * s.f for s in pop])
        p = r / np.sum(r) # probability of selection for each species (sequence)
        if MultinomialOff:
            n = pop_size[t] * p
        else:
            n = np.random.multinomial(pop_size[t], pvals = p) # selected number of each species

        # Update population size and mutate
        newPop = []
        for i in range(len(pop)):
            pop[i].n = n[i] # set new number of each species
            # include mutations, then add mutants to the population array
            p = pop[i].mutate(t)
            for j in range(len(p)):
                unique = True
                for k in range(len(newPop)):
                    if np.array_equal(p[j].sequence, newPop[k].sequence):
                        unique       = False
                        newPop[k].n += p[j].n
                        break
                if unique:
                    newPop.append(p[j])
        pop = newPop

        # Update measurements
        if record == 1:
            nVec.append(np.array([s.n        for s in pop]))
            sVec.append(np.array([s.sequence for s in pop]))            
        elif t % record == 0 and t != 0:    # avoids the problem of recording the initial population, then recording again at t=0
            nVec.append(np.array([s.n        for s in pop]))
            sVec.append(np.array([s.sequence for s in pop]))
        
        if FullTrajectory:
            nVec_full.append(np.array([s.n        for s in pop]))
            sVec_full.append(np.array([s.sequence for s in pop]))  

    # calculate population trajectories
    times = np.array(range(len(nVec))) * record
    traj = np.zeros((len(nVec), L), dtype = float)
    pop_size_record = np.array([np.sum(nVec[t]) for t in range(len(nVec))])
    for t in range(len(nVec)):
        for i in range(L):
            traj[t, i] = np.sum([sVec[t][k][i] * nVec[t][k] for k in range(len(sVec[t]))])/pop_size_record[t]
            
    if FullTrajectory:
        traj_full = np.zeros((len(nVec_full), L), dtype = float)
        for t in range(len(nVec_full)):
            for i in range(L):
                traj_full[t, i] = np.sum([sVec_full[t][k][i] * nVec_full[t][k] for k in range(len(sVec_full[t]))])/pop_size[t] 
    else:
        traj_full = 0
    
    # sample sequence data
    if sample != 0 or arg_list.TVsample != None:
        sVec, nVec = SampleSequences(sVec, nVec, sample_size[times])
    
    """
    The following calculates the covariance matrix and the single and double site allele frequencies, the single site are the sampled trajectories.
    The inference script can also do this and there is more freedom there to adjust parameters.
    """
    if covariance_off==False:
        q = 2
        totalCov = np.zeros((L, L), dtype = float)
            
        #first way to count allele frequencies
        single_freq = np.zeros((len(nVec),L))
        double_freq = np.zeros((len(nVec),L,L))
        single_freq, double_freq = allele_counter(sVec, nVec)
        pop_size_sample = sample_size[times]
        
        # calculate the covariance matrix at each generation
        covar = covariance_calc(single_freq, double_freq)
        
        # calculate integrated covariance matrix 
        if covar_int == True:
            nVec_rel = []
            for t in range(len(nVec)):
                nVec_rel.append(np.array([i/pop_size_sample[t] for i in nVec[t]]))
            cov_dyn = processStandard(sVec, nVec_rel, times, q, totalCov, dynamics, out_str=out_str) 
        
        # calculate the condition numbers of the covariance matrix
        condition_numbers = np.zeros(len(nVec))
        for t in range(len(nVec)):
            condition_numbers[t] = np.linalg.cond(covar[t,:,:])
            
    # end and output total time
    if timed>0:
        end = timer()
        print('\nTotal time: %lfs, average per generation %lfs' % ((end - start),(end - start)/float(tEnd)))
        script_time = end - start
    else:
        script_time = 0
    
    f = open(out_str+'.npz', mode='w+b')
    if DetailOff:
        np.savez_compressed(f, nVec=nVec, sVec=sVec, traj=traj, times=times, script_time=script_time, 
                            pop_size=pop_size, covar_int=totalCov)
    else:
        np.savez_compressed(f, nVec=nVec, sVec=sVec, traj=traj, times=times, single_freq=single_freq, 
                            double_freq=double_freq, covar=covar, generations=T, condition=condition_numbers, 
                            script_time=script_time, pop_size=pop_size, covar_int=totalCov, traj_full=traj_full)
    f.close()



if __name__ == '__main__': 
    main(sys.argv[1:])

    
