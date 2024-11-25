import numpy as np
import scipy.integrate as sp

# 示例 A(x) 和 b(x) 的定义
def A(x):
    return np.array([[np.sin(x), 0], [0, np.cos(x)]])  # 2x2 矩阵，随 x 变化

def b(x):
    return np.array([np.sin(2 * x), np.cos(2 * x)])  # 长度为 L 的向量

# 定义方程的右侧 (一阶微分方程组)
def fun(x, y):
    s = y[:L, :]   # s(x)
    v = y[L:, :]   # v(x) = s'(x)
    dydx = np.zeros_like(y)

    # s'(x) = v(x)
    dydx[:L, :] = v

    # v'(x) = A(x)s(x) + b(x)
    for i, xi in enumerate(x):  # 遍历每个 x 点，分别计算 A(x) 和 b(x)
        dydx[L:, i] = A(xi) @ s[:, i] + b(xi)

    return dydx

# 定义边界条件
def bc(ya, yb):
    # ya: y 在边界 a 的值, yb: y 在边界 b 的值
    # 假设边界条件是 s(0) = [1, 0], s(1) = [0, 0]
    cond = np.zeros(2 * L)
    cond[:L] = ya[:L] - np.array([1, 0])  # s(0) = [1, 0]
    cond[L:] = yb[:L] - np.array([0, 0])  # s(1) = [0, 0]
    return cond

# 初始时间点和解的初始猜测
x = np.linspace(0, 10, 100)  # 时间点
L = 2  # 向量长度
y_guess = np.zeros((2 * L, x.size))  # 初始猜测

# 求解边值问题
solution = sp.solve_bvp(fun, bc, x, y_guess, tol=1e-3, max_nodes=10000)

# 提取解
s_sol = solution.sol(x)[:L, :]  # s(x)
v_sol = solution.sol(x)[L:, :]  # v(x) = s'(x)

# 打印或绘制结果
import matplotlib.pyplot as plt
plt.plot(x, s_sol[0, :], label="s1(x)")
plt.plot(x, s_sol[1, :], label="s2(x)")
plt.legend()
plt.xlabel("x")
plt.ylabel("s(x)")
plt.title("Solution of the BVP")
plt.show()



#!/usr/bin/env python
# coding: utf-8

## nucleotide parameter
NUC = ['-', 'A', 'C', 'G', 'T']
q = len(NUC)


"""Infer time-varying selection coefficients from HIV data"""
############################################################################
################################# function #################################

# calculate mutation flux term at time t
def get_mut_flux_at_t(x,ex,muVec):
    flux = np.zeros(x_length)
    for i in range(seq_length):
        for a in range(q):
            aa = int(muVec[i][a])
            if aa != -1:
                for b in range(q):
                    bb = int(muVec[i][b])
                    if b != a:
                        if bb != -1:
                            flux[aa] +=  muMatrix[b][a] * x[bb] - muMatrix[a][b] * x[aa]
                        else:
                            flux[aa] += -muMatrix[a][b] * x[aa]
    for n in range(ne):
        for nn in range(len(escape_group[n])):
            for a in range(q):
                WT = escape_TF[n][nn]
                index = escape_group[n][nn]
                if a not in WT:
                    for b in WT:
                        flux[x_length-ne+n] += muMatrix[b][a] * (1 - x[x_length-ne+n]) - muMatrix[a][b] * ex[n,index,a]
    return flux

# calculate recombination flux term at time t
def get_rec_flux_at_t(x, r_rates, p_wt, p_mut_k, trait_dis):
    flux = np.zeros(x_length)
    for n in range(ne):
        fluxIn  = 0
        fluxOut = 0

        for nn in range(len(escape_group[n])-1):
            k_index = escape_group[n][0]+nn
            fluxIn  += trait_dis[n][nn] * p_wt[n]*p_mut_k[k_index][0]
            fluxOut += trait_dis[n][nn] * p_mut_k[k_index][1]*p_mut_k[k_index][2]
        
        flux[x_length-ne+n] = r_rates * (fluxIn - fluxOut)

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

# calculate the frequency change at all times
def cal_delta_x(single_freq,times):
    delta_x = np.zeros((len(single_freq),x_length))   # difference between the frequency at time t and time t-1s
    # calculate by np.gradient function
    # for ii in range(x_length):
    #     delta_x[:,ii] = np.gradient(single_freq.T[ii],times)

    # calculate manually
    for t in range(len(single_freq)-1):
        delta_x[t] = (single_freq[t+1] - single_freq[t])/(times[t+1]-times[t])

    # dt for the last time point, make sure the expected x[t+1] is less than 1
    dt_last = times[-1] - times[-2]
    for ii in range(x_length):
        if single_freq[-1,ii] + delta_x[-1,ii]*dt_last> 1:
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

def get_gamma2(last_time):
    # Use a time-varying gamma_prime, gamma_2tv is the middle value, 
    # boundary value is 4 times larger, decrese/increase exponentially within 10% generation.
    gamma2 = np.ones((x_length,len(ExTimes)))*gamma_2c
    gamma_t = np.ones(len(ExTimes))
    tv_range = max(int(round(last_time*0.1/10)*10),1)
    alpha  = np.log(4) / tv_range
    for i, ti in enumerate(ExTimes): # loop over all time points, i: index, ti: time
        if ti <= 0:
            gamma_t[i] = 4
        elif ti >= last_time:
            gamma_t[i] = 4
        elif 0 < ti and ti <= tv_range:
            gamma_t[i] = 4 * np.exp(-alpha * ti)
        elif last_time - tv_range < ti and ti <= last_time:
            gamma_t[i] = 1 * np.exp(alpha * (ti - last_time + tv_range))
        else:
            gamma_t[i] = 1

    # individual site: gamma_2c, escape group and special site: gamma_2tv
    for n in range(ne):# binary trait
        gamma2[x_length-ne+n] = gamma_t * gamma_2tv
    for p_site in p_sites: # special site - time varying
        for qq in range(len(NUC)):
            index = int (muVec[p_site][qq]) 
            if index != -1:
                gamma2[index] = gamma_t * gamma_2tv

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

# Interpolation function definition
################################################################################
######################### time varying inference ###############################

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
    time_step    = rawdata['time_step']
    seq_length   = rawdata['seq_length']

    if cr:
        r_rates = np.ones(len(sample_times)) * 1.4e-5
    else:
        r_rates = rawdata['r_rates']

    # information for escape group
    p_wt         = rawdata['p_wt_freq']
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
t_extend   = int(round(sample_times[-1]*theta/10)*10)
ExTimes = np.concatenate(([-t_extend], sample_times, [sample_times[-1] + t_extend]))

# get gamma_1 and gamma_2
gamma_1 = get_gamma1(sample_times[-1])
gamma_2 = get_gamma2(sample_times[-1])

# get dx
delta_x_raw = cal_delta_x(x, sample_times)

# Use linear interpolates to get the input arrays at any given time point
interp_x   = interp1d(sample_times, x, axis=0, kind='linear', bounds_error=False, fill_value=0)
interp_xx  = interp1d(sample_times, xx, axis=0, kind='linear', bounds_error=False, fill_value=0)
interp_ex  = interp1d(sample_times, ex, axis=0, kind='linear', bounds_error=False, fill_value=0)
interp_wt  = interp1d(sample_times, p_wt, axis=0, kind='linear', bounds_error=False, fill_value=0)
interp_mut = interp1d(sample_times, p_mut_k, axis=0, kind='linear', bounds_error=False, fill_value=0)
interp_dx  = interp1d(sample_times, delta_x_raw, axis=0, kind='linear', bounds_error=False, fill_value=0)
interp_r   = interp1d(sample_times, r_rates, kind='linear', bounds_error=False, fill_value=0)
interp_g2  = interp1d(ExTimes, gamma_2, axis=0, kind='linear', bounds_error=False, fill_value=0)

if print_time:
    start_time = time_module.time()

# solve the bounadry condition ODE to infer selections
def fun(t,s):
    """ Function defining the right-hand side of the system of ODE's"""
    s1                 = s[:x_length,:]   # the actual selection coefficients s1 = s
    s2                 = s[x_length:,:]   # the derivatives of the selection coefficients, s2 = s'
    dsdt               = np.zeros_like(s)  # the RHS of the system of ODE's

    # s' = s2
    dsdt[:x_length, :] = s2

    # s2'(t) = A(t)s1(t) + b(t)
    for i, ti in enumerate(t): # loop over all time points, i: index, ti: time
        
        gamma_2     = interp_g2(ti)

        if ti < 0 or ti > sample_times[-1]:
            # outside the range, only gamma
            A_t = np.diag(gamma_1)
            b_t = np.zeros(x_length)

        else:
            # calculate the frequency at time ti
            single_freq = interp_x(ti)
            double_freq = interp_xx(ti)
            escape_freq = interp_ex(ti)
            p_wt_freq   = interp_wt(ti)
            p_mut_k     = interp_mut(ti)
            r_rate      = interp_r(ti)
            delta_x     = interp_dx(ti)

            # calculate A(t) = C(t) + gamma_1 * I
            C_t = diffusion_matrix_at_t(single_freq, double_freq) # covariance matrix
            A_t = C_t + np.diag(gamma_1)

            # calculate b(t)
            flux_mu  = get_mut_flux_at_t(single_freq, escape_freq, muVec)
            flux_rec = get_rec_flux_at_t(single_freq, r_rate, p_wt_freq, p_mut_k, trait_dis)
            b_t      = flux_mu + flux_rec - delta_x

        # s'' = A(t)s(t) + b(t)
        dsdt[x_length:, i] = A_t @ s[:x_length, i] / gamma_2 + b_t / gamma_2

    return dsdt

# Boundary conditions
# solution to the system of differential equation with the derivative of the selection coefficients zero at the endpoints
def bc(b1,b2):
    # Neumann boundary condition
    return np.ravel(np.array([b1[x_length:],b2[x_length:]])) # s' = 0 at the extended endpoints

# initial guess for the selection coefficients
ss_extend = np.zeros((2*x_length,len(ExTimes)))
# sc_constant  = np.loadtxt('%s/constant/output/sc-%s.dat'%(HIV_DIR,tag))
# for i in range(len(muVec)):
#     for j in range(len(NUC)):
#         index = int(muVec[i][j])
#         if index != -1:
#             ss_extend[index] = sc_constant[i * 5 + j]

try:
    solution = sp.integrate.solve_bvp(fun, bc, ExTimes, ss_extend, max_nodes=100000, tol=1e-3)
except ValueError:
    print("BVP solver has to add new nodes")
    sys.exit()

# Get the solution for sample times
# removes the superfluous part of the array and only save the sampled times points
# including the extended time points
time_all       = np.linspace(ExTimes[0], ExTimes[-1], int(ExTimes[-1]-ExTimes[0]+1))
sc_all         = solution.sol(time_all)
desired_sc_all = sc_all[:x_length,:]

# not include the extended time points
time_sample       = np.linspace(sample_times[0], sample_times[-1], int(sample_times[-1]-sample_times[0]+1))
sc_sample         = solution.sol(time_sample)
desired_sc_sample = sc_sample[:x_length,:]

print(f'CH{tag[-5:]} solver_time : {solution.x}')

if print_time:
    end_time = time_module.time()
    print(f"Execution time for CH{tag[6:]} : {end_time - start_time} seconds")

# save the solution with constant_time-varying selection coefficient
g = open('%s/%s/sc_%s%s.npz'%(HIV_DIR, output_dir, tag, name), mode='w+b')
np.savez_compressed(g, all = desired_sc_all, selection=desired_sc_sample, time=sample_times, ExTimes=ExTimes)
g.close()

if __name__ == '__main__':
main(sys.argv[1:])
