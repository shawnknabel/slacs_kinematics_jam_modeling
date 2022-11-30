# this calculates final kinematics pdf and covariance for all objects in a list by looping through object names.

# 
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import pathlib # to create directory
from time import perf_counter as timer
# register first tick
tick = timer()

#################################################
# objects
obj_names = [#'SDSSJ0029-0055',
             #'SDSSJ0037-0942',
             #'SDSSJ0330-0020',
             #'SDSSJ1112+0826',
             #'SDSSJ1204+0358',
             #'SDSSJ1250+0523',
             'SDSSJ1306+0600',
             'SDSSJ1402+6321',
             'SDSSJ1531-0105',
             'SDSSJ1538+5817',
             'SDSSJ1621+3931',
             'SDSSJ1627-0053',
             'SDSSJ1630+4520',
             'SDSSJ2303+1422'
            ]

#################################################
# functions specific to this script

def weighted_gaussian(xx, mu, sig, c2):
    yy = np.zeros(shape=xx.shape)
    for i in range(len(xx)):
        yy[i] = np.exp(-np.power(xx[i] - mu, 2.) / (2 * np.power(sig, 2.))) * np.exp(-0.5 * c2)
    return yy

def estimate_systematic_error (k_solutions, velocity, random_error, chi2, plot=True):
    
    # take x axis to be the 81 solutions
    x = np.arange(k_solutions)
    # y axis the velocity measurement
    y = velocity
    dy = random_error
    
    if plot==True:
        f, axarr = plt.subplots(1)
        # plot the 81 velocity disperions with error bars
        axarr.errorbar(x, y, yerr=dy, fmt='o', color='black',
                  ecolor='lightgray', elinewidth=1, capsize=0)
        f.set_size_inches(12, 2)
        plt.xlabel('Solution #')
        plt.ylabel(r'$\sigma$ km/s')
        plt.tight_layout()
        plt.show()
    
    # take velocity bounds
    vel_min = y.min() - dy.max() * 3
    vel_max = y.max() + dy.max() * 3
    
    # sum by weighted velocity disperions profiles
    xx = np.linspace(vel_min, vel_max, 1000)
    c2 = chi2

    if plot==True:
        plt.figure(figsize=(8,6))
        plt.xlabel(r'$\sigma$ km/s')
    
    sum_gaussians = np.zeros(shape=xx.shape)
    
    for i in range(len(y)):
        yy = weighted_gaussian(xx, y[i], dy[i], c2[i]) 
        if plot==True:
            plt.plot(xx, yy)    
        sum_gaussians += yy
    
    if plot==True:
        plt.show()
    
    max_likelihood = xx[sum_gaussians.argmax()]
    
    # calculate initial guess for Gaussian fit
    mu_0 = y.mean()
    sigma_0 = dy.mean()
    amp_0 = sum_gaussians.max()

    # fit the summed profile with a Gaussian to get sigma for the bin
    parameters, covariance = curve_fit(weighted_gaussian, xx, sum_gaussians, p0=[mu_0, sigma_0, amp_0])

    # take the parameters for plotting
    fit_mu = parameters[0]
    fit_sigma = parameters[1]
    fit_amp = parameters[2]
    fit_y = weighted_gaussian(xx, fit_mu, fit_sigma, fit_amp)

    # check if max_likelihood and fit_mu are super far apart
    if abs(max_likelihood - fit_mu) > 5:
        print('Check this fit...')
    # plot it all
    
    if plot==True:
        plt.figure(figsize=(8,6))

        for i in range(len(y)):
            yy = weighted_gaussian(xx, y[i], dy[i], c2[i]) 
            plt.plot(xx, yy)    

        plt.plot(xx, sum_gaussians, '-', c='b', label='sum of gaussians')
        plt.axvline(max_likelihood, c='b', linestyle='--', label=f'Max likelihood {np.around(max_likelihood, 2)}')
        plt.plot(xx, fit_y, '-', c='k', label='Gaussian fit to the sum')
        plt.axvline(fit_mu, linestyle='--', c='k', label=f'mean {np.around(fit_mu, 2)}')
        plt.axvline(fit_mu - fit_sigma, linestyle=':', c='k', label=f'sigma {np.around(fit_sigma, 2)}')
        plt.axvline(fit_mu + fit_sigma, linestyle=':', c='k')
        plt.xlabel(r'$\sigma$ km/s')
        plt.legend(loc='upper left')
        plt.show()
        
    return fit_mu, fit_sigma


def estimate_covariance (vel1, vel2, mu1, mu2, chi2_1, chi2_2):
    # calculat covariance matrix from individual solution means (vel1 and vel2) as arrays 
    # and sample means (mu1, mu2) weighted by product of normalized likelihoods (not correct..)
    likelihood_1 = np.exp( -1/2 * (chi2_1))
    normalized_likelihood_1 = likelihood_1/np.sum(likelihood_1)
    likelihood_2 = np.exp( -1/2 * (chi2_2))
    normalized_likelihood_2 = likelihood_2/np.sum(likelihood_2)
    covariance = np.sum( (vel1 - mu1) * (vel2 - mu2) * np.sqrt(normalized_likelihood_1 * normalized_likelihood_2) )
    covariance_2 = np.mean( (vel1 - mu1) * (vel2 - mu2) )
    return covariance#, covariance_2


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

#########################################################################
### loop through the objects

# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'

for obj_name in obj_names:
    
    print('#####################################################')
    print('#####################################################')
    print()
    print(f'Beginning final kinematics script for object {obj_name}.')
    print()
    
    # set abbreviation for directories
    obj_abbr = obj_name[4:9] # e.g. J0029
    # object directory
    dir = f'{data_dir}mosaics/{obj_name}/'
    # set systematics directory
    syst_dir = f'{dir}{obj_name}_systematics/'
    # make new directory for outputs
    output_dir = f'{dir}{obj_name}_final_kinematics/'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
    
    # bring in systematics data
    V = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_V.txt', delimiter=',')
    VD = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_VD.txt', delimiter=',')
    dV = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_dV.txt', delimiter=',')
    dVD = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_dVD.txt', delimiter=',')
    chi2 = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_chi2.txt', delimiter=',')
    
    ###########
    # calculate velocity dispersions
    velocity_dispersions = np.zeros(len(VD))
    sigmas = np.zeros(len(VD))
    #velocities = np.zeros(len(V))

    print('Calculating velocity disperions and sigmas')
    print()
    print()
                             
    for i in range(len(VD)):
        print('###################')
        print(f'Bin number {i}')
        velocity_dispersions[i], sigmas[i] = estimate_systematic_error(81, VD[i], dVD[i], chi2[i], plot=False)
        print(velocity_dispersions[i])
    
    # save velocity dispersions
    np.savetxt(f'{output_dir}{obj_name}_velocity_dispersions_binned.txt', 
               velocity_dispersions,
               delimiter=',')
    
    ####################################
    # calculate covariance matrix
    
    covariance_matrix = np.zeros((len(VD), len(VD)))
                                               
    for i in range(len(VD)):
        for j in range(len(VD)):
            if j == i:
                covariance_matrix[i,j] = sigmas[i]**2
            else:
                covariance_matrix[i,j] = estimate_covariance (VD[i], VD[j], velocity_dispersions[i], velocity_dispersions[j], chi2[i], chi2[j])
    
    # save covariance matrix
    np.savetxt(f'{output_dir}{obj_name}_covariance_matrix.txt', covariance_matrix,
               delimiter=',')

    vmin = np.min(covariance_matrix)
    vmax = np.max(covariance_matrix)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

    plt.figure(figsize=(16,16))
    plt.imshow(covariance_matrix, cmap='seismic', norm=norm)
    plt.title('J0029 covariance matrix', fontsize=16)
    plt.xlabel('bin #', fontsize=16)
    plt.ylabel('bin #', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.colorbar()
    plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_visual.pdf')
    plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_visual.png')
    
    ##################################
    # show it without the diagonal for easier view of the covariance terms
    
    covariance_without_diagonal = covariance_matrix - np.diagonal(covariance_matrix)*np.identity(len(covariance_matrix))

    vmin = np.min(covariance_without_diagonal)
    vmax = np.max(covariance_without_diagonal)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

    plt.figure(figsize=(16,16))
    plt.imshow(covariance_without_diagonal, cmap='seismic', norm=norm)
    plt.title('J0029 covariance matrix', fontsize=16)
    plt.xlabel('bin #', fontsize=16)
    plt.ylabel('bin #', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.colorbar()
    
    plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_no_diag_visual.pdf')
    plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_no_diag_visual.png')
    
    print()
    print('####################################')
    print()
    print(f'{obj_name} finished.')
    print()
    
    tock=timer()
    time_elapsed=(tock-tick)/3600 # hours
    print(f'Time elapsed: {time_elapsed} hours.')
    print()
    
print('#################################################')
print('#################################################')

###########
    # calculate velocity dispersions
    velocity_dispersions = np.zeros(len(VD))
    sigmas = np.zeros(len(VD))
    #velocities = np.zeros(len(V))

    print('Calculating velocity disperions and sigmas')
    print()
    print()
                             
    for i in range(len(VD)):
        print('###################')
        print(f'Bin number {i}')
        velocity_dispersions[i], sigmas[i] = estimate_systematic_error(81, VD[i], dVD[i], chi2[i], plot=False)
        print(velocity_dispersions[i])
    
    # save velocity dispersions
    np.savetxt(f'{output_dir}{obj_name}_velocity_dispersions_binned.txt', 
               velocity_dispersions,
               delimiter=',')
    
    ####################################
    # calculate covariance matrix
    
    covariance_matrix = np.zeros((len(VD), len(VD)))
                                               
    for i in range(len(VD)):
        for j in range(len(VD)):
            if j == i:
                covariance_matrix[i,j] = sigmas[i]**2
            else:
                covariance_matrix[i,j] = estimate_covariance (VD[i], VD[j], velocity_dispersions[i], velocity_dispersions[j], chi2[i], chi2[j])
    
    # save covariance matrix
    np.savetxt(f'{output_dir}{obj_name}_covariance_matrix.txt', covariance_matrix,
               delimiter=',')

    vmin = np.min(covariance_matrix)
    vmax = np.max(covariance_matrix)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

    plt.figure(figsize=(16,16))
    plt.imshow(covariance_matrix, cmap='seismic', norm=norm)
    plt.title('J0029 covariance matrix', fontsize=16)
    plt.xlabel('bin #', fontsize=16)
    plt.ylabel('bin #', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.colorbar()
    plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_visual.pdf')
    plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_visual.png')
    
    ##################################
    # show it without the diagonal for easier view of the covariance terms
    
    covariance_without_diagonal = covariance_matrix - np.diagonal(covariance_matrix)*np.identity(len(covariance_matrix))

    vmin = np.min(covariance_without_diagonal)
    vmax = np.max(covariance_without_diagonal)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

    plt.figure(figsize=(16,16))
    plt.imshow(covariance_without_diagonal, cmap='seismic', norm=norm)
    plt.title('J0029 covariance matrix', fontsize=16)
    plt.xlabel('bin #', fontsize=16)
    plt.ylabel('bin #', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.colorbar()
    
    plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_no_diag_visual.pdf')
    plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_no_diag_visual.png')
    
    print()
    print('####################################')
    print()
    print(f'{obj_name} finished.')
    print()
    
    tock=timer()
    time_elapsed=(tock-tick)/3600 # hours
    print(f'Time elapsed: {time_elapsed} hours.')
    print()

print('#################################################')
print('#################################################')
print('#################################################')
    
print()
print('Script complete.')
