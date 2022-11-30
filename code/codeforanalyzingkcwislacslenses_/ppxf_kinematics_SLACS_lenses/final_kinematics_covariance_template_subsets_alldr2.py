# this calculates final kinematics pdf and covariance for all objects in a list by looping through object names.

# 
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import pathlib # to create directory
from datetime import date
today = date.today().strftime('%d%m%y')
from time import perf_counter as timer
import glob
import pickle
# register first tick
tick = timer()

#################################################
# objects
obj_names = ['SDSSJ0029-0055',
             #'SDSSJ0037-0942',
             #'SDSSJ0330-0020',
             #'SDSSJ1112+0826',
             #'SDSSJ1204+0358',
             #'SDSSJ1250+0523',
             #'SDSSJ1306+0600',
             #'SDSSJ1402+6321',
             #'SDSSJ1531-0105',
             #'SDSSJ1538+5817',
             #'SDSSJ1621+3931',
             #'SDSSJ1627-0053',
             #'SDSSJ1630+4520',
             #'SDSSJ2303+1422'
            ]

#################################################
# functions specific to this script

def weighted_gaussian(xx, mu, sig, c2):
    yy = np.zeros(shape=xx.shape)
    for i in range(len(xx)):
        yy[i] = np.exp(-np.power(xx[i] - mu, 2.) / (2 * np.power(sig, 2.))) * np.exp(-0.5 * c2)
    return yy

def estimate_systematic_error (k_solutions, velocity, random_error, chi2, vel_moment, plot=True):
    
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
        if vel_moment == 'VD':
            plt.ylabel(r'$\sigma$ km/s')
        elif vel_moment == 'V':
            plt.ylabel('Velocity km/s')
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
        if vel_moment == 'VD':
            plt.xlabel(r'$\sigma$ km/s')
        elif vel_moment == 'V':
            plt.xlabel('Velocity km/s')
    
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
        if vel_moment == 'VD':
            plt.xlabel(r'$\sigma$ km/s')
        elif vel_moment == 'V':
            plt.xlabel('Velocity km/s')
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
    
# write a function for getting, sorting, and saving the global template velocities

def create_glob_temp_velocities_array (save_dir, save=True):

    glob_temp_velocities = np.zeros(162)
    models_names = np.zeros(162).astype(str)

    i = 0
    for file in glob.glob(f'{save_dir}**/*l*.pkl'):
        model = file[-41:-33]
        file = open(file,'rb') 
        picle = pickle.load(file)
        glob_temp_velocities[i] = picle.sol[0]
        models_names[i] = model
        i = i+1
    
    glob_temp_vel_sort = np.column_stack((glob_temp_velocities, models_names))
    glob_temp_vel_sort = glob_temp_vel_sort[glob_temp_vel_sort[:,1].argsort()]
    
    glob_temp_vel_array = glob_temp_vel_sort[:,0].astype(float)
    if save == True:
        np.savetxt(f'{save_dir}glob_temp_vel_array.txt', glob_temp_vel_array, delimiter=',')
    
    return glob_temp_vel_array

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
    syst_dir = f'{dir}{obj_name}_systematics_template_subsets_070922/'
    # make new directory for outputs
    output_dir = f'{dir}{obj_name}_final_kinematics_template_subsets_070922/'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
    
    # bring in systematics data
    V = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_V.txt', delimiter=',')
    VD = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_VD.txt', delimiter=',')
    dV = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_dV.txt', delimiter=',')
    dVD = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_dVD.txt', delimiter=',')
    chi2 = np.genfromtxt(f'{syst_dir}{obj_name}_systematics_chi2.txt', delimiter=',')
    
    # calculate and save global template velocities (for ea of 162 models) for velocity subtraction
    global_template_velocities = create_glob_temp_velocities_array( syst_dir )
    
    ###########
    # calculate velocity dispersions
    velocity_dispersions = np.zeros(len(VD))
    VD_sigmas = np.zeros(len(VD))
    velocities = np.zeros(len(V))
    V_sigmas = np.zeros(len(V))
    
    for vel_moment in ['VD','V']: ### ignore the velocities for now
        
        if vel_moment == 'VD':
            velocity_measurements = velocity_dispersions
            vel = VD
            sigmas = VD_sigmas
            dvel = dVD
        elif vel_moment == 'V':
            velocity_measurements = velocities
            vel = V
            sigmas = V_sigmas
            dvel = dV

        print(f'Calculating {vel_moment}s and sigmas')
        print()
        print()

        for i in range(len(vel)): # vel[i] is 81 velocities in ith bin...
            
            # if vel_moment=V, correct for global template velocity
            if vel_moment == 'V':
                vel[i] = vel[i] - global_template_velocities
            
            print('###################')
            print(f'Bin number {i}')
            velocity_measurements[i], sigmas[i] = estimate_systematic_error(162, vel[i], dvel[i], chi2[i],
                                                                            vel_moment,
                                                                            plot=False)
            print(velocity_measurements[i])

        # save velocity measurements
        np.savetxt(f'{output_dir}{obj_name}_{vel_moment}_binned.txt', 
                   velocity_measurements,
                   delimiter=',')

        ####################################
        # calculate covariance matrix

        covariance_matrix = np.zeros((len(vel), len(vel)))

        for i in range(len(vel)):
            for j in range(len(vel)):
                if j == i:
                    covariance_matrix[i,j] = sigmas[i]**2
                else:
                    covariance_matrix[i,j] = estimate_covariance (vel[i], vel[j], 
                                                                  velocity_measurements[i], 
                                                                  velocity_measurements[j], 
                                                                  chi2[i], 
                                                                  chi2[j])

        # save covariance matrix
        np.savetxt(f'{output_dir}{obj_name}_covariance_matrix_{vel_moment}.txt', covariance_matrix,
                   delimiter=',')

        vmin = np.min(covariance_matrix)
        vmax = np.max(covariance_matrix)
        
        if (vmin != 0) & (vmax != 0):
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        else:
            norm = None

        plt.figure(figsize=(16,16))
        plt.imshow(covariance_matrix, cmap='seismic', norm=norm)
        plt.title(f'{obj_name} {vel_moment} covariance matrix', fontsize=16)
        plt.xlabel('bin #', fontsize=16)
        plt.ylabel('bin #', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.colorbar()
        plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_{vel_moment}_visual.pdf')
        plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_{vel_moment}_visual.png')

        ##################################
        # show it without the diagonal for easier view of the covariance terms

        covariance_without_diagonal = covariance_matrix - np.diagonal(covariance_matrix)*np.identity(len(covariance_matrix))

        vmin = np.min(covariance_without_diagonal)
        vmax = np.max(covariance_without_diagonal)

        if (vmin != 0) & (vmax != 0):
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
        else:
            norm = None

        plt.figure(figsize=(16,16))
        plt.imshow(covariance_without_diagonal, cmap='seismic', norm=norm)
        plt.title(f'{obj_name} {vel_moment} covariance matrix', fontsize=16)
        plt.xlabel('bin #', fontsize=16)
        plt.ylabel('bin #', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.colorbar()

        plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_{vel_moment}_no_diag_visual.pdf')
        plt.savefig(f'{output_dir}{obj_name}_covariance_matrix_{vel_moment}_no_diag_visual.png')

        print()
        print('####################################')
        print()
        print(f'{vel_moment} finished.')
        print()
    print('####################################')
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
