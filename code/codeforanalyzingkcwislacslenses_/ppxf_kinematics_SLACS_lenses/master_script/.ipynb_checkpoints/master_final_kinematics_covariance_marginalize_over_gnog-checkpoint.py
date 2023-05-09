###########################################################################
###########################################################################
# 03/03/23 - This is the master script for the "final" kinematics and covariance (i.e. it puts the systematics together)
# iterates through the objects
# 05/02/23 - This has been modified from "master_final_kinematics_covariance" to handle the data from the Hoffman2 run.
# It also will marginalize over G-band variation.
# This will give 108 variations.

# this calculates final kinematics pdf and covariance for all objects in a list by looping through object names.


print('##################################################################################################')
print('##################################################################################################')
print('##################################################################################################')

print()
print('Beginning that there scripty bit.')
print()
        
print('##################################################################################################')
print('##################################################################################################')
print('##################################################################################################')

# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from astropy.io import fits
from scipy.optimize import curve_fit
from statsmodels.stats.correlation_tools import cov_nearest

import pathlib # to create directory
import glob
from pathlib import Path # to create directory
import pickle
from datetime import date
today = date.today().strftime('%d%m%y')
from time import perf_counter as timer
# register first tick
tick = timer()

# command line arguments to select obj_names to be used
import sys
obj_index = np.array(sys.argv[1:], dtype=int)

#################################################
# date and number of initial kinematics run e.g. 2023-02-28_2
date_of_kin = '2023-02-28_2'

#################################################
# objects
obj_names = ['SDSSJ0029-0055',
             'SDSSJ0037-0942',
             'SDSSJ0330-0020',
             'SDSSJ1112+0826',
             'SDSSJ1204+0358',
             'SDSSJ1250+0523',
             'SDSSJ1306+0600',
             'SDSSJ1402+6321',
             'SDSSJ1531-0105',
             'SDSSJ1538+5817',
             'SDSSJ1621+3931',
             'SDSSJ1627-0053',
             'SDSSJ1630+4520',
             'SDSSJ2303+1422'
            ]

# select the active object names to run from command line
active_objects = []
for i in obj_index:
    active_objects.append(obj_names[i])

obj_names = active_objects
    
print(f'Active objects are {obj_names}')
print()

#################################################
# functions specific to this script

def weighted_gaussian(xx, mu, sig, c2):
    yy = np.zeros(shape=xx.shape)
    for i in range(len(xx)):
        yy[i] = np.exp(-np.power(xx[i] - mu, 2.) / (2 * np.power(sig, 2.))) * np.exp(-0.5 * c2)
    return yy

def estimate_systematic_error (k_solutions, velocity, random_error, chi2, vel_moment, bin_number, err_dir, vel_mom_name, plot=True):
    
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
        plt.close()
    
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
        plt.close()

    max_likelihood = xx[sum_gaussians.argmax()]
    
    # calculate initial guess for Gaussian fit
    mu_0 = y.mean()
    sigma_0 = dy.mean()
    amp_0 = sum_gaussians.max()
    
    # try to make it a Gaussian
    try:
        # fit the summed profile with a Gaussian to get sigma for the bin
        parameters, covariance = curve_fit(weighted_gaussian, xx, sum_gaussians, p0=[mu_0, sigma_0, amp_0])

        # take the parameters for plotting
        fit_mu = parameters[0]
        fit_sigma = parameters[1]
        fit_amp = parameters[2]
        fit_y = weighted_gaussian(xx, fit_mu, fit_sigma, fit_amp)

        # check if max_likelihood and fit_mu are super far apart
        if abs(max_likelihood - fit_mu) > 5:
            print('Max likelihood and mean are super far apart. Check this fit...')
            
            # log the error
            cov_error = open(f'{err_dir}error_logs.txt', 'a')
            cov_error.write('########################################## \n')
            cov_error.write(f'{vel_mom_name} - bin number {bin_number} \n')
            cov_error.write('Max likelihood and mean are super far apart. Check this fit...  \n')
            cov_error.write(f'{err_dir}sum_bin_gaussians_mean_max_discrepency_{vel_mom_name}_{bin_number}.png \n')
            cov_error.close()
            
            # plot the 81 velocity disperions with error bars
            f, axarr = plt.subplots(1)
            axarr.errorbar(x, y, yerr=dy, fmt='o', color='black',
                      ecolor='lightgray', elinewidth=1, capsize=0)
            f.set_size_inches(12, 2)
            plt.xlabel('Solution #')
            plt.title(f'Bin number {bin_number}')
            if vel_moment == 'VD':
                plt.ylabel(r'$\sigma$ km/s')
            elif vel_moment == 'V':
                plt.ylabel('Velocity km/s')
            plt.tight_layout()
            plt.savefig(f'{err_dir}bin_gaussians_error_{vel_mom_name}_{bin_number}.png', bbox_inches='tight')
            plt.close()
            
            # plot the gaussians
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
            plt.title(f'Bin number {bin_number}')
            if vel_moment == 'VD':
                plt.xlabel(r'$\sigma$ km/s')
            elif vel_moment == 'V':
                plt.xlabel('Velocity km/s')
            plt.xlabel(r'$\sigma$ km/s')
            plt.legend(loc='upper left')
            plt.savefig(f'{err_dir}sum_bin_gaussians_mean_max_discrepency_{vel_mom_name}_{bin_number}.png', bbox_inches='tight')
            plt.close()
            
        # plot if plot is True

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
            plt.title(f'Bin number {bin_number}')
            if vel_moment == 'VD':
                plt.xlabel(r'$\sigma$ km/s')
            elif vel_moment == 'V':
                plt.xlabel('Velocity km/s')
            plt.xlabel(r'$\sigma$ km/s')
            plt.legend(loc='upper left')
            plt.show()
            plt.close()

    except:
        print('Could not make this Gaussian. Check.')
        
        # Make an error directory
        print('Error plot will be in ', err_dir)
        # log the error
        cov_error = open(f'{kin_dir}error_logs.txt', 'a')
        cov_error.write('########################################## \n')
        cov_error.write(f'{vel_mom_name} - bin number {bin_number} \n')
        cov_error.write(f'Could not make this Gaussian. Check. \n')
        cov_error.write(f'{err_dir}bin_gaussians_error_{vel_mom_name}_{bin_number}.png \n')
        cov_error.close()
        
        # Make plots to view!
        # plot the 81 velocity disperions with error bars
        f, axarr = plt.subplots(1)
        axarr.errorbar(x, y, yerr=dy, fmt='o', color='black',
                  ecolor='lightgray', elinewidth=1, capsize=0)
        f.set_size_inches(12, 2)
        plt.xlabel('Solution #')
        plt.title(f'Bin number {i}')
        if vel_moment == 'VD':
            plt.ylabel(r'$\sigma$ km/s')
        elif vel_moment == 'V':
            plt.ylabel('Velocity km/s')
        plt.tight_layout()
        plt.savefig(f'{err_dir}bin_gaussians_error_{vel_mom_name}_{bin_number}.png', bbox_inches='tight')
        plt.close()
        
        fit_mu = float('nan')
        fit_sigma = float('nan')

    return fit_mu, fit_sigma


def estimate_covariance (vel1, vel2, mu1, mu2, chi2_1, chi2_2):
    # calculate covariance matrix from individual solution means (vel1 and vel2) as arrays #### Something is wrong, I should be estimating covariance between bins
    # and sample means (mu1, mu2) weighted by product of normalized likelihoods (not correct..)
    
    likelihood_1 = np.exp( -1/2 * (chi2_1)) # chi2_1 is 81 chi2 for bin 1, so likelihood_1 is 81 likelihoods for bin 1
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

def create_glob_temp_velocities_array (num_variations, save_dir, save=True):

    glob_temp_velocities = np.zeros(num_variations)
    models_names = np.zeros(num_variations).astype(str)

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

def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)

#------------------------------------------------------------------------------
# Directories and files

# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
tables_dir = f'{data_dir}tables/'
mosaics_dir = f'{data_dir}mosaics/'
kinematics_full_dir = f'{data_dir}kinematics/'
kinematics_dir =f'{kinematics_full_dir}{date_of_kin}/'
print(f'Outputs will be in {kinematics_dir}')
print()

#------------------------------------------------------------------------------
# Kinematics systematics choices

# target SN for voronoi binning
vorbin_SN_targets = np.array([10, 15, 20])

# include G band or no
g_bands = ['no_g','g_band']

#########################################################################
### loop through the objects

for obj_name in obj_names:
    
    print('##########################################################################################################')
    print('##########################################################################################################')
    print('##########################################################################################################')
    print()
    print(f'Beginning final kinematics script for object {obj_name}.')
    print()
    print('##########################################################################################################')

    # set abbreviation for directories
    obj_abbr = obj_name[4:9] # e.g. J0029

    #------------------------------------------------------------------------------
    # object-specific directories

    mos_dir = f'{mosaics_dir}{obj_name}/' 
    # modified because of the file structure I used with the Hoffman2 cluster
    kin_dir = f'{kinematics_dir}{obj_name}/'
    #hoff2_dir = f'{kinematics_dir}systematics_hoffman2_2023-04-27_from_2023-02-28_2/{obj_name}/'

    #KCWI mosaic datacube
    mos_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'

    for vorbin_SN_target in vorbin_SN_targets:
        input_target_dir = f'{kin_dir}target_sn_{vorbin_SN_target}/'
        target_dir = input_target_dir #f'{hoff2_dir}{vorbin_SN_target}/'
        #syst_dir = f'{target_dir}systematics/'
        # make new directory for outputs
        fin_kin_dir = f'{target_dir}{obj_name}_{vorbin_SN_target}_marginalized_gnog_final_kinematics/'
        Path(fin_kin_dir).mkdir(parents=True, exist_ok=True)
        
        # Make an error directory
        err_dir = f'{fin_kin_dir}errors/'#hoff2_systematics_error/'
        Path(err_dir).mkdir(parents=True, exist_ok=True)
        print('Errors plot will be in ', err_dir)
        # create a txt file that records errors in gaussian summation and covariance estimation
        cov_error = open(f'{err_dir}{obj_abbr}_error_logs.txt', 'w')
        cov_error.write('########################################## \n')
        cov_error.write('########################################## \n')
        cov_error.write(f'This is the error log for {obj_abbr} final kinematics script \n \n')
        cov_error.close()

        print('################################################')
        print('################################################')
        print()
        print('Working with S/N target ', vorbin_SN_target)
        print()
        print('################################################')
        print('################################################')

        '''
        Step 0: input the necessary information of the datacube
        ''' 

        ## import voronoi binning data
        voronoi_binning_data = fits.getdata(input_target_dir +'voronoi_binning_' + obj_name + '_data.fits')
        N = voronoi_binning_data.shape[0] # for saving the systematics measurements # number bins, i think?

        '''
        Step 8: systematics tests
        '''
        
        # modifying to marginalize over G-band variations as well.

        # first create combined G-band and no_G directory in final kineamtics
        syst_gband_dir = f'{target_dir}systematics/g_band/'
        syst_nog_dir = f'{target_dir}systematics/no_g/'
        

        # create a mask that indexes good bins and rejects bad ones
        # bad bin is one for which the sum of gaussians doesn't work or the VD > 350 or [V] > 250
        bin_mask = np.ones(N)
        

        if obj_abbr != 'J0330':
            # G-band
            VD_g_band = np.genfromtxt(f'{syst_gband_dir}{obj_name}_g_band_systematics_VD.txt', delimiter=',')
            dVD_g_band = np.genfromtxt(f'{syst_gband_dir}{obj_name}_g_band_systematics_dVD.txt', delimiter=',')
            V_g_band = np.genfromtxt(f'{syst_gband_dir}{obj_name}_g_band_systematics_V.txt', delimiter=',')
            dV_g_band = np.genfromtxt(f'{syst_gband_dir}{obj_name}_g_band_systematics_dV.txt', delimiter=',')
            chi2_g_band = np.genfromtxt(f'{syst_gband_dir}{obj_name}_g_band_systematics_chi2.txt', delimiter=',')
            # subtract the global template velocity
            # calculate and save global template velocities (for ea of 81 models) for velocity subtraction
            global_template_velocities_g_band = create_glob_temp_velocities_array( VD_g_band.shape[1], syst_gband_dir )
            ################
            ####### subtract the global template velocities from each bin velocity (row)
            # create grid of global template velocities (num_bins, 81)
            glob_temp_vel_grid_g_band, _ = np.meshgrid(global_template_velocities_g_band, V_g_band[:,0])
            # subtract global template velocities from V
            V_g_band = V_g_band - glob_temp_vel_grid_g_band
            
            # no G
            VD_no_g = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_VD.txt', delimiter=',')
            dVD_no_g = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_dVD.txt', delimiter=',')
            V_no_g = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_V.txt', delimiter=',')
            dV_no_g = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_dV.txt', delimiter=',')
            chi2_no_g = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_chi2.txt', delimiter=',')
            # subtract the global template velocity
            # calculate and save global template velocities (for ea of 81 models) for velocity subtraction
            global_template_velocities_no_g = create_glob_temp_velocities_array( VD_no_g.shape[1], syst_nog_dir )
            ################
            ####### subtract the global template velocities from each bin velocity (row)
            # create grid of global template velocities (num_bins, 81)
            glob_temp_vel_grid_no_g, _ = np.meshgrid(global_template_velocities_no_g, V_no_g[:,0])
            # subtract global template velocities from V
            V_no_g = V_no_g - glob_temp_vel_grid_no_g
            
            VD = np.concatenate((VD_g_band, VD_no_g), axis=1)
            dVD = np.concatenate((dVD_g_band, dVD_no_g), axis=1)
            V = np.concatenate((V_g_band, V_no_g), axis=1)
            dV = np.concatenate((dV_g_band, dV_no_g), axis=1)
            chi2 = np.concatenate((chi2_g_band, chi2_no_g), axis=1)

        elif obj_abbr == 'J0330':
            VD = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_VD.txt', delimiter=',')
            dVDd = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_dVD.txt', delimiter=',')
            V = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_V.txt', delimiter=',')
            dV = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_dV.txt', delimiter=',')
            chi2 = np.genfromtxt(f'{syst_nog_dir}{obj_name}_no_g_systematics_chi2.txt', delimiter=',')
            
            # calculate and save global template velocities (for ea of 81x2 models) for velocity subtraction
            global_template_velocities = create_glob_temp_velocities_array(len(V), syst_nog_dir )
            ################
            ####### subtract the global template velocities from each bin velocity (row)
            # create grid of global template velocities (num_bins, 81x2)
            glob_temp_vel_grid, _ = np.meshgrid(global_template_velocities, V[:,0])
            # subtract global template velocities from V
            V = V - glob_temp_vel_grid
            
        ################
        # calculate Vrms from VD and V
        Vrms = np.sqrt(V**2 + VD**2)
        dVrms = Vrms * (V*dV + VD*dVD)/(V**2 + VD**2)
        
        
        ###########
        ###########
        # calculate velocities
        velocity_dispersions = np.zeros(len(VD))
        VD_sigmas = np.zeros(len(VD))
        velocities = np.zeros(len(V))
        V_sigmas = np.zeros(len(V))
        rms_velocities = np.zeros(len(Vrms))
        Vrms_sigmas = np.zeros(len(Vrms))
    

        for vel_moment in ['VD','V','Vrms']: ### ignore the velocities for now
        #for vel_moment in ['V','Vrms']: ### ignore the velocities for now

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
            elif vel_moment == 'Vrms':
                velocity_measurements = rms_velocities
                vel = Vrms
                sigmas = Vrms_sigmas
                dvel = dVrms

            vel_mom_name = f'{vorbin_SN_target}_combinedG_{vel_moment}'
            num_bins = len(vel)

            print(f'Calculating {vel_moment}s and errors')
            print()
            print()
            print(f'{num_bins} bins.')

            for i in range(num_bins): # vel[i] is 81 velocities in ith bin...

                ##### 09/15/22 - Doing this to the whole array before this step
                # if vel_moment=V, correct for global template velocity
                #if vel_moment == 'V':
                #    vel[i] = vel[i] - global_template_velocities

                print('###################')
                print(f'Bin number {i}')
                print(len(vel[i]))

                if bin_mask[i] == 0:
                    print('Bin has already been rejected.')


                else: # do the measurements, estimate error
                    # num variations
                    num_variations = VD.shape[1]
                    
                    velocity_measurements[i], sigmas[i] = estimate_systematic_error(num_variations, vel[i], dvel[i], chi2[i],
                                                                                        vel_moment,
                                                                                        bin_number=i,
                                                                                        err_dir=None,
                                                                                        vel_mom_name=vel_mom_name,
                                                                                        plot=True)
                    
                    print(velocity_measurements[i])
                    # update the mask if the bin is bad
                    if (np.isnan(velocity_measurements[i])):
                        bin_mask[i] = 0
                        # log the error
                        cov_error = open(f'{err_dir}error_logs.txt', 'a')
                        cov_error.write('########################################## \n')
                        cov_error.write(f'{vel_mom_name} \n')
                        cov_error.write(f'Masking bin {i} because it failed to be Gaussian. \n')
                        cov_error.close()
                        print(f'Masking bin {i} because it failed to be Gaussian.')
                    elif (vel_moment=='VD' and velocity_measurements[i] > 350):
                        bin_mask[i] = 0
                        # log the error
                        cov_error = open(f'{err_dir}error_logs.txt', 'a')
                        cov_error.write('########################################## \n')
                        cov_error.write(f'{vel_mom_name} \n')
                        cov_error.write(f'Masking bin {i} because VD > 350 km/s. \n')
                        cov_error.close()
                        print(f'Masking bin {i} because VD > 350 km/s.')
                    elif (vel_moment=='VD' and velocity_measurements[i] < 10):
                        bin_mask[i] = 0
                        # log the error
                        print(f'Masking bin {i} because VD < 10 km/s.')
                        cov_error = open(f'{err_dir}error_logs.txt', 'a')
                        cov_error.write('########################################## \n')
                        cov_error.write(f'{vel_mom_name} \n')
                        cov_error.write(f'Masking bin {i} because VD < 10 km/s. \n')
                    elif(vel_moment=='V' and abs(velocity_measurements[i]) > 200):
                        bin_mask[i] = 0
                        # log the error
                        cov_error = open(f'{err_dir}error_logs.txt', 'a')
                        cov_error.write('########################################## \n')
                        cov_error.write(f'{vel_mom_name} \n')
                        cov_error.write(f'Masking bin {i} because V > 200 km/s. \n')
                        cov_error.close()
                        print(f'Masking bin {i} because |V| > 200 km/s.')
                    else:
                        print('Bin is good.')

            # save velocity measurements
            np.savetxt(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_binned.txt', 
                        velocity_measurements,
                        delimiter=',')
                
            print()
            print('####################################')
            print()
            print(f'{obj_name} {vel_mom_name} error estimates finished.')
            print()
                
        # save the bin mask
        np.savetxt(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_bin_mask.txt', 
                       bin_mask,
                           delimiter=',')
            
        print()
        print('####################################')
        print()
        print(f'{obj_name} {vorbin_SN_target} error estimates finished.')
        print(f'Now covariances.')
        print()
            
        #############################################################
        # Covariance matrices with masked bin maps
             
        for vel_moment in ['VD','V','Vrms']: 

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
            elif vel_moment == 'Vrms':
                velocity_measurements = rms_velocities
                vel = Vrms
                sigmas = Vrms_sigmas
                dvel = dVrms

            vel_mom_name = f'{vorbin_SN_target}_combinedG_{vel_moment}'


            print(f'Calculating {vel_mom_name} covariance matrix')
            print()
            print()
            ####################################
            # calculate covariance matrices
                
            # bad bins
            bad_bins = np.argwhere(bin_mask==0)
            num_bad_bins = len(bad_bins)
            print(f'Bins {bad_bins} are bad, out of {num_bins}.')
                
            # mask index
            covariance_matrix_mask = np.ones((len(vel), len(vel)))
            # empty matrix
            covariance_matrix = np.zeros((len(vel), len(vel)))

            for i in range(len(vel)):
                for j in range(len(vel)):
                    # ignore if rejected by max
                    if (bin_mask[i] == 0) or (bin_mask[j] == 0):
                        # if either bin is bad
                        covariance_matrix[i,j] = float('nan')
                        covariance_matrix_mask[i,j] = 0
                    else:
                        if j == i:
                            covariance_matrix[i,j] = sigmas[i]**2
                        else:
                            covariance_matrix[i,j] = estimate_covariance (vel[i], vel[j], 
                                                                              velocity_measurements[i], 
                                                                              velocity_measurements[j], 
                                                                              chi2[i], 
                                                                              chi2[j])
                
            # take a covariance matrix that is masked 
            num_good_bins = num_bins - num_bad_bins
            covariance_matrix_masked = covariance_matrix[covariance_matrix_mask==1].reshape((num_good_bins, num_good_bins))
            
            print('covariance matrix....')
            print(covariance_matrix_masked)
            
            # check if masked covariance matrix is positive semi-definite, correct if not                            
            if is_pos_semidef(covariance_matrix_masked) != True:
                print('Approximating closest positive semi-definite covariance matrix')
                covariance_matrix = cov_nearest(covariance_matrix_masked)
                if is_pos_semidef(covariance_matrix_masked):
                    cov_matrix_header = f'This covariance matrix has been smoothed to the closest positive semi-definite covariance matrix, bad bins {bad_bins}'
                else:
                    print('Approximating again with lower threshold')
                    covariance_matrix_masked = cov_nearest(covariance_matrix_masked, threshold=1e-13)
                    cov_matrix_header = f'This covariance matrix has been smoothed to the closest positive semi-definite covariance matrix, bad bins {bad_bins}'
                # if still not true, save it, note it, move on
                if is_pos_semidef(covariance_matrix_masked) != True:
                    # log the error
                    cov_error = open(f'{err_dir}error_logs.txt', 'a')
                    cov_error.write('########################################## \n')
                    cov_error.write(f'{vel_mom_name} \n')
                    cov_error.write(f"covariance matrix still not positive semi-definite, take a closer look - {fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix.txt \n")
                    cov_error.close()
                    # save the covariance matrix with header saying it is not positive semi-definite
                    cov_matrix_header = f'This covariance matrix is not positive semi-definite, bad bins {bad_bins}'
                    print(f"covariance matrix still not positive semi-definite, take a closer look - {fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix.txt")
                        
                # save covariance matrix
                np.savetxt(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix.txt', covariance_matrix,
                               delimiter=',', header=cov_matrix_header)
                # save masked covariance matrix
                np.savetxt(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_masked.txt', covariance_matrix_masked,
                               delimiter=',', header=cov_matrix_header)


            else:
                # save covariance matrix (with bad bins)
                np.savetxt(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix.txt', covariance_matrix,
                               delimiter=',')
                # save masked covariance matrix
                np.savetxt(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_masked.txt', covariance_matrix_masked,
                               delimiter=',')
                    
            ###########################################
            # Plots
                
            # Plot the masked and the unmasked oness
                
            #####################################
            # masked
                
            print('Plotting masked covariance matrix')

            # normalize plotting colorbar
            vmin = np.nanmin(covariance_matrix_masked)
            vmax = np.nanmax(covariance_matrix_masked)
            print('Colorbar... vmin and vmax ', vmin, vmax)
            if vmin < 0:
                norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
                cmap = 'seismic'
            elif vmin >= 0:
                norm = None
                cmap = 'Reds'
            else:
                print('Colorbar problem... vmin and vmax ', vmin, vmax)

            plt.figure(figsize=(16,16))
            plt.imshow(covariance_matrix_masked, cmap='seismic', norm=norm)
            plt.title(f'{obj_name} {vel_moment} covariance matrix', fontsize=16)
            plt.xlabel('bin #', fontsize=16)
            plt.ylabel('bin #', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.colorbar()
            plt.savefig(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_masked_visual.pdf')
            plt.savefig(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_masked_visual.png')

            ##################################
            # show it without the diagonal for easier view of the covariance terms

            covariance_without_diagonal = covariance_matrix - np.diagonal(covariance_matrix)*np.identity(len(covariance_matrix))

            # normalize plotting colorbar
            vmin = np.nanmin(covariance_without_diagonal)
            vmax = np.nanmax(covariance_without_diagonal)
            print('Colorbar... vmin and vmax ', vmin, vmax)
            if vmin < 0:
                norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
                cmap = 'seismic'
            elif vmin >= 0:
                norm = None
                cmap = 'Reds'
            else:
                print('Colorbar problem... vmin and vmax ', vmin, vmax)

            plt.figure(figsize=(16,16))
            plt.imshow(covariance_without_diagonal, cmap='seismic', norm=norm)
            plt.title(f'{obj_name} {vel_moment} covariance matrix', fontsize=16)
            plt.xlabel('bin #', fontsize=16)
            plt.ylabel('bin #', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.colorbar()

            plt.savefig(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_masked_no_diag_visual.pdf')
            plt.savefig(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_masked_no_diag_visual.png')

            #####################################
            # unmasked              
            print('Plotting masked covariance matrix')
                
            # normalize plotting colorbar
            vmin = np.nanmin(covariance_matrix)
            vmax = np.nanmax(covariance_matrix)
            print('Colorbar... vmin and vmax ', vmin, vmax)
            if vmin < 0:
                norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
                cmap = 'seismic'
            elif vmin >= 0:
                norm = None
                cmap = 'Reds'
            else:
                print('Colorbar problem... vmin and vmax ', vmin, vmax)

            plt.figure(figsize=(16,16))
            plt.imshow(covariance_matrix, cmap='seismic', norm=norm)
            plt.title(f'{obj_name} {vel_moment} covariance matrix', fontsize=16)
            plt.xlabel('bin #', fontsize=16)
            plt.ylabel('bin #', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.colorbar()
            plt.savefig(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_visual.pdf')
            plt.savefig(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_visual.png')

            ##################################
            # show it without the diagonal for easier view of the covariance terms

            covariance_without_diagonal = covariance_matrix - np.diagonal(covariance_matrix)*np.identity(len(covariance_matrix))

            # normalize plotting colorbar
            vmin = np.nanmin(covariance_without_diagonal)
            vmax = np.nanmax(covariance_without_diagonal)
            print('Colorbar... vmin and vmax ', vmin, vmax)
            if vmin < 0:
                norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
                cmap = 'seismic'
            elif vmin >= 0:
                norm = None
                cmap = 'Reds'
            else:
                print('Colorbar problem... vmin and vmax ', vmin, vmax)

            plt.figure(figsize=(16,16))
            plt.imshow(covariance_without_diagonal, cmap='seismic', norm=norm)
            plt.title(f'{obj_name} {vel_moment} covariance matrix', fontsize=16)
            plt.xlabel('bin #', fontsize=16)
            plt.ylabel('bin #', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.colorbar()

            plt.savefig(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_no_diag_visual.pdf')
            plt.savefig(f'{fin_kin_dir}{obj_name}_{vel_mom_name}_covariance_matrix_no_diag_visual.png')

            print()
            print('####################################')
            print()
            print(f'{obj_name} {vel_mom_name} covariance finished.')
            print()
        
        print('####################################')
        print('####################################')
        print()
        print(f'{obj_name} covariance finished.')
        print()

        tock=timer()
        time_elapsed=(tock-tick)/3600 # hours
        print(f'Time elapsed: {time_elapsed} hours.')
        print()

    print('##################################################################################################')
    print('##################################################################################################')
    print('##################################################################################################')

    print()
    print(f'{obj_name} complete.')
    print()
        
    tock=timer()
    time_elapsed=(tock-tick)/3600 # hours
    print(f'Time elapsed: {time_elapsed} hours.')
    print()
        
    print('##################################################################################################')
    print()
