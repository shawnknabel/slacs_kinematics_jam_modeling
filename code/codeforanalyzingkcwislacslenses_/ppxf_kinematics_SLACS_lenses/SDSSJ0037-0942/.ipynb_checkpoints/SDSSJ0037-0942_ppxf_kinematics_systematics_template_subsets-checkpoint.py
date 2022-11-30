#!/usr/bin/env python
##############################################################################
'''
Code for extracting the kinematics information from the
KCWI data with ppxf.

Geoff Chih-Fan Chen, Feb 28 2022 for Shawan Knabel.

Shawn Knabel, Feb 28 2022 editting for my own machine and directories.
03/01/22 - SDSSJ0029-0055
07/12/22 - systematics checks
08/31/22 - changing stellar template subsets to be random subsets of the G789K012 selection that CF made, instead of temperature ranges.

'''

from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ppxf.kcwi_util import register_sauron_colormap
from ppxf.kcwi_util import visualization
from ppxf.kcwi_util import get_datacube
from ppxf.kcwi_util import ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift
from ppxf.kcwi_util import find_nearest
from ppxf.kcwi_util import SN_CaHK
from ppxf.kcwi_util import select_region
from ppxf.kcwi_util import voronoi_binning
from ppxf.kcwi_util import get_voronoi_binning_data
from ppxf.kcwi_util import get_velocity_dispersion_deredshift
from ppxf.kcwi_util import kinematics_map
from ppxf.kcwi_util import stellar_type

import pathlib # to create directory

import pickle

from time import perf_counter as timer
# register first tick
tick = timer()

from datetime import date
today = date.today().strftime('%d%m%y')

register_sauron_colormap()


#------------------------------------------------------------------------------
# data directory.
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
stellar_library_dir = f'{data_dir}xshooter_lib/random_subset_selections/'

# Set 'obj_name', 'z', 'T_exp'
obj_name = 'SDSSJ0037-0942'
obj_abbr = obj_name[4:9] # e.g. J0029
z = 0.195 # lens redshift
T_exp = 1800*3*60 #266 * 60
lens_center_x,lens_center_y = 59, 137

#------------------------------------------------------------------------------
# Kinematics systematics choices

# stellar templates libraries
libraries = ['all_dr2_fits_G789K012_rand_subset_1',
             'all_dr2_fits_G789K012_rand_subset_2',
             'all_dr2_fits_G789K012_rand_subset_3']

# aperture
apertures = ['R0','R1','R2']

# wavelength range
wave_mins = [315,320,325] # narrower, initial, wider
wave_maxs = [435,430,425] 

# degree of the additive Legendre polynomial in ppxf
degrees = [4,5,6] # 110/25 = 4.4 round up

#------------------------------------------------------------------------------
# Information specific to KCWI and templates

## R=3600. spectral resolution is ~ 1.42A
FWHM = 1.42 #1.42

FWHM_tem_xshooter = 0.43 #0.43

## initial estimate of the noise
noise = 0.014

# velocity scale ratio
velscale_ratio = 2

#------------------------------------------------------------------------------
# variable settings in ppxf and utility functions

# cut the datacube at lens center, radius given here
radius_in_pixels = 21

# target SN for voronoi binning
vorbin_SN_target = 20

# global template spectrum chi2 threshold
global_template_spectrum_chi2_threshold = 1

#------------------------------------------------------------------------------

'''
Step 0: input the necessary information of the datacube
'''
    
# object directory
dir = f'{data_dir}mosaics/{obj_name}/'

# make save directory
save_dir = f'{dir}{obj_name}_systematics_template_subsets_{today}/'
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True) 

#KCWI mosaic datacube
name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'

## R=3600. spectral resolution is ~ 1.42A
FWHM = 1.42 #1.42

FWHM_tem_xshooter = 0.43 #0.43

## initial estimate of the noise
noise = 0.014

## velocity scale ratio?
velscale_ratio = 2

## import voronoi binning data
voronoi_binning_data = fits.getdata(dir +'voronoi_binning_' + name + '_data.fits')


'''
Step 8: systematics tests
'''

N = voronoi_binning_data.shape[0]

systematics_VD  = np.zeros(shape=(N,0))
systematics_dVD = np.zeros(shape=(N,0))
systematics_V   = np.zeros(shape=(N,0))
systematics_dV  = np.zeros(shape=(N,0))
systematics_chi2 = np.zeros(shape=(N,0))

# create a pandas dataframe for the name of the model l#a#d#w# and chi2 of global fit
global_fit_chi2_log = pd.DataFrame((np.zeros((3**4,2))), dtype=object, columns=['model_name','chi2'])
n = 0

# check template library, aperture radius, degree, and wavelength ranges for systematics
for l in range(len(libraries)): # 3 libraries
    library = f'{stellar_library_dir}{libraries[l]}/'
    for a in range(len(apertures)): # 2 aperture sizes
        aperture = f'{dir}{obj_abbr}_central_spectrum_{apertures[a]}.fits'
        for d in range(len(degrees)):
            #degree
            degree=degrees[d]
            for w in range(len(wave_mins)):
                wave_min = wave_mins[w]
                wave_max = wave_maxs[w]
                
                VD_name = f'l{l}a{a}d{d}w{w}'
                
                print('################################################')
                print(f'Beginning model {VD_name}')
                print('################################################')
    
                
                # make save directory
                model_dir = f'{save_dir}{VD_name}/'
                pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True) 
                
                templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, quasar = \
                    ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(
                        library,
                        degree=degree,
                        spectrum_aperture=aperture,
                        wave_min=wave_min,
                        wave_max=wave_max,
                        velscale_ratio=velscale_ratio,
                        z=z,
                        noise=noise,
                        templates_name='xshooter',
                        FWHM=FWHM,
                        FWHM_tem=FWHM_tem_xshooter,
                        plot=True)

                # save model name and chi2
                global_fit_chi2_log.iloc[n, :] = VD_name, pp.chi2
                n = n+1 # advance to next model in pandas dataframe

                # plot the spectrum and template fits
                plt.savefig(model_dir + 'global_template_spectrum.png')
                plt.clf()

                # if chi2 < 1 
                # forget this chi2 threshold... it will be reflected in high uncertainty
                # save the pp object as a pickle 
                #if pp.chi2 < 1:
                # if chi2 is bad, just flag it 
                #else:
                if pp.chi2 < 1:
                    print('chi2 < 1')
                    
                # write pickle file
                pickle_file = open(f'{model_dir}global_template_spectrum_fit.pkl', 'wb')
                pickle.dump(pp, pickle_file)
                pickle_file.close()
                
                # ppxf on bins
                nTemplates_xshooter = templates.shape[1]
                global_temp_xshooter = templates @ pp.weights[
                                                   :nTemplates_xshooter]
                
                # run ppxf on all bins
                # the results will be N (# bins) rows of 5 columns
                # columns: V, VD, dV, dVD, chi2
                systematics_results = get_velocity_dispersion_deredshift(degree=degree,
                                                   spectrum_aperture=aperture,
                                                   voronoi_binning_data=voronoi_binning_data,
                                                   velscale_ratio=velscale_ratio,
                                                   z=z,
                                                   noise=noise,
                                                   FWHM=FWHM,
                                                   FWHM_tem_xshooter=FWHM_tem_xshooter,
                                                   dir=dir,
                                                   save_dir=model_dir,
                                                   libary_dir=library,
                                                   global_temp=global_temp_xshooter,
                                                   wave_min=wave_min,
                                                   wave_max=wave_max,
                                                   T_exp=T_exp,
                                                   VD_name=VD_name,
                                                   plot=False)
                    
                # separate into V, VD, dV, dVD and chi2
                # stack them all by bin according to the different models (or "solutions")
                # rows are bins, columns are models
                systematics_V =np.hstack((systematics_V,
                                           systematics_results[:,0:1]))
                systematics_VD =np.hstack((systematics_VD,
                                           systematics_results[:,1:2]))
                systematics_dV =np.hstack((systematics_dV,
                                           systematics_results[:,2:3]))
                systematics_dVD =np.hstack((systematics_dVD,
                                           systematics_results[:,3:4]))
                systematics_chi2 =np.hstack((systematics_chi2,
                                           systematics_results[:,4:5]))
                
                # record time elapsed
                tock = timer()
                time_passed = (tock - tick) / 3600. # hours
                print('Time elapsed since start of script: ' + '{:.4f}'.format(time_passed) + ' hours')
                    

# save the systematics
np.savetxt(f'{save_dir}{obj_name}_systematics_VD.txt',
           systematics_VD, delimiter=',')
np.savetxt(f'{save_dir}{obj_name}_systematics_dVD.txt',
           systematics_dVD, delimiter=',')
np.savetxt(f'{save_dir}{obj_name}_systematics_V.txt', 
           systematics_V, delimiter=',')
np.savetxt(f'{save_dir}{obj_name}_systematics_dV.txt',
           systematics_dV, delimiter=',')
np.savetxt(f'{save_dir}{obj_name}_systematics_chi2.txt',
           systematics_chi2, delimiter=',')

# save the chi2 log
global_fit_chi2_log.to_csv(f'{save_dir}global_fit_chi2_log.csv')
    
# plot the systematics
    
x = np.arange(systematics_VD.shape[1])
Nbin=10
Nstar=0
VD_names = global_fit_chi2_log['model_name']

for i in range(N):
    if np.mod(i, Nbin) == 0:
        f, axarr = plt.subplots(Nbin, sharex=True)
        Nstar=i
    y = systematics_VD[i]
    dy = systematics_dVD[i]
    axarr[i-Nstar].errorbar(x, y, yerr=dy, fmt='o', color='black',
                      ecolor='lightgray', elinewidth=1, capsize=0)
    if (np.mod(i, Nbin) == 9) or (i == N-1):
        f.suptitle('bin (%s-%s)'%(Nstar+1,Nstar+Nbin))
        f.set_size_inches(12, 18)
        plt.xticks(np.arange(len(VD_names)))
        axarr[Nbin-1].set_xticklabels(VD_names, rotation=90)
        plt.savefig(f'{save_dir}{obj_name}_{Nstar+1}{Nstar+Nbin}_systematics.png')
        #plt.show()
        
print('Done.')
print('###############################################################')
