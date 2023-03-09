'''
###################################################################################################
#########
This is modified to run on Hoffman2 in username "knabel" with data in scratch directory
#########
Date is 02/27/23 - Shawn
This is the master script for the systematics test of kinematics 
This systematics check is where we actually take kinematics measurements.
# iterates through the objects
This code puts all the ppxf scripts for each individual object into one master script that can run any number of them iteratively. Hopefully it saves time. We'll see. I should figure out how to use multiple threads. I'm not a good programmer. 
I've left previous notes below.
############################################################
Code for extracting the kinematics information from the
KCWI data with ppxf.
Geoff Chih-Fan Chen, Feb 28 2022 for Shawan Knabel.
Shawn Knabel, Feb 28 2022 editting for my own machine and directories.
03/01/22 - SDSSJ0029-0055
07/12/22 - load and check both R=1 and R=2 center apertures. Favor R=1 and just do R=2 in the systematics check.
#############################################################
This is the Fateful Disaster Part 2 response.
I need to change the following:
1. Make "obj_dir" -> "mos_dir" for mosaic directory. Load from here, but do not save here.
2. Make "kin_dir", where we will save the values from these, and specify the "target_sn_10" (etc) directory for each experiment
3. Fix "Exp_time" -> should not have *60
4. Set the "Vorbin_target_SN" to be the one in the experiment
#############################################################
Let's get on with it, shall we?
'''

print('running')

from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ppxf
from ppxf.kcwi_util import register_sauron_colormap
from ppxf.kcwi_util import visualization
from ppxf.kcwi_util import get_datacube
from ppxf.kcwi_util import ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift
from ppxf.kcwi_util import remove_background_source_from_galaxy_deredshift
from ppxf.kcwi_util2 import getMaskInFitsFromDS9reg
from ppxf.kcwi_util import fitting_SN
from ppxf.kcwi_util import fitting_SN_forshow
from ppxf.kcwi_util import find_nearest
from ppxf.kcwi_util import SN_CaHK
from ppxf.kcwi_util import select_region
from ppxf.kcwi_util import voronoi_binning
from ppxf.kcwi_util import get_voronoi_binning_data
from ppxf.kcwi_util import get_velocity_dispersion_deredshift
from ppxf.kcwi_util import kinematics_map
from ppxf.kcwi_util import stellar_type
from scipy.optimize import minimize

from pathlib import Path # to create directory
import pickle
from datetime import date
today = date.today()
from time import perf_counter as timer
# register first tick
tick = timer()

register_sauron_colormap()

print('#############################################################')
print('#############################################################')
print('Hold onto your butts!')
print()

date_of_kin = '2023-02-28_2'

#------------------------------------------------------------------------------
# Directories and files

# data directory
data_dir = '/u/scratch/k/knabel/data/' #'/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
stellar_library_dir = data_dir #f'{data_dir}xshooter_lib/'
tables_dir = f'{data_dir}tables/'
mosaics_dir = f'{data_dir}mosaics/'
#kinematics_full_dir = f'{data_dir}kinematics/'
cd 
# point to the kinematics of the specific date
kinematics_dir = f'{data_dir}{data_of_kin}/'#f'{kinematics_full_dir}{date_of_kin}/'
# create a systematics directory
systematics_dir = f'{data_dir}systematics/'
Path(systematics_dir).mkdir(parents=True, exist_ok=True)
print(f'Outputs will be in {systematics_dir}')
print()

# tables
obj_names = pd.read_csv(f'{tables_dir}obj_names.csv')
#obj_names = obj_names.iloc[2:] # 030123 starting from J0330 because this is where it failed.
ppxf_inputs = pd.read_csv(f'{tables_dir}ppxf_input_table.csv')

#------------------------------------------------------------------------------
# Kinematics systematics choices

# target SN for voronoi binning
vorbin_SN_targets = np.array([10, 15, 20])

# include G band or no
g_bands = ['no_g','g_band']
# if G_band, 10 nm will be added to wave_max

# stellar templates libraries
libraries = ['all_dr2_fits_G789K012',
             'all_dr2_fits_G789K012_lo_var',
             'all_dr2_fits_G789K012_hi_var']

# aperture
apertures = ['R0','R1','R2']

# wavelength range
# will be determined from ppxf input table

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

# global template spectrum chi2 threshold
global_template_spectrum_chi2_threshold = 1

#------------------------------------------------------------------------------
# iterate over objects

for i, obj_name in enumerate(obj_names.to_numpy()[:,0]):
    
    obj_abbr = obj_name[4:9] # e.g. J0029
    print('##########################################################################################################################')
    print('##########################################################################################################################')
    print()
    print(obj_name, obj_abbr)
    print()
    
    #------------------------------------------------------------------------------
    # ppxf inputs
    
    ppxf_obj_in = ppxf_inputs[ppxf_inputs['obj_name']==obj_name]
    z = ppxf_obj_in['z'].to_numpy()[0] # redshift of lens galaxy
    z_bs = ppxf_obj_in['zbs'].to_numpy()[0] # redshift of background source, 0 if no contamination
    T_exp = ppxf_obj_in['T_exp'].to_numpy()[0] # exposure time in seconds
    lens_center_x = ppxf_obj_in['center_x'].to_numpy()[0] # center x coordinate
    lens_center_y = ppxf_obj_in['center_y'].to_numpy()[0] # center y coordinate
    wave_min = ppxf_obj_in['wave_min'].to_numpy()[0] # minimum wavelength for fitting central spectrum
    wave_max = ppxf_obj_in['wave_max'].to_numpy()[0] # max
    
    # wavelength ranges
    wave_mins = np.array([wave_min+5, wave_min, wave_min-5]) # narrower, initial, wider
    wave_maxs = np.array([wave_max-5, wave_max, wave_max+5]) 

    #------------------------------------------------------------------------------
    # object-specific directories
    
    mos_dir = f'{mosaics_dir}{obj_name}/' 
    kin_dir = f'{kinematics_dir}{obj_name}/'
    
    # create systematics directory for this object
    syst_obj_dir = f'{systematics_dir}{obj_name}/'
    Path(syst_obj_dir).mkdir(parents=True, exist_ok=True)
    
    #KCWI mosaic datacube
    mos_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'
    
    #------------------------------------------------------------------------------
    # Check for contamination from background source
    # if J0330 or J1112
    if obj_abbr in ['J0330','J1112']:
        print(f'{obj_abbr} must be fit with the background source. Be careful.') # the voronoi_binning_data is already background subtracted
        # spectrum template for background source
        background_source_spectrum = f'{mos_dir}{obj_abbr}_spectrum_background_source.fits'
        contam = True
    else:
        background_source_spectrum = None
        contam = False
    
    for vorbin_SN_target in vorbin_SN_targets:
        target_dir = f'{kin_dir}target_sn_{vorbin_SN_target}/'
        # create kin_dir if not exists
        #Path(f'{target_dir}').mkdir(parents=True, exist_ok=True)
        
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

        # make systematics directory for the vorbin sn target
        syst_dir = f'{syst_obj_dir}{vorbin_SN_target}/'
        Path(syst_dir).mkdir(parents=True, exist_ok=True) 

        ## import voronoi binning data
        voronoi_binning_data = fits.getdata(target_dir +'voronoi_binning_' + obj_name + '_data.fits')
        N = voronoi_binning_data.shape[0] # for saving the systematics measurements

        '''
        Step 8: systematics tests
        '''

        # iterate over G-band and not G-band
        for g_band in g_bands:
            
            if (obj_abbr=='J0330') & (g_band=='g_band'):
                print('J0330 does not get G band')
                continue

            print()
            print('Working with ', g_band)
            print()

            # first create G-band and no_G directories
            g_dir = f'{syst_dir}{g_band}/'
            Path(g_dir).mkdir(parents=True, exist_ok=True)

            # set arrays to save the systematics info
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
                    aperture = f'{mos_dir}{obj_abbr}_central_spectrum_{apertures[a]}.fits'
                    for d in range(len(degrees)):
                        #degree
                        degree=degrees[d]
                        for w in range(len(wave_mins)):
                            wave_min = wave_mins[w]
                            wave_max = wave_maxs[w]
                            
                            # if g, add 10 nm
                            if g_band == 'g_band':
                                wave_max = wave_max+20

                            VD_name = f'l{l}a{a}d{d}w{w}'

                            print('################################################')
                            print(f'Beginning model {VD_name}')
                            print('################################################')


                            # make save directory
                            model_dir = f'{g_dir}{VD_name}/'
                            Path(model_dir).mkdir(parents=True, exist_ok=True) 

                            templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, quasar = \
                                ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(
                                    library,
                                    degree=degree,
                                    spectrum_aperture=aperture,
                                    background_source_spectrum=background_source_spectrum,
                                    wave_min=wave_min,
                                    wave_max=wave_max,
                                    velscale_ratio=velscale_ratio,
                                    z=z,
                                    z_background_source=z_bs,
                                    noise=noise,
                                    templates_name='xshooter',
                                    FWHM=FWHM,
                                    FWHM_tem=FWHM_tem_xshooter,
                                    plot=True,
                                    quiet=True)

                            plt.subplots()
                            pp.plot()
                            plt.xlim(wave_min/1000, wave_max/1000)
                            plt.savefig(f'{model_dir}global_template_spectrum.png')
                            plt.savefig(f'{model_dir}global_template_spectrum.pdf')

                            # save model name and chi2
                            global_fit_chi2_log.iloc[n, :] = VD_name, pp.chi2
                            n = n+1 # advance to next model in pandas dataframe

                            # write pickle file
                            pickle_file = open(f'{model_dir}global_template_spectrum_fit.pkl', 'wb')
                            pickle.dump(pp, pickle_file)
                            pickle_file.close()

                            # if chi2 < 1 
                            # forget this chi2 threshold... it will be reflected in high uncertainty
                            # save the pp object as a pickle 
                            #if pp.chi2 < 1:
                            # if chi2 is bad, just flag it 
                            #else:
                            if pp.chi2 < 1:
                                print('chi2 < 1')


                            # ppxf on bins
                            nTemplates_xshooter = templates.shape[1]
                            global_temp_xshooter = templates @ pp.weights[
                                                               :nTemplates_xshooter]

                            # run ppxf on all bins
                            # the results will be N (# bins) rows of 5 columns
                            # columns: V, VD, dV, dVD, chi2
                            systematics_results = get_velocity_dispersion_deredshift(degree=degree,
                                                               spectrum_aperture=aperture,
                                                                background_source_spectrum=background_source_spectrum,
                                                               voronoi_binning_data=voronoi_binning_data,
                                                               velscale_ratio=velscale_ratio,
                                                               z=z,
                                                                z_background_source=z_bs,
                                                               noise=noise,
                                                               FWHM=FWHM,
                                                               FWHM_tem_xshooter=FWHM_tem_xshooter,
                                                               dir=model_dir,
                                                               obj_name=VD_name,
                                                               libary_dir=library,
                                                               global_temp=global_temp_xshooter,
                                                               wave_min=wave_min,
                                                               wave_max=wave_max,
                                                               T_exp=T_exp,
                                                               VD_name=VD_name,
                                                               plot=False,
                                                               quiet=True)

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
                            # close all plots
                            plt.close('all')


            # save the systematics
            np.savetxt(f'{g_dir}{obj_name}_{g_band}_systematics_VD.txt',
                       systematics_VD, delimiter=',')
            np.savetxt(f'{g_dir}{obj_name}_{g_band}_systematics_dVD.txt',
                       systematics_dVD, delimiter=',')
            np.savetxt(f'{g_dir}{obj_name}_{g_band}_systematics_V.txt', 
                       systematics_V, delimiter=',')
            np.savetxt(f'{g_dir}{obj_name}_{g_band}_systematics_dV.txt',
                       systematics_dV, delimiter=',')
            np.savetxt(f'{g_dir}{obj_name}_{g_band}_systematics_chi2.txt',
                       systematics_chi2, delimiter=',')

            # save the chi2 log
            global_fit_chi2_log.to_csv(f'{g_dir}{obj_name}_{g_band}_global_fit_chi2_log.csv')

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
                    plt.savefig(f'{g_dir}{obj_name}_{g_band}_{Nstar+1}{Nstar+Nbin}_systematics.png')
                    #plt.show()

            print('Done.')
            print('###############################################################')
            
print('##########################################################################################################################')
print('##########################################################################################################################')
print('##########################################################################################################################')
print('##########################################################################################################################')
print('Work complete.')
print(':)')