#!/usr/bin/env python
##############################################################################
'''
11/19/23
    Creating a script for going through all the lenses and fitting the global galaxy spectrum with different stellar libraries
    Using ppxf_kinematics_getGlobal_lens_deredshift_library_test from home/shawnknabel/Documents/slacs_kinematics/my_python_packages/ppxf_kcwi_util_022423/kcwi_util.py
    Copied from home/shawnknabel/Documents/slacs_kinematics/code/codeforanalyzingkcwislacslenses_/ppxf_kinematics_SLACS_lenses/SDSSJ0029-0055/SDSSJ0029-0055_ppxf_kinematics_022423.py
------
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
'''

# import libraries
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ppxf
import ppxf.ppxf_util as ppxf_util
import os
# import my packages
import sys
sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages")
from ppxf_kcwi_util_022423.kcwi_util import ppxf_kinematics_getGlobal_lens_deredshift_library_test
from ppxf_kcwi_util_022423.kcwi_util import register_sauron_colormap
from ppxf_kcwi_util_022423.kcwi_util import visualization
from ppxf_kcwi_util_022423.kcwi_util import get_datacube
#from ppxf_kcwi_util_022423.kcwi_util import ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift
from ppxf_kcwi_util_022423.kcwi_util import find_nearest
from ppxf_kcwi_util_022423.kcwi_util import SN_CaHK
from ppxf_kcwi_util_022423.kcwi_util import select_region
from ppxf_kcwi_util_022423.kcwi_util import voronoi_binning
from ppxf_kcwi_util_022423.kcwi_util import get_voronoi_binning_data
from ppxf_kcwi_util_022423.kcwi_util import get_velocity_dispersion_deredshift
from ppxf_kcwi_util_022423.kcwi_util import kinematics_map
from ppxf_kcwi_util_022423.kcwi_util import stellar_type

import pathlib # to create directory
import pickle

register_sauron_colormap()

#------------------------------------------------------------------------------
# Directories and files

# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
#hst_dir = '/data/raw_data/HST_SLACS_ACS/kcwi_kinematics_lenses/'
tables_dir = f'{data_dir}tables/'
mosaics_dir = f'{data_dir}mosaics/'
kinematics_full_dir = f'{data_dir}kinematics/'
kinematics_dir = f'{data_dir}stellar_library_tests/'
if not os.path.exists(kinematics_dir):
    os.mkdir(kinematics_dir)

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

#################################################
# tables
paper_table = pd.read_csv(f'{tables_dir}paper_table_100223.csv')
slacs_ix_table = pd.read_csv(f'{tables_dir}slacs_ix_table3.csv')
zs = paper_table['zlens']

#################################################
# libraries
templates_names = ['xshooter',
                   'fsps',
                  'galaxev',
                  'emiles'
                  ]
#################################################

#------------------------------------------------------------------------------
# Kinematics systematics initial choices

# stellar templates library
# stellar_templates_library = 'all_dr2_fits_G789K012'

# aperture
aperture = 'R2' # largest aperture

# wavelength range
wave_min = 360 # Only the MILES range 3525 – 7500 has high resolution
wave_max = 410 # I am doing this to be uniform across the objects

# degree of the additive Legendre polynomial in ppxf
degree = 2 # 50/25 110/25 = 4.4 round up

#------------------------------------------------------------------------------
# Information specific to KCWI and templates

kcwi_scale = 0.1457

## R=3600. spectral resolution is ~ 1.42A
FWHM = 1.42 #1.42

## initial estimate of the noise
noise = 0.014

# velocity scale ratio
velscale_ratio = 2

#------------------------------------------------------------------------------
# variable settings in ppxf and utility functions

# global template spectrum chi2 threshold
global_template_spectrum_chi2_threshold = 100

#------------------------------------------------------------------------------

#################################################
# Loop through objects:

for i, obj_name in enumerate(obj_names):

    print()
    print('#################################################')
    print('#################################################')
    print()
    print(obj_name)
    
    obj_abbr = obj_name[4:9] # e.g. J0029
    z = zs[i] # lens redshift
    print('z',z)
    
    if obj_abbr == 'J0330':
        print('Skipping J0330.')
        continue
        
    if obj_abbr == 'J1306':
        print('Skipping J1306')
        continue

    # other necessary directories ... Be very careful! This is how we will make sure we are using the correct files moving forward.
    mos_dir = f'{mosaics_dir}{obj_name}/' # files should be loaded from here but not saved
    kin_dir = f'{kinematics_dir}{obj_name}/'
    if not os.path.exists(kin_dir):
        os.mkdir(kin_dir)

    #KCWI mosaic datacube
    mos_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'

    #spectrum from the lens center # using R=1
    spectrum_aperture = f'{mos_dir}{obj_abbr}_central_spectrum_{aperture}.fits' 
    
    ################################################################################
    # 11/19/23 - Template tests, loop through the template sets
    
    for templates_name in templates_names:
        
        print()
        print('#################################################')
        print()
        print(templates_name)
        
        if templates_name=='xshooter':
            # xshooter libary directory # chih-fan spelled wrong :)
            library_dir = f'{data_dir}xshooter_lib/all_dr2_fits/'
            FWHM_tem = 0.43 #0.43
        else:
            library_dir = None
            FWHM_tem = None

        try:
            # fit center spectrum with templates
            templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy = \
                ppxf_kinematics_getGlobal_lens_deredshift_library_test(library_dir=library_dir,
                                                                          degree=degree,
                                                                          spectrum_aperture=spectrum_aperture,
                                                                          wave_min=wave_min,
                                                                          wave_max=wave_max,
                                                                          velscale_ratio=velscale_ratio,
                                                                          z=z,
                                                                          noise=noise,
                                                                          templates_name=templates_name,
                                                                          FWHM=FWHM,
                                                                          FWHM_tem=FWHM_tem,
                                                                          plot=False)

            # plot the spectrum and template fits
            #pp.plot()
            #plt.savefig(kin_dir + obj_name + templates_name + '_global_template_spectrum_full.png')
            #plt.savefig(kin_dir + obj_name + templates_name + '_global_template_spectrum_full.pdf')
            #plt.pause(1)
            #plt.clf()
            pp.plot()
            plt.xlim(wave_min/1000, wave_max/1000) # it's in microns
            plt.title(f'V {np.around(pp.sol[0], 2)} km/s, VD {np.around(pp.sol[1],2)} km/s')
            plt.savefig(kin_dir + obj_name + templates_name + '_global_template_spectrum.png')
            plt.savefig(kin_dir + obj_name + templates_name + '_global_template_spectrum.pdf')
            plt.pause(1)

            # flag and reject if pp.chi2 > threshold
            assert pp.chi2 < global_template_spectrum_chi2_threshold, f'Global template spectrum chi2 {pp.chi2} exceeds threshold of {global_template_spectrum_chi2_threshold}. Try again please.'

            # save the pp object as a pickle
            # write file
            pickle_file = open(f'{kin_dir}{obj_name}_{templates_name}_global_template_spectrum_fit.pkl', 'wb')
            pickle.dump(pp, pickle_file)
            pickle_file.close()

            # save templates object as a pickle
            pickle_file = open(f'{kin_dir}{obj_name}_{templates_name}_global_template_spectrum_fit_templates.pkl', 'wb')
            pickle.dump(templates, pickle_file)
            pickle_file.close()

            # save variables for future use, show number of stars
            #nTemplates = templates.shape[1]
            #global_temp_xshooter = templates @ pp.weights[:nTemplates]
            #pp_weights_2700 = pp.weights[:nTemplates]
            #print('number of stars that have non-zero contribution =', np.sum((~(
              #      pp_weights_2700 == 0))*1))
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error) # An exception occurred: division by zero

print()
print('############################')
print('############################')
print('Done')
print('############################')
print('############################')