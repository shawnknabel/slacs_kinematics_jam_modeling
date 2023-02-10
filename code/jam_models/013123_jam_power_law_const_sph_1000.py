'''
1/31/23 - This script takes inputs from MGE and calculates a first estimate of the JAM dynamical model with the following specifics:
'''

################################################################
# set mass model, anisotropy, and alignment
model = 'power_law'
anisotropy = 'const'
align = 'sph'
# set nstep
nstep=10000
################################################################

################################################################

# import general libraries and modules
import numpy as np
np.set_printoptions(threshold=10000)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 6)
import pandas as pd
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings( "ignore", module = "plotbin\..*" )
import os
from os import path
import pickle
from datetime import datetime

# astronomy/scipy
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import rotate
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares as lsq
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.cosmology import Planck15 as cosmo # I took 15 because for some reason Planck18 isn't in this astropy install #Planck18 as cosmo  # Planck 2018
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# mge fit
import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from mgefit.mge_fit_sectors_regularized import mge_fit_sectors_regularized

# jam
from jampy.jam_axi_proj import jam_axi_proj
from jampy.jam_axi_proj import rotate_points
from jampy.jam_axi_proj import bilinear_interpolate
from jampy.mge_half_light_isophote import mge_half_light_isophote
from plotbin.plot_velfield import plot_velfield
from plotbin.sauron_colormap import register_sauron_colormap
register_sauron_colormap()
from pafit.fit_kinematic_pa import fit_kinematic_pa
from jampy.jam_axi_proj import jam_axi_proj
from jampy.mge_radial_mass import mge_radial_mass
from plotbin.symmetrize_velfield import symmetrize_velfield

# adamet
from adamet.adamet import adamet
from adamet.corner_plot import corner_plot

# my functions
import sys
sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages")
from slacs_mge_jampy import crop_center_image
from slacs_mge_jampy import import_center_crop
#from slacs_mge_jampy import try_fractions_for_find_galaxy
from slacs_mge_jampy import convert_mge_model_outputs
#from slacs_mge_jampy import plot_contours_321
#from slacs_mge_jampy import plot_contours_531
#from slacs_mge_jampy import load_2d_kinematics
#from slacs_mge_jampy import kinematics_map_systematics
#from slacs_mge_jampy import load_2d_kinematics_with_datacube_contours
from slacs_mge_jampy import get_bin_centers
from slacs_mge_jampy import bin_velocity_maps
#from slacs_mge_jampy import plot_kinematics_mge_contours
#from slacs_mge_jampy import show_pa_difference
from slacs_mge_jampy import rotate_bins
#from slacs_mge_jampy import correct_bin_rotation
#from slacs_mge_jampy import find_half_light
#from slacs_mge_jampy import calculate_minlevel
#from slacs_mge_jampy import fit_kcwi_sigma_psf
#from slacs_mge_jampy import optimize_sigma_psf_fit
#from slacs_mge_jampy import estimate_hst_psf
#from slacs_ani_mass_jam import kinematics_map_systematics
from slacs_ani_mass_jam import osipkov_merritt_model
from slacs_ani_mass_jam import osipkov_merritt_generalized_model
from slacs_ani_mass_jam import inner_outer_anisotropy_model
from slacs_ani_mass_jam import nfw_generalized_model
from slacs_ani_mass_jam import dark_halo_mge
from slacs_ani_mass_jam import total_mass_mge
from slacs_ani_mass_jam import jam_lnprob
from slacs_ani_mass_jam import jam_lnprob_power_law
from slacs_ani_mass_jam import jam_lnprob_nfw_constbeta
from slacs_ani_mass_jam import jam_lnprob_nfwgen_constbeta
from slacs_ani_mass_jam import jam_lnprob_nfw_om
from slacs_ani_mass_jam import jam_lnprob_nfwgen_om
from slacs_ani_mass_jam import summary_plot
from slacs_ani_mass_jam import save_fit_parameters
from slacs_ani_mass_jam import get_power_law_slope
from slacs_ani_mass_jam import jampy_details
from slacs_ani_mass_jam import prepare_to_jam
from slacs_ani_mass_jam import space_jam
from slacs_mge_jampy import make_gaussian

################################################################
# some needed information
kcwi_scale = 0.1457  # arcsec/pixel
hst_scale = 0.050 # ACS/WFC

# specify object directory and names
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/' # data directory

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

print('Thank you for choosing MiseryBot 9.3.2, where you will hate everything and probably quit after a short while.')
print('...')
print('...')
print('...')
print('Please hold...')
print('...')

print('')
print('Ah yes, the JAM routine. This one is fun.')
print('...')
print('...')
print('...')
print('But where to start?')
print('...')

print('########################################################')
print('########################################################')

for i, obj_name in enumerate(obj_names):
    
    print(f'Jamming object {obj_name}.')
    print('Brace for jamming.')
    print('########################################################') 
    
    if i == 6:
        print('Object 6 is unsatisfactory.')
        print('On to the next?')
        print('########################################################') 
        print('########################################################') 
        continue
    else:
        space_jam(data_dir, obj_name, model, anisotropy, align, nstep, cosmo)
        
    print('########################################################') 
    print('Jamming complete. On to the next?')
    print('########################################################') 
    print('########################################################') 
    
print('Okay, all done. Thank you for using MiseryBot 9.1.3. We deprecated while you were away. Hope you data is not corrupted.')