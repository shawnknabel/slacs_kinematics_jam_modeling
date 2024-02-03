#!/usr/bin/env python
# coding: utf-8

# # 01/31/24 5:47 pm. - I have gotten the mass to make reasonable things. J0037 is running, but I want to run some code overnight and see if it actually is correct... or at least to have it set up. This notebook tests the modules "space_jam" and "total_mass_mge" in e.g. home/shawnknabel/Documents/slacs_kinematics/my_python_packages/space_jam.py
# # 02/01/24 10:30 am - Making this a script and actually running it.

# In[2]:


# import general libraries and modules
import numpy as np
np.set_printoptions(threshold=10000)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 6)
plt.switch_backend('agg')
import pandas as pd
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings( "ignore", module = "plotbin\..*" )
import os
from os import path
from pathlib import Path
#import pickle
import dill as pickle
from datetime import datetime
def tick():
    return datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
import glob

# astronomy/scipy
from astropy.io import fits
#from astropy.wcs import WCS
#from scipy.ndimage import rotate
#from scipy.ndimage import map_coordinates
#from scipy.optimize import least_squares as lsq
#from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
#from astropy.cosmology import Planck15 as cosmo # I took 15 because for some reason Planck18 isn't in this astropy install #Planck18 as cosmo  # Planck 2018
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
#from scipy.interpolate import interp1d
#from scipy.optimize import fsolve
import astropy.units as u
import astropy.constants as constants

# my functions
import sys
sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages")


################################################################
# some needed information
kcwi_scale = 0.1457  # arcsec/pixel
hst_scale = 0.050 # ACS/WFC

# value of c^2 / 4 pi G
c2_4piG = (constants.c **2 / constants.G / 4 / np.pi).to('solMass/pc')


# In[3]:


# bring in the space_jam and total_mass_mge modules

from space_jam import space_jam
from total_mass_mge import total_mass_mge


# In[4]:


##################################################################################################################################

date_of_kin = '2023-02-28_2'

#------------------------------------------------------------------------------
# Directories and files

# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
hst_dir = '/data/raw_data/HST_SLACS_ACS/kcwi_kinematics_lenses/'
tables_dir = f'{data_dir}tables/'
mosaics_dir = f'{data_dir}mosaics/'
kinematics_full_dir = f'{data_dir}kinematics/'
kinematics_dir =f'{kinematics_full_dir}{date_of_kin}/'
jam_output_dir = f'{data_dir}jam_outputs/'
# create a directory for JAM outputs
#Path(jam_output_dir).mkdir(parents=True, exist_ok=True)
#print(f'Outputs will be in {jam_output_dir}')
print()

# target SN for voronoi binning
vorbin_SN_targets = np.array([10, 15, 20])

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

paper_table = pd.read_csv(f'{tables_dir}paper_table_100223.csv')
slacs_ix_table = pd.read_csv(f'{tables_dir}slacs_ix_table3.csv')
zs = paper_table['zlens']
zlenses = slacs_ix_table['z_lens']
zsources = slacs_ix_table['z_src']
# get the revised KCWI sigmapsf
sigmapsf_table = pd.read_csv(f'{tables_dir}kcwi_sigmapsf_estimates.csv')

# # Start with cylindrical

# In[9]:


for i in range(2):#len(obj_names)):
    obj_name = obj_names[i]
    SN = 15
    mass_model='power_law'
    anisotropy='const'
    geometry='axi'
    align='cyl'
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    cosmo = cosmo
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    p0 = [2.0, 0.4, 0.9, 1.0, 0.5, 7] # gamma0, q0, ratio0, einstein radius, lambda_int
    bounds = [[1.4, 0.051, 0.0, 0.7, 0.0, 5 ], 
              [2.8, 1.0,   1.0, 2.0, 1.0, 10]]
    sigpar = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # crude estimate of uncertainties
    prior_type = ['uniform','uniform','uniform','uniform','uniform','uniform']
    lensprior = False
    nstep = 1
    nwalkers = 12
    ndim = 6
    sampler_args = [nstep, nwalkers, ndim] # 10 walkers
    date_time = datetime.now().strftime("%Y_%m_%d")#-%I_%M_%S_%p"
    run_id = 2.1

    welcome_to_the_jam = space_jam(kinematics_dir, 
                                   jam_output_dir,
                                   obj_name, SN, 
                                   mass_model, 
                                   anisotropy, 
                                   geometry, 
                                   align, 
                                   zlens, 
                                   zsource, 
                                   sigmapsf,
                                   cosmo, 
                                   fast_slow,
                                   p0, 
                                   bounds,
                                   sigpar, 
                                   prior_type,
                                   lensprior, 
                                   sampler_args,
                                   date_time,
                                   run_id,
                                   plot=False, 
                                   overwrite=True, 
                                   test_prior=False,
                                   constant_err=False, 
                                   kinmap_test=None)
    
    # run it
    welcome_to_the_jam.run_mcmc()

    # plot it
    welcome_to_the_jam.summary_plot(save=True)


# # New run, with spherical alignment

# In[17]:


for i in range(2):#len(obj_names)):
    obj_name = obj_names[i]
    SN = 15
    mass_model='power_law'
    anisotropy='const'
    geometry='axi'
    align='sph'
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    cosmo = cosmo
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    p0 = [2.0, 0.4, 0.9, 1.0, 0.5, 7] # gamma0, q0, ratio0, einstein radius, lambda_int
    bounds = [[1.4, 0.051, 0.0, 0.7, 0.0, 5 ], 
              [2.8, 1.0,   1.0, 2.0, 1.0, 10]]
    sigpar = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # crude estimate of uncertainties
    prior_type = ['uniform','uniform','uniform','uniform','uniform','uniform']
    lensprior = False
    nstep = 500
    nwalkers = 12
    ndim = 6
    sampler_args = [nstep, nwalkers, ndim] # 10 walkers
    date_time = datetime.now().strftime("%Y_%m_%d")#-%I_%M_%S_%p"
    run_id = 2.2

    welcome_to_the_jam = space_jam(kinematics_dir, 
                                   jam_output_dir,
                                   obj_name, SN, 
                                   mass_model, 
                                   anisotropy, 
                                   geometry, 
                                   align, 
                                   zlens, 
                                   zsource, 
                                   sigmapsf,
                                   cosmo, 
                                   fast_slow,
                                   p0, 
                                   bounds,
                                   sigpar, 
                                   prior_type,
                                   lensprior, 
                                   sampler_args,
                                   date_time,
                                   run_id,
                                   plot=False, 
                                   overwrite=True, 
                                   test_prior=False,
                                   constant_err=False, 
                                   kinmap_test=None)
    # run it
    welcome_to_the_jam.run_mcmc()

    # plot it
    welcome_to_the_jam.summary_plot(save=True)


# In[ ]:




