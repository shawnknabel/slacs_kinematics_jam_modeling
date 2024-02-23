# import general libraries and modules
import numpy as np
np.set_printoptions(threshold=10000)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 6)
#plt.switch_backend('agg')
#%matplotlib inline
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


########################################################

# bring in the space_jam and total_mass_mge modules

from space_jam import space_jam
from total_mass_mge import total_mass_mge

#########################################################

##################################################################################################################################

date_of_kin = '2024_02_15'

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

#############################################################################

# define the tasks

'''
First task is axisymmetric spherical
'''

def task_1(i):

    i=0
    obj_name = obj_names[i]
    kin_dir = f'{kinematics_dir}{obj_name}/'
    jam_dir = f'{jam_output_dir}{obj_name}/'
    jam_details_file = f'{kin_dir}{obj_name}_details_for_jampy_xshooter_{date_of_kin}.pkl'
    mass_model='power_law'
    anisotropy='const'
    geometry='axi'
    align='sph'
    zlens= zlenses[slacs_ix_table['Name']==obj_name].to_numpy()[0]
    zsource = zsources[slacs_ix_table['Name']==obj_name].to_numpy()[0]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    #cosmo = cosmo
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    systematics_est = 0.05 # %
    covariance_est = 0.02 # %
    p0 = [2.0, 0.4, 0.9, 1.0, 0.5, 7] # gamma0, q0, k_ani, einstein radius, k_mst, a_mst
    bounds = [[1.4, 0.051, 0.0, 0.7, 0.0, 5 ], 
              [2.8, 1.0,   1.0, 2.0, 1.0, 10]]
    sigpar = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # crude estimate of uncertainties
    prior_type = ['uniform','uniform','uniform','uniform','uniform','uniform']
    lensprior = False
    fix_pars = [0, 0, 0, 0, 1, 1]
    lambda_int= 1.0
    nstep = 1
    nwalkers = 12
    ndim = 6
    minimization = 'MCMC'
    sampler_args = [nstep, nwalkers, ndim] # 10 walkers
    date_time = datetime.now().strftime("%Y_%m_%d")#-%I_%M_%S_%p"
    run_id = 1.1

    welcome_to_the_jam = space_jam(jam_dir, jam_details_file,
                                     obj_name, mass_model, anisotropy, geometry, align, 
                                    zlens=zlens, zsource=zsource, cosmo=cosmo, sigmapsf=sigmapsf, fast_slow=fast_slow,
                                   systematics_est=systematics_est, covariance_est=covariance_est,
                                   p0=p0, bounds=bounds, sigpar=sigpar, prior_type=prior_type, lensprior=lensprior, fix_pars=fix_pars, lambda_int=lambda_int,
                                   minimization=minimization, sampler_args=sampler_args, date_time=date_time, run_id=run_id, plot=True, overwrite=True, 
                                   test_prior=False, constant_err=False, kinmap_test=None)

    welcome_to_the_jam.run_mcmc()
    welcome_to_the_jam.save_space_jam()
    welcome_to_the_jam.summary_plot(save=True)
    
'''
Second task is axisymmetric cylindrical
'''

def task_2(i):

    i=0
    obj_name = obj_names[i]
    kin_dir = f'{kinematics_dir}{obj_name}/'
    jam_dir = f'{jam_output_dir}{obj_name}/'
    jam_details_file = f'{kin_dir}{obj_name}_details_for_jampy_xshooter_{date_of_kin}.pkl'
    mass_model='power_law'
    anisotropy='const'
    geometry='axi'
    align='sph'
    zlens= zlenses[slacs_ix_table['Name']==obj_name].to_numpy()[0]
    zsource = zsources[slacs_ix_table['Name']==obj_name].to_numpy()[0]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    #cosmo = cosmo
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    systematics_est = 0.05 # %
    covariance_est = 0.02 # %
    p0 = [2.0, 0.4, 0.9, 1.0, 0.5, 7] # gamma0, q0, k_ani, einstein radius, k_mst, a_mst
    bounds = [[1.4, 0.051, 0.0, 0.7, 0.0, 5 ], 
              [2.8, 1.0,   1.0, 2.0, 1.0, 10]]
    sigpar = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # crude estimate of uncertainties
    prior_type = ['uniform','uniform','uniform','uniform','uniform','uniform']
    lensprior = False
    fix_pars = [0, 0, 0, 0, 1, 1]
    lambda_int= 1.0
    nstep = 1
    nwalkers = 12
    ndim = 6
    minimization = 'MCMC'
    sampler_args = [nstep, nwalkers, ndim] # 10 walkers
    date_time = datetime.now().strftime("%Y_%m_%d")#-%I_%M_%S_%p"
    run_id = 1.2

    welcome_to_the_jam = space_jam(jam_dir, jam_details_file,
                                     obj_name, mass_model, anisotropy, geometry, align, 
                                    zlens=zlens, zsource=zsource, cosmo=cosmo, sigmapsf=sigmapsf, fast_slow=fast_slow,
                                   systematics_est=systematics_est, covariance_est=covariance_est,
                                   p0=p0, bounds=bounds, sigpar=sigpar, prior_type=prior_type, lensprior=lensprior, fix_pars=fix_pars, lambda_int=lambda_int,
                                   minimization=minimization, sampler_args=sampler_args, date_time=date_time, run_id=run_id, plot=True, overwrite=True, 
                                   test_prior=False, constant_err=False, kinmap_test=None)

    welcome_to_the_jam.run_mcmc()
    welcome_to_the_jam.save_space_jam()
    welcome_to_the_jam.summary_plot(save=True)
    
# protect the entry point
from concurrent.futures import ProcessPoolExecutor

if __name__ == '__main__':
    # report a message
    print('Starting task 1...')
    # create the process pool
    with ProcessPoolExecutor(14) as exe:
        # perform calculations
        results = exe.map(task_1, range(14))
    # report a message
    print('Done.')
    
if __name__ == '__main__':
    # report a message
    print('Starting task 2...')
    # create the process pool
    with ProcessPoolExecutor(14) as exe:
        # perform calculations
        results = exe.map(task_2, range(14))
    # report a message
    print('Done.')