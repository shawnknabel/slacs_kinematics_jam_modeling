#!/usr/bin/env python
# coding: utf-8

# # 11/14/23 - Improving upon models from yesterday with new lambda_int parameter k_mst
# ## Changing name of lambda_int to k_mst
# ## Changing the inner and outer slope of the MGE to 2 and 3 (defaults)
# ## Set it to reject any model that somehow still gets negative mass (by setting lambda_int=0 and then rejecting in jam_lnprob if that is the case)
# ## lambda_int will be between [0.8, min(1.2, maxcalculated)] i.e. the minimum of either 1.2 or the max that the model can handle
# ____
# 
# # 11/13/23 - Looking at ways to parametrize the mass sheet transform that won't give negative mass density values and aren't biased in the scale radius.
# ## Likely trying to introduce r_scale as a free parameter a_scale = r_scale/r_eff = [5,10]
# ## Also parametrizing lambda_int = lower_bound + (upper - lower) * k_mst like I do with the anisotropy.
# ## notebook copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/110723_jam.ipynb
# _____
# #### 11/07/23 - JAM models with Chin Yi's updated lens models accounting for systematic uncertainties
# ##### copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/110323_jam.ipynb
# ###### lens_models_table_chinyi_with_sys.csv
# ______
# #### 11/03/23 - JAM models after correcting the MGEs and bin alignments in home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/kinematics_maps/plot_kinematics_contours/110123_redoing_MGE_plotting_kin_contours.ipynb
# ##### copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/103123_jam_without_lenspriors.ipynb
# ##### Running with and without lens priors
# ##### Changed lambda_int bounds to [0.75,1.25]
# ______
# #### 10/31/23 - Testing J0029 and J0037 without lens priors to compare with 10/20/23 models
# ##### copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/103123_jam_without_lenspriors.ipynb
# ##### Changes:
# ##### I'm adding the priors to the plots to show if they deviate from them	
# ##### For now, this will only affect the shape prior
# ##### I'm also adding a line that prints what the priors are after they are updated	
# ##### 2:30 - had to add a line "plt.switch_backend('agg')" for some dumb reason?
# ______
# #### 10/11/23 - Updating, debugging.
# #### 10/06/23 - Copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/100423_jam_milestone_anisotropy_studies.ipynb
# ##### This will be 8 objects with 3 models each (no OM)
# #### Major changes:
# ##### 1. I use shape and anisotropy parameterizations from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/100323_jam_shape_anisotropy_ideas.ipynb
# ##### 2. This uses q_intr as a parameter instead of q_min
# ##### 3. I also change the mass profile ellipticity from the average to the one designated by mge_half_light_radius (all averaged quantities are changed in this way.)
# ##### 4. It also uses k_ani for the anisotropy parameter instead of the ratio itself
# ##### Also note:
# ##### I am still using jam_axi_proj in the spherical limit because that wasn't working
# ##### OM is also not working.
# _____

# #### 10/03/23 - Copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/100223_jam_milestone_anisotropy_studies.ipynb
# #### 10/04/23 - Fixed lambda_int bounds and set correct object names for input, setting lower bound for const anisotropy from q_min lower bound
# ##### This will be 8 objects with 3 models each (no OM)
# ##### Major changes:
# ##### 1. I bound lambda_int to 1.25 (will investigate that more later)
# ##### 2. I incorporate the new lens models from Chin-Yi
# ##### 3. I enforce that slow rotators must have q_min_intr > 0.6, which means slow-rotator MGEs with q_min_obs < 0.6 (i.e. J1531) are not good. I will have to redo those more carefully.
# ##### 4. I am also skipping J1250 because it should actally be aligned along the kinematic axis with "prolate" q (like Anowar did), which I will do later
# ##### Also note:
# ##### I am still using jam_axi_proj in the spherical limit because that wasn't working
# ##### OM is also not working.
# _______________________________

# #### 10/02/23 - Copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/092823_jam_milestone_anisotropy_studies_scratchpad.ipynb
# ##### This is to make the clean script.

# #### 09/28/23 - Copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/091323_jam_milestone_anisotropy_studies.ipynb
# ##### Following meeting with Michele and Tommaso, I am redoing the models with small changes: (JAM has been updated)
# ###### 1. STILL NOT IMPLEMENTED -q_intr_min (being the intrinsic flattening of the flattest gaussian) is > 0.6
# ###### 2. Longer chain - 200 iterations?
# ###### 3. STILL NOT IMPLEMENTED - Plot the convergence of the chain.
# ###### 4. STILL NOT IMPLEMENTED -Save a value that states if the chain converged.
# ###### 5. NO OM CYL.
# ###### 6. Forget the shape distribution from population for J1204 (for consistency's sake)
# ###### 7. Use "logistic" keyword for OM.
# ###### 8. Show effective radius in plots
# ________________
# #### 09/21/23 - Edited
# ##### I messed up the bounds for the constant anisotropy again.
# ##### It has been fixed, and the prior information has been put into the "get_priors_and_labels" function, which was renamed from "get_priors"
# #### 09/13/23  -Copied from 091123_jam_milestone_anisotropy_studies.ipynb
# ##### I had the anisotropy radius for OM with bounds from the constant anisotropy.

# #### 09/11/23 - Copied from 090623_jam_milestone_anisotropy_studies.ipynb
# ##### I had the Vrms error incorrectly saved. This is a really insane thing.
# 
# #### 09/06/23 - Milestone studies for anisotropy constraints. For as many as I can make good models...  4 models each
# ##### Spherical and Axisymmetric
# ##### PL mass profile
# ##### OM and constant anisotropy
# #### 08/10/23 - Changing the break radius and mass sheet radius to check its effect.
# ##### Make sure const ani parameter [0,1] for cylindrical.
# ##### Save chi2 of bestfit model and look at distribution (model comparison, BIC, something with evidence comparison)

# #### 05/19/23 - Copied from 012523_jam_bayes.ipynb. Testing the new implementation of anisotropy profiles from Michele. See https://mail.google.com/mail/u/0/?tab=rm&ogbl#search/osipkov/FMfcgzGrcFfBKDBKxwlvtkWlVRXdvklX
# #### 05/22/23 - Corrected the "details_for_jampy.pkl" - ONLY FOR J0037 SN 15 so far
# - surface brightness is now correctly computed from B-mag (I think)
# - bin x and y coordinates rotated along the photometric axis are now correct (phot maj axis is x axis)
# 
# #### 05/25/23 - Starting a new notebook just to keep things cleaner.
# #### 05/30/23 - Trying to improve things... See
# https://mail.google.com/mail/u/0/?tab=rm&ogbl#sent/KtbxLwgddjZLmPGGgmbJhZRvDWrPbJPXpL
# #### Today I'm going to try to implement it with emcee instead of adamet
# #### 06/02/23 - Running AdaMet and Emcee with corrected photometry.
# #### 06/07/23 - Copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/060623_jam_bayes_emcee_pl_const_5obj.ipynb
# #### 06/09/23 - Copied from home/shawnknabel/Documents/slacs_kinematics/jupyter_notebooks/jampy/060723_jam_whole_story_J1204.ipynb
# #### 06/13/23 - Actually doing the Einstein radius mass scaling
# #### This notebook goes through tests on just J1204 to prepare a "whole story" for the Otranto conference.
# #### Power law and composite, spherical and cylindrical, const and OM, SN 15
# #### Introducing new methods for incorporating the MSD and mass scaling.
# #### Mass scaling will be from the Einstein radius now.
# #### I will now work with convergences and multiply by the critical density to get the surface brightness.
# #### That is where the proper units come in.
# 

# In[1]:


################################################################

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
import pickle
from datetime import datetime
def tick():
    return datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
import glob


# astronomy/scipy
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import rotate
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares as lsq
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
#from astropy.cosmology import Planck15 as cosmo # I took 15 because for some reason Planck18 isn't in this astropy install #Planck18 as cosmo  # Planck 2018
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import astropy.units as u
import astropy.constants as constants

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
from jampy.jam_sph_proj import jam_sph_proj
from jampy.mge_half_light_isophote import mge_half_light_isophote
from plotbin.plot_velfield import plot_velfield
#from plotbin.sauron_colormap import register_sauron_colormap
#register_sauron_colormap()
from pafit.fit_kinematic_pa import fit_kinematic_pa
#from jampy.jam_axi_proj import jam_axi_proj
from jampy.mge_radial_mass import mge_radial_mass
from plotbin.symmetrize_velfield import symmetrize_velfield

# adamet
#from adamet.adamet import adamet
from adamet.corner_plot import corner_plot
# emcee
import emcee
import corner
from IPython.display import display, Math

# my functions
import sys
sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages")


################################################################
# some needed information
kcwi_scale = 0.1457  # arcsec/pixel
hst_scale = 0.050 # ACS/WFC

# value of c^2 / 4 pi G
c2_4piG = (constants.c **2 / constants.G / 4 / np.pi).to('solMass/pc')


# In[2]:


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


# # Lensing details from Chin-Yi's new models (given to me 10/03/23), instead of Anowar Shajib 2020
# ## The uncertainty of the gamma values maybe underestimated so it you want to be conservative I would add 0.1 uncertainty in quadrature (i.e. uncertainty_gamma_corrected = sqrt(gamma_err^2 + 0.1^2) )

# In[3]:


# power law info

lens_models_chinyi = pd.read_csv(f'{tables_dir}lens_models_table_chinyi.csv')
lens_models_chinyi_sys = pd.read_csv(f'{tables_dir}lens_models_table_chinyi_with_sys.csv')
lens_models_anowar = pd.read_csv(f'{tables_dir}lens_models_table_anowar.csv')



# In[6]:


lens_models_chinyi_sys.rename(columns={'dgamma':'gamma_err',
                                       'dgamma_sys':'gamma_sys',
                                      },inplace=True)


# In[7]:


lens_models_chinyi_sys['dgamma'] = np.sqrt( lens_models_chinyi_sys['gamma_err']**2 + lens_models_chinyi_sys['gamma_sys']**2 )
lens_models_chinyi_sys.loc[9, 'dgamma'] = np.sqrt( lens_models_chinyi_sys.loc[9, 'gamma_err']**2 + np.nanmean(lens_models_chinyi_sys['gamma_sys'])**2)
lens_models_chinyi_sys


# In[8]:


lens_models=lens_models_chinyi_sys


# # What I will do for now is ignore the slow rotators with q_min < 0.6 (i.e. SDSSJ1531-0105 so far)

# # Functions

# # Prep Functions

# In[9]:


###############################################################################

# class to collect and save all the attributes I need for jampy
class jampy_details:
    
    def __init__(details, surf_density, mge_sigma, q, kcwi_sigmapsf, Vrms_bin, dVrms_bin, V_bin, dV_bin, xbin_phot, ybin_phot, reff):
        details.surf_density=surf_density 
        details.mge_sigma=mge_sigma
        details.q=q 
        details.kcwi_sigmapst=kcwi_sigmapsf 
        details.Vrms_bin=Vrms_bin 
        details.dVrms_bind=dVrms_bin
        details.V_bin=V_bin 
        details.dV_bin=dV_bin 
        details.xbin_phot=xbin_phot 
        details.ybin_phot=ybin_phot
        details.reff=reff
        
        
###############################################################################


def prepare_to_jam(obj_name, file_dir, SN):

    # take the surface density, etc from mge saved parameters
    with open(f'{file_dir}{obj_name}_{SN}_details_for_jampy.pkl', 'rb') as f:
        tommy_pickles = pickle.load(f)
        
    surf = tommy_pickles.surf_density
    sigma = tommy_pickles.mge_sigma
    qObs = tommy_pickles.q
    #kcwi_sigmapsf = tommy_pickles.kcwi_sigmapst # mistake in name  
    try:
        kcwi_sigmapsf = tommy_pickles.kcwi_sigmapsf
    except:
        kcwi_sigmapsf = tommy_pickles.kcwi_sigmapst
    Vrms_bin = tommy_pickles.Vrms_bin
    try:
        dVrms_bin = tommy_pickles.dVrms_bin 
    except:
        dVrms_bin = tommy_pickles.dVrms_bind # mistake in name
    V_bin = tommy_pickles.V_bin
    dV_bin = tommy_pickles.dV_bin
    xbin_phot = tommy_pickles.xbin_phot
    ybin_phot = tommy_pickles.ybin_phot
    reff = tommy_pickles.reff
    
    return (surf, sigma, qObs, kcwi_sigmapsf, Vrms_bin, dVrms_bin, V_bin, dV_bin, xbin_phot, ybin_phot, reff)


###############################################################################

# function to get the priors if none are given

def get_priors_and_labels (model, anisotropy, surf, sigma, qobs, qobs_eff, geometry, align, fast_slow, p0=None, bounds=None, sigpar=None, prior_type=None):
    
    ###### axisymmetric geometry
    if geometry=='axi':
        try:
            if any(prior_type==None):
                prior_type=['uniform','uniform','uniform','uniform','uniform','uniform']  
        except:
            if prior_type==None:
                prior_type=['uniform','uniform','uniform','uniform','uniform','uniform']  
        
        # mass model labels
        if model=='power_law':
            label0 = r'$\gamma$'
        elif model=='nfw':
            label0 = r"$f_{\rm DM}$"
        
        # intrinsic axis ratio
        # The calculation of the inclination from q_intr involves a few steps
        bound_q_intr_lo, bound_q_intr_hi = get_bounds_on_q_intr_eff (surf, sigma, qobs, qobs_eff)
        bounds[0][1] = bound_q_intr_lo
        bounds[1][1] = bound_q_intr_hi  
        
        label1 =  r"$q_{\rm intr}$"
        # priors based on fast/slow
        if fast_slow == 'fast':
            # make the prior a gaussian from Weijman 2014
            p0[1] = 0.25
            sigpar[1] = 0.14
            prior_type[1]='gaussian'
        elif fast_slow == 'slow':
           # make the prior a gaussian from Li 2018
            p0[1] = 0.74
            sigpar[1] = 0.08
            prior_type[1]='gaussian'  
            # bound it lower by the max of either bound_q_intr_lo or q_intr > 0.6
            bounds[0][1] = np.max([0.6, bound_q_intr_lo])

        # anisotropy priors and labels
        if (align == 'sph') & (anisotropy == 'const'):
            bound_ani_ratio_hi = 2.0
            # bounds[1][2] = 2.0    # anisotropy of spherical can be up to 2.0
            # lower bound by R of the lower bound of q_min
            #bounds[0][2] = np.sqrt(0.3 + 0.7 * bounds[0][1])
            label2 = r"$\sigma_{\theta}/\sigma_r$"
        elif (align == 'cyl') & (anisotropy == 'const'):
            bound_ani_ratio_hi = 1.0
            #bounds[1][2] = 1.0 # anisotropy of cylindrical CANNOT be up to 2.0
            label2 = r"$\sigma_z/\sigma_R$"
        elif anisotropy == 'OM':
            label2 = r"$a_{ani}$"
            
        # shape and anisotropy secondary bounds
        #shape_anis_bounds = np.array([bound_x_lo, bound_x_hi, 0.0, bound_ani_ratio_hi])
        shape_anis_bounds = np.array([0.0, bound_ani_ratio_hi])
        
        # einstein radius is universal across models
        label3 = r"$\theta_E$"

        # 11/14/23 - lambda_int is now a parameter k_mst, but the label will be the same, lambda_int is universal across models
        label4 = r'$\lambda_{int}$'
        
        # r_scale is universal across models
        label5 = r'$a_{MST}$'
        
        labels = [label0, label1, label2, label3, label4, label5]
        
    elif geometry=='sph':
        if any(prior_type==None):
            prior_type=['uniform','uniform','uniform','uniform','uniform']  

        # mass model labels
        if model=='power_law':
            label0 = r'$\gamma$'
        elif model=='nfw':
            label0 = r"$f_{\rm DM}$"
            
        # parameter 1 is q, which is not here...

        # anisotropy priors and labels
        if (align == 'sph') & (anisotropy == 'const'):
            bounds[1][1] = 2.0    # anisotropy of spherical can be up to 2.0
            label2 = r"$\sigma_{\theta}/\sigma_r$"
        elif anisotropy == 'OM':
            label2 = r"$a_{ani}$"
        shape_anis_bounds = bounds[1]#np([0., 1.0])

        # einstein radius is universal across models
        label3 = r"$\theta_E$"

        # 11/14/23 - lambda_int is now a parameter k_mst, but the label will be the same, lambda_int is universal across models
        label4 = r'$\lambda_{int}$'
        
        # r_scale is universal across models
        label5 = r'$a_{MST}$'

        labels = [label0, label2, label3, label4, label5]  
    
    print('Priors are now ', prior_type)
    print('Mean prior values are ', p0)

    return p0, bounds, shape_anis_bounds, sigpar, prior_type, labels


###############################################################################


# Getting lower bound on q_intr_eff is where x = qobs_min**2
##### This function is not useful
def get_bounds_x_and_q_intr_eff (surf, sigma, qobs, qobs_eff):
    
    x_bound_hi = 0.9999999*np.min(qobs)**2
    x_bound_lo = 0
    
    # intrinsic shape bounds
    q_intr_eff_bound_hi = qobs_eff
    q_intrs_bound_lo = np.sqrt( (qobs**2 - x_bound_hi)/(1 - x_bound_hi))
    reff, reff_maj, eps, lum_tot = mge_half_light_isophote(surf, sigma, q_intrs_bound_lo)
    q_intr_eff_bound_lo = 1-eps
    
    return x_bound_lo, x_bound_hi, q_intr_eff_bound_lo, q_intr_eff_bound_hi

def get_bounds_on_q_intr_eff (surf, sigma, qobs, qobs_eff):
    
    # intrinsic shape must be flatter than the observed shape
    q_intr_eff_bound_hi = qobs_eff
    # all MGE components must be flatter than observed
    qobs_min = np.min(qobs)
    inc_min = np.arccos(qobs_min)
    q_intr_eff_bound_lo = np.sqrt( (qobs_eff**2 - qobs_min**2)/(1 - qobs_min**2))
    print('qobs_min ', qobs_min)
    print('q_intr lower bound from qobs_min ', q_intr_eff_bound_lo)
    inc_bound_lo = np.sqrt( np.rad2deg(np.arcsin( (1 - qobs_min**2) / (1 - q_intr_eff_bound_lo**2))) ) # check what this minimum inclination is
    print('minimum inclination from qobs_min ', inc_bound_lo)
    #argmin = ~np.argmin(q_intrs_bound_lo)
    #reff, reff_maj, eps, lum_tot = mge_half_light_isophote(surf[argmin], sigma[argmin], q_intrs_bound_lo[argmin])
    #q_intr_eff_bound_lo = 1-eps
    
    return q_intr_eff_bound_lo, q_intr_eff_bound_hi


# # Auxiliary functions

# In[10]:


###############################################################################
# Write a function for the shape_anis_joint_prior

def propose_initial_walkers (nwalkers, bounds, ndim, anisotropy):
    
    '''
    Proposes the initial state of the walkers for the emcee in parameter space.
    If fitting the anisotropy ratio directly, instead of k_ani...
    Allows the joint prior betweeen q_min_intr and anis_ratio to be introduced.
    anis_ratio > R(q) = sqrt(0.3 + 0.7*q)
    '''
    
    if anisotropy=='const_ratio':
        # propose 5 times the number of walkers we need
        propose_walk0 = np.random.uniform(bounds[0], bounds[1], [nwalkers*10,ndim])

        # calculate the constraint
        R0 = np.sqrt(0.3+0.7*propose_walk0[:,1])
        keep = R0 < propose_walk0[:,2]

        # shuffle and keep only nwalkers
        keep_walk0 = propose_walk0[keep]
        if len(keep_walk0) < nwalkers:
        # propose a bunch more and add them
            print('proposing additional')
            print(len(keep_walk0))
            propose_walk0 = np.random.uniform(bounds[0], bounds[1], [nwalkers*10,ndim])
            # calculate the constraint
            R0 = np.sqrt(0.3+0.7*propose_walk0[:,1])
            keep = R0 < propose_walk0[:,2]
            keep_walk0 = np.concatenate((keep_walk0, propose_walk0[keep]))
        np.random.shuffle(keep_walk0)
        walk0 = keep_walk0[:nwalkers]
        
    else: # using k_ani or osipkov-merritt
        walk0 = np.random.uniform(bounds[0], bounds[1], [nwalkers,ndim])
        
    return walk0


###############################################################################


def check_convergence(samples): # stolen from https://github.com/exoplanet-dev/exoplanet/blob/2e66605f3d51e4cc052759438657c41d646de446/paper/notebooks/scaling/scaling.py#L124
    tau = emcee.autocorr.integrated_time(samples, tol=0)
    num = samples.shape[0] * samples.shape[1]
    converged = np.all(tau * 1 < num)
    converged &= np.all(len(samples) > 50 * tau)
    return converged, num / tau


###############################################################################


# make a 2D gaussian

def make_2d_gaussian_xy (x, y, surf_pot, sigma_pot, qobs_pot):
    gauss = surf_pot * np.exp( - x**2 / (2 * sigma_pot**2 * qobs_pot**2) -  y**2 / (2 * sigma_pot**2))
    return gauss


###############################################################################


# define a function that will circularize the MGE

def circularize_mge (sigma, qobs):
    
    sigma_circ = sigma*np.sqrt(qobs)
    qobs_circ = np.ones_like(qobs)
    
    return sigma_circ, qobs_circ


# # Best fit and plotting functions

# In[25]:


# functions to view previous samplings

###############################################################################

def get_best_param_err (obj_name, SN, model, anisotropy, align, date_time=None, run_id=None):

    '''
    obj_name and number steps to try it out. Start all with the same priors.
    '''

    obj_abbr = obj_name[4:9] # e.g. J0029
    zlens = zs[i]
    distance = cosmo.angular_diameter_distance(zlens).value

    mos_dir = f'{mosaics_dir}{obj_name}/' # directory with all files of obj_name
    kin_dir = f'{kinematics_dir}{obj_name}/'
    jam_dir = f'{jam_output_dir}{obj_name}/'
    
    if obj_abbr=='J0330':
        target_kin_dir = f'{kin_dir}target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        target_kin_dir = f'{kin_dir}target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
    target_jam_dir = f'{jam_dir}target_sn_{SN}/'
    
    # Get model directory
    if date_time is not None and run_id is None:
        model_dir = f'{target_jam_dir}{obj_name}_model_{date_time}_{SN}_{model}_{anisotropy}_{align}/'
    elif date_time is None:
        date_time = datetime.now().strftime("%Y_%m_%d")#-%I_%M_%S_%p")
        model_dir = f'{target_jam_dir}{obj_name}_model_{date_time}_{SN}_{model}_{anisotropy}_{align}/'
    elif date_time is not None and run_id is not None:
        model_dir =  f'{target_jam_dir}{obj_name}_model_{date_time}_v{run_id}_{SN}_{model}_{anisotropy}_{align}/'   
    else:
        print('something wrong')
    
    #
    bestfit = np.genfromtxt(f'{model_dir}{obj_name}_{date_time}_bestfit_parameters.txt', delimiter='')
    err = np.genfromtxt(f'{model_dir}{obj_name}_{date_time}_bestfit_parameters_error.txt', delimiter='')
    pars = np.genfromtxt(f'{model_dir}{obj_name}_{date_time}_parameters_fit.txt', delimiter='')
    lnprob = np.genfromtxt(f'{model_dir}{obj_name}_{date_time}_likelihood.txt', delimiter='') 
    bounds = np.genfromtxt(f'{model_dir}{obj_name}_{date_time}_bounds.txt', delimiter='')
    rms_model = np.genfromtxt(f'{model_dir}{obj_name}_{date_time}_rms_model.txt', delimiter='')
    #print(labels)
    with open(f"{model_dir}{obj_name}_{date_time}_kwargs.pkl", "rb") as f:
        kwargs = pickle.load(f)
        f.close()
    
    #chi2 = -2*jam_lnprob(bestfit, **kwargs)  # Compute chi2 of model at best fit location
    chi2 = np.nansum((rms_model - kwargs.rms)**2/(kwargs.erms)**2)
    
    return bestfit, err, pars, lnprob, chi2, bounds, rms_model#, labels


###############################################################################


def jam_bestfit (pars, **kwargs):
    
    """
    Return the model of the bestfit parameters

    """
    
    surf_lum=kwargs['surf_lum']
    sigma_lum=kwargs['sigma_lum']
    qobs_lum=kwargs['qobs_lum']
    distance=kwargs['distance']
    xbin=kwargs['xbin']
    ybin=kwargs['ybin']
    sigmapsf=kwargs['sigmapsf']
    normpsf=kwargs['normpsf']
    goodbins=kwargs['goodbins']
    rms=kwargs['rms']
    erms=kwargs['erms']
    pixsize=kwargs['pixsize']
    reff=kwargs['reff']
    plot=kwargs['plot']
    model=kwargs['model']
    anisotropy=kwargs['anisotropy']
    geometry=kwargs['geometry']
    align=kwargs['align']
    zlens=kwargs['zlens']
    zsource=kwargs['zsource']
    #rs_mst=kwargs['rs_mst']
    shape_anis_bounds=kwargs['shape_anis_bounds']
    qobs_eff=kwargs['qobs_eff']
    
    
    ##### axisymmetric geometry
    if geometry=='axi':
        # parameters for fitting
        # Mass model
        if model=='power_law':
            gamma, q, anis_param, theta_E, k_mst, rs_mst = pars
            # let f_dm = 0 for a power law
            f_dm = 0
        elif model=='nfw':
            f_dm, q, anis_param, theta_E, k_mst, rs_mst = pars
            # gamma = -1 for NFW
            gamma = -1

        # Anisotropy is dependent on model
        if anisotropy=='const':
            logistic=False
            k_ani = anis_param # k_ani is a parameter [0, 1]
            ratio = anisotropy_ratio_from_k_and_q_intr(k_ani, q, shape_anis_bounds[1]) # allowed ratio depends on the intrinsic q proposed and fast/slow
            beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy, anis_param is the ratio of q_t/q_r
        elif anisotropy=='OM':
            logistic=True
            a_ani = anis_param # anis_param is the anisotropy transition radius in units of the effective radius
            r_a = a_ani*reff
            beta_0 = 0 # fully isotropic
            beta_inf = 1 # fully radially anisotropic
            alpha = 2 # sharpness of transition
            beta = [r_a, beta_0, beta_inf, alpha]

        # Get the inclination from the intrinsic shape
        inc = calculate_inclination_from_qintr_eff (qobs_eff, q)
        
        #print(q, shape_anis_bounds[1])

        # Obtain total mass profile
        surf_pot, sigma_pot, qobs_pot, lambda_int = total_mass_mge(surf_lum, sigma_lum, qobs_lum, qobs_eff,
                                                           reff, model, zlens, zsource,
                                                          gamma, f_dm, inc, theta_E, k_mst=k_mst, rs_mst=rs_mst, lg_ml=None,
                                                           plot=plot)

        surf_potential = np.stack((surf_pot, sigma_pot, qobs_pot))

        # ignore central black hole
        mbh=0.

        print('JAMMING the best fit model')

        # make the JAM model
        jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                           inc, mbh, distance, xbin, ybin, pixsize=pixsize,
                           sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                           beta=beta, logistic=logistic, data=rms, errors=erms, 
                           ml=1, plot=True, quiet=1, nodots=True)
        
    #### spherical geometry
    elif geometry=='sph':
            # parameters for fitting
        # Mass model
        if model=='power_law':
            gamma, anis_param, theta_E, k_mst, rs_mst = pars
            # let f_dm = 0 for a power law
            f_dm = 0
        elif model=='nfw':
            f_dm, anis_param, theta_E, k_mst, rs_mst = pars
            # gamma = -1 for NFW
            gamma = -1

        # Anisotropy is dependent on model
        if anisotropy=='const':
            logistic=False
            ratio = anis_param
            beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy, anis_param is the ratio of q_t/q_r
        elif anisotropy=='OM':
            logistic=True
            a_ani = anis_param # anis_param is the anisotropy transition radius in units of the effective radius
            r_a = a_ani*reff
            beta_0 = 0 # fully isotropic
            beta_inf = 1 # fully radially anisotropic
            alpha = 2 # sharpness of transition
            beta = [r_a, beta_0, beta_inf, alpha]


        # Obtain total mass profile
        surf_pot, sigma_pot, qobs_pot, lambda_int = total_mass_mge(surf_lum, sigma_lum, qobs_lum, qobs_eff,
                                                           reff, model, zlens, zsource,
                                                          gamma, f_dm, inc=90, theta_E=theta_E, k_mst=k_mst, rs_mst=rs_mst, lg_ml=None,
                                                           plot=plot)

        surf_potential = np.stack((surf_pot, sigma_pot, qobs_pot))

        # get radius of bin centers
        #rad_bin = np.sqrt(xbin**2 + ybin**2)
        # ignore central black hole
        mbh=0.

        print('JAMMING the best fit model')
        
        # There is no "goodbins" keyword for jam_axi_sph, so I need to adjust the data as such
        #rms = rms[goodbins]
        #erms = erms[goodbins]
        #rad_bin = rad_bin[goodbins]
        # Now run the jam model

        # make the JAM model
        #jam = jam_sph_proj(surf_lum, sigma_lum, surf_pot, sigma_pot, 
        #                   mbh, distance, rad_bin, #xbin, ybin, align=align, 
        #                    beta=beta, logistic=logistic, rani=r_a,
        #                   data=rms, errors=erms, #goodbins=goodbins,
        #                   pixsize=pixsize, sigmapsf=sigmapsf, normpsf=normpsf, 
        #                   plot=plot, quiet=1, ml=1)#, nodots=True)
        ####### 10/02/23 - for now, we will run jam_axi_proj, which is the same as jam_sph_proj in the spherical limit that q=1
        inc=90
        jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                           inc, mbh, distance, xbin, ybin, 
                            align=align, beta=beta, logistic=logistic,
                           data=rms, errors=erms, goodbins=goodbins,
                           pixsize=pixsize, sigmapsf=sigmapsf, normpsf=normpsf, 
                           plot=plot,  quiet=1, ml=1, nodots=True)

    return jam, surf_potential, lambda_int
 

###############################################################################
    

def summary_plot(obj_name, date_time, model_dir, jam_prob_func, model_name,                      pars=None, lnprob=None, labels=None, bounds=None, lensprior=None,                     kwargs=None, save=False, load=False):
    
    """
    Print the best fitting solution with uncertainties.
    Plot the final corner plot with the best fitting JAM model.
    """
    
    xbin = kwargs['xbin']
    ybin = kwargs['ybin']
    goodbins = kwargs['goodbins']
    rms = kwargs['rms']
    erms = kwargs['erms']
    reff = kwargs['reff']
    shape_anis_bounds = kwargs['shape_anis_bounds']
    geometry = kwargs['geometry']
    anisotropy = kwargs['anisotropy']
    p0 = np.copy(kwargs['p0'])
    
    if len(p0)==5:
        p0[1]=-0.5
        p0[3]=0
        p0[4]=0
        if lensprior=='nolensprior':
            p0[0]=0
            p0[2]=0
    else:
        p0[2]=-0.5
        p0[4]=0
        p0[5]=0
        if lensprior=='nolensprior':
            p0[0]=0
            p0[3]=0
    
    if load == True:
        jam_test_dir = model_dir #f'{data_dir}jam_testing/2023_01_31/'
        pars = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*parameters_fit.txt')[0])
        lnprob = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*likelihood.txt')[0])
        labels = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*labels.txt')[0], delimiter='  ', dtype='<U20')
        bounds = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*bounds.txt')[0])
        bestfit = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*bestfit_parameters.txt')[0])
        perc = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*bestfit_parameters_percentile.txt')[0])
        sig_bestfit = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*bestfit_parameters_error.txt')[0])
        surf_potential = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*surf_potential.txt')[0])
        rms_model = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*rms_model.txt')[0])
        flux_model = np.genfromtxt(glob.glob(f'{jam_test_dir}*{obj_name}*/*flux_model.txt')[0])
    
    else:
        
        bestfit = pars[np.argmax(lnprob)]  # Best fitting parameters

        # jam the best fit
        jam, surf_potential, lambda_int = jam_bestfit(bestfit, **kwargs)
        rms_model = jam.model
        flux_model = jam.flux
    
    lambda_ints = np.zeros(len(pars))
    print('This is now where we are calculating all the lambda_ints')
    # Calculate the actual lambda_internal from the MST parameters [0,1]
    for i, par in enumerate(pars):
        gamma = par[0]
        q = 1# not used for this
        theta_E = par[-3]
        k_mst = par[-2]
        rs_mst = par[-1]
        rbreak=100*reff
        zlens = kwargs['zlens']
        zsource = kwargs['zsource']
        lambda_int = power_law_mge (gamma, theta_E, q, rbreak, reff, k_mst, rs_mst, zlens, zsource, plot=False, return_lambda_int=True)
        lambda_ints[i] = lambda_int
    pars[:,-2] = lambda_ints
    bestfit[-2] = lambda_ints[np.argmax(lnprob)]
    bounds[0][-2] = 0.8
    bounds[1][-2] = 1.2

    # Caclculate uncertainty, extra steps for k_ani
    if (geometry == 'axi') & (anisotropy == 'const'):
        # Calculate the anisotropy ratio from pars[2] (k_ani) and pars[1] (q_intr)
        q_intrs = pars[:,1]
        k_anis = pars[:,2]
        ratios = anisotropy_ratio_from_k_and_q_intr(k_anis, q_intrs, shape_anis_bounds[1])
        pars[:,2] = ratios
        # Do the same for bestfit
        bestfit_ratio = ratios[np.argmax(lnprob)]
        bestfit[2] = bestfit_ratio
        perc = np.percentile(pars, [15.86, 84.14], axis=0)  # 68% interval
        sig_bestfit = np.squeeze(np.diff(perc, axis=0)/2)   # half of interval (1sigma)
        # bounds should be updated from shape_anis_bounds
        bounds[1][2] = shape_anis_bounds[1]
    else:
        perc = np.percentile(pars, [15.86, 84.14], axis=0)  # 68% interval
        sig_bestfit = np.squeeze(np.diff(perc, axis=0)/2)   # half of interval (1sigma)
    
    # For plotting, only show the finite probability points
    finite = np.isfinite(lnprob)

    # Produce final corner plot without trial values and with best fitting JAM
    plt.rcParams.update({'font.size': 14})
    plt.clf()
    corner_plot(pars[finite], lnprob[finite], labels=labels, extents=bounds, truths=p0, truth_color='k', fignum=1)
    logprob = jam_prob_func(bestfit, **kwargs)  # Compute model at best fit location
    chi2 = np.nansum((rms_model-rms)**2/(erms)**2) # -2*logprob
                              
    dx = 0.24
    yfac = 0.87
    fig = plt.gcf()
    fig.set_size_inches((12,12))
    fig.tight_layout()
    
    i = 0                          
    # annotate the model results
    plt.annotate(f'chi2 = {np.around(chi2, 2)}', (0.30, 0.97-(1+len(labels))*0.03), xycoords='figure fraction', fontsize=16)
    for label, best, sig in zip(labels, bestfit, sig_bestfit):
        string = f"{label} = {best:#.4g} ± {sig:#.2g}"
        plt.annotate(string, (0.30, 0.94-i*0.03), xycoords='figure fraction', fontsize=16) 
        i = i+1
                                
    # plot circular reff
    reff_plot = plt.Circle((0,0), reff, color='k', fill=False, linestyle='--')
                                 
    # plot data
    fig.add_axes([0.69, 0.99 - dx*yfac, dx, dx*yfac])  # left, bottom, xsize, ysize
    rms1 = rms.copy()
    rms1[goodbins] = symmetrize_velfield(xbin[goodbins], ybin[goodbins], rms[goodbins])
    vmin, vmax = np.percentile(rms1[goodbins], [0.5, 99.5])
    plot_velfield(xbin, ybin, rms1, vmin=vmin, vmax=vmax, linescolor='w', 
                  colorbar=1, label=r"Data $V_{\rm rms}$ (km/s)", flux=flux_model, nodots=True)
    plt.tick_params(labelbottom=False)
    plt.ylabel('arcsec')
    ax = plt.gca()
    ax.add_patch(reff_plot)
                              
    # plot circular reff again... can only patch one time
    reff_plot = plt.Circle((0,0), reff, color='k', fill=False, linestyle='--')
    
    # plot model
    fig.add_axes([0.69, 0.98 - 2*dx*yfac, dx, dx*yfac])  # left, bottom, xsize, ysize
    plot_velfield(xbin, ybin, rms_model, vmin=vmin, vmax=vmax, linescolor='w',
                  colorbar=1, label=r"Model $V_{\rm rms}$ (km/s)", flux=flux_model, nodots=True)
    #plt.tick_params(labelbottom=False)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    ax = plt.gca()
    ax.add_patch(reff_plot)
                              
    if save==True:
        plt.savefig(f'{model_dir}{obj_name}_corner_plot_{model_name}_{date_time}.png', bbox_inches='tight')
        plt.savefig(f'{model_dir}{obj_name}_corner_plot_{model_name}_{date_time}.pdf', bbox_inches='tight')

    plt.pause(1)
    plt.clf()
    plt.close()
                            
                                                                
        
    return pars, bounds, surf_potential, rms_model, flux_model, bestfit, perc, sig_bestfit, chi2



# In[12]:


# weighted gaussian to compare the models..
def weighted_gaussian(xx, mu, sig, c2):
    yy = np.zeros(shape=xx.shape)
    for i in range(len(xx)):
        yy[i] = np.exp(-np.power(xx[i] - mu, 2.) / (2 * np.power(sig, 2.))) * np.exp(-0.5 * c2)
    return yy


# # Mass profiles

# In[23]:


###############################################################################

def total_mass_mge (surf_lum, sigma_lum, qobs_lum, qobs_eff,
                    reff, model, zlens, zsource,
                    gamma, f_dm, inc, theta_E, k_mst, rs_mst, lg_ml=None,
                     plot=False):
    
    """
    Combine the MGE from a dark halo and the MGE from the stellar surface
    brightness in such a way to have a given dark matter fractions f_dm
    inside a sphere of radius one half-light radius reff
    
    # 06/09/23 - I need to figure out how to scale with the Einstein radius instead of M/L
    """
    
    break_factor = 100 # factor by which reff is multiplied to set truncation of mass profile
    
    if model == 'nfw':
        
        gamma = -1
        rbreak = break_factor*reff # much bigger than data # should this be a free parameter?

        surf_dm, sigma_dm, qobs_dm = dark_halo_mge(gamma, rbreak, k_mst, zlens, zsource, plot)
        #plt.pause(1)

        stars_lum_re = mge_radial_mass(surf_lum, sigma_lum, qobs_lum, inc, reff)
        dark_mass_re = mge_radial_mass(surf_dm, sigma_dm, qobs_dm, inc, reff)

        # Find the scale factor needed to satisfy the following definition
        # f_dm == dark_mass_re*scale/(stars_lum_re + dark_mass_re*scale)
        scale = (f_dm*stars_lum_re)/(dark_mass_re*(1 - f_dm))

        surf_pot = np.append(surf_lum, surf_dm*scale)   # Msun/pc**2. DM scaled so that f_DM(Re)=f_DM
        sigma_pot = np.append(sigma_lum, sigma_dm)      # Gaussian dispersion in arcsec
        qobs_pot = np.append(qobs_lum, qobs_dm)
        
        # Note: I multiply surf_pot by ml=10**lg_ml, while I set the keyword ml=1
        # Both the stellar and dark matter increase by ml and f_dm is unchanged
        surf_pot *= 10**lg_ml
        
        
        
    elif model == 'nfw_general':
        
        gamma = gamma
        rbreak = break_factor*reff # much bigger than data

        surf_dm, sigma_dm, qobs_dm = dark_halo_mge(gamma, rbreak, k_mst, zlens, zsource, plot)

        stars_lum_re = mge_radial_mass(surf_lum, sigma_lum, qobs_lum, inc, reff)
        dark_mass_re = mge_radial_mass(surf_dm, sigma_dm, qobs_dm, inc, reff)

        # Find the scale factor needed to satisfy the following definition
        # f_dm == dark_mass_re*scale/(stars_lum_re + dark_mass_re*scale)
        scale = (f_dm*stars_lum_re)/(dark_mass_re*(1 - f_dm))

        surf_pot = np.append(surf_lum, surf_dm*scale)   # Msun/pc**2. DM scaled so that f_DM(Re)=f_DM
        sigma_pot = np.append(sigma_lum, sigma_dm)      # Gaussian dispersion in arcsec
        qobs_pot = np.append(qobs_lum, qobs_dm)
        
        # Note: I multiply surf_pot by ml=10**lg_ml, while I set the keyword ml=1
        # Both the stellar and dark matter increase by ml and f_dm is unchanged
        surf_pot *= 10**lg_ml
        
        
    elif model == 'power_law':
        
        gamma = gamma
        theta_E = theta_E
        rbreak = break_factor*reff # much bigger than data
        
        # take counts-weighted average of light profile q
        ##### SHOULD THIS BE TAKEN FROM MY INTRINSIC FIT PRIOR INSTEAD??
        #q_mean = np.average(qobs_lum, weights=surf_lum)
        # 10/06/23 - We will use the effective shape instead qobs_eff calcualted from the half-light isophote
        
        surf_pot, sigma_pot, qobs_pot, lambda_int = power_law_mge(gamma, theta_E, qobs_eff, rbreak, reff, k_mst, rs_mst, zlens, zsource, plot)
        #plt.pause(1)
        
        if lg_ml is not None:
            lum_re = mge_radial_mass(surf_lum, sigma_lum, qobs_lum, inc, reff)
            mass_re = mge_radial_mass(surf_pot, sigma_pot, qobs_pot, inc, reff)

            # scale so that mass to light ratio at effective radius is the mass to light ratio input
            #print(mass_re/lum_re)
            ml = 10**lg_ml
            scale = lum_re/mass_re * ml
            # Multiply the surface mass by the scale
            surf_pot *= scale

    return surf_pot, sigma_pot, qobs_pot, lambda_int

###############################################################################

def power_law_mge (gamma, theta_E, q, rbreak, reff, k_mst, rs_mst, zlens, zsource, plot=False, return_lambda_int=False):
    """
    gamma - power law slope (2 = isothermal)
    theta_E - einstein radius
    q - mean q from gaussian components of light profile
    rbreak - some radius... make it big?
    lamdba_int - MST parameter
    zlens/source - redshjifts
    """
    # The fit is performed in log spaced radii from 1" to 10*rbreak
    n = 1000     # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(0.01, rbreak, n)   # logarithmically spaced radii in arcsec
    
    # Now working in convergence
    kappa = (3 - gamma) / 2 * (theta_E/r)**(gamma-1)
    #surf_mass_dens = (3 - gamma) / 2 * (rbreak/r)**(gamma-1)
    
    # transform by lambda_int
    kappa_mst, lambda_int = mass_sheet_transform(kappa, k_mst, rbreak, reff, rs_mst, zlens, zsource)
    # if lambda_int==0 we reject this
    if lambda_int==0:
        return 0, 0, 0, 0
    #surf_mass_dens_mst = mass_sheet_transform(surf_mass_dens, lambda_int, rbreak, zlens, zsource)
    if return_lambda_int==True:
        return lambda_int
    
    if any(kappa_mst<0):
        print('Mass sheet results in a negative surface mass density. Something is wrong.')
    
    # Go from convergence to surface mass density with critical surface density
    # get distances
    DL = cosmo.angular_diameter_distance(zlens).to('pc')
    DS = cosmo.angular_diameter_distance(zsource).to('pc')
    DLS = cosmo.angular_diameter_distance_z1z2(zlens, zsource).to('pc')
    # calculate critical surface density
    sigma_crit = c2_4piG * DS / DL / DLS
    # calculate surface mass density with sigma_crit
    surf_mass_dens = kappa_mst * sigma_crit.value
    
    m = mge_fit_1d(r, surf_mass_dens, ngauss=20, inner_slope=2, outer_slope=3, quiet=1, plot=plot) # this creates a circular gaussian with sigma=sigma_x (i.e. along the major axis)
    
    surf_pot, sigma_pot = m.sol           # total counts of gaussians
    surf_pot = surf_pot / np.sqrt(2*np.pi) / sigma_pot # THIS should give peak surface density
    qobs_pot = np.ones_like(surf_pot)*q   # Multiply by q to convert to elliptical Gaussians where sigma is along the major axis... I'm not sure if this is perfectly correct
    
    return surf_pot, sigma_pot, qobs_pot, lambda_int


###############################################################################

def dark_halo_mge (gamma, rbreak, k_mst, zlens, zsource, plot=False):
    """
    Returns the MGE parameters for a generalized spherical NFW dark halo profile
    https://ui.adsabs.harvard.edu/abs/2001ApJ...555..504W
    Inputs:
        gamma - inner profile slope (logarithmic slope, i.e. slope when plotted as log(rho) to log(r), < 0
                if gamma = -1, this is standard NFW profile
        rbreak - break radius, transition from inner slope to outer slope (of -3)
    Outputs:
        surf_dm, sigma_dm, qobs_dm - MGE parameters of dark halo surface potential (peak surface density, sigma of Gaussians, and axial ratio (1 because it's spherical)
        
    """
    # The fit is performed in log spaced radii from 1" to 10*rbreak
    n = 1000     # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(0.01, rbreak, n)   # logarithmically spaced radii in arcsec
    rho = nfw_generalized_model (r, gamma, rbreak)
    m = mge_fit_1d(r, rho, ngauss=15, inner_slope=2, outer_slope=3, quiet=1, plot=plot)

    surf_dm, sigma_dm = m.sol           # Peak surface density and sigma # 05/22/23 - I think this actually gives the "total counts", not peak surface density
    #surf_dm = surf_dm / np.sqrt(2*np.pi) / sigma_dm # THIS should give peak surface density
    ##### 06/08/23 - I was wrong. Because I am fitting the MGE to the volume density, dens = surf/(np.sqrt(2*np.pi)*Sigma)
    qobs_dm = np.ones_like(surf_dm)     # Assume spherical dark halo

    return surf_dm, sigma_dm, qobs_dm


# create a function to transform by the MST

def mass_sheet_transform (kappa, k_mst, rbreak, reff, rs_mst, zlens, zsource):
    
    '''
    kappa is the convergence profile (surface mass density/critical surface density).
    MST scales by lambda and adds the infinite sheet
    kappa_s is the mass sheet
    rs_mst is a "turnover" radius [0,1] (multiplicative factor of rbreak) where it goes to 0, so that it is physical.
    kappa_s = theta_s**2 / (theta_E**2 + theta_s**2)
    Figure 12 from Shajib2023 https://arxiv.org/pdf/2301.02656.pdf
    11/13/23 now, lambda_int will be parameterized as a value k_mst [0,1] that will be transformed into a parameter space allowed by the rest of the model
    '''
    
    # get kappa_s # radially dependent
    n = 1000     # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(0.01, rbreak, n)   # logarithmically spaced radii in arcsec
    # take scale (core) radius to be fixed 50*reff (i.e. 1/2 break)
    # 08/10/23 - going to change these by a factor rs_mst [0,1]
    r_s = reff*rs_mst
    kappa_s = r_s**2/(r**2 + r_s**2)
    
    # find the maximum lambda_int possible
    lambda_int_min = 0.8
    lambda_int_max = 1.2
    lambda_ints = np.linspace(1.0,1.2,1000)
    for test in lambda_ints:
        kappa_bounds = kappa * test + (1 - test) * kappa_s
        if any(kappa_bounds<0):
            lambda_int_max = test
            break

    # calculate surface mass density with mass sheet transform
    lambda_int = lambda_int_min + (lambda_int_max - lambda_int_min) * k_mst # lambda_int is a value [0,1] so lambda_internal will be between [0.8, lambda_int_max]
    mass_sheet = (1 - lambda_int) * kappa_s
    kappa_int = lambda_int * kappa + mass_sheet
    
    if any(kappa_int<0):
        print('Somehow, we have negative mass even though we set it up not to.')
        lambda_int=0
    
    return(kappa_int, lambda_int)


# # I propose now to do the following:
# ## 1. Fit r_scale as a free parameter bounded by [5, 10] times the effective radius.
# ## 2.  Let lambda_int = lower_bound + (upper_bound - lower-bound) * k_mst and fit for k_mst as a free parameter between  0 and 1.
# ## 3. Lower and upper bounds of lambda_int will be [0.8, max(model)] where the upper is the maximum lambda_int given the other model parameters.

# # Directories and saving functions

# In[14]:


# funciton to create model directory

def create_model_directory (target_jam_dir, obj_name, SN, 
                            model, anisotropy, geometry, align, 
                            sampler, lensprior, rs_mst,
                            date_time=None, overwrite=False, run_id=None):
    
    model_name = f'{SN}_{model}_{anisotropy}_{geometry}_{align}_{lensprior}_{rs_mst}'
    if date_time is None:
        date_time = datetime.now().strftime("%Y_%m_%d")#-%I_%M_%S_%p")   
    if run_id is None:
        model_dir = f'{target_jam_dir}{obj_name}_model_{date_time}_{model_name}/'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        else:
            if overwrite==True:
                print(f'Files in {model_dir} will be overwritten.')
            else:
                print('Do not overwrite your files dummy. Adding 1 to run_id to see if it works.')
                # try run_id 
                run_id = 1
                model_dir = f'{target_jam_dir}{obj_name}_model_{date_time}_v{run_id}_{model_name}/'
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                else:
                    print('Who let you do this?')
                    #print(babaganoug) # bring error
    else:
        model_dir = f'{target_jam_dir}{obj_name}_model_{date_time}_v{run_id}_{model_name}/'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        else:
            if overwrite==True:
                print(f'Files in {model_dir} will be overwritten.')
            else:
                print('Do not overwrite your files dummy. Adding 1 to run_id to see if it works.')
                # try 1 run_id higher
                run_id = run_id + 1
                model_dir = f'{target_jam_dir}{obj_name}_model_{date_time}_v{run_id}_{model_name}/'
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                else:
                    print('Who let you do this?')
                    #print(babaganoug)
                    
    return model_dir, model_name


###############################################################################


def save_fit_parameters(model_dir, model_name, obj_name, date_time, 
                        bestfit, sig_bestfit, percentile, best_chi2,
                        pars, lnprob, p0, sigpar, bounds, shape_anis_bounds, labels, 
                        surf_potential, rms_model, flux_model, kwargs):
    
    # I should save this as a pickle instead.
                              
    # save best fit parameter values
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_bestfit_parameters.txt', bestfit)
    # save best fit parameter values percentiles
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_bestfit_parameters_percentile.txt', percentile)
    # save best fit parameter values sigma error
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_bestfit_parameters_error.txt', sig_bestfit)
    # save best fit chi2 value
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_bestfit_chi2.txt', np.array([best_chi2]))
    # save fit parameters
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_parameters_fit.txt', pars)
    # save likelihoods
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_likelihood.txt', lnprob) 
    # save initial parameters
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_initial_parameters.txt',p0)           
    # save initial error estimates
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_initial_error_estimates.txt',sigpar)
    # save bounds
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_bounds.txt', bounds)
    # save shape_anis_bounds
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_shape_anis_bounds.txt', shape_anis_bounds)
    # save labels
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_labels.txt', labels, fmt='%s')
    # save surface potential
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_surf_potential.txt', surf_potential)
    # save rms_model
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_rms_model.txt', rms_model)
    # save flux_model
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_{model_name}_flux_model.txt', flux_model)
    # create a binary pickle file 
    f = open(f"{model_dir}{obj_name}_{date_time}_{model_name}_kwargs.pkl","wb")
    # write the python object (dict) to pickle file
    pickle.dump(kwargs,f)
    # close file
    f.close()


# # Probability functions

# In[15]:


###############################################################################


def jam_lnprior (pars, bounds, mu, sigma, prior_type):
    '''
    Calculate the prior likelihood for the sampled parameters
    pars
    mu - mean of prior
    sigma - width of prior
    prior_type - uniform, gaussian, log_uniform, log_normal
    '''
    
    pars = np.array(pars)
    mu = np.array(mu)
    sigma = np.array(sigma)

    if any(pars < bounds[0]) or any(pars > bounds[1]):
        lnprior = -np.inf
    
    else:
        lnprior=np.ones_like(pars)
        for i in range(len(pars)):
            if prior_type[i]=='uniform':
                lnprior[i]=0.
            elif prior_type[i]=='gaussian':
                lnprior[i]=np.log(1.0/(np.sqrt(2*np.pi)*sigma[i]))-0.5*(pars[i]-mu[i])**2/sigma[i]**2
            
    return np.sum(lnprior)


###############################################################################
# set up new anisotropy functions and probability functions to be fit

def jam_lnprob (pars, bounds=None, p0=None, sigpar=None, prior_type=None,
                surf_lum=None, sigma_lum=None, qobs_lum=None, 
                qobs_eff=None, shape_anis_bounds=None, distance=None,
                  xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None,
                   rms=None, erms=None, pixsize=None, reff=None, plot=True, 
                 model=None, anisotropy=None, geometry=None, align=None, 
                labels=None, zlens=None, zsource=None, rs_mst=None, test_prior=False):
    
    """
    Return the probability of the model, given the data, assuming priors

    """
    
    lnprior = jam_lnprior (pars, bounds, mu=p0, sigma=sigpar, prior_type=prior_type)
    
    if test_prior == True:
        return lnprior
    
    if np.isinf(lnprior) or np.isnan(lnprior):
        return -np.inf
    
    else:
        # axisymmetric model takes jam_axi_proj
        if geometry == 'axi':
            # parameters for fitting
            # Mass model
            if model=='power_law':
                gamma, q, anis_param, theta_E, k_mst, rs_mst = pars
                # let f_dm = 0 for a power law
                f_dm = 0
            elif model=='nfw':
                f_dm, q, anis_param, lg_ml, k_mst, rs_mst = pars
                # gamma = -1 for NFW
                gamma = -1
            # Anisotropy is dependent on model
            if anisotropy=='const':
                logistic=False
                k_ani = anis_param # k_ani is a parameter [0, 1]
                ratio = anisotropy_ratio_from_k_and_q_intr(k_ani, q, shape_anis_bounds[1]) # allowed ratio depends on the intrinsic q proposed and fast/slow
                beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy, anis_param is the ratio of q_t/q_r
                # lower bound is constrained by q as R(q) = sqrt(0.3 + 0.7*q) # This should already be in place, but reject it in case it's bad
                # This is a constraint from intrinsic shape
                R_q = np.sqrt(0.3 + 0.7*q)
                if ratio < R_q:
                    anisotropy_constraint = 0
                else:
                    anisotropy_constraint = 1
            elif anisotropy=='OM':
                logistic=True
                a_ani = anis_param # anis_param is the anisotropy transition radius in units of the effective radius
                r_a = a_ani*reff
                beta_0 = 0 # fully isotropic
                beta_inf = 1 # fully radially anisotropic
                alpha = 2 # sharpness of transition
                beta = np.array([r_a, beta_0, beta_inf, alpha])
                anisotropy_constraint = 1 # doesn't apply here
            # Continue if good to go
            if anisotropy_constraint == 1: # i.e. the constraint from q is okay
                
                # Get the inclination from the intrinsic shape
                inc = calculate_inclination_from_qintr_eff (qobs_eff, q)

                # Obtain total mass profile
                surf_pot, sigma_pot, qobs_pot, lambda_int = total_mass_mge(surf_lum, sigma_lum, qobs_lum, qobs_eff,
                                                                   reff, model, zlens, zsource,
                                                                  gamma, f_dm, inc, theta_E, k_mst, rs_mst, lg_ml=None,
                                                                   plot=plot)
                # reject if negative mass calculated
                if lambda_int==0:
                    lnprob = -np.inf
                    return lnprob
                # ignore central black hole
                mbh=0.
                # make the JAM model
                jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                                   inc, mbh, distance, xbin, ybin, 
                                    align=align, beta=beta, logistic=logistic,
                                   data=rms, errors=erms, goodbins=goodbins,
                                   pixsize=pixsize, sigmapsf=sigmapsf, normpsf=normpsf, 
                                   plot=plot,  quiet=1, ml=1, nodots=True)

                resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
                chi2 = resid @ resid
                lnprob = -0.5*chi2 + lnprior
            else:
                lnprob = -np.inf # reject this one
                return lnprob
        
        # sph geometry is a different jam function
        elif geometry == 'sph':
            # parameters for fitting
            # Mass model
            if model=='power_law':
                gamma, anis_param, theta_E, k_mst, rs_mst = pars
                # let f_dm = 0 for a power law
                f_dm = 0
            elif model=='nfw':
                f_dm, anis_param, lg_ml, k_mst, rs_mst = pars
                # gamma = -1 for NFW
                gamma = -1
            # Anisotropy is dependent on model
            if anisotropy=='const':
                logistic=False
                ratio = anis_param
                beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy, anis_param is the ratio of q_t/q_r
                rani = None
            elif anisotropy=='OM':
                logistic=True
                a_ani = anis_param # anis_param is the anisotropy transition radius in units of the effective radius
                r_a = a_ani*reff
                # put in r_ani keyword
                beta_0 = 0 # fully isotropic
                beta_inf = 1 # fully radially anisotropic
                alpha = 2 # sharpness of transition
                beta = np.array([r_a, beta_0, beta_inf, alpha])
            # Obtain total mass profile
            surf_pot, sigma_pot, qobs_pot, lambda_int = total_mass_mge(surf_lum, sigma_lum, qobs_lum, qobs_eff,
                                                               reff, model, zlens, zsource,
                                                              gamma, f_dm, inc=90, theta_E=theta_E, k_mst=k_mst, rs_mst=rs_mst, lg_ml=None,
                                                               plot=plot)
            # reject if negative mass calculated
            if lambda_int==0:
                lnprob = -np.inf
                return lnprob
            # get radius of bin centers
            #rad_bin = np.sqrt(xbin**2 + ybin**2)
            # ignore central black hole
            mbh=0.
            # There is no "goodbins" keyword for jam_sph_proj, so I need to adjust the data as such
            #rms = rms[goodbins]
            #erms = erms[goodbins]
            #rad_bin = rad_bin[goodbins]

            # Now run the jam model
            #jam = jam_sph_proj(surf_lum, sigma_lum, surf_pot, sigma_pot, 
            #                   mbh, distance, rad_bin, #xbin, ybin, align=align, 
            #                    beta=beta, logistic=logistic, rani=r_a,
            #                   data=rms, errors=erms, #goodbins=goodbins, # there is no goodbins
            #                   pixsize=pixsize, sigmapsf=sigmapsf, normpsf=normpsf, 
            #                   plot=plot, quiet=1, ml=1)#, nodots=True) # there is no nodots
            ####### 10/02/23 - for now, we will run jam_axi_proj, which is the same as jam_sph_proj in the spherical limit that q=1
            inc=90
            jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                               inc, mbh, distance, xbin, ybin, 
                                align=align, beta=beta, logistic=logistic,
                               data=rms, errors=erms, goodbins=goodbins,
                               pixsize=pixsize, sigmapsf=sigmapsf, normpsf=normpsf, 
                               plot=plot,  quiet=1, ml=1, nodots=True)
            resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
            chi2 = resid @ resid
            lnprob = -0.5*chi2 + lnprior

        return lnprob


# # The main function

# In[16]:


paper_table


# In[17]:


def space_jam (obj_name, SN, model, anisotropy, geometry, align, 
               sampler, sampler_args, rs_mst=None,
               p0=None, bounds=None, sigpar=None, prior_type=None, lensprior=None, 
               date_time=None, overwrite=False, run_id=None, test_prior=False):
    
    #############################################################
    # Basic info
    ##############################################################################
    ##############################################################################
    obj_abbr = obj_name[4:9] # e.g. J0029
    zlens = zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    distance = cosmo.angular_diameter_distance(zlens).value
    fast_slow = paper_table[paper_table['obj_name']==obj_name]['class_for_JAM_models'].values[0]#.to_numpy()[0]
    print('Object is a ', fast_slow, ' rotator.')
    
    #############################################################
    # Directories
    ##############################################################################
    ##############################################################################
    mos_dir = f'{mosaics_dir}{obj_name}/' # directory with all files of obj_name
    kin_dir = f'{kinematics_dir}{obj_name}/'
    jam_dir = f'{jam_output_dir}{obj_name}/'
    # create a directory for JAM outputs
    Path(jam_dir).mkdir(parents=True, exist_ok=True)
    # J0330 has no G-band
    if obj_abbr=='J0330':
        target_kin_dir = f'{kin_dir}target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        target_kin_dir = f'{kin_dir}target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
    target_jam_dir = f'{jam_dir}target_sn_{SN}/'
    # create a directory for JAM outputs
    Path(target_jam_dir).mkdir(parents=True, exist_ok=True)
    # make the model directory
    model_dir, model_name = create_model_directory (target_jam_dir, obj_name, SN, 
                                                    model, anisotropy, geometry, align, 
                                                    sampler, lensprior, rs_mst,
                                                    date_time, overwrite, run_id)       
    print()
    print('Outputs to ', model_dir)
    print()

    #############################################################
    # Preparation
    ##############################################################################
    ##############################################################################
    # prepare inputs
    surf, sigma, qobs, kcwi_sigmapsf, Vrms, dVrms, V, dV, xbin, ybin, reff = prepare_to_jam(obj_name, target_kin_dir, SN)
    print('Observed light MGEs, ', qobs)
    
    # prepare MGEs for axisymmetric or spherical geometry
    if geometry=='axi':
        # Get the effective shape and (not) effective radius from the half-light isophote
        _, _, eps_eff, _ = mge_half_light_isophote(surf, sigma, qobs)
        qobs_eff = 1-eps_eff
    elif geometry=='sph':
        # Circularize MGEs if geometry is spherical
        sigma, qobs = circularize_mge(sigma, qobs)
        qobs_eff = 1
                                                                                
    # get priors for sampling
    p0, bounds, shape_anis_bounds, sigpar, prior_type, labels = get_priors_and_labels (model=model, anisotropy=anisotropy, 
                                                                                       surf=surf, sigma=sigma, qobs=qobs, qobs_eff=qobs_eff, 
                                                                                       geometry=geometry, align=align, fast_slow=fast_slow, 
                                                                                         p0=p0, bounds=bounds, sigpar=sigpar, prior_type=prior_type)
    
    print('Bounds in jam: ', bounds)
    print('Anisotropy bounds: ', shape_anis_bounds)
    
    goodbins = np.isfinite(xbin)  # Here I fit all bins, it's already masked

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf, 'sigma_lum': sigma, 'qobs_lum': qobs, 'qobs_eff': qobs_eff,
              'distance': distance, 'xbin': xbin, 'ybin': ybin, 'sigmapsf': kcwi_sigmapsf,
              'normpsf': 1., 'rms':Vrms, 'erms':dVrms, 'pixsize': kcwi_scale,
              'goodbins': goodbins, 'plot': False, 'reff':reff, 
              'model':model, 'anisotropy':anisotropy, 'geometry':geometry, 'align':align,
              'p0':p0, 'bounds':bounds, 'shape_anis_bounds':shape_anis_bounds, 'sigpar':sigpar, 'prior_type':prior_type, 'labels':labels,
              'zlens':zlens, 'zsource':zsource, 'rs_mst':rs_mst, 'test_prior':test_prior
             }
    
    #############################################################
    # Run JAM
    ##############################################################################
    ##############################################################################

    # For now, we have a single function that will work for const/om, and pl/nfw
    jam_prob_func=jam_lnprob

    if sampler=='adamet':
        # Do the fit
        print("Started AdaMet please wait...")
        print("Progress is printed periodically")
        nstep = sampler_args
        pos0 = p0 + np.random.normal(0, sigpar, len(p0)) # initialize slightly off # Fix this later
        pars, lnprob = adamet(jam_prob_func, pos0, sigpar, bounds, nstep, fignum=1,
                              kwargs=kwargs, nprint=nstep/20, labels=labels, seed=2, plot=False)
        
    elif sampler=='emcee':
        # Do the fit
        print("Started Emcee please wait...")
        print("Progress is printed periodically")
        nstep, nwalkers, ndim = sampler_args
        # set initial walker positions
        walk0 = propose_initial_walkers (nwalkers, bounds, ndim, anisotropy)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_prob_func, kwargs=kwargs)
        sampler.run_mcmc(walk0, nstep, progress=True)
        # save sampler as pickle
        f = open(f"{model_dir}{obj_name}_{date_time}_{model_name}_emcee_sampler.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(sampler,f)
        # close file
        f.close()
        pars = sampler.get_chain(flat=True)
        lnprob = sampler.get_log_prob(flat=True)
        #fig = corner.corner(
        #    pars, labels=labels
        #);
        #for i in range(ndim):
        #    mcmc = np.percentile(pars[:, i], [16, 50, 84])
        #    q = np.diff(mcmc)
        #    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        #    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        #    display(Math(txt))

    print('n accepted unique parameters', len(np.unique(pars[:,0])))
    
    # plot the results, get rms_model and flux_model of best fit
    pars, bounds, surf_potential, rms_model, flux_model,         bestfit, percentiles, sig_bestfit, bestfit_chi2 = summary_plot(obj_name, date_time, model_dir, jam_prob_func, model_name,
                                                                     pars=pars, lnprob=lnprob, labels=labels, bounds=bounds, lensprior=lensprior,
                                                                     kwargs=kwargs, save=True, load=False)

    # save parameters from fit
    save_fit_parameters(model_dir, model_name, obj_name, date_time, bestfit, sig_bestfit, percentiles, bestfit_chi2, pars, lnprob, p0, sigpar, 
                        bounds, shape_anis_bounds, labels, surf_potential, rms_model, flux_model, kwargs)


# # New functions 10/06/23
# 

# In[18]:


# Getting lower bound on q_intr_eff is where x = qobs_min**2


def calculate_inclination_from_qintr_eff (qobs_eff, qintr_eff):#, x_bound_hi ):
   
    #f = lambda x: np.sqrt( (qobs_eff**2 - x)/(1 - x) ) - q_intr
    #xx = fsolve(f, x_bound_hi)[0]
    
    y = (1 - qobs_eff**2)/(1 - qintr_eff**2) # sin^2(inclination)
    inclination = 180/np.pi*np.arcsin(np.sqrt(y))
    
    return inclination


def anisotropy_ratio_from_k_and_q_intr(k, q_intr, bound_anis_ratio_hi):
    
    R_q = np.sqrt(0.3 + 0.7*q_intr)
    
    ratio = R_q + (bound_anis_ratio_hi - R_q) * k
    
    return ratio


# ______________________
# 
# # Dynamical Modeling with JAM
# 

# # Priors:
# ### Power law slope - Uniform [1.4, 2.8] or Gaussian from lensing
# ### Anisotropy constant - Uniform [R(q_min), 1] or [R(q_min), 2] for cylindrical/spherical
# ### Anisotropy OM - Uniform [0.1, 2.0] # 5.0] # 09/13/23 - I changed it to 2.0 because I only have data out to about the effective radius, I think...
# ### Shape/Inclination - Uniform [0.051, q_obs_min]
# ### Einstein radius - Uniform [0.7, 2.0] or Gaussian from lensing
# ###################### M_tot/L_B power law - Uniform [0, 1.5]
# ### f_DM - Uniform [0.01, 0.8]
# ### M_*/L_B composite - Uniform [0, 1.5]

# In[19]:


# set up the starting guesses for each model

# These are all parameters that would make more sense in AdaMet.
# For emcee, sigpar isn't used

# power law constant ani
###### 10/06/23 - Now the constant anisotropy is taken as k_ani instead of the ratio explicitly
p0_pow_const = [2.0, 0.4, 0.9, 1.0, 0.5, 7] # gamma0, q0, ratio0, einstein radius, lambda_int
bounds_pow_const = [[1.4, 0.051, 0.0, 0.7, 0.0, 5], 
                    [2.8, 1.0, 1.0, 2.0, 1.0, 10]]
sigpar_pow_const = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # crude estimate of uncertainties

# power law om ani
p0_pow_om = [2, 0.4, 1.0, 1.0, 1.0] # gamma0, q0, anis_rad0, einstin radius, lambda_int
bounds_pow_om = [[1.6, 0.051, 0.1, 0.7, 0.75],  
                  [2.8, 1.0, 2.0, 2.0, 1.25]]#, 2.0]]
sigpar_pow_om = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # crude estimate of uncertainties

# nfw const ani
p0_nfw_const = [0.3, 0.4, 0.9, 1.0, 1.0] # f_dm0, q0, ratio0, lg_ml0
bounds_nfw_const = [[0.01, 0.051, 0.01, 0., 0.75], 
                  [0.8, 1.0, 1.0, 1.5, 1.25]]#, 2.0]]
sigpar_nfw_const = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # crude estimate of uncertainties

# nfw om ani
p0_nfw_om = [0.3, 0.4, 1.0, 1.0, 1.0] # f_dm0, q0, anis_rad0, lg_ml0
bounds_nfw_om = [[0.01, 0.051, 0.1, 0., 0.75], 
                  [0.8, 1.0, 2.0, 1.5, 1.25]]#, 2.0]]
sigpar_nfw_om = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # crude estimate of uncertainties

prior_type_uniform = ['uniform','uniform','uniform','uniform','uniform','uniform']


# In[20]:


active_obj_names = ['SDSSJ0029-0055',
                     'SDSSJ0037-0942',
                     #'SDSSJ0330-0020',
                     #'SDSSJ1112+0826',
                     'SDSSJ1204+0358',
                     #'SDSSJ1250+0523',
                     #'SDSSJ1306+0600',
                     'SDSSJ1402+6321',
                     #'SDSSJ1531-0105',
                     #'SDSSJ1538+5817',
                     #'SDSSJ1621+3931',
                     #'SDSSJ1627-0053',
                     #'SDSSJ1630+4520',
                     #'SDSSJ2303+1422'
                   ]
                    
#['SDSSJ0029-0055', 'SDSSJ0037-0942', 'SDSSJ1112+0826',
#       'SDSSJ1204+0358', 'SDSSJ1250+0523', 'SDSSJ1306+0600',
#       'SDSSJ1402+6321', 'SDSSJ1531-0105', 'SDSSJ1621+3931',
#       'SDSSJ1627-0053']                    


# ____________________
# 
# # Emcee models
# ## Introducing different mass sheet scale radii as factors of break radius, which is currently 100*reff
# ## Using scale radius 0.3

# In[21]:


# Don't redo the constant ones from 091123.

import glob

SN=15

model='*'#pow' # wildcard should select all of them
anisotropy='OM'
align='*'
sampler='*'#emcee'

date_time='*'#'2023_09_11'
run_id = '1'
#model_dir = 

#model_dirs_skip = np.array(glob.glob(f'{jam_output_dir}/**/**/*_{date_time}_v{run_id}_{model}_{anisotropy}_{align}_{sampler}*'))
#model_dirs_skip = np.sort(model_dirs_skip)


# model_dirs_skip

# skip_models = np.array([0,1,2,3])
# any(skip_models == 0)

# # 11/14/23 - Running four objects, axi sph, w/ prior, new lambda_int stuff

# In[26]:


# set mass model, anisotropy, and alignment
models = ['power_law','nfw']
anis = ['const','OM']
geoms = ['axi', 'sph']
aligns = ['sph', 'cyl']
lenspriors = ['lensprior','nolensprior']
# for a conservative lens prior
conservative_gamma=False#True

#scale_radii = np.arange(0.1, 0.6, 0.1) # 10*reff to 50*reff
#rs_mst = 0.3 # scale radius for mass sheet

# set nstep
nstep= 300#500
nwalkers= 16

print('Thank you for choosing MiseryBot 9.3.2, where you will hate everything and probably quit after a short while.')
print('...')
print('Please hold...')
print('...')
print('Ah yes, the JAM routine. This one is fun.')
print('...')
print('But where to start?')
print('...')

date_time = datetime.now().strftime("%Y_%m_%d")#-%I_%M_%S_%p")
print(f'Current date is {date_time}. Hopefully tomorrow will be better.')


print('########################################################')
print('########################################################')

for i, obj_name in enumerate(active_obj_names):
    
    print(f'Jamming object {obj_name}.')
    print('Brace for jamming.')

    print('########################################################') 
    
    for j, SN in enumerate([vorbin_SN_targets[1]]):
        
        print(f'Designated target is S/N {SN}.')
        
        for k, model in enumerate([models[0]]):
            
            print(f'Beginning {model} models.')
            
            for l, anisotropy in enumerate([anis[0]]):
                
                print(f'Beginning {anisotropy} models.')
                              
                for w, geometry in enumerate([geoms[0]]):
                
                    print(f'Beginning {geometry} geometry models.')
                
                    for m, align in enumerate([aligns[0]]):
                        
                        # skip cylindrical when using spherical geometry
                        if (geometry=='sph') & (align=='cyl'):
                            print('No cylindrical alignment in spherical geometry.')
                            continue
                            
                        if (anisotropy=='OM') & (align=='cyl'):
                            print('No OM for cylindrical alignment.')
                            continue

                        print(f'Beginning {align} aligned models.')
                        

                        for n, lensprior in enumerate([lenspriors[0]]):

                            print(f'Commencing JAM routine with model {model}, anisotropy {anisotropy}, geometry {geometry}, and alignment {align}... {lensprior}... number of steps {nstep}')
                            print(f'Current date and time is {tick()}')
                            print('JAMMING.')
                            print('########################################################') 

                            if model=='power_law' and anisotropy=='const':
                                p0 = np.copy(p0_pow_const)
                                bounds = np.copy(bounds_pow_const)
                                sigpar = np.copy(sigpar_pow_const)
                                # take lens priors lensprior = 'lensprior'
                                if lensprior=='lensprior':
                                    # update lensing priors
                                    lens_model = lens_models[lens_models['obj_name']==obj_name]
                                    p0[0] = lens_model['gamma'].iloc[0] #power_law_slopes.iloc[obj_name].iloc[i, 1]
                                    sigpar[0] = lens_model['dgamma'].iloc[0]
                                    p0[3] = lens_model['theta_E'].iloc[0]
                                    sigpar[3] = lens_model['dtheta_E'].iloc[0]
                                    prior_type=['gaussian','uniform','uniform','gaussian','uniform','uniform']
                                    # add 0.1 in quadrature to dgamma if conservative estimate
                                    if conservative_gamma == True:
                                        sigpar[0] = np.sqrt(sigpar[0]**2+0.1**2)
                                else:
                                    prior_type = np.copy(prior_type_uniform)

                            if model=='power_law' and anisotropy=='OM':
                                p0 = np.copy(p0_pow_om)
                                bounds = np.copy(bounds_pow_om)
                                sigpar = np.copy(sigpar_pow_om)
                                # take lens priors lensprior = 'lensprior'
                                if lensprior=='lensprior':
                                    if obj_name == 'SDSSJ1538+5817':
                                        print('SDSSJ1538+5817 does not have lens prior for now.')
                                        continue
                                    else:
                                        # update lensing priors
                                        lens_model = lens_models[lens_models['obj_name']==obj_name]
                                        p0[0] = lens_model['gamma'].iloc[0] #power_law_slopes.iloc[obj_name].iloc[i, 1]
                                        sigpar[0] = lens_model['dgamma'].iloc[0]
                                        p0[3] = lens_model['theta_E'].iloc[0]
                                        sigpar[3] = lens_model['dtheta_E'].iloc[0]
                                        prior_type=['gaussian','uniform','uniform','gaussian','uniform','uniform']
                                else:
                                    prior_type = np.copy(prior_type_uniform)

                            if model=='nfw' and anisotropy=='const':
                                p0 = np.copy(p0_nfw_const)
                                bounds = np.copy(bounds_nfw_const)
                                sigpar = np.copy(sigpar_nfw_const)
                                prior_type = np.copy(prior_type_uniform)
                                if lensprior=='lensprior':
                                    print('No lens prior for NFW.')
                                    print('########################################################') 
                                    print("No JAM! On to the next?")
                                    print('########################################################') 
                                    print('########################################################')
                                    continue

                            if model=='nfw' and anisotropy=='OM':
                                p0 = np.copy(p0_nfw_om)
                                bounds = np.copy(bounds_nfw_om)
                                sigpar = np.copy(sigpar_nfw_om)
                                prior_type = np.copy(prior_type_uniform)
                                if lensprior=='lensprior':
                                    print('No lens prior for NFW.')
                                    print('########################################################') 
                                    print("No JAM! On to the next?")
                                    print('########################################################') 
                                    print('########################################################')
                                    continue
                                    
                            if geometry=='sph':
                                # remove q parameter at index 1
                                p0 = np.delete(p0, 1)
                                bounds = np.delete(bounds, 1, axis=1)
                                sigpar = np.delete(sigpar, 1)
                                prior_type = np.delete(prior_type, 1)
                                    
                            print('p0', p0)
                            print('bounds', bounds)
                            print('sigpar', sigpar)
                            print('prior_type', prior_type)

                            # sampler args for emcee
                            ndim = len(p0)
                            sampler_args = [nstep, nwalkers, ndim] # 10 walkers

                            space_jam (obj_name, SN, model, anisotropy, geometry, align,                                         sampler='emcee', sampler_args=sampler_args,                                        rs_mst=None,                                        p0=p0, bounds=bounds, sigpar=sigpar, prior_type=prior_type, lensprior=lensprior,                                        date_time=date_time, overwrite=True, run_id=2, test_prior=False)#True)

                            print('########################################################') 
                            print("We've been... JAMMED! On to the next?")
                            print('########################################################') 
                            print('########################################################') 

print(f'Okay, all done as of {tick()}. Thank you for using MiseryBot 9.1.3. We deprecated while you were away. Hope you data is not corrupted.')


# In[ ]:


#squash


# __________________
# __________________
# 
# # BREAK
# __________________
# __________________
