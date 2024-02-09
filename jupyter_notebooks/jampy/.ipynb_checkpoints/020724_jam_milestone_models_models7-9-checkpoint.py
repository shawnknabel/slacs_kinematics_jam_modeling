#!/usr/bin/env python
# coding: utf-8
######################## This script does the models from MFL const sph onward because I forgot to skip J0330 for the spherical models here. (I did skip it during PL models)

# # 02/07/24 - Added spherical modeling and OM anisotropy
# # 02/02/24 - Modified so that I can show the differences between cyl/sph and MFL/PL
# # 02/01/24 - This notebook was copied from 011924_jam_mass_profile_testing.ipynb and will be used to measure the enclossed mass within the einstein radius of mass-follows-light models
# ______________
# # 01/19/24 - This notebook tests my mass profile class "total_mass_mge" in e.g.
# # 01/29/24 - Added looking at the mass profile and brightness profile at Michele's suggestion, using the mass-follows-light to compare with powerlaw
# # 01/30/24 - Added Michele's power law code to see if I can reproduce it.
# ## Shawn wrote this. I've compiled this to be ready for Michele to test.

# In[71]:


# import general libraries and modules
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 6)
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import dill as pickle

# astronomy/scipy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
import astropy.units as u
import astropy.constants as constants

# ppxf/capfit
from ppxf.capfit import capfit

# mge fit
import mgefit
from mgefit.mge_fit_1d import mge_fit_1d

# jampy
from jampy.jam_axi_proj import jam_axi_proj
from jampy.jam_sph_proj import jam_sph_proj
from jampy.mge_half_light_isophote import mge_half_light_isophote
from jampy.mge_half_light_isophote import mge_half_light_radius

# plotbin
from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield
from plotbin.sauron_colormap import register_sauron_colormap
register_sauron_colormap()


# my functions
import sys
sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages")
import slacs_mge_jampy

################################################################
# some needed information
kcwi_scale = 0.1457  # arcsec/pixel

# value of c^2 / 4 pi G
c2_4piG = (constants.c **2 / constants.G / 4 / np.pi).to('solMass/pc')


# In[72]:


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


# In[73]:


#################################################
# directories and tables

date_of_kin = '2023-02-28_2'
# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
hst_dir = '/data/raw_data/HST_SLACS_ACS/kcwi_kinematics_lenses/'
tables_dir = f'{data_dir}tables/'
mosaics_dir = f'{data_dir}mosaics/'
kinematics_full_dir = f'{data_dir}kinematics/'
kinematics_dir =f'{kinematics_full_dir}{date_of_kin}/'
jam_output_dir = f'{data_dir}jam_outputs/2024_02_07_jam_milestone_outputs/'
milestone_dir = f'{data_dir}milestone23_data/'

paper_table = pd.read_csv(f'{tables_dir}paper_table_100223.csv')
slacs_ix_table = pd.read_csv(f'{tables_dir}slacs_ix_table3.csv')
zs = paper_table['zlens']
zlenses = slacs_ix_table['z_lens']
zsources = slacs_ix_table['z_src']
# get the revised KCWI sigmapsf
sigmapsf_table = pd.read_csv(f'{tables_dir}kcwi_sigmapsf_estimates.csv')
lens_models = pd.read_csv(f'{tables_dir}lens_models_table_chinyi.csv')


# In[74]:


lens_models_chinyi = pd.read_csv(f'{tables_dir}lens_models_table_chinyi.csv')
lens_models_chinyi_sys = pd.read_csv(f'{tables_dir}lens_models_table_chinyi_with_sys.csv')
lens_models_anowar = pd.read_csv(f'{tables_dir}lens_models_table_anowar.csv')

lens_models_chinyi_sys.rename(columns={'dgamma':'gamma_err',
                                       'dgamma_sys':'gamma_sys',
                                      },inplace=True)


lens_models_chinyi_sys['dgamma'] = np.sqrt( lens_models_chinyi_sys['gamma_err']**2 + lens_models_chinyi_sys['gamma_sys']**2 )
lens_models_chinyi_sys.loc[9, 'dgamma'] = np.sqrt( lens_models_chinyi_sys.loc[9, 'gamma_err']**2 + np.nanmean(lens_models_chinyi_sys['gamma_sys'])**2)
lens_models = lens_models_chinyi_sys


# # Functions needed to get the data, etc

# In[75]:


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

def calculate_1d_gaussian(r, amp, sigma):
    return amp * np.exp(-1/2 * r**2 / sigma**2)



# # Michele's power law code

# In[76]:


###############################################################################

def total_mass_mge_cap(gamma, rbreak):
    """
    Returns the MGE parameters for a generalized NFW dark halo profile
    https://ui.adsabs.harvard.edu/abs/2001ApJ...555..504W
    - gamma is the inner logarithmic slope (gamma = -1 for NFW)
    - rbreak is the break radius in arcsec

    """
    # The fit is performed in log spaced radii from 0.01" to 10*rbreak
    n = 1000#300     # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(0.01, rbreak, n)   # logarithmically spaced radii in arcsec
    rho = (r/rbreak)**gamma * (0.5 + 0.5*r/rbreak)**(-gamma - 3)  # rho=1 at r=rbreak
    ngauss=30#15
    m = mge_fit_1d(r, rho, ngauss=ngauss, inner_slope=2, outer_slope=3, quiet=1, plot=1)

    surf_dm, sigma_dm = m.sol           # Peak surface density and sigma
    #surf_dm /= np.sqrt(2*np.pi*sigma_dm**2)
    qobs_dm = np.ones_like(surf_dm)     # Assume spherical dark halo

    return surf_dm, sigma_dm, qobs_dm

###############################################################################

def jam_resid(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, 
              distance=None, align=None, goodbins=None,
              xbin=None, ybin=None, sigmapsf=None, normpsf=None, 
              rms=None, erms=None, pixsize=None, plot=True, return_mge=False):

    q, ratio, gamma = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy

    rbreak = 50 # arcsec 20*reff
    mbh = 0                     # Ignore the central black hole
    surf_pot, sigma_pot, qobs_pot = total_mass_mge_cap(gamma, rbreak)

    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=None)
    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = np.sum(resid**2)
    
    if return_mge==True:
        
        # return the mass mge multiplied by the ml from jam
        return surf_pot*jam.ml, sigma_pot, qobs_pot, chi2
    
    else:
        return resid   # ln(likelihood) + cost.



# In[56]:


def calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius, plot=False):
    
    # circularize mass
    pc = distance*np.pi/0.648 # constant factor arcsec -> pc
    sigma = sigma_pot*np.sqrt(qobs_pot) # circularize while conserving mass
    mass = 2*np.pi*surf_pot*(sigma*pc)**2 # keep peak surface brightnesss of each gaussian
    mass_tot = np.sum(mass)
    
    # radii to sample
    nrad = 50
    rmin = 0.01 #np.min(sigma)
    rad = np.geomspace(rmin, radius, nrad) # arcsec
    
    # enclosed mass is integral
    mass_enclosed = (mass*(1 - np.exp(-0.5*(rad[:, None]/sigma)**2))).sum(1)[-1]
    
    # slope is average over interval
    mass_profile_1d = np.zeros_like(rad, dtype=float)
    for i in range(len(surf_pot)):
        amp = surf_pot[i]
        sigma = sigma_pot[i]
        q = qobs_pot[i]
        zz = calculate_1d_gaussian(rad, amp, sigma)
        mass_profile_1d += zz
    # diff in log space minus 1 (because we're in surface mass density    
    gamma_avg = (np.log(mass_profile_1d[-1])-np.log(mass_profile_1d[0]))/(np.log(rad[-1])-np.log(rad[0]))-1
    
    # add up the mass profile
    r = np.geomspace(np.min(sigma_pot), np.max(sigma_pot), nrad) # arcsec
    # slope is average over interval
    mass_profile_1d = np.zeros_like(r, dtype=float)
    for i in range(len(surf_pot)):
        amp = surf_pot[i]
        sigma = sigma_pot[i]
        q = qobs_pot[i]
        zz = calculate_1d_gaussian(r, amp, sigma)
        mass_profile_1d += zz
        
    if plot==True:
        plt.loglog(r, mass_profile_1d)
        plt.pause(1)
        
    return mass_enclosed, gamma_avg, r, mass_profile_1d


# # PL, constant anisotropy


# _____________________________
# _____________________________
# _____________________________
# _____________________________
# # MFL, constant anisotropy


# ## Spherical Geometry MFL

# In[89]:


def jam_resid_mfl_sph(pars, surf_lum=None, sigma_lum=None,
                  distance=None, rbin=None,
                   sigmapsf=None, normpsf=None, 
                  rms=None, erms=None, pixsize=None, plot=True, return_mge=False):

    ratio = pars

    # anisotropy
    beta = np.full_like(surf_lum, 1 - ratio**2)   # assume constant anisotropy
    
    # break radius (how far out we fit it)
    rbreak = 50 # arcsec 20*reff
    mbh = 0 # Ignore the central black hole
    
    # take the surface brigbhtness profile as the mass prfoile, it will be scaled by some M/L
    surf_pot, sigma_pot = surf_lum, sigma_lum
    qobs_pot = np.ones_like(surf_pot)
    
    # do the fit
    jam = jam_sph_proj(surf_lum, sigma_lum, surf_pot, sigma_pot,
                       mbh, distance, rbin,
                       sigmapsf=sigmapsf, normpsf=normpsf,
                      beta=beta, data=rms, errors=erms, 
                       plot=plot, pixsize=pixsize, quiet=1, ml=None)
    
    # get residual and chi2
    resid = (rms - jam.model)/erms
    chi2 = np.sum(resid**2)
    
    if return_mge==True:
        
        # return the mass mge multiplied by the ml from jam
        return surf_pot*jam.ml, sigma_pot, qobs_pot, chi2
    
    else:
        return resid   # ln(likelihood) + cost.


# In[90]:


mfl_const_sphsph_solutions = np.ones((len(obj_names), 8))
mfl_const_sphsph_cov = np.ones((len(obj_names), 1, 1))
mfl_const_sphsph_mass_profiles_r = np.ones((len(obj_names), 50))
mfl_const_sphsph_mass_profiles = np.ones((len(obj_names), 50))


# In[91]:


for i in range(len(obj_names)):

    obj_name = obj_names[i]
    obj_abbr = obj_name[4:9]
    
    print('######################')
    print()
    print(obj_name)
    print()
    
    if obj_abbr=='J0330':
        print('J0330 is not reliable and will not be used here.')
        print()
        continue
    
    SN = 15
    mass_model='mfl'
    anisotropy='const'
    geometry='sph'
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    theta_E = lens_models[lens_models['obj_name']==obj_name]['theta_E'].to_numpy()[0]
    cosmo = cosmo
    distance = cosmo.angular_diameter_distance(zlens).to_numpy()[0]
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    
    # load in the 1D kinematics
    Vrms_1d = pd.read_csv(f'{milestone_dir}{obj_name}/{obj_name}_kinmap.csv')
    Vrms = Vrms_1d['binned_Vrms'].to_numpy()
    rbin = np.mean(Vrms_1d[['bin_inner_edge','bin_outer_edge']].to_numpy(), axis=1)
    Vrms_cov = np.loadtxt(f'{milestone_dir}{obj_name}/{obj_name}_kinmap_cov.csv', delimiter=',', dtype=float)
    dVrms = np.sqrt(Vrms_cov.diagonal())
    
    if obj_abbr=='J0330':
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
        
    surf_lum, sigma_lum, _, _, _, _, _, _, _, _, reff = prepare_to_jam(obj_name, file_dir, SN)
    
    # Starting guesses
    ratio0 = 1          # Anisotropy ratio sigma_z/sigma_R
    
    normpsf= 1  
    p0 = [ratio0]
    bounds = [[0.01], [2.0]]

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum,
              'distance': distance, 'rbin':rbin, 
              'sigmapsf': sigmapsf, 'normpsf': normpsf, 
              'rms': Vrms, 'erms': dVrms, 
              'pixsize': kcwi_scale,#pixsize,
              'plot': 0, 'return_mge': False}

    sol = capfit(jam_resid_mfl_sph, p0, kwargs=kwargs, bounds=bounds, verbose=2, fixed=[0])
    if sol.success==True:
        print()
        print('Success!')
        print()
    else:
        print()
        print('Fit failed to meet convergence criteria.')
        print()
    
    # bestfit
    anis_ratio = sol.x[0]
    print('Best fit,', anis_ratio)
    
    # covariance matrix of free parameters
    cov = sol.cov
    anis_ratio_err = np.sqrt(sol.cov.diagonal()[0])
    
    kwargs['plot'] = 1
    kwargs['return_mge'] = True
    surf_pot, sigma_pot, qobs_pot, chi2 = jam_resid_mfl_sph(sol.x, **kwargs)
    plt.savefig(f'{jam_output_dir}{obj_name}_mfl_const_sphsph_fit.png')
    plt.pause(1)
    
    # take chi2/dof
    #dof = len(Vrms)
    #chi2_red = chi2/dof
    
    # calculate the enclosed mass
    mass_enclosed, gamma_avg, profile_radii, profile_1d = calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius=theta_E, plot=True)
    
    # parameters not fit
    inc = np.nan
    gamma_fit = np.nan
    gamma_err = np.nan
    
    
    print('Mass enclosed, avg gamma', mass_enclosed, gamma_avg)
    
    mfl_const_sphsph_solutions[i] = np.array([inc, anis_ratio, anis_ratio_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2])
    mfl_const_sphsph_cov[i] = cov
    mfl_const_sphsph_mass_profiles_r[i] = profile_radii
    mfl_const_sphsph_mass_profiles[i] = profile_1d


# In[92]:


# save with header
sol_header = '#### 14 objects - inc, anis_ratio, anis_ratio_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2'
np.savetxt(f'{jam_output_dir}jam_mfl_const_sphsph_solutions_02072024.txt', mfl_const_sphsph_solutions, header=sol_header) 
np.savetxt(f'{jam_output_dir}jam_mfl_const_sphsph_covariances_02072024.txt', mfl_const_sphsph_cov.reshape(14,1), header='covariance matrix has been reshaped to 2D, reshape back to (14,3,3)') 

profile_header = '#### radius, surface mass density (Msol/pc2), has been reshaped to 2D, reshape back to (14,50,2)'
mfl_const_sphsph_mass_profile_joined = np.dstack((mfl_const_sphsph_mass_profiles_r, mfl_const_sphsph_mass_profiles)).reshape(14, 100)
np.savetxt(f'{jam_output_dir}jam_mfl_const_sphsph_profiles_02072024.txt', mfl_const_sphsph_mass_profile_joined, header=profile_header) 


# # MFL, Osipkov-Merrit

# In[93]:


###############################################################################

def jam_resid_mfl_sph_OM(pars, surf_lum=None, sigma_lum=None,
                      distance=None, rbin=None,
                       sigmapsf=None, normpsf=None, 
                      rms=None, erms=None, pixsize=None, plot=True, return_mge=False):
    
    # OM uses anisotropy radius instead of the ratio
    a_ani = pars
    rani = reff * a_ani

    rbreak = 50 #20*reff
    mbh = 0                     # Ignore the central black hole
    
    # take the surface brigbhtness profile as the mass prfoile, it will be scaled by some M/L
    surf_pot, sigma_pot = surf_lum, sigma_lum
    qobs_pot = np.ones_like(surf_pot)

    jam = jam_sph_proj(surf_lum, sigma_lum, surf_pot, sigma_pot,
                       mbh, distance, rbin,
                       sigmapsf=sigmapsf, normpsf=normpsf,
                       rani=rani, data=rms, errors=erms, 
                       plot=plot, pixsize=pixsize, quiet=1, ml=None)
    
    resid = (rms - jam.model)/erms
    chi2 = np.sum(resid**2)
    
    if return_mge==True:
        
        # return the mass mge multiplied by the ml from jam
        return surf_pot*jam.ml, sigma_pot, qobs_pot, chi2
    
    else:
        return resid   # ln(likelihood) + cost.


# In[101]:


###############################################################################

def jam_resid_mfl_axi_OM(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, 
                      distance=None, align=None, goodbins=None,
                      xbin=None, ybin=None, sigmapsf=None, normpsf=None, 
                      rms=None, erms=None, pixsize=None, plot=True, return_mge=False):
    
    #### OM should only be done with spherical alignment
    
    q, a_ani = pars
    rani = a_ani * reff
    
    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    beta = [rani, 0, 1, 2]
    
    rbreak = 50 #20*reff
    mbh = 0                     # Ignore the central black hole
    
    # take the surface brigbhtness profile as the mass prfoile, it will be scaled by some M/L
    surf_pot, sigma_pot = surf_lum, sigma_lum
    qobs_pot = np.ones_like(surf_pot)

    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=None, logistic=True)
    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = np.sum(resid**2)
    
    if return_mge==True:
        
        # return the mass mge multiplied by the ml from jam
        return surf_pot*jam.ml, sigma_pot, qobs_pot, chi2
    
    else:
        return resid   # ln(likelihood) + cost.


# In[95]:


# will need the fit parameters, average slopes, enclosed masses, and uncertainties on fit parameters

# np.array([inc, a_ani, a_ani_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2_red])
mfl_om_sphsph_solutions = np.ones((len(obj_names), 8))
mfl_om_sphsph_cov = np.ones((len(obj_names), 1, 1))
mfl_om_sphsph_mass_profiles_r = np.ones((len(obj_names), 50))
mfl_om_sphsph_mass_profiles = np.ones((len(obj_names), 50))

# np.array([inc, a_ani, a_ani_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2_red])
mfl_om_axisph_solutions = np.ones((len(obj_names), 8))
mfl_om_axisph_cov = np.ones((len(obj_names), 2, 2))
mfl_om_axisph_mass_profiles_r = np.ones((len(obj_names), 50))
mfl_om_axisph_mass_profiles = np.ones((len(obj_names), 50))


# ## OM MFL spherical geometry

# In[97]:


for i in range(len(obj_names)):

    obj_name = obj_names[i]
    obj_abbr = obj_name[4:9]
    
    print('######################')
    print()
    print(obj_name)
    print()
    
    if obj_abbr=='J0330':
        print('J0330 is not reliable and will not be used here.')
        print()
        continue
    
    SN = 15
    mass_model='mfl'
    anisotropy='om'
    geometry='sph'
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    theta_E = lens_models[lens_models['obj_name']==obj_name]['theta_E'].to_numpy()[0]
    cosmo = cosmo
    distance = cosmo.angular_diameter_distance(zlens).to_numpy()[0]
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    
    # load in the 1D kinematics
    Vrms_1d = pd.read_csv(f'{milestone_dir}{obj_name}/{obj_name}_kinmap.csv')
    Vrms = Vrms_1d['binned_Vrms'].to_numpy()
    rbin = np.mean(Vrms_1d[['bin_inner_edge','bin_outer_edge']].to_numpy(), axis=1)
    Vrms_cov = np.loadtxt(f'{milestone_dir}{obj_name}/{obj_name}_kinmap_cov.csv', delimiter=',', dtype=float)
    dVrms = np.sqrt(Vrms_cov.diagonal())
    
    if obj_abbr=='J0330':
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
        
    surf_lum, sigma_lum, _, _, _, _, _, _, _, _, reff = prepare_to_jam(obj_name, file_dir, SN)

    # Starting guesses
    ani0 = 0.5          # Anisotropy ratio sigma_z/sigma_R
    
    normpsf= 1  
    p0 = [ani0]
    bounds = [[0.01], [2.0]]

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum,
              'distance': distance, 'rbin':rbin, 
              'sigmapsf': sigmapsf, 'normpsf': normpsf, 
              'rms': Vrms, 'erms': dVrms, 
              'pixsize': kcwi_scale,#pixsize,
              'plot': 0, 'return_mge': False}

    sol = capfit(jam_resid_mfl_sph_OM, p0, kwargs=kwargs, bounds=bounds, verbose=2, fixed=[0])
    if sol.success==True:
        print()
        print('Success!')
        print()
    else:
        print()
        print('Fit failed to meet convergence criteria.')
        print()
    
    # bestfit
    a_ani = sol.x[0]
    print('Best fit,', a_ani)
    
    # covariance matrix of free parameters
    cov = sol.cov
    a_ani_err = np.sqrt(sol.cov.diagonal()[0])
    
    kwargs['plot'] = 1
    kwargs['return_mge'] = True
    surf_pot, sigma_pot, qobs_pot, chi2 = jam_resid_mfl_sph_OM(sol.x, **kwargs)
    plt.savefig(f'{jam_output_dir}{obj_name}_mfl_om_sphsph_fit.png')
    plt.pause(1)
    
    # take chi2/dof
    #dof = len(Vrms)
    #chi2_red = chi2/dof
    
    # calculate the enclosed mass
    mass_enclosed, gamma_avg, profile_radii, profile_1d = calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius=theta_E, plot=True)
    
    # parameters not fit
    inc = np.nan
    gamma_fit = np.nan
    gamma_err = np.nan
    
    
    print('Mass enclosed, avg gamma', mass_enclosed, gamma_avg)
    
    mfl_om_sphsph_solutions[i] = np.array([inc, a_ani, a_ani_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2])
    mfl_om_sphsph_cov[i] = cov
    mfl_om_sphsph_mass_profiles_r[i] = profile_radii
    mfl_om_sphsph_mass_profiles[i] = profile_1d


# In[98]:

# save with header
sol_header = '#### 14 objects - inc, anis_ratio, anis_ratio_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2'
np.savetxt(f'{jam_output_dir}jam_mfl_om_sphsph_solutions_02072024.txt', mfl_om_sphsph_solutions, header=sol_header) 
np.savetxt(f'{jam_output_dir}jam_mfl_om_sphsph_covariances_02072024.txt', mfl_om_sphsph_cov.reshape(14,1), header='covariance matrix has been reshaped to 2D, reshape back to (14,3,3)') 

profile_header = '#### radius, surface mass density (Msol/pc2), has been reshaped to 2D, reshape back to (14,50,2)'
mfl_om_sphsph_mass_profile_joined = np.dstack((mfl_om_sphsph_mass_profiles_r, mfl_om_sphsph_mass_profiles)).reshape(14, 100)
np.savetxt(f'{jam_output_dir}jam_mfl_om_sphsph_profiles_02072024.txt', mfl_om_sphsph_mass_profile_joined, header=profile_header) 


# ## OM MFL axisymmetric geometry

# In[103]:


for i in range(len(obj_names)):

    obj_name = obj_names[i]
    obj_abbr = obj_name[4:9]
    
    print('######################')
    print()
    print(obj_name)
    print()
    
    SN = 15
    mass_model='MFL'
    anisotropy='om'
    geometry='axi'
    align='sph'
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    theta_E = lens_models[lens_models['obj_name']==obj_name]['theta_E'].to_numpy()[0]
    cosmo = cosmo
    distance = cosmo.angular_diameter_distance(zlens).to_numpy()[0]
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    
    if obj_abbr=='J0330':
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
        
    surf_lum, sigma_lum, qobs_lum, wrong_psf, Vrms, dVrms, V_bin, dV_bin, xbin, ybin, reff = prepare_to_jam(obj_name, file_dir, SN)
    
    goodbins = np.isfinite(Vrms/dVrms)
    
    # cut out the bad bins
    Vrms = Vrms[goodbins]
    dVrms = dVrms[goodbins]
    xbin = xbin[goodbins]
    ybin = ybin[goodbins]
    
    # Starting guesses
    q0 = np.median(qobs_lum)
    ani0 = 0.5          # Anisotropy ratio sigma_z/sigma_R
    normpsf= 1
    
    
    qmin = np.min(qobs_lum)
    p0 = [q0, ani0]
    bounds = [[0.051, 0.01], [qmin, 2.0]]
    goodbins = np.isfinite(xbin)  # Here I fit all bins

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum, 'qobs_lum': qobs_lum,
              'distance': distance, 'xbin': xbin, 'ybin': ybin, 'sigmapsf': sigmapsf,
              'normpsf': normpsf, 'rms': Vrms, 'erms': dVrms, 'pixsize': kcwi_scale,#pixsize,
              'goodbins': goodbins, 'align':align, 'plot': 0, 'return_mge': False}

    sol = capfit(jam_resid_mfl_axi_OM, p0, kwargs=kwargs, bounds=bounds, verbose=2, fixed=[0, 0])
    # bestfit paramters
    q = sol.x[0]
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    a_ani = sol.x[1]
    # covariance matrix of free parameters
    cov = sol.cov
    q_err = np.sqrt(sol.cov.diagonal()[0])
    a_ani_err = np.sqrt(sol.cov.diagonal()[1])
    
    print('Best fit,', a_ani)
    
    kwargs['plot'] = 1
    kwargs['return_mge'] = True
    surf_pot, sigma_pot, qobs_pot, chi2 = jam_resid_mfl_axi_OM(sol.x, **kwargs)
    plt.savefig(f'{jam_output_dir}{obj_name}_mfl_om_axisph_fit.png')
    plt.pause(1)
    
    # take chi2/dof
    #dof = len(Vrms)
    #chi2_red = chi2/dof
    
    # calculate the enclosed mass
    mass_enclosed, gamma_avg, profile_radii, profile_1d = calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius=theta_E, plot=True)
    
    print('Mass enclosed, avg gamma', mass_enclosed, gamma_avg)
    
    # parameters not fit
    #inc = np.nan
    gamma_fit = np.nan
    gamma_err = np.nan
    
    mfl_om_axisph_solutions[i] = np.array([inc, a_ani, a_ani_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2])
    mfl_om_axisph_cov[i] = cov
    mfl_om_axisph_mass_profiles_r[i] = profile_radii
    mfl_om_axisph_mass_profiles[i] = profile_1d


# In[ ]:


# save with header
sol_header = '#### 14 objects - inc, anis_ratio, anis_ratio_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2'
np.savetxt(f'{jam_output_dir}jam_mfl_om_axisph_solutions_02072024.txt', mfl_om_axisph_solutions, header=sol_header) 
np.savetxt(f'{jam_output_dir}jam_mfl_om_axisph_covariances_02072024.txt', mfl_om_axisph_cov.reshape(14,4), header='covariance matrix has been reshaped to 2D, reshape back to (14,3,3)') 

profile_header = '#### radius, surface mass density (Msol/pc2), has been reshaped to 2D, reshape back to (14,50,2)'
mfl_om_axisph_mass_profile_joined = np.dstack((mfl_om_axisph_mass_profiles_r, mfl_om_axisph_mass_profiles)).reshape(14, 100)
np.savetxt(f'{jam_output_dir}jam_mfl_om_axisph_profiles_02072024.txt', mfl_om_axisph_mass_profile_joined, header=profile_header) 





