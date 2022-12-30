'''
07/07/22 - Auxilliary functions for mass and anisotropy models, and running JAM with different combinations of those, as well as plotting devices.
'''

# import general libraries and modules
import numpy as np
from numpy import polyfit
np.set_printoptions(threshold=10000)
import matplotlib.pyplot as plt
#plt.rcParams[figure.figsize] = (8, 6)
import pandas as pd
import warnings
#warnings.filterwarnings( ignore, module = matplotlib..* )
#warnings.filterwarnings( ignore, module = plotbin..* )
import os
from os import path
from datetime import datetime
import pickle

#########
date_time_ = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
#########

# mge fit
import mgefit
from mgefit.mge_fit_1d import mge_fit_1d

# jam
from jampy.jam_axi_proj import jam_axi_proj
from jampy.mge_radial_mass import mge_radial_mass
from plotbin.plot_velfield import plot_velfield
from plotbin.sauron_colormap import register_sauron_colormap
from plotbin.symmetrize_velfield import symmetrize_velfield
from pafit.fit_kinematic_pa import fit_kinematic_pa

# adamet
from adamet.corner_plot import corner_plot

# my packages
from slacs_mge_jampy import make_gaussian

##############################################################################
### ANISOTROPY MODELS
##############################################################################

def osipkov_merritt_model (r, a_ani, r_eff):
    
    '''
    Given anisotropy scale factor (?) and effective radius, caluclates the anisotropy at the given radius r.
    Inputs:
        r - radius for calculation (must have same units as r_eff)
        a_ani - (r_ani/r_eff) ratio of anisotropy radius and effective radius
        r_eff - effective radius of galaxy (must have same units as r_eff
    Outputs:
        Beta - anisotropy at given radius
    '''
    
    Beta = 1 / (a_ani**2 * (r_eff/r)**2 + 1)
    
    return Beta

##############################################################################

def osipkov_merritt_generalized_model (r, a_ani, r_eff, Beta_max):
    
    '''
    Given anisotropy scale factor (?) and effective radius, caluclates the anisotropy at the given radius r, constraining by multiplying the function by a Beta_max, (between 0,1) that prevents it from reaching full radial symmetry
    Inputs:
        r - radius for calculation (must have same units as r_eff)
        a_ani - (r_ani/r_eff) ratio of anisotropy radius and effective radius
        r_eff - effective radius of galaxy (must have same units as r_eff
    Outputs:
        Beta - anisotropy at given radius
    '''
    
    Beta = 1 / (a_ani**2 * (r_eff/r)**2 + 1) * Beta_max
    
    return Beta

##############################################################################

def inner_outer_anisotropy_model (r, a_ani, r_eff, Beta_in, Beta_out):
    
    '''
    Given anisotropy scale factor and effective radius, caluclates the anisotropy at the given radius r. Two-component model, inner and outer, free to be any value. Fit will be to a_ani, Beta_in, and Beta_out
    Inputs:
        r - radius for calculation (must have same units as r_eff)
        a_ani - (r_ani/r_eff) ratio of anisotropy radius (transition from inner to outer) and effective radius
        r_eff - effective radius of galaxy (must have same units as r)
        Beta_in - anisotropy of region within r_ani
        Beta_out - anisotropy of region outside r_ani
    Outputs:
        Beta - anisotropy at given radius
    '''
    
    r_ani = a_ani * r_eff
    
    if r < r_ani:
        Beta = Beta_in
    else:
        Beta = Beta_out
    
    return Beta


##############################################################################
### MASS MODELS
##############################################################################

def nfw_generalized_model (r, gamma, rbreak):
    '''
    Given inner slope gamma and rbreak (transition radius), calculates value for radius r according to generalized NFW dark matter halo profile.
    Inputs:
        r - radius for calculations (same units as rbreak)
        gamma - inner profile slope (logarithmic slope, i.e. slope when plotted as log(rho) to log(r), < 0
                if gamma = -1, this is standard NFW profile
        rbreak - break radius, transition from inner slope to outer slope (of -3)
    Outputs:
        rho - mass density at the given radius
    '''
    rho = (r/rbreak)**gamma * (0.5 + 0.5*r/rbreak)**(-gamma - 3)  # rho=1 at r=rbreak
    
    return rho


###############################################################################

def dark_halo_mge (gamma, rbreak, plot=False):
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
    r = np.geomspace(0.01, rbreak*10, n)   # logarithmically spaced radii in arcsec
    rho = nfw_generalized_model (r, gamma, rbreak)
    m = mge_fit_1d(r, rho, ngauss=15, quiet=1, plot=plot)

    surf_dm, sigma_dm = m.sol           # Peak surface density and sigma
    qobs_dm = np.ones_like(surf_dm)     # Assume spherical dark halo

    return surf_dm, sigma_dm, qobs_dm


###############################################################################

def power_law_mge (gamma, rbreak, plot=False):
    """
    Haven't quite worked this one out yet...
    """
    # The fit is performed in log spaced radii from 1" to 10*rbreak
    n = 1000     # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(0.01, rbreak*10, n)   # logarithmically spaced radii in arcsec
    #rho = (r/rbreak)**gamma * (0.5 + 0.5*r/rbreak)**(-gamma - 3)  # rho=1 at r=rbreak
    ### Michele said to make rbreak large and fit only gamma? But that isn't a power law...
    rho = (r/rbreak)**gamma
    
    m = mge_fit_1d(r, rho, ngauss=15, quiet=1, plot=plot)

    surf_pot, sigma_pot = m.sol           # Peak surface density and sigma
    qobs_pot = np.ones_like(surf_pot)     # spherical??? is this valid?

    return surf_pot, sigma_pot, qobs_pot


###############################################################################

def total_mass_mge (surf_lum, sigma_lum, qobs_lum, reff, gamma, f_dm, inc, lg_ml, model, plot=False):
    """
    Combine the MGE from a dark halo and the MGE from the stellar surface
    brightness in such a way to have a given dark matter fractions f_dm
    inside a sphere of radius one half-light radius reff

    """
    
    if model == 'nfw':
        
        gamma = -1
        rbreak = 5*reff # much bigger than data # should this be a free parameter?

        surf_dm, sigma_dm, qobs_dm = dark_halo_mge(gamma, rbreak, plot)

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
        rbreak = 5*reff # much bigger than data

        surf_dm, sigma_dm, qobs_dm = dark_halo_mge(gamma, rbreak, plot)

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
        rbreak = 5*reff # much bigger than data
        
        surf_pot, sigma_pot, qobs_pot = power_law_mge(gamma, rbreak, plot)
        
        # Note: I multiply surf_pot by ml=10**lg_ml, while I set the keyword ml=1
        # Both the stellar and dark matter increase by ml and f_dm is unchanged
        surf_pot *= 10**lg_ml


    return surf_pot, sigma_pot, qobs_pot

##############################################################################
### JAM MODELS
##############################################################################

def jam_lnprob (pars, surf_lum=None, sigma_lum=None, qobs_lum=None, distance=None,
              xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None,
               rms=None, erms=None, pixsize=None, reff=None, plot=True, 
                         model=None, anisotropy=None, align=None):
    
    """
    Return the probability of the model, given the data, assuming constant priors

    """
    #gamma, q, ratio, f_dm, lg_ml = pars
    q, ratio, f_dm, lg_ml = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy

    surf_pot, sigma_pot, qobs_pot = total_mass_mge(surf_lum, sigma_lum, qobs_lum, reff,
                                                   gamma, f_dm, inc, lg_ml, model, plot=plot)
    
    # ignore central black hole
    mbh=0.
    
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=1, nodots=True)

    # These two lines are just used for the final plot
    jam_lnprob.rms_model = jam.model
    jam_lnprob.flux_model = jam.flux
    
     # Stack the surf_potential and save it for later 
    jam_lnprob.surf_potential = np.stack((surf_pot, sigma_pot, qobs_pot))

    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.

###############################################################################
# Constant Anisotropy

def jam_lnprob_nfw_constbeta(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, distance=None,
                              xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None,
                               rms=None, erms=None, pixsize=None, reff=None, plot=True, 
                             model='nfw', anisotropy=None, align='sph'):
    """
    Return the probability of the model, given the data, assuming constant priors

    """
    #gamma, q, ratio, f_dm, lg_ml = pars
    q, ratio, f_dm, lg_ml = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy
    
    # gamma = -1 for NFW
    gamma = -1

    surf_pot, sigma_pot, qobs_pot = total_mass_mge(surf_lum, sigma_lum, qobs_lum, reff,
                                                   gamma, f_dm, inc, lg_ml, model, plot=plot)
    
    # ignore central black hole
    mbh=0.
    
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=1, nodots=True)

    # These two lines are just used for the final plot
    jam_lnprob.rms_model = jam.model
    jam_lnprob.flux_model = jam.flux
    
    # Stack the surf_potential and save it for later 
    jam_lnprob.surf_potential = np.stack((surf_pot, sigma_pot, qobs_pot))

    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.

###############################################################################

def jam_lnprob_nfwgen_constbeta(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, distance=None,
                              xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None,
                               rms=None, erms=None, pixsize=None, reff=None, plot=True, 
                             model='nfw', anisotropy=None, align='sph'):
    """
    Return the probability of the model, given the data, assuming constant priors

    """
    gamma, q, ratio, f_dm, lg_ml = pars
    #q, ratio, f_dm, lg_ml = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy

    surf_pot, sigma_pot, qobs_pot = total_mass_mge(surf_lum, sigma_lum, qobs_lum, reff,
                                                   gamma, f_dm, inc, lg_ml, model, plot=plot)
    
    # ignore central black hole
    mbh=0.
    
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=1, nodots=True)

    # These two lines are just used for the final plot
    jam_lnprob.rms_model = jam.model
    jam_lnprob.flux_model = jam.flux
    
    # Stack the surf_potential and save it for later 
    jam_lnprob.surf_potential = np.stack((surf_pot, sigma_pot, qobs_pot))

    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.

###############################################################################
# Osipkov-Merritt Anisotropy

def jam_lnprob_nfw_om(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, distance=None,
                      xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None,
                       rms=None, erms=None, pixsize=None, reff=None, plot=True, 
                         model='nfw', anisotropy=None, align='sph'):
    """
    Return the probability of the model, given the data, assuming constant priors

    """
    #gamma, q, ratio, f_dm, lg_ml = pars
    q, a_ani, f_dm, lg_ml = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    #beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy
    # create array of Beta values for each Gaussian k
    beta = np.zeros(len(qobs_lum))
    # calculate Beta at each sigma
    for i in range(len(sigma)):
            r = sigma[i]
            beta[i] = osipkov_merritt_model(r, a_ani, r_eff)

    surf_pot, sigma_pot, qobs_pot = total_mass_mge(surf_lum, sigma_lum, qobs_lum, reff,
                                                   gamma, f_dm, inc, lg_ml, model, plot=plot)
    
    # ignore central black hole
    mbh=0.
    
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=1, nodots=True)

    # These two lines are just used for the final plot
    jam_lnprob.rms_model = jam.model
    jam_lnprob.flux_model = jam.flux
    
    # Stack the surf_potential and save it for later 
    jam_lnprob.surf_potential = np.stack((surf_pot, sigma_pot, qobs_pot))

    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.


###############################################################################

def jam_lnprob_nfwgen_om(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, distance=None,
                              xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None,
                               rms=None, erms=None, pixsize=None, reff=None, plot=True, 
                         model='nfw', anisotropy=None, align='sph'):
    """
    Return the probability of the model, given the data, assuming constant priors

    """
    gamma, q, a_ani, f_dm, lg_ml = pars
    #q, ratio, f_dm, lg_ml = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    #beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy
    # create array of Beta values for each Gaussian k
    beta = np.zeros(len(qobs_lum))
    # calculate Beta at each sigma
    for i in range(len(sigma)):
            r = sigma[i]
            beta[i] = osipkov_merritt_model(r, a_ani, r_eff)

    surf_pot, sigma_pot, qobs_pot = total_mass_mge(surf_lum, sigma_lum, qobs_lum, reff,
                                                   gamma, f_dm, inc, lg_ml, model, plot=plot)
    
    # ignore central black hole
    mbh=0.
    
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=1, nodots=True)

    # These two lines are just used for the final plot
    jam_lnprob.rms_model = jam.model
    jam_lnprob.flux_model = jam.flux
    
    # Stack the surf_potential and save it for later 
    jam_lnprob.surf_potential = np.stack((surf_pot, sigma_pot, qobs_pot))

    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.

###############################################################################
# Generalized Osipkov-Merritt Anisotropy

def jam_lnprob_nfw_omgen(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, distance=None,
                          xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None,
                           rms=None, erms=None, pixsize=None, reff=None, plot=True, 
                         model='nfw', anistropy=None, align='sph'):
    """
    Return the probability of the model, given the data, assuming constant priors

    """
    #gamma, q, ratio, f_dm, lg_ml = pars
    q, a_ani, Beta_in, Beta_out, f_dm, lg_ml = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    #beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy
    # create array of Beta values for each Gaussian k
    beta = np.zeros(len(qobs_lum))
    # calculate Beta at each sigma
    for i in range(len(sigma)):
            r = sigma[i]
            beta[i] = osipkov_merritt_generalized_model(r, a_ani, r_eff, Beta_in, Beta_out)

    surf_pot, sigma_pot, qobs_pot = total_mass_mge(surf_lum, sigma_lum, qobs_lum, reff,
                                                   gamma, f_dm, inc, lg_ml, model, plot=plot)
    
    # ignore central black hole
    mbh=0.
    
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=1, nodots=True)

    # These two lines are just used for the final plot
    jam_lnprob.rms_model = jam.model
    jam_lnprob.flux_model = jam.flux
    
    # Stack the surf_potential and save it for later 
    jam_lnprob.surf_potential = np.stack((surf_pot, sigma_pot, qobs_pot))

    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.


###############################################################################

def jam_lnprob_nfwgen_om(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, distance=None,
                              xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None,
                               rms=None, erms=None, pixsize=None, reff=None, plot=True, 
                         model='nfw', anisotropy=None, align='sph'):
    """
    Return the probability of the model, given the data, assuming constant priors

    """
    gamma, q, a_ani, Beta_in, Beta_out, f_dm, lg_ml = pars
    #q, ratio, f_dm, lg_ml = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    #beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy
    # create array of Beta values for each Gaussian k
    beta = np.zeros(len(qobs_lum))
    # calculate Beta at each sigma
    for i in range(len(sigma)):
            r = sigma[i]
            beta[i] = osipkov_merritt_generalized_model(r, a_ani, r_eff, Beta_in, Beta_out)

    surf_pot, sigma_pot, qobs_pot = total_mass_mge(surf_lum, sigma_lum, qobs_lum, reff,
                                                   gamma, f_dm, inc, lg_ml, model, plot=plot)
    
    # ignore central black hole
    mbh=0.
    
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=1, nodots=True)

    # These two lines are just used for the final plot
    jam_lnprob.rms_model = jam.model
    jam_lnprob.flux_model = jam.flux
    
    # Stack the surf_potential and save it for later 
    jam_lnprob.surf_potential = np.stack((surf_pot, sigma_pot, qobs_pot))

    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.


###############################################################################


def summary_plot(xbin, ybin, goodbins, rms, pars, lnprob, labels, bounds, kwargs, obj_name, date_time, model_dir, jam_prob_func, save=False):
    """
    Print the best fitting solution with uncertainties.
    Plot the final corner plot with the best fitting JAM model.

    """
    
    bestfit = pars[np.argmax(lnprob)]  # Best fitting parameters
    perc = np.percentile(pars, [15.86, 84.14], axis=0)  # 68% interval
    sig_bestfit = np.squeeze(np.diff(perc, axis=0)/2)   # half of interval (1sigma)

    print("\nBest-fitting parameters and 1sigma errors:")
    for label, best, sig in zip(labels, bestfit, sig_bestfit):
        print(f"   {label} = {best:#.4g} +/- {sig:#.2g}")
        
    # Produce final corner plot without trial values and with best fitting JAM
    plt.clf()
    corner_plot(pars, lnprob, labels=labels, extents=bounds, fignum=1)
    chi2 = jam_prob_func(bestfit, **kwargs)  # Compute model at best fit location
    
    # save variables, surf_pot, sigma_pot, qobs_pot, rms_model and flux_model
    surf_potential = jam_lnprob.surf_potential
    rms_model = jam_lnprob.rms_model
    flux_model = jam_lnprob.flux_model

    dx = 0.24
    yfac = 0.87
    fig = plt.gcf()
    fig.set_size_inches((12,12))
    fig.tight_layout()

    fig.add_axes([0.69, 0.99 - dx*yfac, dx, dx*yfac])  # left, bottom, xsize, ysize
    rms1 = rms.copy()
    rms1[goodbins] = symmetrize_velfield(xbin[goodbins], ybin[goodbins], rms[goodbins])
    vmin, vmax = np.percentile(rms1[goodbins], [0.5, 99.5])
    plot_velfield(xbin, ybin, rms1, vmin=vmin, vmax=vmax, linescolor='w', 
                  colorbar=1, label=r"Data $V_{\rm rms}$ (km/s)", flux=jam_lnprob.flux_model, nodots=True)
    plt.tick_params(labelbottom=False)
    plt.ylabel('arcsec')

    fig.add_axes([0.69, 0.98 - 2*dx*yfac, dx, dx*yfac])  # left, bottom, xsize, ysize
    plot_velfield(xbin, ybin, rms_model, vmin=vmin, vmax=vmax, linescolor='w',
                  colorbar=1, label=r"Model $V_{\rm rms}$ (km/s)", flux=flux_model, nodots=True)
    plt.tick_params(labelbottom=False)
    plt.ylabel('arcsec')
    if save==True:
        plt.savefig(f'{model_dir}{obj_name}corner_plot{date_time}.png')
        plt.savefig(f'{model_dir}{obj_name}corner_plot{date_time}.pdf')
        
    return surf_potential, rms_model, flux_model


##############################################################################

def save_fit_parameters(model_dir, obj_name, date_time, pars, lnprob, p0, sigpar, bounds, labels, surf_potential, rms_model, flux_model, kwargs):
    
    # save fit parameters
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_parameters_fit.txt', pars)
    # save likelihoods
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_likelihood.txt', lnprob) 
    # save initial parameters
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_initial_parameters.txt',p0)           
    # save initial error estimates
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_initial_error_estimates.txt',sigpar)
    # save bounds
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_bounds.txt', bounds)
    # save labels
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_labels.txt', labels, fmt='%s')
    # save surface potential
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_surf_potential.txt', surf_potential)
    # save rms_model
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_rms_model.txt', rms_model)
    # save flux_model
    np.savetxt(f'{model_dir}{obj_name}_{date_time}_flux_model.txt', flux_model)
    # create a binary pickle file 
    f = open(f"{model_dir}{obj_name}_{date_time}_kwargs.pkl","wb")
    # write the python object (dict) to pickle file
    pickle.dump(kwargs,f)
    # close file
    f.close()
    
    print(f'Parameters saved to directory {model_dir}')


##############################################################################    

def get_power_law_slope (surf_potential, reff, surf_lum, obj_name, model_dir, date_time, save=False):
    
    surf_pot, sigma_pot, qobs_pot = surf_potential
    rbreak = 5*reff
    
    n = 1000  # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(10**(-2), 10*rbreak, n)

    plt.figure()

    min_maus = 0 

    total_mass_profile = np.zeros(len(r))
    stellar_only = np.zeros(len(r))
    dark_only = np.zeros(len(r))

    for i in range(len(surf_pot)):
        gauss = make_gaussian(r, surf_pot[i], sigma_pot[i], qobs_pot[i])
        if np.min(gauss) > min_maus:
            min_maus = np.min(gauss)
        if i < len(surf_lum):
            color = 'r'
            stellar_only += gauss
        else:
            color='k'
            dark_only += gauss
        plt.loglog(r, gauss, color=color) # plot the gaussian
        total_mass_profile += gauss
    
    # labeling purposes
    plt.plot(r, np.zeros(len(r)), 'r', label='stellar') # just to add labels and not have it loop through all
    plt.plot(r, np.zeros(len(r)), 'k', label='dark')
    
    # plot the total mass, stellar, and dark profiles from Gaussians
    plt.loglog(r, total_mass_profile, linestyle='--', color='b', label='total')
    plt.loglog(r, stellar_only, linestyle='-.',color='r')
    plt.loglog(r, dark_only, linestyle='-.',color='k')
    
    # fit for power law slope
    power_law_slope, intercept = polyfit(np.log10(r), np.log10(total_mass_profile), deg=1)
    power_law = r**power_law_slope * 10**intercept
    max_maus = np.max(power_law)
    
    # plot the power law
    plt.loglog(r, power_law, linewidth=4, color='b', label=f'slope = {np.around(power_law_slope, 3)}')

    plt.ylim([min_maus,10**25.1])
    plt.xlim([10**(-1.2),np.max(r)])
    plt.xlabel('arcsec')
    plt.ylabel('mass density')

    plt.legend()

    plt.title(f'{obj_name} Mass Model {date_time}')
    if save == True:
        plt.savefig(f'{model_dir}{obj_name}_{date_time}_mass_profile.png')
        plt.savefig(f'{model_dir}{obj_name}_{date_time}_mass_profile.pdf')
    
    return power_law_slope, intercept


    ### But how do I circularize the ellipticals

##############################################################################