#!/usr/bin/env python3
"""
    This example illustrates how to fit a galaxy model with dark matter halo
    using the JAM package in combination with the AdaMet Bayesian package.

    V1.0.0: Michele Cappellari, Oxford, 2022

"""
from os import path
import matplotlib.pyplot as plt
import numpy as np

# All packages below are available at https://pypi.org/user/micappe/

from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield

import jampy as jam_package
from jampy.jam_axi_proj import jam_axi_proj
from jampy.mge_half_light_isophote import mge_half_light_radius
from jampy.mge_radial_mass import mge_radial_mass

from adamet.adamet import adamet
from adamet.corner_plot import corner_plot

from mgefit.mge_fit_1d import mge_fit_1d

###############################################################################

def summary_plot(xbin, ybin, goodbins, rms, pars, lnprob, labels, bounds, kwargs):
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
    chi2 = jam_lnprob(bestfit, **kwargs)  # Compute model at best fit location

    dx = 0.24
    yfac = 0.87
    fig = plt.gcf()

    fig.add_axes([0.69, 0.99 - dx*yfac, dx, dx*yfac])  # left, bottom, xsize, ysize
    rms1 = rms.copy()
    rms1[goodbins] = symmetrize_velfield(xbin[goodbins], ybin[goodbins], rms[goodbins])
    vmin, vmax = np.percentile(rms1[goodbins], [0.5, 99.5])
    plot_velfield(xbin, ybin, rms1, vmin=vmin, vmax=vmax, cmap='viridis', linescolor='w',
                  colorbar=1, label=r"Data $V_{\rm rms}$ (km/s)", flux=jam_lnprob.flux_model)
    plt.tick_params(labelbottom=False)
    plt.ylabel('arcsec')

    fig.add_axes([0.69, 0.98 - 2*dx*yfac, dx, dx*yfac])  # left, bottom, xsize, ysize
    plot_velfield(xbin, ybin, jam_lnprob.rms_model, vmin=vmin, vmax=vmax, cmap='viridis', linescolor='w',
                  colorbar=1, label=r"Model $V_{\rm rms}$ (km/s)", flux=jam_lnprob.flux_model)
    plt.tick_params(labelbottom=False)
    plt.ylabel('arcsec')

###############################################################################

def dark_halo_mge(gamma, rbreak):
    """
    Returns the MGE parameters for a generalized NFW dark halo profile
    https://ui.adsabs.harvard.edu/abs/2001ApJ...555..504W
    - gamma is the inner logarithmic slope (gamma = -1 for NFW)
    - rbreak is the break radius in arcsec

    """
    # The fit is performed in log spaced radii from 1" to 10*rbreak
    n = 300     # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(1, rbreak*10, n)   # logarithmically spaced radii in arcsec
    rho = (r/rbreak)**gamma * (0.5 + 0.5*r/rbreak)**(-gamma - 3)  # rho=1 at r=rbreak
    m = mge_fit_1d(r, rho, ngauss=15, quiet=1, plot=0)

    surf_dm, sigma_dm = m.sol           # Peak surface density and sigma
    qobs_dm = np.ones_like(surf_dm)     # Assume spherical dark halo

    return surf_dm, sigma_dm, qobs_dm

###############################################################################

def total_mass_mge(surf_lum, sigma_lum, qobs_lum, gamma, rbreak, f_dm, inc):
    """
    Combine the MGE from a dark halo and the MGE from the stellar surface
    brightness in such a way to have a given dark matter fractions f_dm
    inside a sphere of radius one half-light radius reff

    """
    surf_dm, sigma_dm, qobs_dm = dark_halo_mge(gamma, rbreak)

    reff = mge_half_light_radius(surf_lum, sigma_lum, qobs_lum)[0]
    stars_lum_re = mge_radial_mass(surf_lum, sigma_lum, qobs_lum, inc, reff)
    dark_mass_re = mge_radial_mass(surf_dm, sigma_dm, qobs_dm, inc, reff)

    # Find the scale factor needed to satisfy the following definition
    # f_dm == dark_mass_re*scale/(stars_lum_re + dark_mass_re*scale)
    scale = (f_dm*stars_lum_re)/(dark_mass_re*(1 - f_dm))

    surf_pot = np.append(surf_lum, surf_dm*scale)   # Msun/pc**2. DM scaled so that f_DM(Re)=f_DM
    sigma_pot = np.append(sigma_lum, sigma_dm)      # Gaussian dispersion in arcsec
    qobs_pot = np.append(qobs_lum, qobs_dm)

    return surf_pot, sigma_pot, qobs_pot

###############################################################################

def jam_lnprob(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, distance=None,
              xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None,
              rms=None, erms=None, pixsize=None, plot=True):
    """
    Return the probability of the model, given the data, assuming constant priors

    """
    q, ratio, f_dm, lg_ml = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy

    # In this example I keep the halo slope and break radius fixed.
    # These parameters could be fitted from good and spatially-extended data.
    gamma = -1                  # Adopt fixed NFW inner halos slope
    rbreak = 20e3               # Adopt fixed halo break radius of 20 kpc (much larger than the data)
    pc = distance*np.pi/0.648   # Constant factor to convert arcsec --> pc
    rbreak /= pc                # Convert the break radius from pc --> arcsec
    mbh = 0                     # Ignore the central black hole
    surf_pot, sigma_pot, qobs_pot = total_mass_mge(surf_lum, sigma_lum, qobs_lum, gamma, rbreak, f_dm, inc)

    # Note: I multiply surf_pot by ml=10**lg_ml, while I set the keyword ml=1
    # Both the stellar and dark matter increase by ml and f_dm is unchanged
    surf_pot *= 10**lg_ml
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align='cyl',
                       beta=beta, data=rms, errors=erms, ml=1)

    # These two lines are just used for the final plot
    jam_lnprob.rms_model = jam.model
    jam_lnprob.flux_model = jam.flux

    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.

##############################################################################

def jam_dark_halo_adamet_example():

    # MGE model of the galaxy M32 from Table B1 of Cappellari et al. (2006)
    # http://adsabs.harvard.edu/abs/2006MNRAS.366.1126C
    surf_lum = 10**np.array([6.187, 5.774, 5.766, 5.613, 5.311, 4.774, 4.359, 4.087, 3.682, 3.316, 2.744, 1.618])
    sigma_lum = 10**np.array([-1.762, -1.143, -0.839, -0.438, -0.104, 0.232, 0.560, 0.835, 1.160, 1.414, 1.703, 2.249])
    qobs_lum = np.array([0.790, 0.741, 0.786, 0.757, 0.720, 0.724, 0.725, 0.743, 0.751, 0.838, 0.835, 0.720])

    # Read mock kinematics with realistic parameters and noise
    jam_dir = path.dirname(path.realpath(jam_package.__file__))
    xbin, ybin, rms, erms, flux = np.loadtxt(jam_dir + "/examples/jam_mock_kinematics_dark_halo.txt", unpack=True)
    distance = 0.7   # M32 Distance

    # The following line tries to *approximately* account for systematic errors.
    # See beginning of Sec.6.1 of Mitzkus+17 for an explanation
    # https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4789M
    erms *= (2*rms.size)**0.25

    # Starting guess, e.g. from a previous least-squares fit
    q0 = 0.55               # Axial ratio of the flattest MGE Gaussian
    ratio0 = 0.9            # Anisotropy ratio sigma_z/sigma_R
    f_dm0 = 0.15            # Dark matter fraction inside a sphere of radius Re
    lg_ml0 = np.log10(1.4)  # I sample the M/L logarithmically

    pixsize = 2             # spaxel size in arcsec (before Voronoi binning)
    sigmapsf = 4/2.355      # sigma PSF in arcsec (=FWHM/2.355)
    normpsf = 1

    # I adjusted the fitting range below after an initial fit which
    # gave me an idea of some suitable ranges for the parameters
    qmin = np.min(qobs_lum)
    p0 = [q0, ratio0, f_dm0, lg_ml0]
    bounds = [[0.051, 0.5, 0, lg_ml0-0.2], [qmin, 1, 0.5, lg_ml0+0.2]]
    labels = [r"$q_{\rm min}$", r"$\sigma_z/\sigma_R$", r"$f_{\rm DM}$", r"$\lg(M_\ast/L)$"]

    goodbins = np.isfinite(xbin)  # Here I fit all bins

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum, 'qobs_lum': qobs_lum,
              'distance': distance, 'xbin': xbin, 'ybin': ybin, 'sigmapsf': sigmapsf,
              'normpsf': normpsf, 'rms': rms, 'erms': erms, 'pixsize': pixsize,
              'goodbins': goodbins, 'plot': 0}

    # This is a rather small number of steps for illustration.
    # But the distribution has already qualitatively converged
    nstep = 1000
    sigpar = np.array([0.15, 0.15, 0.05, 0.05])  # crude estimate of uncertainties

    # The acceptance rate approaches the optimal theoretical value of 28% in 4-dim
    print("Started AdaMet please wait...")
    print("This example takes about 15 min on a 2GHz CPU")
    print("Progress is printed roughly every minute")
    pars, lnprob = adamet(jam_lnprob, p0, sigpar, bounds, nstep, fignum=1,
                          kwargs=kwargs, nprint=nstep/10, labels=labels, seed=2)

    summary_plot(xbin, ybin, goodbins, rms, pars, lnprob, labels, bounds, kwargs)

##############################################################################

if __name__ == '__main__':

    # This example takes about 15 min on a 2GHz CPU
    jam_dark_halo_adamet_example()
