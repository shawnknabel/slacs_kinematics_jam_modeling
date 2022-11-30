#!/usr/bin/env python3
"""
    This example illustrates how to fit the mass of a supermassive black hole
    using the JAM package in combination with the AdaMet Bayesian package.

    V1.0.0: Michele Cappellari, Oxford, 04 May 2018
    V1.1.0: Use the new jampy.jam_axi_proj. MC, Oxford, 28 April 2021

"""

from os import path
import matplotlib.pyplot as plt
import numpy as np

# All packages below are available at http://purl.org/cappellari/software
from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield
import jampy as jam_package
from jampy.jam_axi_proj import jam_axi_proj
from adamet.adamet import adamet
from adamet.corner_plot import corner_plot

###############################################################################

def summary_plot(xbin, ybin, goodbins, rms, pars, lnprob, labels, bounds, kwargs):
    """
    Print the best fitting solution with errors.
    Plot the final corner plot with the best fitting JAM model.

    """
    bestfit = pars[np.argmax(lnprob)]  # Best fitting parameters
    perc = np.percentile(pars, [15.86, 84.14], axis=0)
    sig_bestfit = np.squeeze(np.diff(perc, axis=0)/2)   # half of 68% interval

    print("\nBest-fitting parameters and 1sigma errors:")
    for label, best, sig in zip(labels, bestfit, sig_bestfit):
        print(f"   {label} = {best:#.4g} +/- {sig:#.2g}")
        
    # Produce final corner plot wihout trial values and with best fitting JAM
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

def jam_lnprob(pars, surf_lum=None, sigma_lum=None, qobs_lum=None,
              surf_pot=None, sigma_pot=None, qobs_pot=None, dist=None, 
              xbin=None, ybin=None, sigmapsf=None, normpsf=None, goodbins=None, 
              rms=None, erms=None, pixsize=None, plot=True):
                      
    q, beta, mbh, ml = pars
    
    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan(np.sqrt((1 - qmin**2)/(qmin**2 - q**2))))

    # Note: surf_pot is multiplied by ml, while I set the keyword ml=1
    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot*ml, sigma_pot, qobs_pot,
                       inc, mbh, dist, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align='cyl',
                       beta=np.full_like(qobs_lum, beta), data=rms, errors=erms, ml=1)

    # These two lines are just for the final plot
    jam_lnprob.rms_model = jam.model
    jam_lnprob.flux_model = jam.flux

    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = resid @ resid

    return -0.5*chi2  # ln(likelihood) + cost.

##############################################################################

def jam_bh_adamet_example():

    # MGE model of M32 from Table B1 of Cappellari et al. (2006)
    # http://adsabs.harvard.edu/abs/2006MNRAS.366.1126C
    surf = 10**np.array([6.187, 5.774, 5.766, 5.613, 5.311, 4.774, 4.359, 4.087, 3.682, 3.316, 2.744, 1.618])
    sigma = 10**np.array([-1.762, -1.143, -0.839, -0.438, -0.104, 0.232, 0.560, 0.835, 1.160, 1.414, 1.703, 2.249])
    qObs = np.array([0.790, 0.741, 0.786, 0.757, 0.720, 0.724, 0.725, 0.743, 0.751, 0.838, 0.835, 0.720])

    # Read mock kinematics with realistic parameters and noise
    jam_dir = path.dirname(path.realpath(jam_package.__file__))
    xbin, ybin, rms, erms, flux = np.loadtxt(jam_dir + "/examples/jam_bh_mock_kinematics.txt", unpack=True)
    distance = 0.7   # M32 Distance

    # Here assume mass follows light
    surf_lum = surf_pot = surf
    sigma_lum = sigma_pot = sigma
    qobs_lum = qobs_pot = qObs

    # Starting guess, e.g. from previous least-squares fit
    q0 = 0.4
    bh0 = 2.5e6
    ml0 = 1.4
    beta0 = 0.0

    # Typical Adaptive-Optics PSF: narrow core + broad wings
    sigmapsf = [0.04, 0.4]  # sigma PSF in arcsec
    normpsf = [0.7, 0.3]
    pixsize = 0.05

    qmin = np.min(qObs)
    p0 = [q0, beta0, bh0, ml0]
    bounds = [[0.051, -0.4, bh0/1.3, ml0/1.1], [qmin, 0.4, bh0*1.3, ml0*1.1]]
    labels = [r"$q_{\rm min}$", r"$\beta_z$", r"$M_{BH}$", r"$(M/L)_{\rm tot}$"]

    goodbins = np.isfinite(xbin)  # Here we fit all bins

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum, 'qobs_lum': qobs_lum,
              'surf_pot': surf_pot, 'sigma_pot': sigma_pot, 'qobs_pot': qobs_pot,
              'dist': distance, 'xbin': xbin, 'ybin': ybin, 'sigmapsf': sigmapsf,
              'normpsf': normpsf, 'rms': rms, 'erms': erms, 'pixsize': pixsize,
              'goodbins': goodbins, 'plot': 0}

    # This is a rather small number of steps for illustration.
    # But the distribution has already qualitatively converged
    nstep = 1000
    sigpar = np.array([0.1, 0.1, bh0*0.1, ml0*0.02])  # crude estimate of uncertainties

    # The acceptance rate approaches the optimal theoretical value of 28% in 4-dim
    pars, lnprob = adamet(jam_lnprob, p0, sigpar, bounds, nstep, fignum=1,
                          kwargs=kwargs, nprint=nstep/10, labels=labels, seed=2)

    summary_plot(xbin, ybin, goodbins, rms, pars, lnprob, labels, bounds, kwargs)
            
##############################################################################

if __name__ == '__main__':

    # This example takes about 8 min on a 2GHz CPU
    jam_bh_adamet_example()
