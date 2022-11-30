"""
    Copyright (C) 2019-2021, Michele Cappellari

    E-mail: michele.cappellari_at_physics.ox.ac.uk

    Updated versions of the software are available from my web page
    http://purl.org/cappellari/software

CHANGELOG
---------

V1.1.0: MC, Oxford, 16 July 2020
    - Compute both Vrms and LOS velocity.
V1.0.1: MC, Oxford, 21 April 2020
    - Made a separate file
V1.0.0: Michele Cappellari, Oxford, 08 November 2019
    - Written and tested

"""
import numpy as np
import matplotlib.pyplot as plt

from jampy.jam_axi_proj import jam_axi_proj

##############################################################################

def jam_axi_proj_example():
    """
    Usage example for jam_axi_proj.
    It takes about 2s on a 3GHz CPU

    """
    np.random.seed(123)
    xbin, ybin = np.random.uniform(low=[-55, -40], high=[55, 40], size=[1000, 2]).T

    inc = 60.                                                # Assumed galaxy inclination
    r = np.sqrt(xbin**2 + (ybin/np.cos(np.radians(inc)))**2) # Radius in the plane of the disk
    a = 40                                                   # Scale length in arcsec
    vr = 2000*np.sqrt(r)/(r + a)                             # Assumed velocity profile (v_c of Hernquist 1990)
    vel = vr * np.sin(np.radians(inc))*xbin/r                # Projected velocity field
    sig = 8700/(r + a)                                       # Assumed velocity dispersion profile
    rms = np.sqrt(vel**2 + sig**2)                           # Vrms field in km/s

    # Until here I computed some fake input kinematics to fit with JAM.
    # Ina real application, instead of the above lines one will read the
    # measured stellar kinematics, e.g. from integral-field spectroscopy

    surf = np.array([39483., 37158., 30646., 17759., 5955.1, 1203.5, 174.36, 21.105, 2.3599, 0.25493])
    sigma = np.array([0.153, 0.515, 1.58, 4.22, 10, 22.4, 48.8, 105, 227, 525])
    qObs = np.full_like(sigma, 0.57)

    distance = 16.5     # Assume Virgo distance in Mpc (Mei et al. 2007)
    mbh = 1e8           # Black hole mass in solar masses
    beta = np.full_like(surf, 0.2)

    # Below I assume mass follows light, but in a real application one
    # will generally include a dark halo in surf_pot, sigma_pot, qobs_pot.
    # See e.g. Cappellari (2013) for an example
    # https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.1709C

    surf_lum = surf_pot = surf
    sigma_lum = sigma_pot = sigma
    qobs_lum = qobs_pot = qObs

    sigmapsf = [0.6, 1.2]
    normpsf = [0.7, 0.3]
    pixsize = 0.8
    goodbins = r > 10  # Arbitrarily exclude the center to illustrate how to use goodbins

    # I use a loop below, just to higlight the fact that all parameters
    # remain the same for the two JAM calls, except for 'moment' and 'data'
    plt.figure(1)
    for moment, data in zip(['zz', 'z'], [rms, vel]):

        print(" ")
        # The model is by design similar but not identical to the adopted kinematics!
        m = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                         inc, mbh, distance, xbin, ybin, plot=True, data=data,
                         sigmapsf=sigmapsf, normpsf=normpsf, beta=beta, pixsize=pixsize,
                         moment=moment, goodbins=goodbins, align='cyl', ml=None)
        plt.pause(3)
        plt.figure(2)
        surf_pot *= m.ml  # Scale the density by the best fitting M/L from the previous step

##############################################################################

if __name__ == '__main__':

    jam_axi_proj_example()
