#!/usr/bin/env python
"""
    In this example/test I fit a one dimensional MGE to the density
    of a Hernquist (1990, ApJ, 356, 359; hereafter H90) model.
    I then compute the following quantities for a spherical model:
    1) The circular velocity with MGE_VCIRC;
    2) The projected sigma of an isotropic H90 model;
    3) The projected sigma of a fully tangential H90 model;
    4) The projected sigma of an ansiotropic H90 model with black hole;
    I use both JAM_SPH_RMS and JAM_AXI_PROJ with align='cyl'/'sph';
    The solutions are compared with the analytic results by H90 and
    with the solution presented in Fig.4.20 of Binney & Tremaine (2008).

    - IMPORTANT: One needs the MGE_FIT_1D routine from the `mgefit`
      package available from http://purl.org/cappellari/software.

    V1.0.0: Michele Cappellari, Oxford, 28 November 2008
    V1.1.0: Included test with anisotropy and black holes.
      MC, Oxford, 17 June 2010
    V2.0.0: Translated from IDL into Python. MC, Oxford, 9 April 2014
    V2.0.1: Fixed RuntimeWarning. MC, Oxford, 17 March 2017
    V2.0.2: Changed imports for jam as a package. MC, Oxford, 17 April 2018
    V2.1.0: Use the new jampy.jam_axi_proj. MC, Oxford, 28 April 2021
    V2.2.0 Included align='sph' models and plot legend.
        MC, Oxford, 30 April 2021

"""

import numpy as np
import matplotlib.pyplot as plt

from mgefit.mge_fit_1d import mge_fit_1d
from jampy.mge_vcirc import mge_vcirc
from jampy.jam_axi_proj import jam_axi_proj
from jampy.legacy.jam_sph_rms import jam_sph_rms

#----------------------------------------------------------------------------

def jam_hernquist_model_example():

    M = 1e11          # Total mass in Solar Masses
    a = 1e3           # Break radius in pc
    distance = 16.5   # Assume Virgo distance in Mpc (Mei et al. 2007)
    pc = distance*np.pi/0.648 # Constant factor to convert arcsec --> pc
    G = 0.004301      # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    mbh = 0

    n = 300  # Number of values to sample the H90 profile for the fit
    r = np.logspace(-2.5, 2, n)*a   # logarithmically spaced radii in pc
    rho = M*a/(2*np.pi*r*(r+a)**3)  # Density in Msun/pc**3 (H90 equation 2)

    m = mge_fit_1d(r, rho, ngauss=16, plot=True)
    plt.pause(1)

    surf = m.sol[0]       # Surface density in Msun/pc**2
    sigma = m.sol[1]/pc   # Gaussian dispersion in arcsec
    qObs = np.full_like(surf, 1)         # Assume spherical model
    inc = 90                # Edge-on view
    npix = 100
    rad = np.geomspace(0.01, 50, npix) # desired output radii in arcsec (avoid R=0)
    r = rad*pc

    ################## Circular Velocity #################

    plt.clf()
    plt.subplot(211)
    plt.xlabel('R (arcsec)')
    plt.ylabel(r'$V_{circ}$, $V_{rms}$ (km/s)')
    plt.title('Reproduces projected analytic results by Hernquist (1990)')

    vcirc = mge_vcirc(surf, sigma, qObs, inc, mbh, distance, rad)
    plt.plot(rad, vcirc, 'C4', label='MGE V$_{\\rm circ}$')

    # Compare with analytic result
    #
    vc = np.sqrt(G*M*r)/(r+a) # H90 equation (16)
    plt.plot(rad, vc, 'C0-.', label='H90')
    plt.text(30, 310, '$V_{circ}$')

    #################### Isotropic Vrms ###################

    # Spherical isotropic H90 model
    #
    sigp = jam_sph_rms(surf, sigma, surf, sigma, mbh, distance, rad)[0]
    plt.plot(rad, sigp, 'C1', label='JAM$^{\\rm sph}_{\\rm sph}$')

    # Axisymmetric isotropic model in the spherical limit.
    # This is plotted on the major axis, but the Vrms has circular symmetry
    #
    vrms = jam_axi_proj(surf, sigma, qObs, surf, sigma, qObs,
                        inc, mbh, distance, rad, rad*0, align='cyl').model
    plt.plot(rad, vrms, 'C2--', label='JAM$^{\\rm axi}_{\\rm cyl}$')

    vrms = jam_axi_proj(surf, sigma, qObs, surf, sigma, qObs,
                        inc, mbh, distance, rad, rad*0, align='sph').model
    plt.plot(rad, vrms, 'C3:', label='JAM$^{\\rm axi}_{\\rm sph}$')

    # Analytic surface brightness from H90
    #
    s = r/a
    w = s < 1
    xs = np.hstack([np.arccosh(1/s[w]) / np.sqrt(1 - s[w]**2),    # H90 eq. (33)
                    np.arccos(1/s[~w]) / np.sqrt(s[~w]**2 - 1)])  # H90 eq. (34)
    IR = M*((2 + s**2)*xs - 3) / (2*np.pi*a**2*(1 - s**2)**2)     # H90 eq. (32)

    # Projected second moments of isotropic model from H90
    #
    sigp = np.sqrt(G*M**2/(12*np.pi*a**3*IR) # H90 equation (41)
                 *(0.5/(1 - s**2)**3
                 *(-3*s**2*xs*(8*s**6 - 28*s**4 + 35*s**2 - 20)
                 - 24*s**6 + 68*s**4 - 65*s**2 + 6) - 6*np.pi*s))
    plt.plot(rad, sigp, 'C0-.')
    plt.text(20, 90, 'isotropic $V_{rms}$')

    ################### Anisotropic Vrms ##################

    # Projected second moments for a H90 model with sigma_R=0.
    # This implies beta=-Infinity but I adopt as an approximation
    # below a large negative beta. This explains why te curves do not
    # overlap perfectly.
    #
    beta = np.full_like(surf, -20)
    sigp = jam_sph_rms(surf, sigma, surf, sigma, mbh, distance, rad, beta=beta)[0]
    plt.plot(rad, sigp, 'C1')

    # Axisymmetric anisotropic model in the spherical limit.
    # The spherical Vrms is the quadratic average of major
    # and minor axes of the axisymmetric model.
    #
    vrms_maj = jam_axi_proj(surf, sigma, qObs, surf, sigma, qObs,
                            inc, mbh, distance, rad, rad*0, beta=beta, align='cyl').model
    vrms_min = jam_axi_proj(surf, sigma, qObs, surf, sigma, qObs,
                            inc, mbh, distance, rad*0, rad, beta=beta, align='cyl').model
    vrms_r = jam_axi_proj(surf, sigma, qObs, surf, sigma, qObs,
                          inc, mbh, distance, rad, rad*0, beta=beta, align='sph').model
    plt.plot(rad, np.sqrt((vrms_maj**2 + vrms_min**2)/2), 'C2--')
    plt.plot(rad, vrms_r, 'C3:')

    # Projected second moments of fully tangential model from H90
    #
    sigp = np.sqrt(G*M**2*r**2/(2*np.pi*a**5*IR) # H90 equation (42)
                 *(1./(24*(1 - s**2)**4)
                 *(-xs*(24*s**8 - 108*s**6 + 189*s**4 - 120*s**2 + 120)
                 - 24*s**6 + 92*s**4 - 117*s**2 + 154) + 0.5*np.pi/s))
    plt.plot(rad, sigp, 'C0-.')
    plt.text(20, 200, 'fully tangential $V_{rms}$')

    plt.legend()

    ############### Anisotropic models with Black Hole ###############

    # Reproduces Fig.4.20 of Binney J., & Tremaine S.D., 2008,
    # Galactic Dynamics, 2nd ed., Princeton University Press
    # See https://books.google.co.uk/books?id=6mF4CKxlbLsC&pg=PA352

    plt.subplot(212)
    plt.xlabel('R/a')
    plt.ylabel('$\\sigma\, (G M / a)^{-1/2}$')
    plt.title('Reproduces Fig.4.20 bottom by Binney & Tremaine (2008)')

    cost = np.sqrt(G*M/a)
    rad = np.logspace(-2.3, 1, 50)*a/pc # desired output radii in arcsec
    bhs = np.array([0., 0.002, 0.004])*M
    betas = np.array([-0.51, 0, 0.51]) # Avoids singularity at beta=+/-0.5
    colors = ['C2', 'C1', 'C0']

    # Test the anisotropic jam_axi_proj with Black Hole in the spherical limit
    #
    for beta in betas:
        betaj = np.full_like(surf, beta)
        for bh in bhs:
            sigp = jam_sph_rms(surf, sigma, surf, sigma, bh, distance, rad, beta=betaj)[0]
            plt.semilogx(rad/a*pc, sigp/cost, 'C1')
            vrms = jam_axi_proj(surf, sigma, qObs, surf, sigma, qObs,
                                inc, bh, distance, rad, rad*0, beta=betaj, align='sph').model
            plt.semilogx(rad/a*pc, vrms/cost, 'C3:')

    # Test the isotropic jam_axi_proj with Black Hole in the spherical limit
    #
    for bh in bhs:
        vrms = jam_axi_proj(surf, sigma, qObs, surf, sigma, qObs,
                            inc, bh, distance, rad, rad*0, align='cyl').model
        plt.semilogx(rad/a*pc, vrms/cost, 'C2--')

    plt.axis([0.006, 10, 0, 0.6]) # x0, x1, y0, y1
    plt.tight_layout()

    plt.text(0.03, 0.15,'$\\beta$=-0.5 (tangential)')
    plt.text(.03, 0.35,'$\\beta$=0 (isotropic)')
    plt.text(0.03, 0.5,'$\\beta$=0.5 (radial)')
    plt.pause(1)  # allow the plot to appear in certain situations

#----------------------------------------------------------------------------

if __name__ == '__main__':
    jam_hernquist_model_example()
