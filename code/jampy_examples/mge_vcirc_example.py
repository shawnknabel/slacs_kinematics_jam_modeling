#!/usr/bin/env python

"""
Computes the circular velocity for the MGE approximation of the
axisymmetric Satoh (1980) model given in Table 1 of Cappellari (2020)
https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C
and compares it with the analytic result.

V1.0.0: Michele Cappellari, Oxford, 27 July 2020

"""

from os import path
import numpy as np
import matplotlib.pyplot as plt

import jampy as jampy_package
from jampy.mge_vcirc import mge_vcirc

def mge_vcirc_example():
    """
    Usage example for mge_vcirc()
    It takes a fraction of a second on a 2GHz computer
    
    """
    inc = 60.  # Inclination in degrees
    distance = 0.648/np.pi  # Mpc (adopt distance where 1" = 1pc)
    rad = np.geomspace(0.1, 100, 30)  # Radii in arscec where Vcirc has to be computed

    # Compute a projected MGE from the intrinsic one in Table 1 of Cappellari (2020)
    jampy_dir = path.dirname(path.realpath(jampy_package.__file__))
    lg_dens, lg_sigma, qintr = np.loadtxt(jampy_dir + '/examples/cappellari2020_table1.txt', unpack=True)
    dens, sigma, inc_rad = 10**lg_dens, 10**lg_sigma, np.radians(inc)
    qobs = np.sqrt((qintr*np.sin(inc_rad))**2 + np.cos(inc_rad)**2)  # Eq.(35) Cappellari (2020)
    surf = np.sqrt(2*np.pi)*dens*qintr*sigma/qobs                    # Eq.(38) Cappellari (2020)

    G = 0.004301  # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    mbh = 0.01        # Assume a BH 1% of the galaxy mass M=1
    vtrue = np.sqrt(G*rad**2/(3 + rad**2)**1.5 + G*mbh/rad)  # Analytic V_c for M = a = b = 1
    vcirc = mge_vcirc(surf, sigma, qobs, inc, mbh, distance, rad)

    plt.plot(np.log10(rad), vcirc, '-', label="Analytic")
    plt.plot(np.log10(rad), vcirc, 'o', label="MGE approximation")
    plt.xlabel('lg R (arcsec)')
    plt.ylabel(r'$V_{circ}$ (km/s)')
    plt.title(r"Circular velocity of Satoh model with $M=1$ and $M_{\rm BH}=M/100$")
    plt.legend(loc=3)

#----------------------------------------------------------------------

if __name__ == '__main__':
    
    plt.clf()
    mge_vcirc_example()
    plt.pause(1)
