# 04/05/2022 - Trying out Gaussian MGE on J0037 by modifying fit_ngc4342 from mge_fit_example.py
# from mge_fit examples

################################################################

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from os import path

import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from mgefit.mge_fit_sectors_twist import mge_fit_sectors_twist
from mgefit.sectors_photometry_twist import sectors_photometry_twist
from mgefit.mge_print_contours_twist import mge_print_contours_twist

################################################################


"""
This procedure reproduces Figures 8-9 in Cappellari (2002)
This example illustrates a simple MGE fit to one single HST/WFPC2 image.

"""

# write file path
#file_dir = path.dirname(path.realpath(mgefit.__file__))  # path of mgefit
file_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/CF_mosaics/SDSSJ0037-0942'

file = file_dir + "/KCWI_J0037_icubes_mosaic_0.1457_crop_2Dintegrated.fits"

hdu = fits.open(file)
img = hdu[0].data

# sky and psf?
# autfwhm from one of the exposures. 1.105087 pix
skylev = 0.55   # counts/pixel
img -= skylev   # subtract sky
scale = 0.0455  # arcsec/pixel
minlevel = 0.2  # counts/pixel
ngauss = 12

# Here we use an accurate four gaussians MGE PSF for
# the HST/WFPC2/F814W filter, taken from Table 3 of
# Cappellari et al. (2002, ApJ, 578, 787)

sigmapsf = [0.494, 1.44, 4.71, 13.4]      # In PC1 pixels
normpsf = [0.294, 0.559, 0.0813, 0.0657]  # total(normpsf)=1

# Here we use FIND_GALAXY directly inside the procedure. Usually you may want
# to experiment with different values of the FRACTION keyword, before adopting
# given values of Eps, Ang, Xc, Yc.
plt.clf()
f = find_galaxy(img, fraction=0.04, plot=1)
plt.pause(1)  # Allow plot to appear on the screen

# Perform galaxy photometry
plt.clf()
s = sectors_photometry(img, f.eps, f.theta, f.xpeak, f.ypeak,
                       minlevel=minlevel, plot=1)
plt.pause(1)  # Allow plot to appear on the screen

# Do the actual MGE fit
# *********************** IMPORTANT ***********************************
# For the final publication-quality MGE fit one should include the line
# "from mge_fit_sectors_regularized import mge_fit_sectors_regularized"
# at the top of this file, rename mge_fit_sectors() into
# mge_fit_sectors_regularized() and re-run the procedure.
# See the documentation of mge_fit_sectors_regularized for details.
# *********************************************************************
plt.clf()
m = mge_fit_sectors(s.radius, s.angle, s.counts, f.eps,
                    ngauss=ngauss, sigmapsf=sigmapsf, normpsf=normpsf,
                    scale=scale, plot=1, bulge_disk=0, linear=0)
plt.pause(1)  # Allow plot to appear on the screen

# Show contour plots of the results
plt.clf()
plt.subplot(121)
mge_print_contours(img.clip(minlevel), f.theta, f.xpeak, f.ypeak, m.sol, scale=scale,
                   binning=7, sigmapsf=sigmapsf, normpsf=normpsf, magrange=9)

# Extract the central part of the image to plot at high resolution.
# The MGE is centered to fractional pixel accuracy to ease visual comparson.

n = 50
img = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
plt.subplot(122)
mge_print_contours(img, f.theta, xc, yc, m.sol,
                   sigmapsf=sigmapsf, normpsf=normpsf, scale=scale)
plt.pause(1)  # Allow plot to appear on the screen