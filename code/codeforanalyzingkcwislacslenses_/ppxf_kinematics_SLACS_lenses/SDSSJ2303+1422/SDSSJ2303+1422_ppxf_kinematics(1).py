#!/usr/bin/env python
##############################################################################
'''
Code for extracting the kinematics information from the
KCWI data with ppxf.

Geoff Chih-Fan Chen, Feb 28 2022 for Shawan Knabel.

Shawn Knabel, Feb 28 2022 editting for my own machine and directories.
03/01/22 - SDSSJ2303+1422
07/12/22 - kinematics only, systematics in other script
'''

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from ppxf.kcwi_util import register_sauron_colormap
from ppxf.kcwi_util import visualization
from ppxf.kcwi_util import get_datacube
from ppxf.kcwi_util import ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift
from ppxf.kcwi_util import find_nearest
from ppxf.kcwi_util import SN_CaHK
from ppxf.kcwi_util import select_region
from ppxf.kcwi_util import voronoi_binning
from ppxf.kcwi_util import get_voronoi_binning_data
from ppxf.kcwi_util import get_velocity_dispersion_deredshift
from ppxf.kcwi_util import kinematics_map
from ppxf.kcwi_util import stellar_type

import pathlib

register_sauron_colormap()


#------------------------------------------------------------------------------

# data directory # local because duplo is down.
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'

# Set 'obj_name', 'z', 'T_exp'
obj_name = 'SDSSJ2303+1422'
obj_abbr = obj_name[4:9] # e.g. J0029
z = 0.155 # lens redshift
T_exp = 1800*3*60 #266 * 60
lens_center_x,lens_center_y = 58, 124

#------------------------------------------------------------------------------

'''
Step 0: input the necessary information of the datacube
'''
#libary directory # chih-fan spelled wrong :)
libary_dir_xshooter = f'{data_dir}xshooter_lib/all_dr2_fits_G789K012/'
library_dir_xshooter_all = f'{data_dir}xshooter_lib/all_dr2_fits/'
    
# object directory
dir = f'{data_dir}mosaics/{obj_name}/'

# make save directory
save_dir = f'{dir}{obj_name}_kinematics/'
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True) 

#KCWI mosaic datacube
name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'

#spectrum from the lens center # using R=1
spectrum_aperture = f'{dir}{obj_abbr}_central_spectrum_R1.fits' 
spectrum_aperture_R2 = f'{dir}{obj_abbr}_central_spectrum_R2.fits'

## R=3600. spectral resolution is ~ 1.42A
FWHM = 1.42 #1.42

FWHM_tem_xshooter = 0.43 #0.43

## initial estimate of the noise
noise = 0.014

# degree of the additive Legendre polynomial in ppxf
degree = 2


'''
Step 1: visualization of the KCWI mosaic datacube
'''
# visualize the entire mosaic data
hdu = fits.open(dir + name + ".fits")
visualization(hdu)
plt.savefig(dir + obj_name + '_mosaic.png')

# cut the datacube at lens center, radius given here
radius_in_pixels = 21

# crop, plot, save
plt.figure()
data_crop = get_datacube(hdu, lens_center_x, lens_center_y, radius_in_pixels)
data_crop.writeto(dir + name + '_crop.fits', overwrite=True)
plt.savefig(dir + obj_name + '_crop.png')


'''
Step 2: obtain the global template of the lensing galaxy
'''

# set parameters for ppxf
wave_min = 320
wave_max = 428
velscale_ratio = 2

# fit center spectrum with templates
plt.figure()
templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, quasar = \
    ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(libary_dir_xshooter,
                                                      degree=degree,                                                   spectrum_aperture=spectrum_aperture,
                                                      wave_min=wave_min,
                                                      wave_max=wave_max,
                                                      velscale_ratio=velscale_ratio,
                                                      z=z,
                                                      noise=noise,
                                                      templates_name='xshooter',
                                                      FWHM=FWHM,
                                                      FWHM_tem=FWHM_tem_xshooter,
                                                      plot=True)

# plot the spectrum and template fits
plt.savefig(save_dir + obj_name + '_global_template_spectrum.png')
plt.show()

# save variables for future use, show number of stars
nTemplates_xshooter = templates.shape[1]
global_temp_xshooter = templates @ pp.weights[:nTemplates_xshooter]
logLam2_xshooter = logLam2
pp_weights_2700 = pp.weights[:nTemplates_xshooter]
print('number of stars that have non-zero contribution =', np.sum((~(
        pp_weights_2700 == 0))*1))


'''
Step3: checking the stellar type in the library
'''

dir_temperture = f'{libary_dir_xshooter}t_eff.xlsx'

# plot temperature
plt.figure()
stellar_type(libary_dir_xshooter, dir_temperture, pp_weights_2700, bins=100)
plt.xlim(4200, 5200)
plt.xlabel('temperature (K)')
plt.show()


'''
Step 4: create the S/N map
'''

hdu_noquasar = fits.open(dir + name + '_crop.fits')
data_no_quasar = hdu_noquasar[0].data

#estimate the noise from the blank sky
noise_from_blank = data_no_quasar[3000:4000, 4-3:4+2,4-3:4+2]
std = np.std(noise_from_blank)
s = np.random.normal(0, std, data_no_quasar.flatten().shape[0])
noise_cube = s.reshape(data_no_quasar.shape)


## in the following, I use the noise spectrum and datacube with no quasar
# light produced in the previous steps to estimate the S/N per AA. Since KCWI
#  is  0.5AA/pixel, I convert the value to S/N per AA. Note that I use only the
# region of CaH&K to estimate the S/N ratio (i.e. 4800AA - 5100AA).
lin_axis_sky = np.linspace(lamRange1[0], lamRange1[1], data_no_quasar.shape[0])
ind_min_SN = find_nearest(lin_axis_sky*(1+z), 4800)
ind_max_SN = find_nearest(lin_axis_sky*(1+z), 5100)

plt.figure()
SN_per_AA, flux_per_AA, sigma_poisson = SN_CaHK(ind_min_SN,ind_max_SN,
                                           data_no_quasar,
                                noise_cube, T_exp)

fits.writeto(dir + name + '_SN_per_AA.fits', SN_per_AA, overwrite=True)
plt.savefig(dir + obj_name + '_SN_per_AA.png')


'''
Step 5: Select the lens regions where they has good S/N to do the voronoi 
binning
'''

## in the following, I select the region where the signal to noise ratio is
# larger than S/N = 1 to do the voronoi binning.

from numpy import unravel_index
SN_y_center, SN_x_center = unravel_index(SN_per_AA.argmax(), SN_per_AA.shape)
max_radius = 50
target_SN = 1.

origin_imaging_data_perAA = np.mean(hdu[0].data[ind_min_SN:ind_max_SN,:,:],
                                 axis=0)*2

# select the region for binning
plt.figure()
select_region(dir, origin_imaging_data_perAA, SN_per_AA,
              SN_x_center,SN_y_center, radius_in_pixels,
              max_radius,
              target_SN, name)
plt.savefig(dir + obj_name + '_selected_region.png')
plt.pause(1)
plt.clf()

## conduct the voronoi binning (produce the map for mapping pixels to bins)
plt.figure()
voronoi_binning(20, dir, name)
plt.tight_layout()
plt.savefig(dir + obj_name + '_voronoi_binning.png')
plt.pause(1)
plt.clf()

## get voronoi_binning_data based on the map
get_voronoi_binning_data(dir, name)


'''
Step 6: measure the kinematics from the voronoi binning data
'''

voronoi_binning_data = fits.getdata(dir +'voronoi_binning_' + name + '_data.fits')

# wavelength range for measuring kinematics
wave_min = 340
wave_max = 430
get_velocity_dispersion_deredshift(degree=degree,
                                   spectrum_aperture=spectrum_aperture,
                                   voronoi_binning_data=voronoi_binning_data,
                                   velscale_ratio=velscale_ratio,
                                   z=z,
                                   noise=noise,
                                   FWHM=FWHM,
                                   FWHM_tem_xshooter=FWHM_tem_xshooter,
                                   dir=dir,
                                   save_dir=save_dir,
                                   libary_dir=libary_dir_xshooter,
                                   global_temp=global_temp_xshooter,
                                   wave_min=wave_min,
                                   wave_max=wave_max,
                                   T_exp=T_exp,
                                   VD_name=None,
                                   plot=False)


'''
Step 7: plot the kinematics measurements.
'''

# velocity dispersion and error, velocity and error
VD_2d, dVD_2d, V_2d, dV_2d = kinematics_map(dir, save_dir, name, radius_in_pixels)

# plot each
plt.figure()
plt.imshow(VD_2d,origin='lower',cmap='sauron')
cbar1 = plt.colorbar()
cbar1.set_label(r'$\sigma$ [km/s]')
plt.savefig(save_dir + obj_name + '_sigma.png')
plt.pause(1)
plt.clf()

plt.figure()
plt.imshow(dVD_2d, origin='lower', cmap='sauron',vmin=0, vmax=40)
cbar2 = plt.colorbar()
cbar2.set_label(r'd$\sigma$ [km/s]')
plt.savefig(save_dir + obj_name + '_delta_sigma.png')
plt.pause(1)
plt.clf()

# mean is bulk velocity, maybe?
mean = np.nanmedian(V_2d)
#
plt.figure()
plt.imshow(V_2d-mean,origin='lower',cmap='sauron',vmin=-100, vmax=100)
cbar3 = plt.colorbar()
cbar3.set_label(r'Vel [km/s]')
plt.title("Velocity map")
plt.savefig(save_dir + obj_name + '_velocity.png')
plt.pause(1)
plt.clf()

plt.figure()
plt.imshow(V_2d-mean,origin='lower',cmap='sauron_r')
cbar4 = plt.colorbar()
cbar4.set_label(r'Vel [km/s]')
plt.title("reversed colormap")
plt.pause(1)
plt.clf()

print('Done. Run systematics now.')
print('###########################################################')


