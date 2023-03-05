#!/usr/bin/env python
##############################################################################
'''
Code for extracting the kinematics information from the
KCWI data with ppxf.

Geoff Chih-Fan Chen, Feb 28 2022 for Shawan Knabel.
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

register_sauron_colormap()


#------------------------------------------------------------------------------


'''
Step 0: input the necessary information of the datacube
'''
#libary directory
libary_dir_xshooter = '/Users/Geoff/anaconda3/envs/py39/lib/python3.9/site' \
                      '-packages/ppxf/all_dr2_fits_G789K012/'
#data directory
dir = "/Volumes/Seagate/drizzlepac_test/drizzle_file_KCWI_J1630/"

#KCWI mosaic datacube
name = "KCWI_J1630_icubes_mosaic_0.1457"

#spectrum from the lens center
spectrum_aperture = dir + 'galaxy.fits'

#redshift of the lens
z = 0.248

## R=3600. spectral resolution is ~ 1.42A
FWHM = 1.42

FWHM_tem_xshooter = 0.43

## initial estimate of the noise
noise = 0.014

# degree of the additive Legendre polynomial in ppxf
degree = 2

# the exposure time in second for the KCWI datasets.
T_exp = 266 * 60


'''
Step 1: visualization of the KCWI mosaic datacube
'''
# visualize the entire mosaic data
hdu = fits.open(dir + name + ".fits")
visualization(hdu)

# cute the datacube centered at 56, 133
lens_center_x,lens_center_y = 59, 130
radius_in_pixels = 21
data_crop = get_datacube(hdu, lens_center_x, lens_center_y, radius_in_pixels)
data_crop.writeto(dir + name + '_crop.fits', overwrite=True)


'''
Step 2: obtain the global template of the lensing galaxy
'''

wave_min = 320
wave_max = 428
velscale_ratio = 2

templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, quasar = \
    ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(libary_dir_xshooter,
                                                      degree=degree,
                                                      spectrum_aperture=spectrum_aperture,
                                                      wave_min=wave_min,
                                                      wave_max=wave_max,
                                                      velscale_ratio=velscale_ratio,
                                                      z=z,
                                                      noise=noise,
                                                      templates_name='xshooter',
                                                      FWHM=FWHM,
                                                      FWHM_tem=FWHM_tem_xshooter,
                                                      plot=True)

plt.show()


nTemplates_xshooter = templates.shape[1]
global_temp_xshooter = templates @ pp.weights[:nTemplates_xshooter]
logLam2_xshooter = logLam2
pp_weights_2700 = pp.weights[:nTemplates_xshooter]
print('number of stars that have non-zero contribution =', np.sum((~(
        pp_weights_2700 == 0))*1))


'''
Step3: checking the stellar type in the library
'''


dir_temperture = '/Users/Geoff/anaconda3/envs/py39/lib/python3.9/site' \
                '-packages/ppxf/all_dr2_fits/t_eff.xlsx'

stellar_type(libary_dir_xshooter, dir_temperture, pp_weights_2700, bins=100)

plt.xlim(4200, 5200)
plt.xlabel('temperature (K)')
plt.show()



'''
Step 4: create the S/N map given "data without quasar" and "noise datacube"
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

SN_per_AA, flux_per_AA, sigma_poisson = SN_CaHK(ind_min_SN,ind_max_SN,
                                           data_no_quasar,
                                noise_cube, T_exp)

fits.writeto(dir + name + '_SN_per_AA.fits', SN_per_AA, overwrite=True)



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

select_region(dir, origin_imaging_data_perAA, SN_per_AA,
              SN_x_center,SN_y_center, radius_in_pixels,
              max_radius,
              target_SN, name)


## conduct the voronoi binning (produce the map for mapping pixels to bins)
voronoi_binning(20, dir, name)
plt.tight_layout()
plt.pause(1)

## get voronoi_binning_data based on the map
get_voronoi_binning_data(dir, name)


'''
Step 6: measure the kinematics from the voronoi binning data
'''

voronoi_binning_data = fits.getdata(dir +'voronoi_binning_' + name + '_data.fits')

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
VD_2d, dVD_2d, V_2d, dV_2d = kinematics_map(dir, name, radius_in_pixels)

plt.imshow(VD_2d,origin='lower',cmap='viridis')
cbar = plt.colorbar()
cbar.set_label(r'$\sigma$ [km/s]')
plt.show()

plt.imshow(dVD_2d, origin='lower', cmap='viridis',vmin=0, vmax=40)
cbar = plt.colorbar()
cbar.set_label(r'd$\sigma$ [km/s]')
plt.show()

mean = np.nanmedian(V_2d)
#
plt.imshow(V_2d-mean,origin='lower',cmap='sauron',vmin=-100, vmax=100)
cbar = plt.colorbar()
cbar.set_label(r'Vel [km/s]')
plt.title("Velocity map")
plt.show()

plt.imshow(V_2d-mean,origin='lower',cmap='sauron_r')
cbar = plt.colorbar()
cbar.set_label(r'Vel [km/s]')
plt.title("reversed colormap")
plt.show()



'''
Step 8: systematics tests
'''
for i in range(0,1):
    #degree
    degree=i
    for w in range(0,1):
        if w==0:
            wave_min = 340
            wave_max = 430
        elif w==1:
            wave_min = 335
            wave_max = 425
        elif w==2:
            wave_min = 330
            wave_max = 420
        else:
            wave_min = 340
            wave_max = 420

        templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, quasar = \
            ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(
                libary_dir_xshooter,
                degree=degree,
                spectrum_aperture=spectrum_aperture,
                wave_min=wave_min,
                wave_max=wave_max,
                velscale_ratio=velscale_ratio,
                z=z,
                noise=noise,
                templates_name='xshooter',
                FWHM=FWHM,
                FWHM_tem=FWHM_tem_xshooter,
                plot=True)

        plt.clf()

        nTemplates_xshooter = templates.shape[1]
        global_temp_xshooter = templates @ pp.weights[
                                           :nTemplates_xshooter]

        VD_name = 'd%sr%s' % (i, w)

        get_velocity_dispersion_deredshift(degree=degree,
                                           spectrum_aperture=spectrum_aperture,
                                           voronoi_binning_data=voronoi_binning_data,
                                           velscale_ratio=velscale_ratio,
                                           z=z,
                                           noise=noise,
                                           FWHM=FWHM,
                                           FWHM_tem_xshooter=FWHM_tem_xshooter,
                                           dir=dir,
                                           libary_dir=libary_dir_xshooter,
                                           global_temp=global_temp_xshooter,
                                           wave_min=wave_min,
                                           wave_max=wave_max,
                                           T_exp=T_exp,
                                           VD_name=None,
                                           plot=False)


        print(degree, wave_max, wave_min, VD_name)

N = voronoi_binning_data.shape[0]

systematics_VD  = np.zeros(shape=(N,0))
systematics_dVD = np.zeros(shape=(N,0))
systematics_V   = np.zeros(shape=(N,0))
systematics_dV  = np.zeros(shape=(N,0))

for i in range(0,1):
    for w in range(0,1):
        VD_name = 'd%sr%s' % (i, w)
        systematics_results = np.loadtxt(dir + 'VD_%s.txt' %
                                        VD_name)
        systematics_V =np.hstack((systematics_V,
                                   systematics_results[:,0:1]))
        systematics_VD =np.hstack((systematics_VD,
                                   systematics_results[:,1:2]))
        systematics_dv =np.hstack((systematics_dV,
                                   systematics_results[:,2:3]))
        systematics_dVD =np.hstack((systematics_dVD,
                                   systematics_results[:,3:4]))

        print(VD_name)


x = np.arange(systematics_VD.shape[1])
Nbin=2
Nstar=0
f, axarr = plt.subplots(Nbin)
for i in range(Nbin):
    y = systematics_VD[i+Nstar]
    dy = systematics_dVD[i+Nstar]
    axarr[i].errorbar(x, y, yerr=dy, fmt='o', color='black',
                      ecolor='lightgray', elinewidth=1, capsize=0)
f.suptitle('bin (%s-%s)'%(Nstar,Nstar+10))
f.set_size_inches(4.5, 6)
plt.show()
