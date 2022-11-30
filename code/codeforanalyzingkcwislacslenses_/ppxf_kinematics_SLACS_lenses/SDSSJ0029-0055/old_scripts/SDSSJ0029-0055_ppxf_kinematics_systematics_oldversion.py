#!/usr/bin/env python
##############################################################################
'''
Code for extracting the kinematics information from the
KCWI data with ppxf.

Geoff Chih-Fan Chen, Feb 28 2022 for Shawan Knabel.

Shawn Knabel, Feb 28 2022 editting for my own machine and directories.
03/01/22 - SDSSJ0029-0055
07/12/22 - systematics checks

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

import pathlib # to create directory

register_sauron_colormap()


#------------------------------------------------------------------------------

# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'

# Set 'obj_name', 'z', 'T_exp'
obj_name = 'SDSSJ0029-0055'
obj_abbr = obj_name[4:9] # e.g. J0029
z = 0.227 # lens redshift
T_exp = 1800*5*60 #266 * 60
lens_center_x,lens_center_y = 60, 129

#------------------------------------------------------------------------------

'''
Step 0: input the necessary information of the datacube
'''
#libary directory # chih-fan spelled wrong :)
library_dir_xshooter = f'{data_dir}xshooter_lib/all_dr2_fits_G789K012/'
library_dir_xshooter_all = f'{data_dir}xshooter_lib/all_dr2_fits/'
libraries = [library_dir_xshooter, library_dir_xshooter_all]
    
# object directory
dir = f'{data_dir}mosaics/{obj_name}/'

# make save directory
save_dir = f'{dir}{obj_name}_systematics/'
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True) 

#KCWI mosaic datacube
name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'

#spectrum from the lens center # using R=1
spectrum_aperture = f'{dir}{obj_abbr}_central_spectrum_R1.fits' 
spectrum_aperture_R2 = f'{dir}{obj_abbr}_central_spectrum_R2.fits'
apertures = [spectrum_aperture, spectrum_aperture_R2]

## R=3600. spectral resolution is ~ 1.42A
FWHM = 1.42 #1.42

FWHM_tem_xshooter = 0.43 #0.43

## initial estimate of the noise
noise = 0.014

## velocity scale ratio?
velscale_ratio = 2

## import voronoi binning data
voronoi_binning_data = fits.getdata(dir +'voronoi_binning_' + name + '_data.fits')


'''
Step 8: systematics tests
'''

# check template library, aperture radius, degree, and wavelength ranges for systematics
for l in range(len(libraries)): # 2 libraries
    library = libraries[l]
    for a in range(len(apertures)): # 2 aperture sizes
        aperture = apertures[a]
        for d in range(0,2):
            #degree
            degree=d
            for w in range(0,4):
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
                        library,
                        degree=degree,
                        spectrum_aperture=aperture,
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

                VD_name = f'l{l}a{a}d{d}w{w}'

                get_velocity_dispersion_deredshift(degree=degree,
                                                   spectrum_aperture=aperture,
                                                   voronoi_binning_data=voronoi_binning_data,
                                                   velscale_ratio=velscale_ratio,
                                                   z=z,
                                                   noise=noise,
                                                   FWHM=FWHM,
                                                   FWHM_tem_xshooter=FWHM_tem_xshooter,
                                                   dir=dir,
                                                   save_dir=save_dir,
                                                   libary_dir=library,
                                                   global_temp=global_temp_xshooter,
                                                   wave_min=wave_min,
                                                   wave_max=wave_max,
                                                   T_exp=T_exp,
                                                   VD_name=VD_name,
                                                   plot=False)


                print(degree, wave_max, wave_min, VD_name)

N = voronoi_binning_data.shape[0]

systematics_VD  = np.zeros(shape=(N,0))
systematics_dVD = np.zeros(shape=(N,0))
systematics_V   = np.zeros(shape=(N,0))
systematics_dV  = np.zeros(shape=(N,0))

VD_names = []
# stack them all together
for l in range(len(libraries)):
    for a in range(len(apertures)):
        for d in range(0,2):
            for w in range(0,4):
                VD_name = f'l{l}a{a}d{d}w{w}'
                systematics_results = np.loadtxt(save_dir + 'VD_%s.txt' %
                                                VD_name)
                systematics_V =np.hstack((systematics_V,
                                           systematics_results[:,0:1]))
                systematics_VD =np.hstack((systematics_VD,
                                           systematics_results[:,1:2]))
                systematics_dv =np.hstack((systematics_dV,
                                           systematics_results[:,2:3]))
                systematics_dVD =np.hstack((systematics_dVD,
                                           systematics_results[:,3:4]))
                VD_names.append(VD_name)
                print(VD_name)

# save the systematics
np.savetxt(f'{save_dir}{obj_name}_systematics_VD.txt',
           systematics_VD, delimiter=',')
np.savetxt(f'{save_dir}{obj_name}_systematics_dVD.txt',
           systematics_dVD, delimiter=',')
np.savetxt(f'{save_dir}{obj_name}_systematics_V.txt', 
           systematics_V, delimiter=',')
np.savetxt(f'{save_dir}{obj_name}_systematics_dV.txt',
           systematics_VD, delimiter=',')
        
x = np.arange(systematics_VD.shape[1])
Nbin=10
Nstar=0

for i in range(N):
    if np.mod(i, Nbin) == 0:
        f, axarr = plt.subplots(Nbin, sharex=True)
        Nstar=i
    y = systematics_VD[i]
    dy = systematics_dVD[i]
    axarr[i-Nstar].errorbar(x, y, yerr=dy, fmt='o', color='black',
                      ecolor='lightgray', elinewidth=1, capsize=0)
    if (np.mod(i, Nbin) == 9) or (i == N-1):
        f.suptitle('bin (%s-%s)'%(Nstar+1,Nstar+Nbin))
        f.set_size_inches(12, 18)
        plt.xticks(np.arange(len(VD_names)))
        axarr[Nbin-1].set_xticklabels(VD_names, rotation=90)
        #plt.savefig(f'{save_dir}{obj_name}_{Nstar+1}{Nstar+Nbin}_systematics.png')
        plt.show()
print('Done.')
print('###############################################################')