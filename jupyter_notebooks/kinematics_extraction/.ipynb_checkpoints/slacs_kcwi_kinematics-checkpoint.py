from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pathlib # to create directory

from ppxf.ppxf import ppxf
from pathlib import Path
from scipy import ndimage
from urllib import request
from scipy import ndimage
from time import perf_counter as clock
from scipy import interpolate
from astropy.visualization import simple_norm
import astropy.units as u
from vorbin.voronoi_2d_binning import voronoi_2d_binning

# my functions
import sys
sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages/ppxf_kcwi_util_022423")
#register_sauron_colormap()
from templates_util import Xshooter
from kcwi_util import register_sauron_colormap
from kcwi_util import poisson_noise
from kcwi_util import find_nearest
from kcwi_util import SN_CaHK
register_sauron_colormap()

import ppxf.ppxf_util as ppxf_util
from os import path
ppxf_dir = path.dirname(path.realpath(ppxf_util.__file__))
import ppxf.sps_util as sps_util

c = 299792.458 # km/s

##############

class slacs_kcwi_kinematics:
    
    def __init__(self,
                 mos_dir,
                 kin_dir,
                 obj_name,
                 kcwi_datacube_file,
                 central_spectrum_file,
                 background_spectrum_file,
                 zlens,
                 exp_time,
                 lens_center_x,
                 lens_center_y,
                 aperture,
                 wave_min,
                 wave_max,
                 degree,
                 sps_name,
                 pixel_scale,
                 FWHM,
                 noise,
                 velscale_ratio,
                 radius_in_pixels,
                 SN,
                 plot,
                 quiet):
        
        self.mos_dir=mos_dir
        self.kin_dir=kin_dir
        self.obj_name=obj_name
        self.kcwi_datacube_file=kcwi_datacube_file
        self.central_spectrum_file=central_spectrum_file
        self.background_spectrum_file=background_spectrum_file
        self.zlens=zlens
        self.exp_time=exp_time
        self.lens_center_x=lens_center_x
        self.lens_center_y=lens_center_y
        self.aperture=aperture
        self.wave_min=wave_min
        self.wave_max=wave_max
        self.degree=degree
        self.sps_name=sps_name
        self.pixel_scale=pixel_scale
        self.FWHM=FWHM
        self.noise=noise
        self.velscale_ratio=velscale_ratio
        self.radius_in_pixels=radius_in_pixels
        self.SN=SN
        self.plot=plot
        self.quiet=plot    
        
###########################################################

    def run_slacs_kcwi_kinematics(self, plot_bin_fits=False):
        # to run all the steps at ones
        print(f'pPXF will now consume your soul and use it to measure the kinematics of {self.obj_name}.')
        self.datacube_visualization()
        self.log_rebin_central_spectrum()
        self.log_rebin_background_spectrum()
        self.get_templates()
        self.set_up_mask()
        self.ppxf_central_spectrum()
        self.crop_datacube()
        self.create_SN_map()
        self.select_region()
        self.voronoi_binning()
        self.ppxf_bin_spectra(plot_bin_fits=plot_bin_fits)
        self.make_kinematic_maps()
        self.plot_kinematic_maps()
        print("Job's finished!")
        
    def datacube_visualization(self):
        '''
        :param datacube: datacube, cropped or not
        :return: no return. This function is just for visualization
        '''
        # visualize the entire mosaic data
        data_hdu = fits.open(self.kcwi_datacube_file)
        datacube=data_hdu[0].data
        data_hdu.close()
        norm = simple_norm(np.nansum(datacube, axis=0), 'sqrt')
        plt.imshow(np.nansum(datacube, axis=0), origin="lower", norm=norm)
        plt.title('KCWI data')
        plt.colorbar(label='flux')
        plt.pause(1)

    def log_rebin_central_spectrum(self):
        
        hdu = fits.open(self.central_spectrum_file)
        gal_lin = hdu[0].data
        h1 = hdu[0].header
        lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1'] - 1)])
        
        # restframe wavelengths and rebin
        self.rest_wave_range = lamRange1/(1+self.zlens) # Compute approximate restframe wavelength range
        self.rest_FWHM = self.FWHM/(1+self.zlens)   # Adjust resolution in Angstrom
        self.central_spectrum, self.rest_wave_log, self.central_velscale = ppxf_util.log_rebin(self.rest_wave_range, gal_lin)
        self.rest_wave = np.exp(self.rest_wave_log)
        
    def log_rebin_background_spectrum(self):
        # for now, no bakcground spectrum, just return None
        self.background_spectrum = None
        
    def get_templates(self):
        
        # take wavelength range of templates to be slightly larger than that of the galaxy restframe
        self.wave_range_templates = self.rest_wave_range[0]/1.2, self.rest_wave_range[1]*1.2
        
        # bring in the tempaltes
        basename = f"spectra_{self.sps_name}_9.0.npz"
        filename = path.join(ppxf_dir, 'sps_models', basename)
        sps = sps_util.sps_lib(filename, 
                               self.central_velscale/self.velscale_ratio, 
                               self.rest_FWHM, 
                               wave_range=self.wave_range_templates)
        templates= sps.templates
        self.templates = templates.reshape(templates.shape[0], -1)
        self.templates_wave = sps.lam_temp

    def set_up_mask(self):
        
        # after de-redshift, the initial redshift is zero.
        goodPixels = ppxf_util.determine_goodpixels(self.rest_wave_log, self.wave_range_templates, 0)
        
        # find the indices of the restframe wavelengths that are closest to the min and max I want
        ind_min = find_nearest(self.rest_wave, self.wave_min)
        ind_max = find_nearest(self.rest_wave, self.wave_max)

        mask=goodPixels[goodPixels<ind_max]
        mask = mask[mask>ind_min]

        boolen = ~((2956 < mask) & (mask < 2983))  # mask the Mg II
        mask = mask[boolen]
        boolen = ~((2983 < mask) & (mask < 3001))  # mask the Mg II
        self.mask = mask[boolen]
    
    def ppxf_central_spectrum(self):
        # Here the actual fit starts. The best fit is plotted on the screen.
        # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
        #
        vel = c*np.log(1 + 0)   # eq.(8) of Cappellari (2017)
        start = [vel, 250.]  # (km/s), starting guess for [V, sigma]
        #bounds = [[-500, 500],[50, 450]]
        t = clock()

        # create a noise
        noise_array = np.full_like(self.central_spectrum, self.noise) # Assume constant noise per
        
        pp = ppxf(self.templates, 
                  self.central_spectrum, 
                  noise_array, 
                  self.central_velscale, 
                  start, 
                  plot=True,
                  moments=2, 
                  goodpixels=self.mask,
                  degree=self.degree,
                  velscale_ratio=self.velscale_ratio,
                  sky=self.background_spectrum, 
                  lam=self.rest_wave,
                  lam_temp=self.templates_wave,
                 )
        plt.xlim(self.wave_min*1e-4, self.wave_max*1e-4)
        plt.ylim(np.nanmin(self.central_spectrum[self.mask])/1.1, np.nanmax(self.central_spectrum[self.mask])*1.1)
        plt.pause(1)

        print("Formal errors:")
        print("     dV    dsigma   dh3      dh4")
        print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

        print('Elapsed time in pPXF: %.2f s' % (clock() - t))
        
        self.central_spectrum_ppxf = pp
        self.nTemplates = pp.templates.shape[1]
        self.global_template = pp.templates @ pp.weights[:self.nTemplates]
        self.global_template_wave = pp.lam_temp
        
    def crop_datacube(self):
        '''
        :param hdu: the fits file of the entir mosaic data
        :param lens_center_x: x coordinate of the lens center
        :param lens_center_y: y coordinate of the lens center
        :param radius_in_pixels: the size where we want to extract the kinematics
        :return: new fits file with smaller size
        '''
        r = self.radius_in_pixels
        data_hdu = fits.open(self.kcwi_datacube_file)
        datacube=data_hdu[0].data
        data_hdu.close()
        self.cropped_datacube = datacube[:, self.lens_center_y - r-1:self.lens_center_y +
                                                              r, self.lens_center_x
                                                                   - r -1:
                                                                 self.lens_center_x + r]
        norm = simple_norm(np.nansum(self.cropped_datacube, axis=0), 'sqrt')
        plt.imshow(np.nansum(self.cropped_datacube, axis=0), origin="lower", norm=norm)
        plt.title('KCWI data')
        plt.colorbar(label='flux')

    def create_SN_map(self):

        #estimate the noise from the blank sky
        noise_from_blank = self.cropped_datacube[3000:4000, 4-3:4+2,4-3:4+2]
        std = np.std(noise_from_blank)
        s = np.random.normal(0, std, self.cropped_datacube.flatten().shape[0])
        noise_cube = s.reshape(self.cropped_datacube.shape)

        ## in the following, I use the noise spectrum and datacube with no quasar
        # light produced in the previous steps to estimate the S/N per AA. Since KCWI
        #  is  0.5AA/pixel, I convert the value to S/N per AA. Note that I use only the
        # region of CaH&K to estimate the S/N ratio (i.e. 4800AA - 5100AA).
        lin_axis_sky = np.linspace(self.rest_wave_range[0], self.rest_wave_range[1], self.cropped_datacube.shape[0])
        # find indices for SN
        ind_min_SN = find_nearest(lin_axis_sky*(1+self.zlens), 4800)
        ind_max_SN = find_nearest(lin_axis_sky*(1+self.zlens), 5100)

        # first, I need to estimate the flux/AA
        flux_per_half_AA = np.nanmedian(self.cropped_datacube[ind_min_SN:ind_max_SN, :, :],
                                      axis=0)

        #  convert from signal/0.5 A to signal/A
        flux_per_AA = 2 * flux_per_half_AA

        # show flux/AA
        plt.imshow(flux_per_AA, origin="lower")
        plt.title('flux per AA')
        plt.colorbar()
        plt.legend()
        plt.show()

        # then, I estimate the noise/AA.
        sigma_per_half_pixel = np.std(noise_cube[ind_min_SN:ind_max_SN,:,:], axis=0)
        sigma = np.sqrt(2) * sigma_per_half_pixel
        # some weired pattern so we find average in the black region around 36, 36
        #sigma_mean = np.mean(sigma [36-6:36+5, 36-6:36+5])
        #sigma = np.ones(sigma.shape)*sigma_mean
        plt.imshow(sigma,origin='lower')
        plt.show()

        # then, estimate the poisson noise
        sigma_poisson = poisson_noise(self.exp_time, flux_per_AA, sigma, per_second=True)
        plt.imshow(sigma_poisson,origin="lower")
        plt.title('poisson noise')
        plt.colorbar()
        plt.show()
        
        # save the SN_per_AA to self
        self.SN_per_AA = flux_per_AA / sigma_poisson
        plt.imshow(self.SN_per_AA, origin="lower")
        plt.title('S/N ratio')
        plt.colorbar()
        plt.show()

    def select_region(self):
        
        SN_y_center, SN_x_center = np.unravel_index(self.SN_per_AA.argmax(), self.SN_per_AA.shape)
        max_radius = 50
        target_SN = 1.

        xx = np.arange(self.radius_in_pixels * 2 + 1)
        yy = np.arange(self.radius_in_pixels * 2 + 1)
        xx, yy = np.meshgrid(xx, yy)
        
        dist = np.sqrt((xx - SN_x_center) ** 2 + (yy - SN_y_center) ** 2)

        SN_mask = (self.SN_per_AA > target_SN) & (dist < max_radius)

        xx_1D = xx[SN_mask]
        yy_1D = yy[SN_mask]
        SN_1D = self.SN_per_AA[SN_mask]
        self.voronoi_binning_input = np.vstack((xx_1D, yy_1D, SN_1D, np.ones(SN_1D.shape[0])))

        plt.imshow(SN_mask, origin="lower", cmap='gray')
        plt.imshow(self.SN_per_AA, origin="lower", alpha=0.9)  #
        plt.title('region selected for voronoi binning (S/N > %s)' % target_SN)
        plt.axis('off')
        plt.colorbar()
        plt.show()
        
    def voronoi_binning(self):
        
        x, y, signal, noise = self.voronoi_binning_input
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
                x, y, signal, noise, self.SN, plot=1, quiet=1)
        
        plt.tight_layout()
        #plt.savefig(target_dir + obj_name + '_voronoi_binning.png')
        #plt.savefig(target_dir + obj_name + '_voronoi_binning.pdf') 
        plt.pause(1)
        plt.clf()
        
        self.voronoi_binning_output = np.column_stack([x, y,binNum])
        ## get voronoi_binning_data based on the map
        self.voronoi_binning_data = np.zeros((int(np.max(self.voronoi_binning_output.T[2]))+1,self.cropped_datacube.shape[0])) #construct the binning
        #  data
        check = np.zeros(self.cropped_datacube[0, :, :].shape)
        for i in range(self.voronoi_binning_output.shape[0]):
            #print(i)
            wx = int(self.voronoi_binning_output[i][0])
            wy = int(self.voronoi_binning_output[i][1])
            num = int(self.voronoi_binning_output[i][2])
            self.voronoi_binning_data[num]=self.voronoi_binning_data[num]+self.cropped_datacube[:,wy,wx]
            check[wy, wx] = num+1

        self.nbins = self.voronoi_binning_data.shape[0]
        print("Number of bins =", self.nbins)
        plt.imshow(check, origin="lower", cmap='sauron')
        plt.colorbar()
        #for (j, i), label in np.ndenumerate(check):
        #    plt.text(i, j, label, ha='center', va='center')
        plt.show()
        
    def ppxf_bin_spectra(self, plot_bin_fits=False):
        self.bin_kinematics = np.zeros(shape=(0,5))
        for i in range(self.nbins):
            bin_spectrum = self.voronoi_binning_data[i]

            galaxy, logLam1, velscale = ppxf_util.log_rebin(self.rest_wave_range, bin_spectrum)
            lam = np.exp(logLam1)
            
            galaxy = galaxy[lam>self.wave_min] 
            lam = lam[lam>self.wave_min]
            galaxy = galaxy[lam<self.wave_max]
            lam = lam[lam<self.wave_max]
            logLam1 = np.log(lam)

            lam_range_global_temp = np.array([self.global_template_wave.min(), self.global_template_wave.max()])
            # after de-redshift, the initial redshift is zero.
            goodPixels = ppxf_util.determine_goodpixels(np.log(lam), lam_range_global_temp, 0)

            ind_min = find_nearest(np.exp(logLam1), self.wave_min)
            ind_max = find_nearest(np.exp(logLam1), self.wave_max)

            mask=goodPixels[goodPixels<ind_max]
            mask = mask[mask>ind_min]

            boolen = ~((2956 < mask) & (mask < 2983))  # mask the Mg II
            mask = mask[boolen]
            boolen = ~((2983 < mask) & (mask < 3001))  # mask the Mg II
            mask = mask[boolen]
            # Here the actual fit starts. The best fit is plotted on the screen.
            # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
            #
            vel = c*np.log(1 + 0)   # eq.(8) of Cappellari (2017)
            start = [vel, 250.]  # (km/s), starting guess for [V, sigma]
            bounds = [[-500, 500],[50, 450]]
            t = clock()

            noise = np.full_like(galaxy, 0.0047)

            pp = ppxf(self.global_template, galaxy, noise, velscale, start, plot=plot_bin_fits, quiet=self.quiet,
                        moments=2, goodpixels=mask,
                        degree=5,
                        velscale_ratio=self.velscale_ratio,
                        lam=lam,
                        lam_temp=self.global_template_wave,
                        )
            if plot_bin_fits==True:
                plt.pause(1)
                plt.clf()
            self.bin_kinematics = np.vstack((self.bin_kinematics, np.hstack( (pp.sol[:2],
                                                           (pp.error*np.sqrt(pp.chi2))[:2],
                                                                      pp.chi2) 
                                                                      )
                                         )
                                        )
            
    def make_kinematic_maps(self):
        
        VD_array    =np.zeros(self.voronoi_binning_output.shape[0])
        dVD_array   =np.zeros(self.voronoi_binning_output.shape[0])
        V_array     =np.zeros(self.voronoi_binning_output.shape[0])
        dV_array    =np.zeros(self.voronoi_binning_output.shape[0])


        for i in range(self.voronoi_binning_output.shape[0]):
            num=int(self.voronoi_binning_output.T[2][i])
            vd = self.bin_kinematics[num][1]
            dvd = self.bin_kinematics[num][3]
            v = self.bin_kinematics[num][0]
            dv = self.bin_kinematics[num][2]

            VD_array[i]=vd
            dVD_array[i]=dvd
            V_array[i]=v
            dV_array[i]=dv

        final=np.vstack((self.voronoi_binning_output.T, VD_array, dVD_array, V_array, dV_array))

        dim = self.radius_in_pixels*2+1

        self.VD_2d=np.zeros((dim, dim))
        self.VD_2d[:]=np.nan
        for i in range(final.shape[1]):
            self.VD_2d[int(final[1][i])][int(final[0][i])]=final[3][i]

        self.dVD_2d=np.zeros((dim, dim))
        self.dVD_2d[:]=np.nan
        for i in range(final.shape[1]):
            self.dVD_2d[int(final[1][i])][int(final[0][i])]=final[4][i]


        self.V_2d=np.zeros((dim, dim))
        self.V_2d[:]=np.nan
        for i in range(final.shape[1]):
            self.V_2d[int(final[1][i])][int(final[0][i])]=final[5][i]

        self.dV_2d=np.zeros((dim, dim))
        self.dV_2d[:]=np.nan
        for i in range(final.shape[1]):
            self.dV_2d[int(final[1][i])][int(final[0][i])]=final[6][i]

    def plot_kinematic_maps(self):
        # plot each
        plt.figure()
        plt.imshow(self.VD_2d,origin='lower',cmap='sauron')
        cbar1 = plt.colorbar()
        cbar1.set_label(r'$\sigma$ [km/s]')
        #plt.savefig(target_dir + obj_name + '_VD.png')
        plt.pause(1)
        plt.clf()

        plt.figure()
        plt.imshow(self.dVD_2d, origin='lower', cmap='sauron',vmin=0, vmax=40)
        cbar2 = plt.colorbar()
        cbar2.set_label(r'd$\sigma$ [km/s]')
        #plt.savefig(target_dir + obj_name + '_dVD.png')
        plt.pause(1)
        plt.clf()

        # mean is bulk velocity, maybe?
        mean = np.nanmedian(self.V_2d)
        #
        plt.figure()
        plt.imshow(self.V_2d-mean,origin='lower',cmap='sauron',vmin=-100, vmax=100)
        cbar3 = plt.colorbar()
        cbar3.set_label(r'Vel [km/s]')
        plt.title("Velocity map")
        #plt.savefig(target_dir + obj_name + '_V.png')
        plt.pause(1)
        plt.clf()

        plt.figure()
        plt.imshow(self.dV_2d,origin='lower',cmap='sauron')
        cbar4 = plt.colorbar()
        cbar4.set_label(r'dVel [km/s]')
        plt.title("error on velocity")
        plt.pause(1)
        plt.clf()
        