'''
##########################################################
##########################################################

01/04/2023 - Shawn Knabel (shawnknabel@gmail.com)

This python script creates a class called slacs_kcwi_kinematics.
Its purpose is to take a mosaic'ed datacube of a SLACS lens galaxy and create kinematic maps.
Several pieces must be done beforehand and input.

I will eventually make this a universal package to be used from kinematics through the dynamical Jeans modeling.

##########################################################
##########################################################
'''

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
import pyregion

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
# Utility functions

def de_log_rebin(delog_axi, value, lin_axi):
    '''
    :param delog_axi: input the value by np.exp(logLam1)
    :param value: flux at the location of np.exp(logLam1) array
    :param lin_axi: linear space in wavelength that we want to intepolate
    :return: flux at the location of linear space in wavelength
    '''
    inte_sky = interpolate.interp1d(delog_axi, value, bounds_error=False)
    sky_lin = inte_sky(lin_axi)
    return sky_lin

def getMaskInFitsFromDS9reg(input,shape,hdu):
    r = pyregion.open(input)
    mask = r.get_mask(shape=shape,hdu=hdu)
    return mask

##############

class slacs_kcwi_kinematics:
    
    '''
    space_jam Purpose:
    -------------------
    For a SLACS galaxy with reduced and mosaic'ed IFU datacube, take some inputs to create a stellar kinematic map.
    
    Calling Sequence:
    -------------------
    
    .. code-block:: python
        from slacs_kcwi_kinematics import slacs_kcwi_kinematics

        # data directory
        data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
        # Set 'obj_name', 'z', 'T_exp'
        obj_name = 'SDSSJ0029-0055'
        obj_abbr = obj_name[4:9] # e.g. J0029
        zlens = 0.227 # lens redshift
        T_exp = 1800*5 # exposure time in seconds... this is where I made the disastrous mistake
        lens_center_x,lens_center_y = 61, 129
        # other necessary directories ... Be very careful! This is how we will make sure we are using the correct files moving forward.
        mos_dir = f'{data_dir}mosaics/{obj_name}/' # files should be loaded from here but not saved
        kin_dir = f'{data_dir}kinematics/{obj_name}/'
        #------------------------------------------------------------------------------
        # Kinematics systematics initial choices
        # aperture
        aperture = 'R2'
        # wavelength range
        wave_min = 3400
        wave_max = 4300 # CF set to 428
        # degree of the additive Legendre polynomial in ppxf
        degree = 5 # 110/25 = 4.4 round up
        #------------------------------------------------------------------------------
        # Information specific to KCWI and templates
        kcwi_scale = 0.1457
        ## R=3600. spectral resolution is ~ 1.42A
        FWHM = 1.42 #1.42
        ## initial estimate of the noise
        noise = 0.014
        # velocity scale ratio
        velscale_ratio = 2
        #------------------------------------------------------------------------------
        # variable settings in ppxf and utility functions
        # cut the datacube at lens center, radius given here
        radius_in_pixels = 21
        # target SN for voronoi binning
        #vorbin_SN_targets = np.array([10, 15, 20])
        SN = 15
        #KCWI mosaic datacube
        mos_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'
        kcwi_datacube = f'{mos_dir}{mos_name}.fits'
        #spectrum from the lens center # using R=2
        central_spectrum_file = f'{mos_dir}{obj_abbr}_central_spectrum_{aperture}.fits' 
        background_spectrum_file = f'{mos_dir}{obj_abbr}_spectrum_background_source.fits'
        background_source_mask_file = f'{mos_dir}{obj_abbr}_background_source_mask.reg'
        sps_name = 'emiles'

        j0029_kinematics = slacs_kcwi_kinematics(
                                                 mos_dir=mos_dir,
                                                 kin_dir=kin_dir,
                                                 obj_name=obj_name,
                                                 kcwi_datacube_file=kcwi_datacube,
                                                 central_spectrum_file=central_spectrum_file,
                                                 background_spectrum_file=None,
                                                 background_source_mask_file=background_source_mask_file,
                                                 zlens=zlens,
                                                 exp_time=T_exp,
                                                 lens_center_x=lens_center_x,
                                                 lens_center_y=lens_center_y,
                                                 aperture=aperture,
                                                 wave_min=wave_min,
                                                 wave_max=wave_max,
                                                 degree=degree,
                                                 sps_name=sps_name,
                                                 pixel_scale=kcwi_scale,
                                                 FWHM=FWHM,
                                                 noise=noise,
                                                 velscale_ratio=velscale_ratio,
                                                 radius_in_pixels=radius_in_pixels,
                                                 SN=SN,
                                                 plot=True,
                                                 quiet=False
                                                    )

        # convenience function will do all the below automatically
        #j0029_kinematics.run_slacs_kcwi_kinematics()
        # Visualize the summed datacube
        j0029_kinematics.datacube_visualization()
        # rebin the central spectrum in log wavelengths and prepare for fitting
        j0029_kinematics.log_rebin_central_spectrum()
        # same with background spectrum
        j0029_kinematics.log_rebin_background_spectrum()
        # prepare the templates from the sps model
        j0029_kinematics.get_templates()
        # set up the wavelengths that will be fit, masks a couple gas lines
        j0029_kinematics.set_up_mask()
        # fit the central spectrum to create the global_template
        j0029_kinematics.ppxf_central_spectrum()
        # crop the datacube to a smaller size
        j0029_kinematics.crop_datacube()
        # create a S/N map to get the Voronoi binning going
        j0029_kinematics.create_SN_map()
        # select the spaxels S/N > 1 that will be binned
        j0029_kinematics.select_region()
        # bin the selected spaxels to the target S/N
        j0029_kinematics.voronoi_binning()
        # fit each bin spectrum with global_template
        j0029_kinematics.ppxf_bin_spectra(plot_bin_fits=plot_bin_fits)
        # create the 2d kinematic maps from ppxf fits
        j0029_kinematics.make_kinematic_maps()
        # plot those maps
        j0029_kinematics.plot_kinematic_maps()
        
        ###
        Nothing will be saved. Individual outputs can be saved as arrays, or the whole class instance can be saved to, e.g. a pkl file with python's pickle or dill module.
        ###
        import dill as pickle
        # to save as a pickle
        with open('test_pickle.pkl', 'wb') as file:
            pickle.dump(j0029_kinematics, file)
        # to reload the pickle
        with open('test_pickle.pkl', 'rb') as file:
            tommy_pickles = pickle.load(file)

    Input Parameters
    ------------------
    
    mos_dir: str
        Path to directory containing mosaic datacubes and extracted central and background source spectra.
        
    kin_dir: str
        Path to directory where outputs will be saved.
        
    obj_name: str
        Object identifier for filenames, plotting, etc.
        
    kcwi_datacube_file: str
        Path to .fits file containing the mosaic'ed KCWI datacube.
    
    central_spectrum_file: str
        Path to .fits file containing the 1D extracted spectrum from the center of the galaxy (e.g. with the program QFitsView)
        
    background_spectrum_file: str
        Path to .fits file containing the 1D extracted spectrum from the background source (e.g. with the program QFitsView)
        
    zlens: float
        Redshift of the foreground deflector lens galaxy
        
    exp_time: float or int
        Total exposure time in seconds of the mosaic'ed datacube (seconds)
        
    lens_center_x, lens_center_y: int
        Row index (x-coordinate) and column index (y-coord) of "roughly" central pixel of foreground lens galaxy, e.g. from DS9 or QFitsView, used for centering
        
    aperture: int
        Size of the aperture in pixels used to extract the central spectrum of the foreground lens galaxy "central_spectrum_file" from e.g. QFitsView (pixels)
        
    wave_min, wave_max: float or int
        Wavelength in Angstroms of the minimum and maximum of the desired fitting range. Actual fitting will ignore any part of the spectrum not inside this range (Angstroms)
        
    degree: int
        Degree of the additive polynomial used to help fit the continuum during kinematic fitting. *Very* rough rule of thumb is (wave_max - wave_min) / 250 Angstroms, rounded up. It will overfit and bias the kinematics if degree is too high. It will be unable to correct wiggles in the continuum if too low.
        
    sps_name: str
        Name of the simple stellar population used with ppxf, e.g. 'emiles'. Other templates can be used, but this is the easiest. I will likely add functionality to do what we originally did with Xshooter (more flexibility, extra steps). The sps model files will be in the ppxf module directory (should be at least). Maybe check the latest version of ppxf from github.
        
    pixel_scale: float
        Pixel scale of the datacube. KCWI is 0.1457 arcseconds per pixel (after we have made the pixels square) (arcseconds / pixel)
        
    FWHM: float
        Estimate of instrument spectral FWHM in Angstroms. KCWI is 1.42 (Angstroms)
        
    noise: float
        Rough initial estimate of the noise (I might have a bug here... Maybe I need to redo it the way CF did)
        
    velscale_ratio: int
        The ratio of desired resolution of the template spectra with relation to the datacube spectra. We tend to use 2, which means the template spectra are sampled at twice the resolution of the data
        
    radius_in_pixels: int
        Radius in pixels taken for cropping the datacube to a smaller square. We tend to use 21 pixels, which is just over 3 arcseconds. This is about the range at which we are able to get S/N per pixel > 1, generally (pixels)
        
    SN: int
        Target signal-to-noise ratio for Voronoi binning. The spaxels (spatial pixels, used interchangeably with pixels here) will be binned so that they are close to this target S/N before fitting kinematics to each bin. We tend to use 15
        
    plot: boolean
        Plot the steps throughout, True or False
        
    quiet: boolean
        Suppress some of the wordier outputs, True or False
    
    Output Parameters
    -----------------

    Stored as attributes of the ``slacs_kcwi_kinematics`` class:
    
    .rest_wave_range: tuple (2,)
        Deredshifted wavelength range of datacube in restframe of foreground deflector lens
    
    .rest_FWHM: float
        Adjusted datacube resolution in Angstroms to restframe of foreground deflector lens
        
    .central_spectrum: array (N,)
        Deredshifted (to foreground deflector restframe), log-rebinned 1D spectrum of foreground deflector galaxy from self.central_spectrum_file, of size N (size of the datacube in spectral elements)
    
    .rest_wave_log: array (N,)
        Log of log-rebinned wavelengths in restframe of datacube spectra (log Angstroms), of size N (size of the datacube in spectral elements)
        
    .rest_wave: array (N,)
        Log-re-binned wavelengths in restframe of datacube spectra (Angstroms), of size N (size of the datacube in spectral elements)
    
    .central_velscale: float
        Resolution in km/s of the datacube
        
    .background_spectrum: array (N,)
        Deredshifted (to foreground deflector restframe), log-rebinned 1D spectrum of background source galaxy from self.background_spectrum_file, of size N (size of the datacube in spectral elements)
    
    .wave_range_templates: array (size slightly larger than N)
        Restframe wavelength range of stellar templates to be used for fitting, should be slightly larger than the range of the galaxy we want to fit, but can be much larger (just takes more time)
        
    .templates: array (number of templates, templates_wave.size) # could be opposite :)
        Array containing the stellar templates from sps models, used for fitting the global template spectrum (i.e. the central_spectrum), size will be the number of templates along one axis and the range of wavelengths of templates along the other (self.templates_wave), which will be the range of templates multiplied by the input velscale (diff(wave_range_templates)*velscale, for the total number of sampled wavelengths of the templates)
        
    .templates_wave: array(diff(wave_range_templates)*velscale,)
        Array of template wavelengths, sampled at velscale times the resolution of the datacube spectra.
        
    .mask: array (unsure of size)
        Array of indices for the wavelengths of spectra that will be included in the fit, masking so that it is between wave_min and wave_max, keeping good pixels, and excluding MgII lines at ~2956-3001, input for ppxf
        
    .central_spectrum_ppxf: instance of ppxf
        Instance of ppxf with information about the central spectrum fit, can be used to recover, e.g., the weights of the stellar templates in the fit
        
    .nTemplates: int
        Number of stellar template spectra used to fit the central foreground galaxy spectrum (global_template)
        
    .global_template: array (N,)
        Weighted sum of stellar template spectra that make up the model of the central galaxy spectrum (central_spectrum); does not include the polynomial, ackground_source components, or kinematics. Will be used as the single template to fit the individual spatially-binned spectra to make the kinematic map. Essentially, this keeps the weights of the stellar templates uniform throughout the bins, which saves time and avoids over-fitting for stellar population deviations. Each bin could also be fit individually with new weights for the templates, but this project does not require that (and the data isn't good enough to determine different stellar populations).
        
    .global_template_wave: array (same as templates_wave)
        Probably redundant, just the wavelengths of the sum of templates, should be the same as the variable templates_wave
        
    .cropped_datacube: array(N, 2*radius_in_pixels+1, 2*radius_in_pixels+1)
        Datacube cropped spatially to square dimensions determined by radius_in_pixels. Wavelengths and fluxes are unaffected.
        
    .SN_per_AA: array_like(cropped_datacube)...(2*radius_in_pixels+1, 2*radius_in_pixels+1)
        Signal-to-noise map of cropped datacube spaxels, for Voronoi binning
    
    .voronoi_binning_input: array (Npix = num pixels S/N > 1, 4)
        Array containing x-coord, y-coord, signal-to-noise, and dummy noise variable for pixels in region where SN_per_AA > 1. np.vstack((xx_1D, yy_1D, SN_1D, np.ones(SN_1D.shape[0]))). Input for voronoi_2d_binning in method self.voronoi_binning
        
    .voronoi_binning_output: array (Npix = num pixels S/N > 1, 3)
        Array of outputs from Voronoi binning, which contains the x-coordinate, y-coordinate, and assigned Voronoi bin number for each of the spaxels that are in the binning (S/N > 1). This allows one to connect any measured bin values, e.g. velocity dispersion, to the individual spaxels that belong to the bin where it was measured.
        
    .voronoi_binning_data: array (nbins, N)
        Array containing the summed spectra (size N) of all of the spaxels that make up each Voronoi bin. This is the "stacked" bin spectrum that is fitted with the global_template in order to measure the kinematics of the bin.
        
    .nbins: int
        Number of spatial Voronoi bins
        
    .bin_kinematics: array (nbins, 5)
        Array of velocity dispersion VD, velocity V, error dVD, error dV, chi2 for each of the spatial Voronoi bins. Errors are determined in a bit of a funky way and will be replaced by sampling at some point. For now, errors are the "formal errors" from ppxf multiplied by the sqrt(chi2) for the fit.
        
    .VD_2d, .V_2d, .dVD_2d, .dV_2d: array_like(cropped_datacube)
        Arrays of same shape as cropped_datacube (2*radius_in_pixels+1, square), with kinematic information assigned to each of the pixels individually (through voronoi_binning_output). The 2D kinematic map that is ready to be plotted. Unfit spaxels are Nan.
        s
    '''

    
    def __init__(self,
                 mos_dir,
                 kin_dir,
                 obj_name,
                 kcwi_datacube_file,
                 central_spectrum_file,
                 background_spectrum_file,
                 background_source_mask_file,
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
        self.background_source_mask_file=background_source_mask_file
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
        '''
        Convenience function runs all the steps in a row for ease.
        plot_bin_fits- if True, will plot each ppxf fit for the bins, default is False (to save time)
        '''
        print(f'pPXF will now consume your soul and use it to measure the kinematics of {self.obj_name}.')
        # Visualize the summed datacube
        self.datacube_visualization()
        # rebin the central spectrum in log wavelengths and prepare for fitting
        self.log_rebin_central_spectrum()
        # same with background spectrum
        self.log_rebin_background_spectrum()
        # prepare the templates from the sps model
        self.get_templates()
        # set up the wavelengths that will be fit, masks a couple gas lines
        self.set_up_mask()
        # fit the central spectrum to create the global_template
        self.ppxf_central_spectrum()
        # crop the datacube to a smaller size
        self.crop_datacube()
        # create a S/N map to get the Voronoi binning going
        self.create_SN_map()
        # select the spaxels S/N > 1 that will be binned
        self.select_region()
        # bin the selected spaxels to the target S/N
        self.voronoi_binning()
        # fit each bin spectrum with global_template
        self.ppxf_bin_spectra(plot_bin_fits=plot_bin_fits)
        # create the 2d kinematic maps from ppxf fits
        self.make_kinematic_maps()
        # plot those maps
        self.plot_kinematic_maps()
        print("Job's finished!")
        
    def datacube_visualization(self):
        '''
        Function shows the mosaic'ed datacube summed over the wavelength axis. This is mostly just to get oriented to where things are in the image.
        '''
        # visualize the entire mosaic data
        # open the fits file and get the data
        data_hdu = fits.open(self.kcwi_datacube_file)
        datacube=data_hdu[0].data
        data_hdu.close()
        # norm for plotting sake
        norm = simple_norm(np.nansum(datacube, axis=0), 'sqrt')
        # plot
        plt.imshow(np.nansum(datacube, axis=0), origin="lower", norm=norm)
        plt.title('KCWI data')
        plt.colorbar(label='flux')
        plt.pause(1)

    def log_rebin_central_spectrum(self):
        '''
        Function to deredshift and rebin the central foreground galaxy spectrum to log space and prepare the restframe wavelengths for proper fitting with the stellar template spectra later.
        '''
        # open the fits file and get the data
        hdu = fits.open(self.central_spectrum_file)
        # galaxy spectrum with linear wavelength spacing
        gal_lin = hdu[0].data
        h1 = hdu[0].header
        # wavelength range from fits header
        lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1'] - 1)])
        
        # Compute approximate restframe wavelength range
        self.rest_wave_range = lamRange1/(1+self.zlens)
        # Adjust resolution in Angstrom
        self.rest_FWHM = self.FWHM/(1+self.zlens)  
        # rebin to log wavelengths and calculate velocity scale (resolution)
        self.central_spectrum, self.rest_wave_log, self.central_velscale = ppxf_util.log_rebin(self.rest_wave_range, gal_lin)
        # keep the rebinned wavelengths in Angstroms
        self.rest_wave = np.exp(self.rest_wave_log)
        
    def log_rebin_background_spectrum(self):
        '''
        Function to deredshift (to the foreground deflector galaxy redshift zlens) and rebin the background source galaxy spectrum to log space and prepare the restframe wavelengths for proper fitting with the "sky" keyword in ppxf.
        '''
        if self.background_spectrum_file is not None:
            # open the fits file and get the data
            hdu = fits.open(self.background_spectrum_file)
            background_source_lin = hdu[0].data 
            # rebin to log wavelengths
            background_source, _, _ = ppxf_util.log_rebin(self.rest_wave_range, background_source_lin)
            # Normalize spectrum to avoid numerical issues
            self.background_spectrum = background_source/np.median(background_source)  
        else:
            self.background_spectrum = np.zeros_like(self.central_spectrum)
            
    def get_templates(self):
        '''
        Function prepares the stellar template spectra from the sps model identified by sps_name.
        '''
        # take wavelength range of templates to be slightly larger than that of the galaxy restframe
        self.wave_range_templates = self.rest_wave_range[0]/1.2, self.rest_wave_range[1]*1.2
        # bring in the templates from ppxf/sps_models/
        basename = f"spectra_{self.sps_name}_9.0.npz"
        filename = path.join(ppxf_dir, 'sps_models', basename)
        # created template library will be sampled at data resolution times the velscale_ratio in the given wavelength range
        sps = sps_util.sps_lib(filename, 
                               self.central_velscale/self.velscale_ratio, 
                               self.rest_FWHM, 
                               wave_range=self.wave_range_templates)
        templates= sps.templates
        self.templates = templates.reshape(templates.shape[0], -1)
        self.templates_wave = sps.lam_temp

    def set_up_mask(self):
        '''
        Function prepares a mask for ppxf that determines the correct wavelength range and masks some gas lines.
        '''
        # after de-redshift, the initial redshift is zero.
        goodPixels = ppxf_util.determine_goodpixels(self.rest_wave_log, self.wave_range_templates, 0)
        # find the indices of the restframe wavelengths that are closest to the min and max we want
        ind_min = find_nearest(self.rest_wave, self.wave_min)
        ind_max = find_nearest(self.rest_wave, self.wave_max)
        mask=goodPixels[goodPixels<ind_max]
        mask = mask[mask>ind_min]
        # mask gas lines
        boolen = ~((2956 < mask) & (mask < 2983))  # mask the Mg II
        mask = mask[boolen]
        boolen = ~((2983 < mask) & (mask < 3001))  # mask the Mg II
        self.mask = mask[boolen]
    
    def ppxf_central_spectrum(self):
        '''
        Function fits the central_spectrum with the stellar template spectra, a polynomial of specified degree, and the background source as the "sky" component.
        '''
        # some setup
        vel = c*np.log(1 + 0)   # eq.(8) of Cappellari (2017)
        start = [vel, 250.]  # (km/s), starting guess for [V, sigma]
        #bounds = [[-500, 500],[50, 450]] # not necessary
        t = clock()
        # create a noise array # Assume constant noise per AA
        noise_array = np.full_like(self.central_spectrum, self.noise) 
        # fit with ppxf
        pp = ppxf(self.templates, # templates for fitting
                  self.central_spectrum,  # spectrum to be fit
                  noise_array,
                  self.central_velscale, # resolution
                  start, # starting guess
                  plot=False, # no need to plot here, will plot after
                  moments=2, # VD and V, no others
                  goodpixels=self.mask, # mask we made
                  degree=self.degree, # degree of polynomial we specified
                  velscale_ratio=self.velscale_ratio, # resolution of templates wrt. data
                  sky=self.background_spectrum, # background source spectrum
                  lam=self.rest_wave, # wavelengths for fitting
                  lam_temp=self.templates_wave, # wavelenghts of templates
                 )

        #plot the fit
        # model
        model = pp.bestfit
        # background source
        background = self.background_spectrum * pp.weights[-1]
        # data
        data = pp.galaxy
        # linearize the wavelengths for plotting
        log_axis = self.rest_wave
        lin_axis = np.linspace(self.rest_wave_range[0], self.rest_wave_range[1], data.size)
        # rebin in linear space
        back_lin = de_log_rebin(log_axis, background, lin_axis)
        model_lin = de_log_rebin(log_axis, model, lin_axis)
        data_lin = de_log_rebin(log_axis, data, lin_axis)
        noise_lin = data_lin - model_lin
        # find the indices of the restframe wavelengths that are closest to the min and max we want for plot limits
        plot_ind_min = find_nearest(lin_axis, self.wave_min)
        plot_ind_max = find_nearest(lin_axis, self.wave_max)
        # make the figure
        plt.figure(figsize=(8,6))
        plt.plot(lin_axis, data_lin, 'k-', label='data')
        plt.plot(lin_axis, model_lin, 'r-', label='model ('
                                                     'lens+background)')
        plt.plot(lin_axis, data_lin - back_lin, 'm-',
                 label='remove background source from data', alpha=0.5)
        plt.plot(lin_axis, back_lin + np.full_like(back_lin, 0.9e-5), 'c-',label='background source', alpha=0.7)
        plt.plot(lin_axis, noise_lin + np.full_like(back_lin, 0.9e-5), 'g-',
                 label='noise (data - best model)', alpha=0.7)
        plt.legend(loc='best')
        plt.ylim(np.nanmin(noise_lin[plot_ind_min:plot_ind_max])/1.1, np.nanmax(data_lin[plot_ind_min:plot_ind_max])*1.1)
        plt.xlim(self.wave_min, self.wave_max)
        plt.xlabel('wavelength (A)')
        plt.ylabel('relative flux')
        plt.title(f'Velocity dispersion - {int(pp.sol[1])} km/s')
        plt.show()
        plt.pause(1)
        # show results
        print("Formal errors:")
        print("     dV    dsigma   dh3      dh4")
        print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

        print('Elapsed time in pPXF: %.2f s' % (clock() - t))
        # take the fit as attributes for future
        self.central_spectrum_ppxf = pp
        # number of templates
        self.nTemplates = pp.templates.shape[1]
        # global_template is what we use to fit the bins
        self.global_template = pp.templates @ pp.weights[:self.nTemplates]
        self.global_template_wave = pp.lam_temp
        
    def crop_datacube(self):
        '''
        Function crops the datacube to a size determined by radius_in_pixels.
        '''
        r = self.radius_in_pixels
        data_hdu = fits.open(self.kcwi_datacube_file)
        datacube=data_hdu[0].data
        data_hdu.close()
        if self.background_source_mask_file is not None:
            self.background_source_mask = ~getMaskInFitsFromDS9reg(self.background_source_mask_file, datacube.shape[1:], data_hdu[0])*1
            norm = simple_norm(np.nansum(datacube, axis=0), 'sqrt')
            plt.imshow(np.nansum(datacube, axis=0)*self.background_source_mask, origin="lower", norm=norm)
            plt.title('Masked KCWI data')
            plt.colorbar(label='flux')
            plt.pause(1)
        else:
            self.background_source_mask = np.ones(datacube.shape[1:])
        self.background_source_mask = self.background_source_mask[self.lens_center_y - r-1:self.lens_center_y + r, 
                                         self.lens_center_x - r -1:self.lens_center_x + r]
        self.cropped_datacube = datacube[:, 
                                         self.lens_center_y - r-1:self.lens_center_y + r, 
                                         self.lens_center_x - r -1:self.lens_center_x + r] * self.background_source_mask
        norm = simple_norm(np.nansum(self.cropped_datacube, axis=0), 'sqrt')
        plt.imshow(np.nansum(self.cropped_datacube, axis=0)*self.background_source_mask, origin="lower", norm=norm)
        plt.title('Cropped KCWI data')
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
        lin_axis = np.linspace(self.rest_wave_range[0], self.rest_wave_range[1], self.cropped_datacube.shape[0])
        # find indices for SN
        ind_min_SN = find_nearest(lin_axis*(1+self.zlens), 4800)
        ind_max_SN = find_nearest(lin_axis*(1+self.zlens), 5100)

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

            galaxy, log_wavelengths, velscale = ppxf_util.log_rebin(self.rest_wave_range, bin_spectrum)
            wavelengths = np.exp(log_wavelengths)
            
            
            galaxy = galaxy[wavelengths>self.wave_min] 
            background_source = self.background_spectrum.copy()
            background_source = background_source[wavelengths>self.wave_min]
            wavelengths = wavelengths[wavelengths>self.wave_min]
            galaxy = galaxy[wavelengths<self.wave_max]
            background_source = background_source[wavelengths<self.wave_max]
            wavelengths = wavelengths[wavelengths<self.wave_max]
            log_wavelengths = np.log(wavelengths)

            lam_range_global_temp = np.array([self.global_template_wave.min(), self.global_template_wave.max()])
            # after de-redshift, the initial redshift is zero.
            goodPixels = ppxf_util.determine_goodpixels(log_wavelengths, lam_range_global_temp, 0)
            
            ind_min = find_nearest(wavelengths, self.wave_min)
            ind_max = find_nearest(wavelengths, self.wave_max)
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

            pp = ppxf(self.global_template, 
                      galaxy, 
                      noise, 
                      velscale, 
                      start, 
                      sky=background_source, 
                      plot=False,#plot_bin_fits, 
                      quiet=self.quiet,
                        moments=2, 
                      goodpixels=mask,
                        degree=5,
                        velscale_ratio=self.velscale_ratio,
                        lam=wavelengths,
                        lam_temp=self.global_template_wave,
                        )
            if plot_bin_fits==True:
                #plot the fit
                background = background_source * pp.weights[-1]
                data = pp.galaxy#spectrum_perpixel
                model = pp.bestfit
                log_axis = wavelengths
                lin_axis = np.linspace(self.wave_min, self.wave_max, data.size)
                sky_lin = de_log_rebin(log_axis, background, lin_axis)
                model_lin = de_log_rebin(log_axis, model, lin_axis)
                data_lin = de_log_rebin(log_axis, data, lin_axis)
                noise_lin = data_lin - model_lin
                # find the indices of the restframe wavelengths that are closest to the min and max we want for plot limits
                plot_ind_min = find_nearest(lin_axis, self.wave_min)
                plot_ind_max = find_nearest(lin_axis, self.wave_max)
                # make the figure
                plt.figure(figsize=(8,6))
                plt.plot(lin_axis, data_lin, 'k-', label='data')
                plt.plot(lin_axis, model_lin, 'r-', label='model ('
                                                             'lens+background)')
                plt.plot(lin_axis, data_lin - back_lin, 'm-',
                         label='remove background source from data', alpha=0.5)
                plt.plot(lin_axis, back_lin + np.full_like(back_lin, 0.9e-5), 'c-',label='background source', alpha=0.7)
                plt.plot(lin_axis, noise_lin + np.full_like(back_lin, 0.9e-5), 'g-',
                         label='noise (data - best model)', alpha=0.7)
                plt.legend(loc='best')
                plt.ylim(np.nanmin(noise_lin[plot_ind_min:plot_ind_max])/1.1, np.nanmax(data_lin[plot_ind_min:plot_ind_max])*1.1)
                plt.xlim(self.wave_min, self.wave_max)
                plt.xlabel('wavelength (A)')
                plt.ylabel('relative flux')
                plt.title(f'Velocity dispersion - {int(pp.sol[1])} km/s')
                plt.show()
                plt.pause(1)
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
        