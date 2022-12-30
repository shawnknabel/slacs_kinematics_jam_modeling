'''
05/14/22 -  used for mgefit and jampy on SLACS lenses from notebooks.
'''

# import general libraries and modules
import os
import numpy as np
np.set_printoptions(threshold=10000)
import matplotlib.pyplot as plt
#plt.rcParams[figure.figsize] = (8, 6)
import pandas as pd
import warnings
#warnings.filterwarnings( ignore, module = matplotlib..* )
#warnings.filterwarnings( ignore, module = plotbin..* )
from os import path
from datetime import datetime
import glob

#########
date_time_ = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
#########

# astronomy/scipy
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import rotate
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
#from astropy.cosmology import Planck18 as cosmo  # Planck 2018
from scipy.optimize import least_squares as lsq
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

# mge fit
import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from mgefit.mge_fit_sectors_twist import mge_fit_sectors_twist
from mgefit.sectors_photometry_twist import sectors_photometry_twist
from mgefit.mge_print_contours_twist import mge_print_contours_twist

# jam
from jampy.jam_axi_proj import jam_axi_proj
from jampy.jam_axi_proj import rotate_points
from jampy.mge_half_light_isophote import mge_half_light_radius
from jampy.mge_radial_mass import mge_radial_mass
from plotbin.plot_velfield import plot_velfield
from plotbin.sauron_colormap import register_sauron_colormap
from plotbin.symmetrize_velfield import symmetrize_velfield
from pafit.fit_kinematic_pa import fit_kinematic_pa


##############################################################################

def crop_center_image (img, radius, scale, method='center'):
    
    '''
    Takes image, crops at argmax, and returns a 2radius x 2radius square image centered at the lower left corner of the center pixel
    
    img - (n,n) image with nxn pixels
    
    radius - radius in arcsec to which the 
    
    scale - pixel scale (arcsec/pix)
    
    method - str, default 'center' does not recenter, 'argmax' recenters to maximum pixel argument
    '''
    
    # take center pixel
    if method == 'center':
        # take center of input image
        central_pix_x = int(np.floor(img.shape[0]/2))
        central_pix_y = int(np.floor(img.shape[1]/2))
    elif method == 'argmax':
        # take center of image at argmax 
        central_pix = np.unravel_index(np.argmax(img, axis=None), img.shape)
        central_pix_x = central_pix[1]
        central_pix_y = central_pix[0]   
    
    # take radius in pixels
    radius = int(np.around(radius / scale))
    
    # crop to radius
    cropped_img = img[central_pix_y - radius:central_pix_y + radius, central_pix_x - radius:central_pix_x + radius]
    
    return(cropped_img, central_pix_x, central_pix_y)


##############################################################################

def import_center_crop (file_dir, obj_name, obj_abbr, data_source='HST', plot=True):

    '''
    This function imports a file from the object directory, crops the image to 2 arcsec, and returns both images. 

    Inputs:
        file_dir - contains the directory containing the object's image files
        obj_name - SDSS name, e.g. SDSSJ0037-094
        obj_abbr - SDSS name, e.g. SDSSJ0037-0942 abbreviated to J0037
        data_source - which image is used, by default HST; if kcwi_datacube, use image integrated over the spectrom so it's a 2D image instead of the cube
        plot - Return both images in line

    Returns:
        _img - file image with arcsec axes, in counts
        _3arc_img = image cropped to 3 arcsecond radius, in counts
        header = input fits file header, in counts
    '''

     ##############################################################################
    # kcwi datacube

    if data_source == 'kcwi_datacube':

        file = file_dir + f"KCWI_{obj_abbr}_icubes_mosaic_0.1457_2Dintegrated.fits"
        hdu = fits.open(file)
        kcwi_img = hdu[0].data
        header = hdu[0].header
        
        # pixel scale
        kcwi_scale = 0.1457  # arcsec/pixel r_eff_V
        img_half_extent = kcwi_img.shape[0]/2 * kcwi_scale

        # crop the image to ~ 3 arcsec radius
        kcwi_3arc_img, central_pix_x, central_pix_y = crop_center_image(kcwi_img, 3, kcwi_scale, 'argmax')
        
        if plot == True:
            # plot full image
            plt.clf()
            plt.subplot(121)
            plt.imshow(kcwi_img, origin='lower',
                       extent = [-img_half_extent,img_half_extent,-img_half_extent,img_half_extent])
            plt.title('KCWI datacube')
            # plot cropped image
            plt.subplot(122)
            plt.imshow(kcwi_3arc_img, origin='lower', extent=[-3,3,-3,3])#, extent=[0,50,0,50])
            plt.contour(kcwi_3arc_img, colors='grey', extent=[-3,3,-3,3])
            plt.title('KCWI datacube')
            plt.pause(1)

        return(kcwi_img, kcwi_3arc_img, header, central_pix_x, central_pix_y)

   ###################################################################################
    # HST cutout

    elif data_source == 'HST':
        
        # take the F435 file if it exists, else take the F814 (if two entries for same filter take first)
        files_F435 = glob.glob(f'{file_dir}*{obj_abbr}*435*.fits')
        files_F814 = glob.glob(f'{file_dir}*{obj_abbr}*814*.fits')
        if files_F435:
            file = files_F435[0] # take the first entry of F435
            filter_name = 'F435'
        elif files_F814:
            file = files_F814[0] # take the first entry of F814
            filter_name = 'F814'
        
        hdu = fits.open(file)
        hst_img = hdu[0].data #### HST data is in counts/second
        header = hdu[0].header
        
        # multiply by exp_time to get counts
        exp_time = header['EXPTIME']
        hst_img = hst_img * exp_time

        # pixel scale
        hst_scale = 0.050 # ACS/WFC
        img_half_extent = hst_img.shape[0]/2 * hst_scale

        # crop the image to 2 arcsec
        hst_3arc_img, central_pix_x, central_pix_y = crop_center_image(hst_img, 3, hst_scale, 'center')

        if plot == True:
            # plot the image
            plt.clf()
            plt.subplot(121)
            plt.imshow(hst_img, origin='lower',
                       extent = [-img_half_extent,img_half_extent,-img_half_extent,img_half_extent]) 
            plt.title(f'HST {filter_name}')
            # plot cropped image   
            plt.subplot(122)
            plt.imshow(hst_3arc_img, origin='lower', extent=[-3,3,-3,3])
            plt.contour(hst_3arc_img, colors='k', extent=[-3,3,-3,3])
            plt.title(f'HST {filter_name}')
            plt.pause(1)

        return(hst_img, hst_3arc_img, header, central_pix_x, central_pix_y)

    
##############################################################################

def try_fractions_for_find_galaxy (img):

    '''
    This function helps to figure out the pixel fraction best to use by showing the region over a range of typical fractions... f.theta is PA in degrees from the NEGATIVE x-axis.
    Inputs:
        img - 2 arcsec image to determine the central fraction
    '''

    # take different values of pixel fractions
    lower, upper, steps = (0.01, 0.10, 10)
    fractions = np.linspace(lower, upper, steps)
        
    for frac in fractions:
        #print(f'Calculating fraction {frac}')
        frac = np.around(frac, 2)
        mid = np.around((upper+lower)/2, 2)
        plt.clf()
        #plt.clf()
        f = find_galaxy(img, fraction=frac, plot=1, quiet=True)
        plt.title(f'{frac} - PA {f.theta}')
        plt.pause(1)

        
##############################################################################

def convert_mge_model_outputs (model, exp_time, extinction, photometric_zeropoint, data_source='F435W'):

    '''
    This function takes model outputs and converts them to what is needed for jampy.
    sigma is converted from pixels to arcsec
    surface brightness is converted to surface brightness density (L_sol_I pc−2)
    Inputs:
        model - output object from mge_fit_sectors, brightness in counts/pixels
        exp_time - exposure time of image in seconds
        data_source - where the image came from, by default F435W
    
    '''

    if data_source=='F435W':
        scale = 0.050 # arcsec HST ACS/WFC
    else:
        print("Don't know what that data source is, change the slacs_mge_jampy.py script")
        
    m = model

    # convert sigma from pixels to arcsec
    sigma_pix = m.sol[1]
    sigma = sigma_pix * scale

    # q 
    q = m.sol[2]

    # surface brightness
    total_counts = m.sol[0]
    # calculate peak surface brightness of each gaussian
    peak_surf_br = total_counts/(2*np.pi*q*sigma_pix**2)
    # correct for extinction and change to surface density
    #### From mgefit/readme_mge_fit_sectors.pdf
    # convert to johnson i band
    # Here 20.840 is the photometric zeropint,
    # for surface brightness measurements, and AI is the extinction in the I-band
    
    Bband_surf_br = photometric_zeropoint - 2.5 * np.log10(peak_surf_br/(exp_time*scale**2)) - extinction
    
    # convert to surface density (L_sol_I pc−2)
    M_sol_B = 4.83
    surf_density = (64800/np.pi)**2 * 10**( 0.4 * (M_sol_B - Bband_surf_br))

    return sigma, surf_density, q


##############################################################################

def plot_contours_321 (img, find_gal, model, sigmapsf, normpsf, contour_alpha=0.5, data_source='HST', plot_img=True):
    
    '''
    Plots the results the results of MGE fitting to the cropped 3 arcsec, 2 arcsec, and 2 arcsec images.
    KCWI kinematics are to ~ 3 arcsec
    Inputs:
        img - the full-sized galaxy image
        central_pix_x - central pixel x from crop_center_image
        central_pix_y - central pixel y from crop_center_image
        find_gal - object created by find_galaxy
        model - object created by mge_fit_sectors
        sigmapsf - array of Gaussian sigma_k of PSF determined from MGE fitting
        normpsf - normalized Gaussian amplitude of PSF determined from MGE fitting
        contour_alpha - alpha (transparency) of contour lines for easier visualization # This is not alpha! This is a binning variable
        data_source - default HST F435W Filter image
    '''
    
    f = find_gal
    m = model
    
    if data_source=='HST':
        scale = 0.050 # arcsec / pixel HST ACS/WFC
    elif data_source=='KCWI':
        scale = 0.147 # arcsec / pix
    else:
        print('We do not have the correct information')
        
    plt.figure()
    plt.tight_layout()

    # 3 arcsec
    n = int(np.around(3/scale))
    xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
    img_3arc = img#[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    plt.subplot(131)
    mge_print_contours(img_3arc, f.theta, xc, yc, m.sol, contour_alpha,
                       sigmapsf=sigmapsf, normpsf=normpsf, 
                       scale=scale)#, plot_img=plot_img)

    # 2 arcsec
    n = int(np.around(2/scale))
    xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
    img_2arc = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    plt.subplot(132)
    mge_print_contours(img_2arc, f.theta, xc, yc, m.sol, contour_alpha,
                       sigmapsf=sigmapsf, normpsf=normpsf, 
                       scale=scale)#, plot_img=plot_img)

    # 1 arcsec
    n = int(np.around(1/scale))
    img_cen = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
    plt.subplot(133)
    mge_print_contours(img_cen, f.theta, xc, yc, m.sol, contour_alpha,
                       sigmapsf=sigmapsf, normpsf=normpsf, 
                       scale=scale)#, plot_img=plot_img)
    plt.subplots_adjust(bottom=0.1, right=2, top=0.9)
    plt.pause(1)  # Allow plot to appear on the screen
 

##############################################################################
       
def load_2d_kinematics (file_dir, obj_name, img, find_gal, model, sigmapsf, normpsf, contour_alpha=0.7, mge_binning=1, mge_magrange=5, data_source='F435W', plot=True, plot_img=True):
    
    '''
    Shows the 2D velocity maps from ppxf fitting.
    Inputs
        file_dir - contains the directory containing the object's image files
        obj_name - SDSS name, e.g. SDSSJ0037-0942
        img - full-sized galaxy image (default is HST F435W)
        find_gal - object created by find_galaxy
        model - object created by mge_fit_sectors
        sigmapsf - array of Gaussian sigma_k of PSF determined from MGE fitting
        normpsf - normalized Gaussian amplitude of PSF determined from MGE fitting
        contour_alpha - alpha (transparency) of contour lines for easier visualization
        mge_binning - binning for Gaussian MGE contours
        mge_magrange - steps of 1 mag / arcsec^2 for each contour of Gaussian MGE
        data_source - default HST F435W Filter image
    Outputs (values defined at each pixel)
        V - line of sight velocity map
        VD - line of sight velocity dispersion map
        Vrms - rms line of sight veloicty map
        dV, dVD, dVrms - uncertainty on each above quantity
    '''
    
    f = find_gal
    m = model

    ####################
    # read in kinematics files for view (3 arcsec, 21 pixels)

    # velocity
    V = np.genfromtxt(file_dir + obj_name + '_V_2d.txt', delimiter=',')
    # find barycenter velocity (intrinsic velocity)
    center_axis_index = int(np.floor(V.shape[0]/2))
    Vbary = V[center_axis_index, center_axis_index]
    # subtract the barycenter velocity
    V = V - Vbary

    # velocity dispersion
    VD = np.genfromtxt(file_dir + obj_name + '_VD_2d.txt', delimiter=',')

    # uncertainties
    dV = np.genfromtxt(file_dir + obj_name + '_dV_2d.txt', delimiter=',')
    dVD = np.genfromtxt(file_dir + obj_name + '_dVD_2d.txt', delimiter=',')

    # rms velocity
    Vrms = np.sqrt(V**2 + VD**2)
    dVrms = np.sqrt((dV*V)**2 + (dVD*VD)**2)/Vrms

    # set scale
    if data_source=='F435W':
        scale = 0.050 # arcsec / pixel HST ACS/WFC
    else:
        print('We do not have the correct information')
    
    ####################
    if plot==True:
        # plot each with surface brightness contours
        # at three arcsec
        cmap='sauron'
        register_sauron_colormap()
        
        fontsize=16
        
        n = int(np.around(3/scale))
        xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
        #img = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
        
        # V
        plt.clf()
        fig = plt.figure(figsize=(12,4))
        fig.add_subplot(131)
        mge_print_contours(img, f.theta, xc, yc, m.sol, mge_binning, mge_magrange,
                           sigmapsf=sigmapsf, normpsf=normpsf, 
                           scale=scale, plot_img=plot_img)
        plt.imshow(V, 
                   extent = [-3, 3, -3, 3],
                   origin='lower',
                  alpha=1,
                  zorder=0,
                  cmap=cmap)#'bwr')
        plt.xlabel('arcsec', fontsize=fontsize)
        plt.ylabel('arcsec', fontsize=fontsize)
        plt.title('V', fontsize=fontsize)

        # VD
        fig.add_subplot(132)
        cnt = mge_print_contours(img, f.theta, xc, yc, m.sol, mge_binning, mge_magrange,
                           sigmapsf=sigmapsf, normpsf=normpsf, 
                           scale=scale, plot_img=plot_img)
        plt.imshow(VD, 
                   extent = [-3, 3, -3, 3],
                   origin='lower',
                  alpha=1,
                  cmap=cmap)#'bwr')
        plt.xlabel('arcsec', fontsize=fontsize)
        plt.ylabel('arcsec', fontsize=fontsize)
        plt.title('VD', fontsize=fontsize)

        # Vrms
        fig.add_subplot(133)
        mge_print_contours(img, f.theta, xc, yc, m.sol, mge_binning, mge_magrange,
                           sigmapsf=sigmapsf, normpsf=normpsf, 
                           scale=scale, plot_img=plot_img)
        plt.imshow(Vrms, 
                   extent = [-3, 3, -3, 3],
                   origin='lower',
                  alpha=1,
                  cmap=cmap)#'bwr')
        plt.title('Vrms', fontsize=fontsize)
        plt.xlabel('arcsec', fontsize=fontsize)
        plt.ylabel('arcsec', fontsize=fontsize)
        fig.tight_layout()
        plt.pause(1)  # Allow plot to appear on the screen
    
    return V, VD, Vrms, dV, dVD, dVrms, Vbary, center_axis_index

##########################################################################

# get bin centers
def get_bin_centers (bin_arrays, num_bins):
    bin_y_means = np.zeros(num_bins)
    bin_x_means = np.zeros(num_bins)
    
    for i in range(num_bins):
        # get bins and x, y for each pixel
        bin_pixels = bin_arrays[bin_arrays[:,2]==i]
        bin_x = bin_pixels[:,0] - 21 # subtract 21 pixels to center at 0,0
        bin_y = bin_pixels[:,1] - 21
        # calculate mean x and y
        mean_x = np.mean(bin_x)
        mean_y = np.mean(bin_y)
        # update array
        bin_x_means[i] = mean_x
        bin_y_means[i] = mean_y
        
    return bin_x_means, bin_y_means


##############################################################################
    
def bin_velocity_maps (file_dir, obj_abbr, Vbary, center_axis_index, data_source='KCWI'):
    
    '''
    Takes velocity measurements from ppxf and assigns to bin-center coordinates.
    Inputs
        file_dir - contains the directory containing the object's image files
        obj_abbr - SDSS name, e.g. SDSSJ0037-0942 abbreviated to J0037
        Vbary - central velocity (intrinsic or barycenter velocity) of the 2D map
        center_axis_index - axis index of the central pixel
    Outputs (values defined at each bin center)
        V_bin - velocity map
        VD_bin - velocity dispersion map
        Vrms_bin - rms velocity map
        dV_bin, dVD_bin, dVrms_bin - uncertainty in above quantities
        xbin_arcsec, ybin_arcsec - x and y components of bin centers in arcsec
    '''
    
    if data_source == 'KCWI':
        scale = 0.1457  # arcsec/pixel
    else:
        print('We have the wrong information')
    
    # bring in the velocity measurements and voronoi binning info

    vel_meas = np.genfromtxt(file_dir + '/VD.txt') # Array with columns 0-3 - Vel, sigma, dv, dsigma
    bins = np.arange(len(vel_meas))
    V_bin = vel_meas[:,0]
    V_bin = V_bin - Vbary # correct to barycenter velocity
    VD_bin = vel_meas[:,1]
    dV_bin = vel_meas[:,2]
    dVD_bin = vel_meas[:,3]
    Vrms_bin = np.sqrt(V_bin**2 + VD_bin**2)
    dVrms_bin = np.sqrt((dV_bin*V_bin)**2 + (dVD_bin*VD_bin)**2)/Vrms_bin
    
    #######################################
    # Changes - 11/30/22
    #######################################
    ## import voronoi binning data
    voronoi_binning_data = fits.getdata(dir +'voronoi_binning_' + name + '_data.fits')
    vorbin_pixels = np.genfromtxt(f'{dir}voronoi_2d_binning_{name}_output.txt',
                     delimiter='')
    # sort the voronoi bin pixel data by bin
    vorbin_pixels = vorbin_pixels[vorbin_pixels[:,2].argsort()]
    
     #######################################
    ## import voronoi binning data
    voronoi_binning_data = fits.getdata(dir +'voronoi_binning_' + name + '_data.fits')
    vorbin_pixels = np.genfromtxt(f'{dir}voronoi_2d_binning_{name}_output.txt',
                     delimiter='')
    # sort the voronoi bin pixel data by bin
    vorbin_pixels = vorbin_pixels[vorbin_pixels[:,2].argsort()]
    
    ########################################
    # find bin centers
    xbin, ybin = get_bin_centers (vorbin_pixels, len(voronoi_binning_data))
    
     #######################################
    # Changes - 11/30/22
    #######################################

    # bring in vor_bins x, y, bin
    vor_bins = np.genfromtxt(file_dir + f'/voronoi_2d_binning_KCWI_{obj_abbr}_icubes_mosaic_0.1457_output.txt')
    vor_bins_df = pd.DataFrame(vor_bins, columns=['x','y','bins'])

    # loop through all bins and attach central bins
    x_cen_bins = []
    y_cen_bins = []

    for bins in bins:

        # take all x and y in this bin
        xs = vor_bins_df[vor_bins_df.bins==bins]['x']
        ys = vor_bins_df[vor_bins_df.bins==bins]['y']

        # take mean of these coords
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)

        x_cen_bins.append(x_mean)
        y_cen_bins.append(y_mean)

    # convert to arrays    
    x_cen_bins = np.array(x_cen_bins)
    y_cen_bins = np.array(y_cen_bins)

    # center
    xbin_00 = x_cen_bins - center_axis_index
    ybin_00 = y_cen_bins - center_axis_index

    # convert to arcsec # kcwi!
    xbin_arcsec = xbin_00 * scale
    ybin_arcsec = ybin_00 * scale

    return V_bin, VD_bin, Vrms_bin, dV_bin, dVD_bin, dVrms_bin, xbin_arcsec, ybin_arcsec

######################################################################

def show_pa_difference(PA_kin, PA_phot, V_map, img):
    
    kcwi_scale=0.1457
    
    print(f'Kinematic PA : {PA_kin}') # PA from kinematics
    print(F'Photometric PA: {PA_phot}') # PA from photometry

    # Look at photometric and kinematic major axis offsets

    # plot with arcsec
    width =  V_map.shape[0]/2 * kcwi_scale
    extent = [-width,width,-width,width]
    plt.figure(figsize=(6,6))
    plt.imshow(V_map, origin='lower', extent=extent, cmap='sauron')
    #plt.scatter(xbin_arcsec, ybin_arcsec, color='k', marker='.')
    plt.contour(img,
                extent=[-3,3,-3,3],
                linewidths=2,
                colors='white')

    # plot the major axes
    # photometric
    x = np.linspace(-3, 3, 1000)
    yph = -np.tan(np.radians(PA_phot))*x
    plt.plot(x,yph, 
             label='Photometric Major Axis', 
             c='g',
            linestyle='--',
            linewidth=3)
    # kinematic
    ykin = -np.tan(np.radians(PA_kin))*x
    plt.plot(x,ykin, 
             label='Kinematic Major Axis', 
             c='orange',
            linestyle='--',
            linewidth=3)

    plt.ylim(-3,3)
    plt.legend(loc='lower left')


##############################################################################

def rotate_bins (PA, xbin_arcsec, ybin_arcsec, V_bin, plot=True):

    '''
    Rotate x and y bins by PA from find_gal model to align major axis with x-axis.
    Inputs
        PA - position angle to rotate (from either photometry or kinematics)
        xbin_arcsec, ybin_arcsec - bin center locations in arcsec from bin_velocity_maps
        V_bin - velocity map by bin center from bin_velocity_maps
    Outputs
        xbin, ybin - rotated 
    '''

    xbin = np.zeros(len(xbin_arcsec))
    ybin = np.zeros(len(ybin_arcsec))

    # rotate the coordinates and append to array
    for i in range(len(xbin_arcsec)):
        xbin[i], ybin[i] = rotate_points(xbin_arcsec[i], ybin_arcsec[i], PA) 

    if plot==True: # check that it worked
        PA_kin_rot, dPA_kin_rot, velocity_offset_rot = fit_kinematic_pa(xbin, ybin, V_bin)
        PA_kin_correction = 90 - PA_kin_rot
       
    return xbin, ybin, PA_kin_correction

##############################################################################

def correct_bin_rotation (PA_kin, PA_kin_correction, xbin_kin, ybin_kin, V_bin):
    n = 0
    
    while PA_kin_correction != 0:
        print(f'Correction {PA_kin_correction}')
        xbin_kin, ybin_kin, PA_kin_correction = rotate_bins (PA_kin_correction, xbin_kin, ybin_kin, 
                                                             V_bin, plot=True)
        plt.pause(1)
        plt.clf()
        PA_kin += PA_kin_correction
        print(f'New kinematic PA is {PA_kin}')
        n = n + 1
        if n >= 10:
            print('Took longer than ten tries... Check things.')
            break # break if takes longer than 10 tries
        
    return xbin_kin, ybin_kin, PA_kin

##############################################################################

def osipkov_merritt_model (r, a_ani, r_eff):
    
    '''
    Given anisotropy scale factor (?) and effective radius, caluclates the anisotropy at the given radius r.
    Inputs:
        r - radius for calculation (must have same units as r_eff)
        a_ani - (r_ani/r_eff) ratio of anisotropy radius and effective radius
        r_eff - effective radius of galaxy (must have same units as r_eff
    Outputs:
        Beta - anisotropy at given radius
    '''
    
    Beta = 1 / (a_ani**2 * (r_eff/r)**2 + 1)
    
    return Beta


##############################################################################

def find_half_light (lum_ks, sigma_ks, q_ks, r_bound_l=0.5, r_bound_u=3.0, plot=True):
    '''
    Takes half total luminosity as sum of Gaussian_k counts from mge_fit_sectors model (m.sol[0]). Iterates over a range of possible radii (default 0.5-3.0 arcsec, calculates total enclosed luminosity at those radii by summing the contributions of each Gaussian_k. Then subtracts the half total luminosity and interpolates the function from the x, y values. Root is half the light radius.
    Inputs:
        lum_ks - total counts of Gaussian components
        sigma_ks - disperions of Gaussian components, in arcseconds
        q_ks - axial ratio of Gaussian components
        r_bound_l, r_bound_u - upper and lower bounds on radius for calculating the enclosed luminosities, default 0.5 and 3.0
    Outputs:
        half_light_radius - the radius in arcsec at which half the total light is enclosed    
    '''
    
    # half light is half total luminosity, found by summing gaussian components and dividing by 2
    half_lum_tot = np.sum(lum_ks)/2
    
    # set values of r in reasonable range
    r = np.linspace(r_bound_l, r_bound_u, 1000)
    # create equally shaped array for luminosities
    lums = np.zeros(r.shape)
    
    # iterate over all values of r to get the luminosity enclosed at that radius
    for i in range(len(r)):
        lum = 0
        for k in range(len(lum_ks)):
            lum += lum_ks[k] * (1 - np.exp(-0.5 * (r[i] / sigma_ks[k])**2 / q_ks[k]) )
        lums[i] = lum
    
    # subtract the half light luminosity, now the root is the half light radius
    residual = lums - half_lum_tot  
    
    # interpolate a function over the values
    function = interp1d(r, residual)
    
    # use that function to solve for the root, which is the half-light radius
    half_light_radius = fsolve(function, 2)[0]
     
    if plot == True: # plot to verify
        plt.plot(r, residual)
        plt.scatter(half_light_radius, 0, color = 'r', label=f'half-light radius={np.around(half_light_radius,3)}')
        plt.axhline(0,3,0, linestyle='--', color='k')
        plt.legend()

    return half_light_radius


##############################################################################

def calculate_minlevel (img, size, plot=True):
    '''
    Takes a square cut of the image in all four corners and takes the one with the lowest mean. The std of that square is taken to be the background noise level.
    Inputs:
        img - the image that will be fit for photometry
        size - size of one side of the square (so it will be size x size)
    Outputs:
        minlevel - 1/2 the std of the background, for input to sectors_photometry
    '''
    
    # Take a size x size square from each corner
    sq0= img[:size,:size]
    sq1= img[:size,-size:]
    sq2= img[-size:,:size]
    sq3= img[-size:,-size:]
    # make list
    sqs = [sq0, sq1, sq2, sq3]
    
    # take square from middle

    # calculate mean and take lowest
    means = [np.mean(sq0), np.mean(sq1), np.mean(sq2), np.mean(sq3)]
    lowest_mean = np.min(means)
    index_lowest = np.argmin(means)

    # take the square with lowest means
    square = sqs[index_lowest]

    # calculate std of the square with lowest mean # why the std and not just the mean?
    std = np.std(square)
    
    # minlevel is 1/2 this
    minlevel = 0.5*std
    
    if plot==True:
        # make it the same size as the image by padding zeros
        square_pad = np.zeros(img.shape)
        
        # add the corner square
        if index_lowest == 0:
            square_pad[:size,:size] = square
        if index_lowest == 1:
            square_pad[:size,-size:] = square
        if index_lowest == 2:
            square_pad[-size:,:size] = square
        elif index_lowest == 3:
            square_pad[-size:,-size:] = square
        
        plt.clf()
        plt.imshow(square_pad, origin='lower', cmap='binary', alpha=0.6,zorder=1)
        plt.imshow(img, origin='lower')
        plt.pause(1)
        
    return minlevel, lowest_mean
    
    
#############################################################################
# function to fit the kcwi image with a convolution of hst image and a gaussian psf

def fit_kcwi_sigma_psf (sigma_psf, hst_img, kcwi_img, hst_scale=0.05, kcwi_scale=0.1457, plot=False):
    '''
    Fits the KCWI image with a convolution of the HST image and a Gaussian PSF with given sigma.
    Inputs:
        sigma_psf - float, sigma of Gaussian PSF, fitting parameter for optimization 
        hst_img - array (size n), 3 arcsec HST image
        kcwi_img - array (size m), 3 arcsec KCWI image
        hst_scale - float, pixel scale of HST image, default 0.05 "/pix
        kcwi_scale - float, pixel scale of KCWI image, default 0.1457 "/pix
    Outputs:
        residual - array (size m), subtraction of kcwi_img from convolved hst model
                    We will optimize with least squares
    '''

    # make gaussian kernel with sigma_psf
    gaussian_2D_kernel = Gaussian2DKernel(sigma_psf)
    
    if plot == True:
        # show the kernel
        plt.imshow(gaussian_2D_kernel, interpolation='none', origin='lower')
        plt.xlabel('x [pixels]')
        plt.ylabel('y [pixels]')
        plt.colorbar()
        plt.show()
        plt.clf()

    # convolve the hst image and the psf kernel
    convolved_img = convolve(hst_img, gaussian_2D_kernel)
    
    if plot == True:
        # show the images and the convolved image
        plt.imshow(hst_img, origin='lower')
        plt.title('HST')
        plt.pause(1)
        plt.clf()

        plt.imshow(convolved_img, origin='lower')
        plt.title('HST convolved')
        plt.pause(1)
        plt.clf()

        plt.imshow(kcwi_img, origin='lower')
        plt.title('KCWI')
        plt.pause(1)

    # make grid with kcwi_img shape times 3... and multiply by ratio of kcwi_scale/hst_scale, divide by 3
    x = np.arange(3*kcwi_img.shape[0])*kcwi_scale/hst_scale/3 # add offset values to adjust centering
    y = np.arange(3*kcwi_img.shape[1])*kcwi_scale/hst_scale/3
    # make grid
    yv, xv = np.meshgrid(x, y)
    
    # map the convolved image to the new grid
    mapped_img = map_coordinates(convolved_img, np.array([xv, yv]), mode='nearest')
    
    # create new array of shape kcwi_img and sum 3x3 sections of the mapped image
    int_mapped_img = np.sum(mapped_img.reshape(kcwi_img.shape[0], 3, kcwi_img.shape[1], 3), axis=(1,3))

    if plot == True:
        plt.imshow(int_mapped_img, origin='lower')
        plt.title('Integrated mapped image')
        plt.pause(1)
        plt.clf()
        plt.imshow(int_conv_img, origin='lower')
        plt.title('Integrated image without mapping')
        plt.pause(1)
        plt.clf()
        plt.imshow(diff, origin='lower')
        plt.title('Residual between two integrated images')
        plt.pause(1)
        plt.clf()

    # normalize the images
    # integrated mapped image
    int_mapped_img_norm = int_mapped_img / np.max(int_mapped_img)
    # kcwi image
    kcwi_img_norm = kcwi_img / np.max(kcwi_img)

    # take residual of normed images
    residual = int_mapped_img_norm - kcwi_img_norm
    
    plt.imshow(residual, origin='lower')
    plt.title('Residual')
    
    # return the residual flattened
    return residual.ravel()


########################################################################################

def optimize_sigma_psf_fit (fit_kcwi_sigma_psf, sigma_psf_guess, # use offset 
                            hst_img, kcwi_img, hst_scale=0.050, kcwi_scale=0.1457, plot=True):
    '''
    Function to optimize with least squares optimization the fit of KCWI sigma_psf by convolving the HST img
    with Gaussian PSF.
    Inputs:
        - fit_kcwi_sigma_psf - function that fits the KCWI image with HST image convolved with a sigma_psf
        - sigma_psf_guess - float, first guess at the sigma_psf value, pixels
        - hst_img 
        - kcwi_img
    '''
    
    # optimize the function
    result = lsq(fit_kcwi_sigma_psf, x0=sigma_psf_guess, kwargs={'hst_img':hst_img,'kcwi_img':kcwi_img,
                                                                 'plot':False})
    
    # state the best-fit sigma-psf and loss function value
    best_fit = result.x[0]*hst_scale
    loss = result.cost
    print(f'Best fit sigma-PSF is {best_fit} arcsec')
    print(f'Best fit loss function value is {loss}')
    
    # take best_residual
    best_residual = result.fun.reshape(kcwi_img.shape)
    
    # show residual
    if plot == True:
        plt.imshow(best_residual, origin='lower')
        plt.title('Best fit residual')
        
    return best_fit, loss, best_residual
    
    
    
#########################################################################

def estimate_hst_psf (file_dir, obj_name, obj_abbr, ngauss=12, frac=1., scale=0.050, minlevel=0., qbounds=[0.98,1.]):
    '''
    Opens fits file containing psf model from M Auger and estimates the psf with Gaussian MGE formalism
    Inputs:
        - file_dir - directory containing all files for object
        - obj_name - name of object, e.g. SDSSJ0037-0942
        - obj_abbr - abbreviation of obj, e.g. J0037
        - ngauss - int, number of Gaussians to try, will likely not need all of them, default 12
        - frac - float, fraction of pixels to use, default 1.
        - scale - hst pixel scale, 0.050
        - minlevel - float, minimum level pixel value to include in photometric fit; raise it to mask noise
        - qbound - boundaries for the axial ratio q; psf should be ~ 1
    '''
    
    # bring in fits file
    # take the F435 file if it exists, else take the F814 (if two entries for same filter take first)
    files_F435 = glob.glob(f'{file_dir}*{obj_abbr}*435*.fits')
    files_F814 = glob.glob(f'{file_dir}*{obj_abbr}*814*.fits')
    if files_F435:
        file = files_F435[0] # take the first entry of F435
        filter_name = 'F435'
    elif files_F814:
        file = files_F814[0] # take the first entry of F814
        filter_name = 'F814'
            
    hdul = fits.open(file)

    ###########################################################################
    # 4th hdu is psf
    
    psf_hdu = hdul[3]
    hst_psf_model = psf_hdu.data
    hst_psf_header = psf_hdu.header
    #print(hst_psf_header) # header is not useful
    plt.imshow(hst_psf_model)

    ###########################################################################
    # Model the central light ellipse

    plt.clf()
    #plt.clf()
    f = find_galaxy(hst_psf_model, fraction=frac, plot=1, quiet=True)
    eps = f.eps
    theta = f.theta
    cen_y = f.ypeak
    cen_x = f.xpeak
    plt.title(f'{frac}')
    plt.pause(1)

    ###########################################################################
    # run sectors photometry
    
    plt.clf()
    s = sectors_photometry(hst_psf_model, eps, theta, cen_x, cen_y, minlevel=minlevel, plot=1)
    plt.pause(1)  # Allow plot to appear on the screen

    ###########################################################################
    # fit and plot

    plt.clf()
    m = mge_fit_sectors(s.radius, s.angle, s.counts, eps,
                        ngauss=ngauss, qbounds=qbounds,
                        scale=scale, plot=1, bulge_disk=0, linear=0)
    plt.pause(1) 

    #############################################################################
    # take the output weights and sigmas for each Gaussian

    hst_psf_weights = m.sol[0] # unnormalized weights in counts of each Gaussian
    hst_normpsf = hst_psf_weights / np.sum(hst_psf_weights) # normalized weights for psf
    hst_sigmapsf = m.sol[1] # sigma of each Gaussian

    print('How good is the fit? Should be low (~ 0.02)... ' + str(m.absdev))

    # Datapoints that drop off sharply at large radii worsen the fit and should be removed as skylevel.
    
    return hst_sigmapsf, hst_normpsf

##############################################################################

# write function to create the gaussians from surf_pot, etc
def make_gaussian(r, surf_pot, sigma_pot, qobs_pot):
    gauss = 10**20*surf_pot/np.sqrt(2.*np.pi*sigma_pot**2*qobs_pot)*np.exp(-r**2/(2*sigma_pot**2))
    return(gauss)

