'''
05/14/22 - Modules used for mgefit and jampy on SLACS lenses from notebooks.
Updated 2/1/23
'''

################################################################

# import general libraries and modules
import numpy as np
np.set_printoptions(threshold=10000)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams.update({'font.size': 14})
import pandas as pd
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings( "ignore", module = "plotbin\..*" )
from os import path
import glob
#import Image from PIL
from PIL import Image
import pickle

# astronomy/scipy
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import rotate
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares as lsq
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.cosmology import Planck15 as cosmo  # Originally I did Planck 2018, but it seems this one isn't in the version of astropy we have on here and I'm not 
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# mge fit
import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from mgefit.mge_fit_sectors_regularized import mge_fit_sectors_regularized


# jam
from jampy.jam_axi_proj import jam_axi_proj
from jampy.jam_axi_proj import rotate_points
from jampy.jam_axi_proj import bilinear_interpolate
from jampy.mge_half_light_isophote import mge_half_light_isophote
from plotbin.plot_velfield import plot_velfield
from plotbin.sauron_colormap import register_sauron_colormap
register_sauron_colormap()
from pafit.fit_kinematic_pa import fit_kinematic_pa

################################################################


# functions

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
    This function imports a file from the object directory, crops the image to 5 and 3 arcsec, and returns all images. 
    *****
    This is modified from other versions to pull in the SLACS bspline model instead of the observed HST image
    *****

    Inputs:
        file_dir - contains the directory containing the object's image files
        obj_name - SDSS name, e.g. SDSSJ0037-0942
        obj_abbr - SDSS name, e.g. SDSSJ0037-0942 abbreviated to J0037
        data_source - which image is used, by default HST; if kcwi_datacube, use image integrated over the spectrom so it's a 2D image instead of the cube
        plot - Return both images in line

    Returns:
        _img - file image with arcsec axes, in counts
        _5arc_img
        _3arc_img = image cropped to 3 arcsecond radius, in counts
        header = input fits file header, in counts
    '''

     ##############################################################################
    # kcwi datacube

    if data_source == 'kcwi_datacube':

        file = file_dir + f"KCWI_{obj_abbr}_icubes_mosaic_0.1457_2D_integrated.fits"
        hdu = fits.open(file)
        kcwi_img = hdu[0].data
        header = hdu[0].header
        
        # get kcwi position angle N thru E
        kcwi_pa = header['ROTDEST'] # I still haven't determined what this angle is relative to. I *believe* it's N->E
        
        # pixel scale
        kcwi_scale = 0.1457  # arcsec/pixel r_eff_V
        img_half_extent = kcwi_img.shape[0]/2 * kcwi_scale
        
        # crop the image to ~ 5 arcsec radius
        kcwi_5arc_img, central_pix_x, central_pix_y = crop_center_image(kcwi_img, 5, kcwi_scale, 'argmax')

        # crop the image to ~ 3 arcsec radius
        kcwi_3arc_img, central_pix_x_3, central_pix_y_3 = crop_center_image(kcwi_img, 3, kcwi_scale, 'argmax')
        
        if plot == True:
            # plot full image
            plt.clf()
            plt.figure(figsize=(12,4))
            plt.subplot(131)
            plt.imshow(kcwi_img, origin='lower',
                       extent = [-img_half_extent,img_half_extent,-img_half_extent,img_half_extent])
            plt.title('KCWI datacube')
            # plot cropped images
            plt.subplot(132)
            plt.imshow(kcwi_5arc_img, origin='lower', extent=[-5,5,-5,5])#, extent=[0,50,0,50])
            plt.contour(kcwi_5arc_img, colors='grey', extent=[-5,5,-5,5])
            plt.title('KCWI datacube')
            plt.subplot(133)
            plt.imshow(kcwi_3arc_img, origin='lower', extent=[-3,3,-3,3])#, extent=[0,50,0,50])
            plt.contour(kcwi_3arc_img, colors='grey', extent=[-3,3,-3,3])
            plt.title('KCWI datacube')
            plt.pause(1)

        return(kcwi_img, kcwi_5arc_img, kcwi_3arc_img, header, central_pix_x, central_pix_y, kcwi_pa)

   ###################################################################################
    # HST cutout

    elif data_source == 'HST':
        
        # take the F435 file if it exists, else take the F814 (if two entries for same filter take first)
        files_F435 = glob.glob(f'{file_dir}*{obj_abbr}*435*.fits')
        files_F814 = glob.glob(f'{file_dir}*{obj_abbr}*814*.fits')
        if files_F435:
            file = files_F435[0] # take the first entry of F435
            filter_name = 'F435W'
        elif files_F814:
            file = files_F814[0] # take the first entry of F814
            filter_name = 'F814W'
        else:
            print('no file')
        
        hdu = fits.open(file)
        hst_img = hdu[0].data #### HST data is in counts/second
        bspl_img = hdu[8].data #### HST data is in counts/second # index 8 is the b-spline model
        header = hdu[0].header
        
        # multiply by exp_time to get counts
        exp_time = header['EXPTIME']
        hst_img = hst_img * exp_time
        bspl_img = bspl_img * exp_time

        # pixel scale
        hst_scale = 0.050 # ACS/WFC
        img_half_extent = hst_img.shape[0]/2 * hst_scale
        
        # crop the image to 5 arcsec
        hst_5arc_img, central_pix_x, central_pix_y = crop_center_image(hst_img, 5, hst_scale, 'center')
        bspl_5arc_img, _, _ = crop_center_image(bspl_img, 5, hst_scale, 'center')
        
        # crop the image to 3 arcsec
        hst_3arc_img, central_pix_x_3, central_pix_y_3 = crop_center_image(hst_img, 3, hst_scale, 'center')
        bspl_3arc_img, _, _ = crop_center_image(bspl_img, 5, hst_scale, 'center')

        if plot == True:
            # plot the image
            plt.clf()
            plt.figure(figsize=(12,4))
            plt.subplot(131)
            plt.imshow(hst_img, origin='lower',
                       extent = [-img_half_extent,img_half_extent,-img_half_extent,img_half_extent]) 
            plt.title(f'HST {filter_name}')
            # plot cropped image   
            plt.subplot(132)
            plt.imshow(hst_5arc_img, origin='lower', extent=[-5,5,-5,5])
            plt.contour(hst_5arc_img, colors='k', extent=[-5,5,-5,5])
            plt.title(f'HST {filter_name}')
            plt.subplot(133)
            plt.imshow(hst_3arc_img, origin='lower', extent=[-3,3,-3,3])
            plt.contour(hst_3arc_img, colors='k', extent=[-3,3,-3,3])
            plt.title(f'HST {filter_name}')
            plt.pause(1)
            
            #plot the bspline model
            plt.clf()
            plt.figure(figsize=(12,4))
            plt.subplot(131)
            plt.imshow(bspl_img, origin='lower',
                       extent = [-img_half_extent,img_half_extent,-img_half_extent,img_half_extent]) 
            plt.title(f'HST bspline {filter_name}')
            # plot cropped image   
            plt.subplot(132)
            plt.imshow(bspl_5arc_img, origin='lower', extent=[-5,5,-5,5])
            plt.contour(bspl_5arc_img, colors='k', extent=[-5,5,-5,5])
            plt.title(f'HST bspline {filter_name}')
            plt.subplot(133)
            plt.imshow(bspl_3arc_img, origin='lower', extent=[-3,3,-3,3])
            plt.contour(bspl_3arc_img, colors='k', extent=[-3,3,-3,3])
            plt.title(f'HST bspline {filter_name}')
            plt.pause(1)

        return(hst_img, hst_5arc_img, hst_3arc_img, bspl_img, bspl_5arc_img, bspl_3arc_img, header, central_pix_x, central_pix_y, exp_time)

    

def try_fractions_for_find_galaxy (img):

    '''
    This function helps to figure out the pixel fraction best to use by showing the region over a range of typical fractions
    Inputs:
        img - 2 arcsec image to determine the central fraction
    '''

    # take different values of pixel fractions
    lower, upper, steps = (0.03, 0.12, 10)
    fractions = np.linspace(lower, upper, steps)
        
    for frac in fractions:
        #print(f'Calculating fraction {frac}')
        frac = np.around(frac, 2)
        mid = np.around((upper+lower)/2, 2)
        plt.clf()
        #plt.clf()
        f = find_galaxy(img, fraction=frac, plot=1, quiet=True)
        plt.title(f'{frac}')
        plt.pause(1)



def convert_mge_model_outputs (model, exp_time, extinction, q, data_source='F435W'):

    '''
    This function takes model outputs and converts them to what is needed for jampy.
    sigma is converted from pixels to arcsec
    surface brightness is converted to surface density (L_sol_I pc−2)
    Inputs:
        model - output object from mge_fit_sectors
        exp_time - exposure time of image in seconds
        extinction - dust extinction AI (I band Jonshon-Cousins)
        data_source - where the image came from, by default F435W
    
    '''

    if (data_source=='F435W') or (data_source=='F814W'):
        scale = 0.050 # arcsec HST ACS/WFC
    else:
        print("Don't know what that data source is, change the slacs_mge_jampy.py script")
        
    m = model

    # convert sigma from pixels to arcsec
    sigma_pix = m.sol[1]
    sigma = sigma_pix * scale

    # q 
    #q = m.sol[2]

    # surface brightness
    total_counts = m.sol[0]
    # calculate peak surface brightness of each gaussian
    peak_surf_br = total_counts/(2*np.pi*q*sigma_pix**2)
    
    #########################################################################################################
    # correct for extinction and change to surface density
    #### From readme_mge_fit_sectors.pdf
    ####$ The surface brightnessC0in counts pixels−1canbe converted into a Johnson-CousinsI-band surfacebrightnessμIin mag arcsec−2using standard photom-etry formulas. In the case of the WFPC2/F814W filterthe equation is to a first approximation (Holtzman etal. 1995, PASP, 107, 1065)μI= 20.840 + 0.1 + 5 log(SCALE)+ 2.5 log(EXPTIME)−2.5 logC0−AI.(2)Here 20.840 is the photometric zeropint, 0.1 is a cor-rection for infinite aperture to be applied for surfacebrightness measurements, andAIis the extinction intheI-band.Finally one goes from the surface brightnessμIinmag arcsec−2to the surface densityI′inLpc−2withthe equation1I′=(64800π)2100.4(M,I−μI).
    # convert to johnson i band (johnson-cousins is normalized to Vega=0.03)
    # Here 20.840 is the photometric zeropint, 0.1 is a correction for infinite aperture to be applied 
    # for surface brightness measurements, and AI is the extinction in the I-band
    #### dust extinction from https://irsa.ipac.caltech.edu/applications/DUST/
    #### from holzman 1995 https://articles.adsabs.harvard.edu/pdf/1995PASP..107.1065H
    # To convert to UBVRI magnitudes/square arcsec, convert the count rates to instrumental magnitudes per square arcsec. 
    # Use the transformation relations given by Eq. (8) with the coefficients presented in Table 7 (observed transformations for primary filters) or 
    # those given by Eq. (9) with the coefficients in Table 10 (synthetic transformations). Add a constant of approximately 0.1 mag to each zero point 
    # to correct to infinite aperture (see Sec. 2.5 and H95).
    # SMAG= - 2,5 X log(DN s"1 ) + iFiS X SCOL + 2 psXSCOL2 ^2fç2.5 log GRf, (8)
    #### Zeropoints https://acszeropoints.stsci.edu/ (VEGAmag)
    # On 2004-09-18 (J0037 observed)
    # F435W zeropoint: 25.793
    # F814W zeropoint: 25.520
    ###############################################################################################################
    
    # for surface density translation
    if data_source=='F435W':
        zeropoint = 25.793
    elif data_source=='F814W':
        zeropoint = 25.520
    inf_ap_correction = 0.1
    
    # convert to iband surface brightness
    iband_surf_br = zeropoint + inf_ap_correction + 5 * np.log10(scale) + 2.5 * np.log10(exp_time) - 2.5 * np.log10(peak_surf_br) - extinction
    
    # convert to surface density (L_sol_I pc−2)
    M_sol_I = 4.08
    # final conversion to surface density
    surf_density = (64800/np.pi)**2 * 10**( 0.4 * (M_sol_I - iband_surf_br))

    return sigma, surf_density


def plot_contours_531 (img, find_gal, model, sigmapsf, normpsf, contour_alpha=0.5, data_source='HST', plot_img=True):
    
    '''
    Plots the results the results of MGE fitting to the cropped 5 arcsec, 3 arcsec, and 1 arcsec images.
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

    # 5 arcsec
    n = int(np.around(5/scale))
    extent = np.array([-n, n, -n, n])
    xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
    img_5arc = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    plt.subplot(131)
    mge_print_contours(img_5arc, f.theta, xc, yc, m.sol, contour_alpha,
                       sigmapsf=sigmapsf, normpsf=normpsf, 
                       scale=scale, extent=extent)#, plot_img=plot_img)
    
    # 3 arcsec
    n = int(np.around(3/scale))
    extent = np.array([-n, n, -n, n])
    xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
    img_3arc = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    plt.subplot(132)
    mge_print_contours(img_3arc, f.theta, xc, yc, m.sol, contour_alpha,
                       sigmapsf=sigmapsf, normpsf=normpsf, 
                       scale=scale, extent=extent)#, plot_img=plot_img)

    # 1 arcsec
    n = int(np.around(1/scale))
    extent = np.array([-n, n, -n, n])
    img_1arc = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
    plt.subplot(133)
    mge_print_contours(img_1arc, f.theta, xc, yc, m.sol, contour_alpha,
                       sigmapsf=sigmapsf, normpsf=normpsf, 
                       scale=scale, extent=extent)#, plot_img=plot_img)
    plt.subplots_adjust(bottom=0.1, right=2, top=0.9)
    plt.pause(1)  # Allow plot to appear on the screen
    

def plot_contours_321 (img, central_pix_x, central_pix_y, find_gal, model, sigmapsf, data_source='F435W'):
    
    '''
    Plots the results the results of MGE fitting to the cropped 3 arcsec, 2 arcsec, and 2 arcsec images.
    KCWI kinematics are to ~ 3 arcsec
    Inputs:
        img - the full-sized galaxy image
        central_pix_x - central pixel x from crop_center_image
        central_pix_y - central pixel y from crop_center_image
        find_gal - object created by find_galaxy
        model - object created by mge_fit_sectors
        sigmapsf - sigma of PSF determined from MGE fitting, I think?
        data_source - default HST F435W Filter image
    '''
    
    f = find_gal
    m = model
    
    if data_source=='F435W':
        scale = 0.050 # arcsec / pixel HST ACS/WFC
    else:
        print('We do not have the correct information')

    # plot 3 arcsec
    n = int(np.around(3/scale))
    xc, yc = central_pix_x, central_pix_y
    img_3arc = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    plt.subplot(122)
    mge_print_contours(img_3arc, f.theta, xc, yc, m.sol,
                       sigmapsf=sigmapsf, #normpsf=normpsf, 
                       scale=scale)
    plt.imshow(img, 
               origin='lower',
               extent = [-3, 3, -3, 3],
              alpha=0.7)
    plt.pause(1)  # Allow plot to appear on the screen

    # plot 2 arcsec
    n = int(np.around(2/scale))
    xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
    img_2arc = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    plt.subplot(122)
    mge_print_contours(img_2arc, f.theta, xc, yc, m.sol,
                       sigmapsf=sigmapsf, #normpsf=normpsf, 
                       scale=scale)
    plt.imshow(img, 
               origin='lower',
               extent = [-2, 2, -2, 2],
              alpha=0.7)
    plt.pause(1)  # Allow plot to appear on the screen

    # plot 1 arcsec
    n = int(np.around(1/scale))
    img_cen = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
    plt.subplot(122)
    mge_print_contours(img_cen, f.theta, xc, yc, m.sol,
                       sigmapsf=sigmapsf, #normpsf=normpsf, 
                       scale=scale)
    plt.imshow(img_cen, 
               origin='lower',
               extent = [-1, 1, -1, 1],
              alpha=0.7)
    plt.pause(1)  # Allow plot to appear on the screen

#######################################################
# Kinematics

def kinematics_map_systematics(dir, name, radius_in_pixels=21):
    
    '''
    dir - file directory for individual object
    name - object name in SDSS, e.g. SDSSJ####-####
    this code modifies CF's kinematics_map function to remap the kinematics measurements above into 2D array from my systematics scripts.
    :return: 2D velocity dispersion, uncertainty of the velocity dispersion, velocity, and the uncertainty of the velocity
    '''
    
    #KCWI mosaic datacube
    obj_abbr = name[4:9] # e.g. J0029
    mosaic_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'

    VD=np.genfromtxt(f'{dir}{name}_final_kinematics/{name}_VD_binned.txt',
                 delimiter=',')
    VD_covariance = np.genfromtxt(f'{dir}{name}_final_kinematics/{name}_covariance_matrix_VD.txt',
                                 delimiter=',')
    dVD = np.sqrt(np.diagonal(VD_covariance)) 
# diagonal is sigma**2
    
    
    V=np.genfromtxt(f'{dir}{name}_final_kinematics/{name}_V_binned.txt',
                 delimiter=',')
    V_covariance = np.genfromtxt(f'{dir}{name}_final_kinematics/{name}_covariance_matrix_V.txt',
                                 delimiter=',')
    dV = np.sqrt(np.diagonal(VD_covariance)) # diagonal is sigma**2
    
    # Vel, sigma, dv, dsigma
    output=np.loadtxt(dir +'voronoi_2d_binning_' + mosaic_name + '_output.txt')

    VD_array    =np.zeros(output.shape[0])
    dVD_array   =np.zeros(output.shape[0])
    V_array     =np.zeros(output.shape[0])
    dV_array    =np.zeros(output.shape[0])

    for i in range(output.shape[0]):
        num=int(output.T[2][i])
        VD_array[i] = VD[num]
        dVD_array[i] = dVD[num]
        V_array[i] = V[num]
        dV_array[i] = dV[num]
    
    final=np.vstack((output.T, VD_array, dVD_array, V_array, dV_array))
    
    dim = radius_in_pixels*2+1

    VD_2d=np.zeros((dim, dim))
    VD_2d[:]=np.nan
    for i in range(final.shape[1]):
        VD_2d[int(final[1][i])][int(final[0][i])]=final[3][i]
        
    dVD_2d=np.zeros((dim, dim))
    dVD_2d[:]=np.nan
    for i in range(final.shape[1]):
        dVD_2d[int(final[1][i])][int(final[0][i])]=final[4][i]


    V_2d=np.zeros((dim, dim))
    V_2d[:]=np.nan
    for i in range(final.shape[1]):
        V_2d[int(final[1][i])][int(final[0][i])]=final[5][i]

    dv_2d=np.zeros((dim, dim))
    dv_2d[:]=np.nan
    for i in range(final.shape[1]):
        dv_2d[int(final[1][i])][int(final[0][i])]=final[6][i]
    
    return VD_2d, dVD_2d, V_2d, dv_2d


def load_2d_kinematics (file_dir, obj_name, img, find_gal, sigmapsf, data_source='F435W', plot=True):
    
    '''
    Shows the 2D velocity maps from ppxf fitting.
    Inputs
        file_dir - contains the directory containing the object's image files
        obj_name - SDSS name, e.g. SDSSJ0037-0942
        img - full-sized galaxy image (default is HST F435W)
        find_gal - object created by find_galaxy
        sigmapsf - sigma of the PSF determined by MGE fitting, I think?
        data_source - default HST F435W Filter image
    Outputs (values defined at each pixel)
        V - line of sight velocity map
        VD - line of sight velocity dispersion map
        Vrms - rms line of sight veloicty map
        dV, dVD, dVrms - uncertainty on each above quantity
    '''
    
    f = find_gal

    ####################
    # read in kinematics files for view (3 arcsec, 21 pixels)

    # velocity
    V = np.genfromtxt(file_dir + obj_name + '_V_2d.txt', delimiter=',')
    # find barycenter velocity (intrinsic velocity)
    center_axis_index = int(np.floor(V.shape[0]/2))
    #Vbary = V[center_axis_index, center_axis_index]
    # subtract the barycenter velocity #### 01/25/23 - I should subtract this after doing the PA fit. It's a better estimate of the correction
    #V = V - Vbary

    # velocity dispersion
    VD = np.genfromtxt(file_dir + obj_name + '_VD_2d.txt', delimiter=',')

    # uncertainties
    dV = np.genfromtxt(file_dir + obj_name + '_dV_2d.txt', delimiter=',')
    dVD = np.genfromtxt(file_dir + obj_name + '_dVD_2d.txt', delimiter=',')

    # rms velocity ##### 01/25/23 - I should calculate this after doing the PA fit. It's a better estimate of the correction for barycenter velocity
    #Vrms = np.sqrt(V**2 + VD**2)
    #dVrms = np.sqrt((dV*V)**2 + (dVD*VD)**2)/Vrms

    # set scale
    if data_source=='F435W':
        scale = 0.050 # arcsec / pixel HST ACS/WFC
    else:
        print('We do not have the correct information')
    
    ####################
    if plot==True:
        # plot each with surface brightness contours
        # at three arcsec

        n = int(np.around(3/scale))
        xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
        img_3arc = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]

        # V
        plt.clf()
        plt.subplot(122)
        mge_print_contours(img_3arc, f.theta, xc, yc, m.sol,
                           sigmapsf=sigmapsf, #normpsf=normpsf, 
                           scale=scale)
        plt.imshow(V, 
                   extent = [-3, 3, -3, 3],
                   origin='lower',
                  alpha=1)
        plt.title('V')
        plt.pause(1)  # Allow plot to appear on the screen

        # VD
        plt.clf()
        plt.subplot(122)
        mge_print_contours(img_3arc, f.theta, xc, yc, m.sol,
                           sigmapsf=sigmapsf, #normpsf=normpsf, 
                           scale=scale)
        plt.imshow(VD, 
                   extent = [-3, 3, -3, 3],
                   origin='lower',
                  alpha=1)
        plt.title('VD')
        plt.pause(1)  # Allow plot to appear on the screen

        # Vrms
        plt.clf()
        plt.subplot(122)
        mge_print_contours(img_3arc, f.theta, xc, yc, m.sol,
                           sigmapsf=sigmapsf, #normpsf=normpsf, 
                           scale=scale)
        plt.imshow(Vrms, 
                   extent = [-3, 3, -3, 3],
                   origin='lower',
                  alpha=1)
        plt.title('Vrms')
        plt.pause(1)  # Allow plot to appear on the screen
    
    return V, VD, dV, dVD, #Vrms, dVrms


def load_2d_kinematics_with_datacube_contours (file_dir, obj_name, img, find_gal, radius, contour_alpha=0.7, data_source='HST', plot=True, plot_img=True):
    
    '''
    Shows the 2D velocity maps from ppxf fitting.
    Inputs
        file_dir - contains the directory containing the object's image files
        obj_name - SDSS name, e.g. SDSSJ0037-0942
        img - full-sized galaxy image (default is HST)
        find_gal - object created by find_galaxy, with photometric PA
        radius - arcsec radius of image
        contour_alpha - alpha (transparency) of contour lines for easier visualization
        data_source - default HST F435W Filter image
    Outputs (values defined at each pixel)
        V - line of sight velocity map
        VD - line of sight velocity dispersion map
        Vrms - rms line of sight veloicty map # calculate Vrms afterwards!
        dV, dVD, dVrms - uncertainty on each above quantity
    '''
    
    f = find_gal

    ####################
    # read in kinematics files for view (3 arcsec, 21 pixels)
    
    # Create 2D maps from 1D binned velocity measurements
    VD, dVD, V, dV = kinematics_map_systematics(file_dir, obj_name, radius_in_pixels=21)

    # velocity
    # V = np.genfromtxt(file_dir + obj_name + '_V_2d.txt', delimiter=',')
    # find barycenter velocity (intrinsic velocity)
    #center_axis_index = int(np.floor(V.shape[0]/2))
    #Vbary_cen = V[center_axis_index, center_axis_index]
    #Vbary_mean = np.mean(V)
    # subtract the barycenter velocity
    #V = V - Vbary # don't subtract until after the pa fit. it's better

    # velocity dispersion
    #VD = np.genfromtxt(file_dir + obj_name + '_VD_2d.txt', delimiter=',')

    # uncertainties
    #dV = np.genfromtxt(file_dir + obj_name + '_dV_2d.txt', delimiter=',')
    #dVD = np.genfromtxt(file_dir + obj_name + '_dVD_2d.txt', delimiter=',')

    # rms velocity # don't calculate until after the barycenter subtraction with the PA fit
    Vrms = np.sqrt(V**2 + VD**2)
    dVrms = np.sqrt((dV*V)**2 + (dVD*VD)**2)/Vrms

    # set scale
    if data_source=='HST':
        scale = 0.050 # arcsec / pixel HST ACS/WFC
    elif data_source=='KCWI':
        scale = 0.147 # arcsec / pixel
    else:
        print('We do not have the correct information')
    
    ####################
    if plot==True:
        # plot each with surface brightness contours
        # at three arcsec
        cmap='sauron'
        register_sauron_colormap()
        
        fontsize=16
        
        # get extent of HST image at 5 arcsec
        n = int(np.around(radius/hst_scale)) # n is the radius of image in pixels
        img_extent = [-radius, radius, -radius, radius]
        
        # get extent of kinematics from pixels
        width =  V.shape[0]/2 * kcwi_scale
        kin_extent = [-width,width,-width,width]
        
        # V
        plt.clf()
        fig = plt.figure(figsize=(12,4))
        fig.add_subplot(131)
        #mge_print_contours(img, f.theta, xc, yc, m.sol, contour_alpha,
        #                   sigmapsf=sigmapsf, normpsf=normpsf, 
        #                   scale=scale, plot_img=plot_img)
        plt.imshow(V, 
                   extent = kin_extent,
                   origin='lower',
                  alpha=1,
                  #zorder=0,
                  cmap=cmap)#'bwr')
        #plt.contour(img, colors='grey', extent=img_extent, levels=[0.1, 0.2, 0.4, 0.7, 1.0])
        plt.xlabel('arcsec', fontsize=fontsize)
        plt.ylabel('arcsec', fontsize=fontsize)
        plt.title('V', fontsize=fontsize)

        # VD
        fig.add_subplot(132)
        #cnt = mge_print_contours(img, f.theta, xc, yc, m.sol, contour_alpha,
        #                   sigmapsf=sigmapsf, normpsf=normpsf, 
        #                   scale=scale, plot_img=plot_img)
        plt.imshow(VD, 
                   extent = kin_extent,
                   origin='lower',
                  alpha=1,
                  #zorder=0,
                  cmap=cmap)#'bwr')
        #plt.contour(img, colors='grey', extent=img_extent, levels=[0.1, 0.2, 0.4, 0.7, 1.0])
        plt.xlabel('arcsec', fontsize=fontsize)
        plt.ylabel('arcsec', fontsize=fontsize)
        plt.title('VD', fontsize=fontsize)

        # Vrms
        fig.add_subplot(133)
        #mge_print_contours(img, f.theta, xc, yc, m.sol, contour_alpha,
        #                   sigmapsf=sigmapsf, normpsf=normpsf, 
        #                   scale=scale, plot_img=plot_img)
        plt.imshow(Vrms, 
                   extent = kin_extent,
                   origin='lower',
                  alpha=1,
                  #zorder=0,
                  cmap=cmap)#'bwr')
        #plt.contour(img, colors='grey', extent=img_extent, levels=[0.1, 0.2, 0.4, 0.7, 1.0])
        plt.title('Vrms', fontsize=fontsize)
        plt.xlabel('arcsec', fontsize=fontsize)
        plt.ylabel('arcsec', fontsize=fontsize)
        fig.tight_layout()
        plt.pause(1)  # Allow plot to appear on the screen
    
    return V, VD, Vrms, dV, dVD, dVrms#, Vbary_cen, Vbary_mean, center_axis_index
    

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

    
def bin_velocity_maps (file_dir, obj_name, data_source='KCWI'):
    
    '''
    Takes velocity measurements from ppxf and assigns to bin-center coordinates.
    Inputs
        file_dir - contains the directory containing the object's image files
        obj_name - SDSS name, e.g. SDSSJ0037-0942
        Vbary - central velocity (intrinsic or barycenter velocity) of the 2D map
        center_axis_index - axis index of the central pixel
    Outputs (values defined at each bin center)
        V_bin - velocity map
        VD_bin - velocity dispersion map
        Vrms_bin - rms velocity map
        dV_bin, dVD_bin, dVrms_bin - uncertainty in above quantities
        xbin_arcsec, ybin_arcsec - x and y components of bin centers in arcsec
    '''
    
    #KCWI mosaic datacube
    obj_abbr = obj_name[4:9] # e.g. J0029
    name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'
    
    if data_source == 'KCWI':
        scale = 0.1457  # arcsec/pixel
    else:
        print('We have the wrong information')
    
     #######################################
    # get velocity dispersion data
    VD_bin=np.genfromtxt(f'{file_dir}{obj_name}_final_kinematics/{obj_name}_VD_binned.txt',
                 delimiter=',')
    VD_cov = np.genfromtxt(f'{file_dir}{obj_name}_final_kinematics/{obj_name}_covariance_matrix_VD.txt',
                     delimiter=',')
    dVD_bin = np.sqrt(np.diagonal(VD_cov)) # to have error bars
    
    #######################################
    # get velocity data 
    V_bin=np.genfromtxt(f'{file_dir}{obj_name}_final_kinematics/{obj_name}_V_binned.txt',
                 delimiter=',')
    #V_bin = V-np.mean(Vbary)
    V_cov = np.genfromtxt(f'{file_dir}{obj_name}_final_kinematics/{obj_name}_covariance_matrix_V.txt',
                     delimiter=',')
    dV_bin = np.sqrt(np.diagonal(V_cov)) # to have error bars
    
    #######################################
    # rms velocity
    #Vrms_bin = np.sqrt(V_bin**2 + VD_bin**2)
    #dVrms_bin = np.sqrt((dV_bin*V_bin)**2 + (dVD_bin*VD_bin)**2)/Vrms_bin
    
    #######################################
    ## import voronoi binning data
    voronoi_binning_data = fits.getdata(file_dir +'voronoi_binning_' + name + '_data.fits')
    vorbin_pixels = np.genfromtxt(f'{file_dir}voronoi_2d_binning_{name}_output.txt',
                     delimiter='')
    # sort the voronoi bin pixel data by bin
    vorbin_pixels = vorbin_pixels[vorbin_pixels[:,2].argsort()]
    
     #######################################
    ## import voronoi binning data
    voronoi_binning_data = fits.getdata(file_dir +'voronoi_binning_' + name + '_data.fits')
    vorbin_pixels = np.genfromtxt(f'{file_dir}voronoi_2d_binning_{name}_output.txt',
                     delimiter='')
    # sort the voronoi bin pixel data by bin
    vorbin_pixels = vorbin_pixels[vorbin_pixels[:,2].argsort()]
    
    ########################################
    # find bin centers
    xbin, ybin = get_bin_centers (vorbin_pixels, len(voronoi_binning_data))
    
     #######################################
    # Changes - 11/30/22
    #######################################

    # convert to arcsec # kcwi!
    xbin_arcsec = xbin * scale
    ybin_arcsec = ybin * scale

    return V_bin, VD_bin, dV_bin, dVD_bin, xbin_arcsec, ybin_arcsec


def rotate_bins (find_gal, xbin_arcsec, ybin_arcsec, Vrms_bin, plot=True):

    '''
    Rotate x and y bins by PA from find_gal model to align major axis with x-axis.
    Inputs
        find_gal - object created by find_galaxy
        xbin_arcsec, ybin_arcsec - bin center locations in arcsec from bin_velocity_maps
        Vrms_bin - rms velocity map by bin center from bin_velocity_maps
    Outputs
        xbin, ybin - rotated 
    '''
    
    f = findgal
    
    # set PA from mean photometry fitting
    PA = f.theta

    xbin = np.zeros(len(xbin_arcsec))
    ybin = np.zeros(len(ybin_arcsec))

    # rotate the coordinates and append to array
    for i in range(len(xbin_arcsec)):
        xbin[i], ybin[i] = rotate_points(xbin_arcsec[i], ybin_arcsec[i], PA) 

    if plot==True:
        plt.clf()
        plt.figure(figsize=(8,8))
        plt.plot(xbin_arcsec, ybin_arcsec)
        plt.plot(xbin, ybin)
        plt.scatter(xbin,ybin,c=Vrms_bin)
        plt.pause(1)
        plt.pause(1)
        
    return xbin, ybin


##################################################################
# psf fits

def fit_kcwi_sigma_psf (sigma_psf, hst_img, kcwi_img, hst_scale=0.050, kcwi_scale=0.147, plot=False):
    '''
    Fits the KCWI image with a convolution of the HST image and a Gaussian PSF with given sigma.
    Inputs:
        sigma_psf - float, sigma of Gaussian PSF, fitting parameter for optimization 
        hst_img - array (size n), 3 arcsec HST image
        kcwi_img - array (size m), 3 arcsec KCWI image
        hst_scale - float, pixel scale of HST image, default 0.05 "/pix
        kcwi_scale - float, pixel scale of KCWI image, default 0.147 "/pix
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

    # make grid with kcwi_img.shape and multiply by ratio of kcwi_scale/hst_scale
    x = np.arange(kcwi_img.shape[0])*kcwi_scale/hst_scale
    y = np.arange(kcwi_img.shape[1])*kcwi_scale/hst_scale
    # make grid
    yv, xv = np.meshgrid(x, y)
    
    # map the convolved image to the new grid
    mapped_img = map_coordinates(convolved_img, np.array([xv, yv]), mode='nearest')

    if plot == True:
        plt.imshow(mapped_img, origin='lower')
        plt.pause(1)
        plt.clf()
        plt.imshow(kcwi_img, origin='lower')
        
    # make sure the images are aligned
    assert np.argmax(mapped_img) == np.argmax(kcwi_img), 'Mapped convolved image and KCWI image not aligned to same argmax'

    # normalize the images
    # mapped image
    mapped_img_norm = mapped_img / np.max(mapped_img)
    # kcwi image
    kcwi_img_norm = kcwi_img / np.max(kcwi_img)

    # take residual of normed images
    residual = mapped_img_norm - kcwi_img_norm
    
    if plot == True:
        plt.imshow(residual, origin='lower')
        plt.title('Residual')
    
    # return the residual flattened
    return residual.ravel()


# function to optimize the sigma_psf fit

def optimize_sigma_psf_fit (fit_kcwi_sigma_psf, sigma_psf_guess, hst_img, kcwi_img, plot=True):
    '''
    Function to optimize with least squares optimization the fit of KCWI sigma_psf by convolving the HST img
    with Gaussian PSF.
    Inputs:
        - fit_kcwi_sigma_psf - function that fits the KCWI image with HST image convolved with a sigma_psf
        - sigma_psf_guess - float, first guess at the sigma_psf value, pixels
        - hst_img 
        - kcwi_img
    '''
    hst_scale=0.050
    
    # optimize the function
    result = lsq(fit_kcwi_sigma_psf, x0=10, args=(hst_img, kcwi_img))
    
    # state the best-fit sigma-psf and loss function value
    best_fit = result.x[0]*hst_scale
    loss = result.cost
    print(f'Best fit sigma-PSF is {best_fit} arcsec')
    print(f'Best fit loss function value is {loss}')
    
    if best_fit < 0.01:
        print('KCWI PSF sigma is too small. Redo this.')
        plot=True
    
    # show residual
    if plot == True:
        best_residual = result.fun.reshape(kcwi_img.shape)
        plt.imshow(best_residual, origin='lower')
        plt.title('Best fit residual')
        
    return best_fit, loss, best_residual
    
##################################################################
# more things (should have used class all the time)    
# class to collect and save all the attributes I need for jampy
class jampy_details:
    
    def __init__(details, surf_density, mge_sigma, q, kcwi_sigmapst, Vrms_bin, dVrms_bin, V_bin, dV_bin, xbin_phot, ybin_phot, reff):
        details.surf_density=surf_density 
        details.mge_sigma=mge_sigma
        details.q=q 
        details.kcwi_sigmapst=kcwi_sigmapst 
        details.Vrms_bin=Vrms_bin 
        details.dVrms_bind=Vrms_bin
        details.V_bin=V_bin 
        details.dV_bin=dV_bin 
        details.xbin_phot=xbin_phot 
        details.ybin_phot=ybin_phot
        details.reff=reff
        
###################################################################

# define function to do all of the steps together

def plot_kinematics_mge_contours (obj_name, frac, levels, binning, magsteps, magrange,
                                  align_phot=False, add_rot_phot=False, add_rot_kin=False, skip_kcwi=False, debug=False):

    print('#####################################################################################################################')
    print('#####################################################################################################################')
    print()
    print(f'Beginning final kinematics visualization and plotting for object {obj_name}.')
    print()

    obj_abbr = obj_name[4:9] # e.g. J0029
    file_dir = f'{data_dir}mosaics/{obj_name}/' # directory with all files of obj_name

    # import image, center, and crop
    #######################################################################################
    # kcwi datacube
    
    print('################################################')
    print('Getting KCWI datacube')

    kcwi_img, kcwi_5arc_img, kcwi_3arc_img, kcwi_header, \
        kcwi_central_pix_x, kcwi_central_pix_y, kcwi_pa = import_center_crop(file_dir, obj_name, obj_abbr, 
                                                                              data_source='kcwi_datacube', plot=True)

    #######################################################################################
    # hst cutout
    
    print('################################################')
    print('Getting HST cutout')

    hst_full_img, hst_5arc_img, hst_3arc_img, bspl_full_img, bspl_5arc_img, bspl_3arc_img, hst_header, \
        central_pix_x, central_pix_y, exp_time = import_center_crop(hst_dir, obj_name, obj_abbr, 
                                                          data_source='HST', plot=True)


    #######################################################################################
    # take 5 arcsec images
    # TAKE THE FULL bspline IMAGE FOR MGE FITTING. THE CONTOURS ARE NOT RIGHT WHEN I FIT THE 5 ARCSEC IMAGE
    
    img = bspl_full_img # This is the image that will be fit with MGE
    hst_img = hst_5arc_img # These three will be used to plot, e.g. the log image
    bspl_img = bspl_5arc_img
    kcwi_img = kcwi_5arc_img

    #######################################################################################
    # calculate the minimum level for inclusion in the photometry fitting from the background
    # minlevel is half the std of the sky patch, this value is for the sectors_photometry function
    # skylevel (std of sky patch) is for smoothing the log image for plotting next to the contours
    
    print('################################################')
    print('Calculating sky level')

    size=50
    minlevel, noise = calculate_minlevel(img, size) 
    skylevel = minlevel*2

    ################################################
    # smooth the log image for viewing
    
    print('################################################')
    print('Smoothed log images for viewing')

    log_hst_img = np.log10(hst_img.clip(skylevel))
    plt.imshow(log_hst_img, extent=[-5,5,-5,5], origin='lower')
    plt.title(f'{obj_name} HST log image')
    plt.savefig(f'{file_dir}{obj_name}_hst_log_image.png')
    plt.pause(1)
    plt.clf()
    
    log_img = np.log10(bspl_img.clip(skylevel))
    plt.imshow(log_img, extent=[-5,5,-5,5], origin='lower')
    plt.title(f'{obj_name} bspline log image')
    plt.savefig(f'{file_dir}{obj_name}_hst_bspline_log_image.png')
    plt.pause(1)
    plt.clf()

    #######################################################################################
    # estimate psfs
    
    print('################################################')
    print('Estimating PSFs')

    #################################################
    # estimate kcwi psf
    
    print('################################################')
    print('KCWI PSF from hst image')
    
    # debug block
    if debug==False:
        kcwi_sigmapsf, loss, residual = optimize_sigma_psf_fit (fit_kcwi_sigma_psf, hst_img=hst_img, kcwi_img=kcwi_img)
    else:
        kcwi_sigmapsf=-1
    
    # calculate the fwhm and print it, if too low, log it
    kcwi_psf_fwhm = 2.355*np.mean(kcwi_sigma_psf)
    
    print(f'KCWI PSF FWHM = {kcwi_psf_fwhm} arcsec')
    if kcwi_psf_fwhm < 0.1:
        print('KCWI PSF should not be that low.')
        # write lines for log
        l1 = '######################################'
        l2 = obj_name
        l3 = f'KCWI PSF FWHM: {kcwi_psf_fwhm}'
        file = open(f'{file_dir}flag_log.txt', 'wt')
        file.write([l1, l2, l3])
    
    #################################################
    # estimate hst psf
    
    print('################################################')
    print('HST PSF')
    
    # estimate without lower minvalue 
    if debug==False:
        estimate_hst_psf(hst_dir, obj_name, obj_abbr)
        

    #################################################
    # Check the minlevel by eye
    print('Check this minlevel')
    minlevel = 2*1e-6 # this value is typically much lower than needed

    # find sigmapsf and normpsf of hst image
    hst_sigmapsf, hst_normpsf = estimate_hst_psf(hst_dir, obj_name, obj_abbr, minlevel=minlevel)
    # circular psf is weighted mean
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! normpsf is ', hst_normpsf)
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! sigmapsf is ', hst_sigmapsf)
    #hst_sigmapsf_circ = np.average(hst_sigmapsf, weights=hst_normpsf)
    #print('mean psf is ', hst_sigmapsf_circ)
    hst_psf_fwhm = 2.355*np.mean(hst_sigma_psf)
    print(f'HST PSF FWHM = {hst_psf_fwhm} arcsec')
    if hst_psf_fwhm < 0.1:
        print('HST PSF should not be that low.')
        # write lines for log
        l1 = '######################################'
        l2 = obj_name
        l3 = f'HST PSF FWHM: {hst_psf_fwhm}'
        file = open(f'{file_dir}flag_log.txt', 'wt')
        file.write([l1, l2, l3])

    #######################################################################################
    # fit the photometry
    
    print('################################################')
    print('Fitting photometry')

    #################################################
    # Model the central light ellipse
    
    print('################################################')
    print('First fit of central light ellipse')

    plt.clf()
    #plt.clf()
    f = find_galaxy(img, fraction=frac, plot=1, quiet=True)
    eps = f.eps
    theta = f.theta
    cen_y = f.ypeak
    cen_x = f.xpeak
    #plt.contour(img, colors='grey', extent=[-3,3,-3,3], levels=[0.1, 0.2, 0.4, 0.7, 1.0], zorder=1)
    plt.title(f'fract: {frac}, theta: {np.around(theta,2)}')
    plt.pause(1)

    ################################################
    # Perform galaxy photometry with full image
    
    print('################################################')
    print('Calculating photometry of sectors')

    plt.clf()
    s = sectors_photometry(img, eps, theta, cen_x, cen_y, minlevel=minlevel, plot=1)
    plt.pause(1)  # Allow plot to appear on the screen

    ################################################
    # Do the first MGE fit
    
    print('################################################')
    print('First MGE fit')

    # select number of gaussians to fit
    ngauss = 12

    # pixel scale
    scale = hst_scale

    # psf - take from the MGE psf model above
    sigmapsf = hst_sigmapsf
    normpsf = hst_normpsf
    #seeing_fwhm = 0.8 # arcsec
    #sigmapsf = seeing_fwhm / scale / 2.355 # pixels, 2.355 is fwhm/sigma
    #normpsf = 1

    # exposure time
    #exp_time = hstF435_header['EXPTIME']

    # fit and plot

    plt.clf()
    m = mge_fit_sectors(s.radius, s.angle, s.counts, eps,
                        ngauss=ngauss, sigmapsf=sigmapsf, normpsf=normpsf,
                        scale=scale, plot=1, bulge_disk=0, linear=0,
                       quiet=True)
    plt.pause(1)

    # plot the contours on the image
    plot_contours_531 (img, f, m, sigmapsf, normpsf, contour_alpha=0.4, data_source='HST')

    ######################################################
    # Do the actual MGE fit
    
    print('################################################')
    print('Regularized MGE fit')

    # fit and plot

    plt.clf()
    m = mge_fit_sectors_regularized(s.radius, s.angle, s.counts, eps,
                        ngauss=ngauss, sigmapsf=sigmapsf, normpsf=normpsf,
                        scale=scale, plot=1, bulge_disk=0, linear=0,
                                   quiet=True)
    plt.pause(1) 
    
        
    # update the ellipticity measurement
    # m solution has set of Gaussians, weight by the normalized total counts and average to get the best estimate
    weights = m.sol[0]/np.sum(m.sol[0])
    qs = m.sol[2]
    q = np.sum(weights*qs)
    ellipticity = 1 - q
    print(f'Best fit ellipticity = {ellipticity}')

    # plot the contours on the image
    plot_contours_531 (img, f, m, sigmapsf, normpsf, contour_alpha=0.4, data_source='HST')
    
    ########### NEW THING FROM JAM NOTEBOOK###############################
    print('################################################')
    print('Converting to surface density and real units!')
    # bring in dust extinction table
    extinctions = pd.read_csv(f'{data_dir}slacs_Iband_extinctions.csv')
    extinction = extinctions.loc[extinctions.obj_name == obj_name, 'A_I'].values[0]
    # convert sigma (of gaussian components) from pixels to arcsec and surface brightness to surface density
    mge_sigma, surf_density = convert_mge_model_outputs (m, exp_time, extinction, qs, data_source='F435W')
    
    #######################################
    # get reff
    print('#################################################')
    print('Calculating half-light isophote and circularized half-light radius')
    reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(surf_density, mge_sigma, qs)
    # slacs reff
    slacs_table = np.genfromtxt(f'{data_dir}slacs_tableA1.txt', delimiter='', dtype='U10')
    slacs_table_name = obj_name[4:]
    slacs_reffs = slacs_table[:,7].astype(float)
    reff_slacs = slacs_reffs[slacs_table[:,0]==slacs_table_name]
    print('effective radius mge: ', reff)
    print('effective radius slacs: ', reff_slacs)
    if reff < 0.5*reff_slacs:
        print('Effective radius should not be that small')
        # write lines for log
        l1 = '######################################'
        l2 = obj_name
        l3 = f'Effective radius from MGE: {reff}'
        l4 = f'Effective radius from SLACS: {reff_slacs}'
        file = open(f'{file_dir}flag_log.txt', 'wt')
        file.write([l1, l2, l3, l4])
        
    #######################################################################################
    # bring in kinematics
    
    print('################################################')
    print('Loading kinematics')

    V, VD, Vrms, \
    dV, dVD, dVrms = load_2d_kinematics_with_datacube_contours (file_dir, obj_name, img/np.max(img), f,
                                                                           radius=5, contour_alpha=0.7, data_source='HST', 
                                                                           plot=True, plot_img=False)
    
    # remove pixels that have VD > 350

    ################################################
    # bin velocity maps
    
    print('################################################')
    print('Binning velocity maps')

    #V_bin, VD_bin, Vrms_bin, \
    #dV_bin, dVD_bin, dVrms_bin, \
    V_bin, VD_bin,  \
    dV_bin, dVD_bin, \
    xbin_arcsec, ybin_arcsec = bin_velocity_maps (file_dir, obj_name, data_source='KCWI')

    # Check binning

    # plot with arcsec
    width =  V.shape[0]/2 * kcwi_scale
    extent = [-width,width,-width,width]
    plt.figure(figsize=(8,8))
    plt.imshow(V, origin='lower', extent=extent, cmap='sauron')
    plt.scatter(xbin_arcsec, ybin_arcsec, color='k', marker='.')

    ##################################################
    # correct barycenter velocity and PA
    
    print('################################################')
    print('Correcting barycentre velocity and calculating PA_kin')
    
    # kinematic PA is measured from the kinematic map's N thru E (ie. "counter clockwise of up", if KCWI aperture position angle was not set to 0, this will not be really north)
    PA_kin_, dPA_kin, velocity_offset = fit_kinematic_pa(xbin_arcsec, ybin_arcsec, V_bin)#, debug=debug)
    # convert to N thru E
    PA_kin = PA_kin_ + kcwi_pa
    
    plt.pause(1)
    plt.clf()

    # Correct the velocity
    V = V - velocity_offset
    V_bin = V_bin - velocity_offset

    # PA_phot is theta from find_galaxy, measured from negative x-axis
    # convert to N thru E (this will be true north because HST images are oriented that way [I think...])
    PA_phot = 270 - theta

    ##################################################
    # rotate the bins by the kinematics PA and the photometric PA
    # plot the rotation with the "non-symmetrized velocity field"
    
    #print('################################################')
    print('Checking PA by rotating bins by each PA')

    # kinematics
    xbin_kin, ybin_kin, _ = rotate_bins (PA_kin+180, xbin_arcsec, ybin_arcsec, V_bin, plot=True)
    plt.pause(1)
    plt.clf()

    # phomotometry
    xbin_phot, ybin_phot, _ = rotate_bins (PA_phot+180, xbin_arcsec, ybin_arcsec, V_bin, plot=True)
    plt.pause(1)
    plt.clf()
    
    # both kinematic and photometric PA should be less than 180
    print(f'KCWI PA : {kcwi_pa}') # PA of KCWI aperture relative to N
    print(f'Kinematic PA: {PA_kin}')
    print(f'Photometric PA: {PA_phot}') # PA from photometry
    
    #######################################
    # rms velocity
    print('################################################')
    print('Calculating rms velocity')
    Vrms_bin = np.sqrt(V_bin**2 + VD_bin**2)
    dVrms_bin = np.sqrt((dV_bin*V_bin)**2 + (dVD_bin*VD_bin)**2)/Vrms_bin
    
    #################################################################################################################
    ### make final plots
    ##################################################
    # plot the velocity map without the axis, save figure
    
    print('################################################')
    print('Plotting and saving velocity map without axis')
    
    plt.rcParams.update({'font.size': 18})
    
    ###################################################
    # velocity map without contours
    # first rotate the map by the KCWI PA to position it with y axis up pointing N
    
    # velocity_range
    vel_range = np.nanmax(np.abs(V))
    # get extent of kinematics from pixels
    width =  V.shape[0]/2 * kcwi_scale
    kin_extent = [-width,width,-width,width]

    plt.figure(figsize=(8,8))
    plt.imshow(V, origin='lower', extent=kin_extent, cmap='sauron', vmin=-vel_range-5, vmax=vel_range+5)

    plt.axis('off')

    plt.savefig(f'{file_dir}{obj_name}_velocity_map_no_axis.png', bbox_inches='tight', pad_inches=0.0)
    
    ###################################################
    # velocity DISPERSION map without contours
    # first rotate the map by the KCWI PA to position it with y axis up pointing N
    
    # velocity_range
    vel_max = np.nanmax(VD)
    vel_min = np.nanmin(VD)
    # get extent of kinematics from pixels
    #width =  V.shape[0]/2 * kcwi_scale
    #kin_extent = [-width,width,-width,width]

    plt.figure(figsize=(8,8))
    plt.imshow(VD, origin='lower', extent=kin_extent, cmap='sauron', vmin=vel_min-5, vmax=vel_max+5)

    plt.axis('off')

    plt.savefig(f'{file_dir}{obj_name}_VD_map_no_axis.png', bbox_inches='tight', pad_inches=0.0)
    
    ##################################################
    # load the velocity map sans axis, rotate by KCWI PA, then add to a figure with the contours
    
    print('################################################')
    print('Loading velocity map and rotating, adding to figure')

    #read the image and map
    input_map = Image.open(f'{file_dir}{obj_name}_velocity_map_no_axis.png')
    angle = kcwi_pa
    output_map = input_map.rotate(angle)

    plt.figure(figsize=(8,8))
    plt.imshow(output_map, origin='upper', extent=kin_extent, cmap='sauron', vmin=-vel_range-5, vmax=vel_range+5)
    
    # get image extent and cut to 5 
    n = int(np.around(5/scale))
    img_extent = np.array([-n, n, -n, n])
    mge_print_contours(img, f.theta, cen_x, cen_y, m.sol, binning=binning, magsteps=magsteps, magrange=magrange,
                       sigmapsf=sigmapsf, normpsf=normpsf, 
                       scale=hst_scale, extent=img_extent)

    plt.axis('off')

    plt.savefig(f'{file_dir}{obj_name}_velocity_map_bspline_contours_no_axis.png', bbox_inches='tight', pad_inches=0.0)
    
    ###################################################
    # do the same with VD
    print('################################################')
    print('Loading velocity map and rotating, adding to figure')

    #read the image and map
    input_map = Image.open(f'{file_dir}{obj_name}_VD_map_no_axis.png')
    angle = kcwi_pa
    output_map = input_map.rotate(angle)

    plt.figure(figsize=(8,8))
    plt.imshow(output_map, origin='upper', extent=kin_extent, cmap='sauron', vmin=vel_min, vmax=vel_max)
    
    # get image extent and cut to 5 
    n = int(np.around(5/scale))
    img_extent = np.array([-n, n, -n, n])
    mge_print_contours(img, f.theta, cen_x, cen_y, m.sol, binning=binning, magsteps=magsteps, magrange=magrange,
                       sigmapsf=sigmapsf, normpsf=normpsf, 
                       scale=hst_scale, extent=img_extent)

    plt.axis('off')

    plt.savefig(f'{file_dir}{obj_name}_VD_map_bspline_contours_no_axis.png', bbox_inches='tight', pad_inches=0.0)
    
    
    ###################################################
    # do the same with the log-smoothed hst image and bspline model
    
    plt.figure(figsize=(8,8))
    plt.imshow(log_img, origin='lower')
    plt.axis('off')
    plt.savefig(f'{file_dir}{obj_name}_hst_bspline_log_img_no_axis.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.imshow(log_hst_img, origin='lower')
    plt.axis('off')
    plt.savefig(f'{file_dir}{obj_name}_hst_log_img_no_axis.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()

    ##################################################
    # load the velocity map and hst log image sans axis, rotate by photometric PA, then add to a figure
    
    print('################################################')
    print('Loading velocity map and hst image and rotating, adding to figures')
          
    #read the image and map
    input_V_map = Image.open(f'{file_dir}{obj_name}_velocity_map_bspline_contours_no_axis.png')
    input_VD_map = Image.open(f'{file_dir}{obj_name}_VD_map_bspline_contours_no_axis.png')
    input_bspl_image = Image.open(f'{file_dir}{obj_name}_hst_bspline_log_img_no_axis.png')
    input_hst_image = Image.open(f'{file_dir}{obj_name}_hst_log_img_no_axis.png')

    ############################################
    #rotate map and image along PA_phot
    
    print('################################################')
    print('Aligning map and image along photometric PA')
    
    angle = 270-PA_phot
    if add_rot_phot == True: # try to get the red on the right
        angle=angle-180
    output_V_map = input_V_map.rotate(angle)
    output_VD_map = input_VD_map.rotate(angle)
    output_bspl_image = input_bspl_image.rotate(angle)
    output_hst_image = input_hst_image.rotate(angle)

    arcsec_width = 10#scale*43
    output_pixel_scale = arcsec_width/output_map.size[0]
    arcsec_in_pixels = 1/output_pixel_scale
    ticks_arcsec = np.linspace(-5,5,11)
    ticks = []

    for i in ticks_arcsec:
        ticks.append(output_map.size[0]/2 + arcsec_in_pixels * i)
    
    ##########################################
    # plot V map
    plt.figure(figsize=(8,8))
    V_plot = plt.imshow(V, extent=extent, cmap='sauron', vmin=-vel_range-5, vmax=vel_range+5)
    plt.imshow(output_V_map)
    #plt.gca().set_visible(False)
    cbar = plt.colorbar(V_plot, shrink=0.85)
    cbar.set_label(r'V [km/s]')

    plt.xticks(ticks, labels=ticks_arcsec)
    plt.yticks(ticks, labels=ticks_arcsec)

    # plot the major axes
    # difference in position angle
    print('PA_kin ', PA_kin)
    print('PA_phot ', PA_phot)
    delta_PA = int(PA_kin - PA_phot)
    abs_min_delta_PA = np.min(np.abs([delta_PA,180-np.abs(delta_PA)])) # print the angle that is less than 90 degrees
    print('delta_PA ', delta_PA)
    print('delta_PA < 90', abs_min_delta_PA)
    # kinematic
    x = np.linspace(-arcsec_width/2, arcsec_width/2, 1000)
    ykin = -np.tan(np.radians(delta_PA))*x # I think it's negative because the image is "technically" upside down
    y_pix = ykin * arcsec_in_pixels + 434/2
    x_pix = np.linspace(0,434,1000)
    plt.plot(x_pix,y_pix, 
             #label='Kinematic', 
             c='k',
            linestyle=':',
            linewidth=1)
    # photometric
    yphot = np.ones(1000)*434/2
    plt.plot(x_pix,yphot, 
             #label='Photometric', 
             c='k',
            linestyle=':',
            linewidth=1)
    # line for contours in legend
    ycontour = -np.ones(len(x_pix))
    plt.plot(x_pix, ycontour, label=f'{magsteps} mag', color='k')
    # add the difference in angle and the ellipticity
    delta_PA_symb = r'$\Delta_{PA}$'
    degree_symb = r'$^\circ$'
    ellipt_symb = r'$\epsilon$'
    # add the difference
    plt.annotate(f'{delta_PA_symb} = {abs_min_delta_PA}{degree_symb}±{np.rint(dPA_kin).astype(int)}', (10,424), color='k')
    plt.annotate(f'{ellipt_symb} = {np.around(ellipticity,2)}', (10,404), color='k')
    
    plt.ylim(434,0)
    plt.xlim(0,434)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.legend(loc='upper right', frameon=False)

    #plt.legend(loc='upper left',
    #        fontsize=12)

    #plt.axis('off')

    plt.savefig(f'{file_dir}{obj_name}_bspline_velocity_map_aligned_phot_axis.png', bbox_inches='tight')
    plt.savefig(f'{file_dir}{obj_name}_bspline_velocity_map_aligned_phot_axis.pdf', bbox_inches='tight')
    plt.pause(1)
    plt.clf()
    
    ##########################################
    # plot VD map
    plt.figure(figsize=(8,8))
    VD_plot = plt.imshow(VD, extent=extent, cmap='sauron', vmin=vel_min, vmax=vel_max)
    plt.imshow(output_VD_map)
    #plt.gca().set_visible(False)
    cbar = plt.colorbar(VD_plot, shrink=0.85)
    cbar.set_label(r'V [km/s]')

    plt.xticks(ticks, labels=ticks_arcsec)
    plt.yticks(ticks, labels=ticks_arcsec)

    # plot the major axes
    # difference in position angle
    #print('PA_kin ', PA_kin)
    #print('PA_phot ', PA_phot)
    delta_PA = int(PA_kin - PA_phot)
    abs_min_delta_PA = np.min(np.abs([delta_PA,180-np.abs(delta_PA)])) # print the angle that is less than 90 degrees
    #print('delta_PA ', delta_PA)
    #print('delta_PA < 90', abs_min_delta_PA)
    # kinematic
    x = np.linspace(-arcsec_width/2, arcsec_width/2, 1000)
    ykin = -np.tan(np.radians(delta_PA))*x # I think it's negative because the image is "technically" upside down
    y_pix = ykin * arcsec_in_pixels + 434/2
    x_pix = np.linspace(0,434,1000)
    plt.plot(x_pix,y_pix, 
             #label='Kinematic', 
             c='k',
            linestyle=':',
            linewidth=1)
    # photometric
    yphot = np.ones(1000)*434/2
    plt.plot(x_pix,yphot, 
             #label='Photometric', 
             c='k',
            linestyle=':',
            linewidth=1)
    # line for contours in legend
    ycontour = -np.ones(len(x_pix))
    plt.plot(x_pix, ycontour, label=f'{magsteps} mag', color='k')
    # add the difference in angle and the ellipticity
    delta_PA_symb = r'$\Delta_{PA}$'
    degree_symb = r'$^\circ$'
    ellipt_symb = r'$\epsilon$'
    # add the difference
    plt.annotate(f'{delta_PA_symb} = {abs_min_delta_PA}{degree_symb}±{np.rint(dPA_kin).astype(int)}', (10,424), color='k')
    plt.annotate(f'{ellipt_symb} = {np.around(ellipticity,2)}', (10,404), color='k')
    
    plt.ylim(434,0)
    plt.xlim(0,434)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.legend(loc='upper right', frameon=False)

    plt.savefig(f'{file_dir}{obj_name}_bspline_VD_map_aligned_phot_axis.png', bbox_inches='tight')
    plt.savefig(f'{file_dir}{obj_name}_bspline_VD_map_aligned_phot_axis.pdf', bbox_inches='tight')
    plt.pause(1)
    plt.clf()

    ################################################
    # save log-smoothed image rotated by photometric axis
    
    fig, axs = plt.subplots(figsize=(6.65,6.65)) # this size will make it the same size as the map
    plt.imshow(output_bspl_image)
    axs.set_facecolor('k')
    
    plt.xticks(ticks, labels=ticks_arcsec)
    plt.yticks(ticks, labels=ticks_arcsec)

    # plot the major axes
    # difference in position angle
    #print('PA_kin ', PA_kin)
    #print('PA_phot ', PA_phot)
    #print('delta_PA ', abs_min_delta_PA)
    # kinematic
    x = np.linspace(-arcsec_width/2, arcsec_width/2, 1000)
    ykin = -np.tan(np.radians(delta_PA))*x # I think it's negative because the image is "technically" upside down
    y_pix = ykin * arcsec_in_pixels + 434/2
    x_pix = np.linspace(0,434,1000)
    plt.plot(x_pix,y_pix, 
             #label='Kinematic', 
             c='w',
            linestyle=':',
            linewidth=1)
    # photometric
    yphot = np.ones(1000)*434/2
    plt.plot(x_pix,yphot, 
             #label='Photometric', 
             c='w',
            linestyle=':',
            linewidth=1)
    # add the difference
    plt.annotate(f'{delta_PA_symb} = {abs_min_delta_PA}{degree_symb}±{np.rint(dPA_kin).astype(int)}', (10,424), color='w')
    plt.annotate(f'{ellipt_symb} = {np.around(ellipticity,2)}', (10,404), color='w')
    # plot north and east
    angle=np.radians(angle)
    vertex = np.array([354, 354])
    r = 70 #radius of arrows, pixels
    N = vertex + r*np.array([-np.sin(angle), -np.cos(angle)])
    E = vertex + r*np.array([-np.cos(angle), np.sin(angle)])
    plt.annotate('N', vertex, N, color='yellowgreen', ha='center', va='center', arrowprops=dict(arrowstyle="<|-", linewidth=3, color='yellowgreen'))
    plt.annotate('E', vertex, E, color='yellowgreen', ha='center', va='center', arrowprops=dict(arrowstyle="<|-", linewidth=3, color='yellowgreen'))
    #plt.scatter(vertex[0], vertex[1], color='yellowgreen')

    plt.ylim(434,0)
    plt.xlim(0,434)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')

    plt.savefig(f'{file_dir}{obj_name}_hst_bspline_log_img_aligned_phot_axis.png', bbox_inches='tight')
    plt.savefig(f'{file_dir}{obj_name}_hst_bspline_log_img_aligned_phot_axis.pdf', bbox_inches='tight')
    plt.pause(1)
    plt.clf()
    
    ################################################
    # save log-smoothed image rotated by photometric axis
    
    fig, axs = plt.subplots(figsize=(6.65,6.65)) # this size will make it the same size as the map
    plt.imshow(output_hst_image)
    axs.set_facecolor('k')
    
    plt.xticks(ticks, labels=ticks_arcsec)
    plt.yticks(ticks, labels=ticks_arcsec)

    # plot the major axes
    # difference in position angle
    #print('PA_kin ', PA_kin)
    #print('PA_phot ', PA_phot)
    #print('delta_PA ', abs_min_delta_PA)
    # kinematic
    x = np.linspace(-arcsec_width/2, arcsec_width/2, 1000)
    ykin = -np.tan(np.radians(delta_PA))*x # I think it's negative because the image is "technically" upside down
    y_pix = ykin * arcsec_in_pixels + 434/2
    x_pix = np.linspace(0,434,1000)
    plt.plot(x_pix,y_pix, 
             #label='Kinematic', 
             c='w',
            linestyle=':',
            linewidth=1)
    # photometric
    yphot = np.ones(1000)*434/2
    plt.plot(x_pix,yphot, 
             #label='Photometric', 
             c='w',
            linestyle=':',
            linewidth=1)
    # add the difference
    plt.annotate(f'{delta_PA_symb} = {abs_min_delta_PA}{degree_symb}±{np.rint(dPA_kin).astype(int)}', (10,424), color='w')
    plt.annotate(f'{ellipt_symb} = {np.around(ellipticity,2)}', (10,404), color='w')
    # plot north and east
    #angle=np.radians(angle)
    #vertex = np.array([354, 354])
    #r = 70 #radius of arrows, pixels
    #N = vertex + r*np.array([-np.sin(angle), -np.cos(angle)])
    #E = vertex + r*np.array([-np.cos(angle), np.sin(angle)])
    plt.annotate('N', vertex, N, color='yellowgreen', ha='center', va='center', arrowprops=dict(arrowstyle="<|-", linewidth=3, color='yellowgreen'))
    plt.annotate('E', vertex, E, color='yellowgreen', ha='center', va='center', arrowprops=dict(arrowstyle="<|-", linewidth=3, color='yellowgreen'))
    #plt.scatter(vertex[0], vertex[1], color='yellowgreen')

    plt.ylim(434,0)
    plt.xlim(0,434)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')

    plt.savefig(f'{file_dir}{obj_name}_hst_log_img_aligned_phot_axis.png', bbox_inches='tight')
    plt.savefig(f'{file_dir}{obj_name}_hst_log_img_aligned_phot_axis.pdf', bbox_inches='tight')
    plt.pause(1)
    plt.clf()

    ############################################
    # now rotate image along PA_kin
    #### IF YOU DECIDE YOU NEED THIS, JUST GET IT FROM A PREVIOUS NOTEBOOK
    
    #############################################
    print('#########################################')
    print('Saving details for jampy use')
    
    details_for_jampy = jampy_details(surf_density, mge_sigma, qs, kcwi_sigmapsf, Vrms_bin, dVrms_bin, V_bin, dV_bin, xbin_phot, ybin_phot, reff)
    # save to pickle
    with open(f'{file_dir}{obj_name}_details_for_jampy.pkl', 'wb') as f:
        pickle.dump(details_for_jampy, f)
    
    print()
    print()
    print('################################################################################################################################')
    print(f"Job's finished!")
    
    # return kinematic PA and error, photometric PA, difference in PA, ellipticity, kcwi_sigmapsf, hst_sigmapsf
    return PA_kin, dPA_kin, PA_phot, delta_PA, ellipticity, kcwi_sigmapsf #, mge_sigma, surf_density, q, V_bin, dV_bin#, hst_sigmapsf_circ