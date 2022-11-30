'''
05/14/22 - Modules used for mgefit and jampy on SLACS lenses from notebooks.
'''

# import general libraries and modules
import numpy as np
np.set_printoptions(threshold=10000)
import matplotlib.pyplot as plt
plt.rcParams[figure.figsize] = (8, 6)
import pandas as pd
import warnings
#warnings.filterwarnings( ignore, module = matplotlib..* )
#warnings.filterwarnings( ignore, module = plotbin..* )
from os import path

# astronomy/scipy
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import rotate
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares as lsq
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.cosmology import Planck18 as cosmo  # Planck 2018


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
from plotbin.plot_velfield import plot_velfield

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



def import_center_crop (file_dir, obj_abbr, data_source='F435W', plot=True):

    '''
    This function imports a file from the object directory, crops the image to 2 arcsec, and returns both images. 

    Inputs:
        file_dir - contains the directory containing the object's image files
        obj_abbr - SDSS name, e.g. SDSSJ0037-0942 abbreviated to J0037
        data_source - which image is used, by default HST F435W filter; if kcwi_datacube, use image integrated over the spectrom so it's a 2D image instead of the cube
        plot - Return both images in line

    Returns:
        _img - file image with arcsec axes
        _2arc_img = image cropped to 2 arcsecond radius
        header = input fits file header
    '''

     ##############################################################################
    # kcwi datacube

    if data_source == 'kcwi_datacube"

        file = file_dir + f"/KCWI_{obj_abbr}_icubes_mosaic_0.1457_2Dintegrated.fits"
        hdu = fits.open(file)
        kcwi_img = hdu[0].data
        header = hdu[0].header
        
        # pixel scale
        kcwi_scale = 0.147  # arcsec/pixel r_eff_V
        img_half_extent = kcwi_img.shape[0]/2 * kcwi_scale

        # crop the image to ~ 2 arcsec radius
        kcwi_2arc_img, _, _ = crop_center_image(kcwi_img, 2, kcwi_scale, 'argmax')
        
        if plot == True:
            # plot full image
            plt.clf()
            plt.imshow(kcwi_img, extent = [-img_half_extent,img_half_extent,-img_half_extent,img_half_extent])
            plt.title('KCWI datacube')
            plt.pause(1)
            # plot cropped image
            plt.clf()
            plt.imshow(kcwi_2arc_img, origin='lower', extent=[-2,2,-2,2])#, extent=[0,50,0,50])
            plt.contour(kcwi_2arc_img, colors='grey', extent=[-2,2,-2,2])
            plt.title('KCWI datacube')
            plt.pause(1)

        return kcwi_img, kcwi_2arc_img, header

   ###################################################################################
    # F435W cutout

    elif data_source == 'F435W'

        file = f'/data/end_product/SLACS{obj_name}/cutouts{obj_name}_F435W.fits'
        hdu = fits.open(file)
        hstF435_img = hdu[0].data
        header = hdu[0].header

        # pixel scale
        hst_scale = 0.050 # ACS/WFC
        img_half_extent = hst_img.shape[0]/2 * hst_scale

        # crop the image to 2 arcsec
        hstF435_2arc_img, _, _  = crop_center_image(hstF435_img, 2, hst_scale, 'center')

        if plot == True:
            # plot the image
            plt.clf()
            plt.imshow(hstF435_img, extent = [-img_half_extent,img_half_extent,-img_half_extent,img_half_extent]) 
            plt.title('HST F435W')
            plt.pause(1)
            # plot cropped image   
            plt.clf()
            plt.imshow(hstF435_2arc_img, origin='lower', extent=[-2,2,-2,2])
            plt.contour(hstF435_2arc_img, colors='k', extent=[-2,2,-2,2])
            plt.title('HST F435W')
            plt.pause(1)

        return hstF435_img, hstF435_2arc_img, header


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



def convert_mge_model_outputs (model, exp_time, data_source='F435W')

    '''
    This function takes model outputs and converts them to what is needed for jampy.
    sigma is converted from pixels to arcsec
    surface brightness is converted to surface density (L_sol_I pc−2)
    Inputs:
        model - output object from mge_fit_sectors
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
    #### 4/19/22 I don't know where I got this surface brightness and extinction stuff...
    # convert to johnson i band
    # Here 20.840 is the photometric zeropint, 0.1 is a correction for infinite aperture to be applied 
    # for surface brightness measurements, and AI is the extinction in the I-band
    # dust extinction ~ 0.05 from https://irsa.ipac.caltech.edu/workspace/TMP_lFD64I_6198/DUST/SDSSJ0037-0942.v0002/extinction.html
    AI = 0.05
    iband_surf_br = 20.840 + 0.1 + 5 * np.log10(scale) + 2.5 * np.log10(exp_time) - 2.5 * np.log10(peak_surf_br) - AI
    # convert to surface density (L_sol_I pc−2)
    M_sol_I = 4.08
    surf_density = (64800/np.pi)**2 * 10**( 0.4 * (M_sol_I - iband_surf_br))

    return sigma, surf_density


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

            
def load_2d_kinematics (file_dir, obj_name, img, find_gal, sigmapsf, data_source=='F435W', plot=True):
    
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
    
    return V, VD, Vrms, dV, dVD, dVrms
    

    
def bin_velocity_maps (file_dir, obj_abbr, data_source='KCWI', plot=True):
    
    '''
    Takes velocity measurements from ppxf and assigns to bin-center coordinates.
    Inputs
        file_dir - contains the directory containing the object's image files
        obj_abbr - SDSS name, e.g. SDSSJ0037-0942 abbreviated to J0037
    Outputs (values defined at each bin center)
        V_bin - velocity map
        VD_bin - velocity dispersion map
        Vrms_bin - rms velocity map
        dV_bin, dVD_bin, dVrms_bin - uncertainty in above quantities
        xbin_arcsec, ybin_arcsec - x and y components of bin centers in arcsec
    '''
    
    if data_source = 'KCWI':
        scale = 0.147  # arcsec/pixel
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

    
    ################
    if plot==True:
        # plot with pixels first
        plt.figure(figsize=(8,8))
        plt.imshow(Vrms, origin='lower')
        plt.scatter(x_cen_bins, y_cen_bins, color='k', marker='.')

        # plot with arcsec
        extent = [-3,3,-3,3]
        plt.figure(figsize=(8,8))
        plt.imshow(Vrms, origin='lower', extent=extent)
        plt.scatter(xbin_arcsec, ybin_arcsec, color='k', marker='.')

    return V_bin, VD_bin, Vrms_bin, dV_bin, dVD_bin, dVrms_bin, xbin_arcsec, ybin_arcsec


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


#######################################################################

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
    
    # show residual
    if plot == True:
        best_residual = result.fun.reshape(kcwi_img.shape)
        plt.imshow(best_residual, origin='lower')
        plt.title('Best fit residual')
        
    return best_fit, loss, best_residual
    
    
