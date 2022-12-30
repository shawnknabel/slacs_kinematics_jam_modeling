# 11/30/22 - This notebook plots the velocity and velocity dispersions with the KCWI datacube contours and photometric/kinematic position angles.

# Modified from 062322_J0037_mge_jam_starttofinish.ipynb



################################################################

# import general libraries and modules
import numpy as np
np.set_printoptions(threshold=10000)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams.update({'font.size': 14})
import pandas as pd
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings( "ignore", module = "plotbin\..*" )
from os import path

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
from pafit.fit_kinematic_pa import fit_kinematic_pa

# my functions
from slacs_mge_jampy import crop_center_image
from slacs_mge_jampy import import_center_crop
from slacs_mge_jampy import try_fractions_for_find_galaxy
from slacs_mge_jampy import convert_mge_model_outputs
from slacs_mge_jampy import plot_contours_321
from slacs_mge_jampy import load_2d_kinematics
from slacs_mge_jampy import bin_velocity_maps
from slacs_mge_jampy import rotate_bins
from slacs_mge_jampy import osipkov_merritt_model
from slacs_mge_jampy import find_half_light
from slacs_mge_jampy import calculate_minlevel
from slacs_mge_jampy import fit_kcwi_sigma_psf
from slacs_mge_jampy import optimize_sigma_psf_fit
from slacs_mge_jampy import estimate_hst_psf

################################################################
# some needed information
kcwi_scale = 0.147  # arcsec/pixel
hst_scale = 0.050 # ACS/WFC

# info for J0037
# B band (F435W) dust extinction ~ 0.116 from https://irsa.ipac.caltech.edu/applications/DUST/
#extinction = 0.116
### photometric zeropoint for F435W as of 2007 was 25.745
#photometric_zeropoint = 25.745
# redshift, convert to angular diameter dist in Mpc
#z = 0.195
#distance = cosmo.angular_diameter_distance(z).value

# specify object directory and name
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/' # data directory

#################################################
# objects
obj_names = ['SDSSJ0029-0055',
             'SDSSJ0037-0942',
             'SDSSJ0330-0020',
             'SDSSJ1112+0826',
             'SDSSJ1204+0358',
             'SDSSJ1250+0523',
             'SDSSJ1306+0600',
             'SDSSJ1402+6321',
             'SDSSJ1531-0105',
             'SDSSJ1538+5817',
             'SDSSJ1621+3931',
             'SDSSJ1627-0053',
             'SDSSJ1630+4520',
             'SDSSJ2303+1422'
            ]

#################################################

# functions

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

# define a function that plots the contours from the datacube instead of the mge contours

def load_2d_kinematics_with_datacube_contours (file_dir, obj_name, img, find_gal, contour_alpha=0.7, data_source='F435W', plot=True, plot_img=True):
    
    '''
    Shows the 2D velocity maps from ppxf fitting.
    Inputs
        file_dir - contains the directory containing the object's image files
        obj_name - SDSS name, e.g. SDSSJ0037-0942
        img - full-sized galaxy image (default is HST F435W)
        find_gal - object created by find_galaxy, with photometric PA
        contour_alpha - alpha (transparency) of contour lines for easier visualization
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
    
    # Create 2D maps from 1D binned velocity measurements
    VD, dVD, V, dV = kinematics_map_systematics(file_dir, obj_name, radius_in_pixels=21)

    # velocity
    # V = np.genfromtxt(file_dir + obj_name + '_V_2d.txt', delimiter=',')
    # find barycenter velocity (intrinsic velocity)
    center_axis_index = int(np.floor(V.shape[0]/2))
    Vbary = V[center_axis_index, center_axis_index]
    # subtract the barycenter velocity
    V = V - Vbary

    # velocity dispersion
    #VD = np.genfromtxt(file_dir + obj_name + '_VD_2d.txt', delimiter=',')

    # uncertainties
    #dV = np.genfromtxt(file_dir + obj_name + '_dV_2d.txt', delimiter=',')
    #dVD = np.genfromtxt(file_dir + obj_name + '_dVD_2d.txt', delimiter=',')

    # rms velocity
    Vrms = np.sqrt(V**2 + VD**2)
    dVrms = np.sqrt((dV*V)**2 + (dVD*VD)**2)/Vrms

    # set scale
    if data_source=='F435W':
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
        
        n = int(np.around(3/scale))
        #xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
        #img = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
        
        # V
        plt.clf()
        fig = plt.figure(figsize=(12,4))
        fig.add_subplot(131)
        #mge_print_contours(img, f.theta, xc, yc, m.sol, contour_alpha,
        #                   sigmapsf=sigmapsf, normpsf=normpsf, 
        #                   scale=scale, plot_img=plot_img)
        plt.contour(img, colors='grey', extent=[-3,3,-3,3], levels=[0.1, 0.2, 0.4, 0.7, 1.0])
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
        #cnt = mge_print_contours(img, f.theta, xc, yc, m.sol, contour_alpha,
        #                   sigmapsf=sigmapsf, normpsf=normpsf, 
        #                   scale=scale, plot_img=plot_img)
        plt.contour(img, colors='grey', extent=[-3,3,-3,3], levels=[0.1, 0.2, 0.4, 0.7, 1.0])
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
        #mge_print_contours(img, f.theta, xc, yc, m.sol, contour_alpha,
        #                   sigmapsf=sigmapsf, normpsf=normpsf, 
        #                   scale=scale, plot_img=plot_img)
        plt.contour(img, colors='grey', extent=[-3,3,-3,3], levels=[0.1, 0.2, 0.4, 0.7, 1.0])
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


obj_name = 'SDSSJ0037-0942' # e.g. SDSSJ0037-0942
obj_abbr = obj_name[4:9] # e.g. J0029
file_dir = f'{data_dir}mosaics/{obj_name}/' # directory with all files of obj_name

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
    
def bin_velocity_maps (file_dir, obj_name, Vbary, center_axis_index, data_source='KCWI'):
    
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
    V=np.genfromtxt(f'{file_dir}{obj_name}_final_kinematics/{obj_name}_V_binned.txt',
                 delimiter=',')
    V_bin = V-np.mean(V)
    V_cov = np.genfromtxt(f'{file_dir}{obj_name}_final_kinematics/{obj_name}_covariance_matrix_V.txt',
                     delimiter=',')
    dV_bin = np.sqrt(np.diagonal(V_cov)) # to have error bars
    
    #######################################
    # get reff
    slacs_table = np.genfromtxt(f'{data_dir}slacs_tableA1.txt', delimiter='', dtype='U10')
    slacs_table_name = obj_name[4:]
    slacs_reffs = slacs_table[:,7].astype(float)
    reff = slacs_reffs[slacs_table[:,0]==slacs_table_name]
    
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

    return V_bin, VD_bin, Vrms_bin, dV_bin, dVD_bin, dVrms_bin, xbin_arcsec, ybin_arcsec

############################################################################################################################
############################################################################################################################
############################################################################################################################

for obj_name in obj_names:
    
    obj_abbr = obj_name[4:9] # e.g. J0029
    file_dir = f'{data_dir}mosaics/{obj_name}/' # directory with all files of obj_name
    
    print('#####################################################')
    print('#####################################################')
    print()
    print(f'Beginning final kinematics visualization and plotting script for object {obj_name}.')
    print()

    ## First look at the KCWI integrated datacube and HSTF435W image, crop to 3 arcsec

    # import image, center, and crop

    #######################################################################################
    # kcwi datacube

    kcwi_img, kcwi_3arc_img, kcwi_header, \
        kcwi_central_pix_x, kcwi_central_pix_y = import_center_crop(file_dir, obj_name, obj_abbr, 
                                                              data_source='kcwi_datacube', plot=True)





    # Fit photometry of 3 arcsec KCWI image

    ## Take 3 arcsec KCWI image for find_galaxy initial estimates of PA and ellipticity

    # take 3 arcsec hst image for find_galaxy initial estimates of PA and ellipticity

    img = kcwi_3arc_img

    ############################################################################################################################
    # figure out the pixel fraction best to use

    #try_fractions_for_find_galaxy(img)

    # set the fraction to be used as frac
    frac = 0.04

    #################################################
    # Model the central light ellipse

    plt.clf()
    #plt.clf()
    f = find_galaxy(img, fraction=frac, plot=1, quiet=True)
    eps = f.eps
    theta = f.theta
    cen_y = f.ypeak
    cen_x = f.xpeak
    #plt.contour(img, colors='grey', extent=[-3,3,-3,3], levels=[0.1, 0.2, 0.4, 0.7, 1.0], zorder=1)
    plt.title(f'{frac}')
    plt.pause(1)



    # Kinematics

    ## Import the 2D kinematics maps


    V, VD, Vrms, \
    dV, dVD, dVrms, \
    Vbary, center_axis_index = load_2d_kinematics_with_datacube_contours (file_dir, obj_name, img/np.max(img), f,
                                                                           contour_alpha=0.7, data_source='KCWI', 
                                                                           plot=True, plot_img=False)

    # Bin the velocity maps.

    ##########################################################################


    V_bin, VD_bin, Vrms_bin, \
    dV_bin, dVD_bin, dVrms_bin, \
    xbin_arcsec, ybin_arcsec = bin_velocity_maps (file_dir, obj_name, Vbary, 
                                                  center_axis_index, data_source='KCWI')

    # Check binning

    # plot with arcsec
    width =  V.shape[0]/2 * kcwi_scale
    extent = [-width,width,-width,width]
    plt.figure(figsize=(8,8))
    plt.imshow(V, origin='lower', extent=extent, cmap='sauron')
    plt.scatter(xbin_arcsec, ybin_arcsec, color='k', marker='.')

    ## Determine correction to intrinsic (barycenter) velocity and kinematics PA with PAFit.

    PA_kin, dPA_kin, velocity_offset = fit_kinematic_pa(xbin_arcsec, ybin_arcsec, V_bin)

    # Vbary_new = Vbary+correction
    # V_new = V - correction
    V_bin = V_bin - velocity_offset

    # set kinematic PA from the negative x-axis
    PA_kin = 270 - PA_kin

    # PA_phot is theta from find_galaxy, measure from negative x-axis
    PA_phot = theta

    # rotate the bins by the kinematics PA and the photometric PA
    # plot the rotation with the "non-symmetrized velocity field"

    # kinematics
    xbin_kin, ybin_kin, _ = rotate_bins (PA_kin+180, xbin_arcsec, ybin_arcsec, V_bin, plot=True)

    plt.pause(1)
    plt.clf()

    # phomotometry
    xbin_phot, ybin_phot, _ = rotate_bins (PA_phot+180, xbin_arcsec, ybin_arcsec, V_bin, plot=True)

    ## Align by photometric position angles

    goober = plt.imshow(V, origin='lower', extent=extent, cmap='sauron')

    plt.rcParams.update({'font.size': 14})

    print(f'Kinematic PA : {PA_kin}') # PA from kinematics
    print(F'Photometric PA: {PA_phot}') # PA from photometry

    # velocity_range
    vel_range = np.nanmax(np.abs(V))

    # plot with arcsec
    width =  V.shape[0]/2 * kcwi_scale
    extent = [-width,width,-width,width]
    plt.figure(figsize=(8,8))
    plt.imshow(V, origin='lower', extent=extent, cmap='sauron', vmin=-vel_range-5, vmax=vel_range+5)
    cbar = plt.colorbar(shrink=0.85)
    cbar.set_label(r'V [km/s]')
    #plt.scatter(xbin_arcsec, ybin_arcsec, color='k', marker='.')
    plt.contour(img/np.max(img),
                extent=[-3,3,-3,3],
                linewidths=2,
                colors='white',
                levels=[0.1, 0.2, 0.4, 0.7, 1.0]
               )

    # plot the major axes
    # kinematic
    ykin = -np.tan(np.radians(PA_kin))*x
    plt.plot(x,ykin, 
             label='Kinematic Major Axis', 
             c='k',
            linestyle='--',
            linewidth=3)

    plt.legend(loc='best')
    # photometric
    ykin = -np.tan(np.radians(PA_phot))*x
    plt.plot(x,ykin, 
             label='Photometric Major Axis', 
             c='k',
            linestyle=':',
            linewidth=3)


    plt.ylim(-3,3)
    plt.xlim(-3,3)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')

    plt.legend(loc='upper left')