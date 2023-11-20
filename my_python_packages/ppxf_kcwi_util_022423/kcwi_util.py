import numpy as np
import glob
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import colors
#import sys
#sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages")
from ppxf_kcwi_util_022423.templates_util import Xshooter, miles, Munari05_subset, indo_US # 11/17/23 - Shawn changed this from ppxf.templates_util, because ppxf updated and no longer uses the same templates_util.py, ppxf_kcwi_util_022423 saves it in a place I keep and point the path to
from pathlib import Path
from os import path
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as sps_util
from urllib import request
from scipy import ndimage
from time import perf_counter as clock
from scipy import interpolate
from astropy.visualization import simple_norm
from astropy.modeling.models import Sersic2D
from vorbin.voronoi_2d_binning import voronoi_2d_binning
import pandas as pd

############################################# 7/13/22 - Shawn: I have replaced "quasar" with "background_source" because my background sources are not quasars.

def register_sauron_colormap():
    """
    Regitsr the 'sauron' and 'sauron_r' colormaps in Matplotlib

    """
    cdict = {'red':[(0.000,   0.01,   0.01),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.4,    0.4),
                 (0.414,   0.5,    0.5),
                 (0.463,   0.3,    0.3),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.7,    0.7),
                 (0.590,   1.0,    1.0),
                 (0.668,   1.0,    1.0),
                 (0.834,   1.0,    1.0),
                 (1.000,   0.9,    0.9)],
        'green':[(0.000,   0.01,   0.01),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.85,   0.85),
                 (0.414,   1.0,    1.0),
                 (0.463,   1.0,    1.0),
                 (0.502,   0.9,    0.9),
                 (0.541,   1.0,    1.0),
                 (0.590,   1.0,    1.0),
                 (0.668,   0.85,   0.85),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.9,    0.9)],
         'blue':[(0.000,   0.01,   0.01),
                 (0.170,   1.0,    1.0),
                 (0.336,   1.0,    1.0),
                 (0.414,   1.0,    1.0),
                 (0.463,   0.7,    0.7),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.0,    0.0),
                 (0.590,   0.0,    0.0),
                 (0.668,   0.0,    0.0),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.9,    0.9)]
         }

    rdict = {'red':[(0.000,   0.9,    0.9),
                 (0.170,   1.0,    1.0),
                 (0.336,   1.0,    1.0),
                 (0.414,   1.0,    1.0),
                 (0.463,   0.7,    0.7),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.3,    0.3),
                 (0.590,   0.5,    0.5),
                 (0.668,   0.4,    0.4),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.01,   0.01)],
        'green':[(0.000,   0.9,    0.9),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.85,   0.85),
                 (0.414,   1.0,    1.0),
                 (0.463,   1.0,    1.0),
                 (0.502,   0.9,    0.9),
                 (0.541,   1.0,    1.0),
                 (0.590,   1.0,    1.0),
                 (0.668,   0.85,   0.85),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.01,   0.01)],
         'blue':[(0.000,   0.9,    0.9),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.0,    0.0),
                 (0.414,   0.0,    0.0),
                 (0.463,   0.0,    0.0),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.7,    0.7),
                 (0.590,   1.0,    1.0),
                 (0.668,   1.0,    1.0),
                 (0.834,   1.0,    1.0),
                 (1.000,   0.01,   0.01)]
         }

    sauron = colors.LinearSegmentedColormap('sauron', cdict)
    sauron_r = colors.LinearSegmentedColormap('sauron_r', rdict)
    plt.register_cmap(cmap=sauron)
    plt.register_cmap(cmap=sauron_r)



def stellar_type(libary_dir_xshooter, dir_temperture, pp_weights_2700, bins,
                 ):
    xshooter = glob.glob(libary_dir_xshooter + '/*uvb.fits')
    df = pd.DataFrame(pd.read_excel(dir_temperture))
    info=df.to_numpy()
    col = 1
    t_eff=info[np.argsort(info[:,col])]

    number = np.zeros(len(xshooter))
    for i in range(number.shape[0]):
        number[i] = int(xshooter[i][-12:-9])

    temperture = np.zeros(len(xshooter))
    for i in range(temperture.shape[0]):
        if (~(t_eff.T[0] == number[i])).all():
            temperture[i] = 0
        else:
            temperture[i]= t_eff.T[1][t_eff.T[0] == number[i]]
            print(temperture[i], number[i])
    stellar_type=np.stack((temperture,pp_weights_2700)).T
    stellar_type_reorder = stellar_type[np.argsort(stellar_type[:,0])]

    plt.hist(stellar_type_reorder[:, 0], weights=stellar_type_reorder[:, 1],
             bins=bins)
    plt.xlabel('temperture (K)')
    plt.ylabel('weight')


def ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(libary_dir,
                                                      degree,
                                                    spectrum_aperture,
                                                      wave_min, wave_max,
                                                    z,
                                                      noise, 
                                                      FWHM,
                                                      FWHM_tem,
                                                      templates_name,
                                                      velscale_ratio,
                                                      background_source_spectrum=None,
                                                    z_background_source=None,
                                                   global_template_lens=None, plot=False, 
                                                      spectrum_perpixel=None,
                                                     wavelength_goobers=False):
    
    '''
    This function returns the information of the ppxf fitting given the
    data spectrum (either single pixel or voronoi binning spectrum) and
    the background_source spectrum (I use background_source spectrum in the "sky" keyword in ppxf).
    The noise can be either single value (not accurate) or poisson noise (
    the accurate noise input).
    '''

    # Read a galaxy spectrum and define the wavelength range
    file = spectrum_aperture
    hdu = fits.open(file)
    if spectrum_perpixel is not None:
        gal_lin = spectrum_perpixel
    else:
        gal_lin = hdu[0].data
    h1 = hdu[0].header
    lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1'] - 1)])
    print('CRVAL1 is', h1['CRVAL1'])
    print('CDELT1 is', h1['CDELT1'])
    print('NAXIS1 is', h1['NAXIS1'], gal_lin.shape[0])
    FWHM_gal = FWHM

    z = z # Initial estimate of the galaxy redshift
    lamRange1 = lamRange1/(1+z) # Compute approximate restframe wavelength range
    FWHM_gal = FWHM_gal/(1+z)   # Adjust resolution in Angstrom
    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_lin)
    lam = np.exp(logLam1)
    print("velscale of the data is", velscale)
    # de redshift


    # create a noise
    if isinstance(noise,np.ndarray):
        noise = noise
    else:
        noise = np.full_like(galaxy, noise) # Assume constant noise per
        # pixel here

    if background_source_spectrum is not None:
        z_bs = z_background_source

        # CF's method...
        # Read a background_source spectrum and define the wavelength range
        file_q = background_source_spectrum
        hdu_q = fits.open(file_q) # open the fits file (if it's a fits file)
        h1_q = hdu_q[0].header # take header
        lamRange1_q = h1_q['CRVAL1'] + np.array( # wavelength range
            [0., h1_q['CDELT1'] * (h1_q['NAXIS1'] - 1)])
        background_source_lin = hdu_q[0].data # lin??? just the data?

        lamRange1_q = lamRange1_q/(1+z_bs) # Compute approximate restframe wavelength # shouldn't the background source template already be at restframe wavelength?

        # My method 7/13/22 - using Kinney 1996 galaxy templates given to me by Tommaso for my LinKS project (in dir KECK_KCWI.../kinney_1996_galaxy_templates/)
        # Read a background_source spectrum and define the wavelength range
        #background_template = np.loadtxt(background_source_spectrum, unpack=True)
        #lamRange1_q = np.array([background_template[0,0], background_template[0,-1]]) # wavelength range
        #background_source_lin = background_template[1,:] # spectrum

        # Compute the approximate wavelength redshifted by z_bs - z
        #z_ls = z_bs - z # redshift between lens and source
        #lamRange1_q = lamRange1_q*(1+z_ls) # Compute approximate restframe wavelength # shouldn't the background source template already be at restframe wavelength?

        background_source, logLam1_q, velscale_q = util.log_rebin(lamRange1_q, background_source_lin)
        background_source = background_source/np.median(background_source)  # Normalize spectrum to avoid numerical


    else:
        background_source = 1 # this is only for returning the value and does not have
        # any meaning
        print('no sky spectrum (i.e., no background_source)')



    # Read the list of filenames from Xshooter library
    if templates_name == 'xshooter':
        xshooter = glob.glob(libary_dir + '/*uvb.fits')
        #FWHM_tem = 0.43  # Xshooter spectra have a constant resolution FWHM of
        # 1/9200*3950 = 0.43
        ssp, lamRange2, h2 = Xshooter(xshooter[0])
        print('h2 =', h2)

    elif templates_name == 'miles':
        vazdekis = glob.glob(libary_dir + '/Mun1.30*.fits')
        #FWHM_tem = 2.51  # Vazdekis+10 spectra have a constant resolution
        # FWHM of 2.51A.
        ssp, lamRange2, h2 = miles(vazdekis[0])
        print('h2 =', h2)
    elif templates_name == 'Munari05_subset':
        Munari05 = glob.glob(libary_dir + '/*.ASC')
        wavelength= libary_dir + '/LAMBDA_R20.DAT'
        #FWHM_tem = 0.1972  # R=20000
        ssp, lamRange2, h2 = Munari05_subset(Munari05[0], wavelength)
        print('h2 =', h2)
    elif templates_name == 'indo_US':
        Indo_US = glob.glob(libary_dir + '/*.txt')
        #FWHM_tem = 2.51  # Vazdekis+10 spectra have a constant resolution
        # FWHM of 2.51A.
        ssp, lamRange2, h2 = indo_US(Indo_US[0])
        print('h2 =', h2)

    elif templates_name == None:
        pass
    else:
        print('either xshooter or miles or Munari05_subset')

    velscale_ratio = velscale_ratio  # adopts 2x higher spectral sampling for templates
    # than for galaxy
    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                    velscale=velscale / velscale_ratio)
    
    if wavelength_goobers == True:
        wavelength_goobers = np.exp(logLam2) # I'm saving this to have the proper wavelength range for replotting later
        print(f'Wavelength goobers! {wavelength_goobers}')
        np.savetxt(f'{libary_dir}/templates_vs{velscale_ratio}_wavelength_array.txt', wavelength_goobers, delimiter=',')
        print('velscale of the templates is ', velscale_temp)


    # create the templates
    if global_template_lens is not None:
        templates = global_template_lens
    else:
        if templates_name == 'xshooter':
            # checking the templates existing or not
            check_templates = Path(libary_dir+'/templates_vs%s.fits'
                                   %velscale_ratio)
            if check_templates.is_file():
                templates = fits.getdata(libary_dir+'/templates_vs%s.fits'
                                         %velscale_ratio)
                print('get templates from '+
                      libary_dir+'/templates_vs%s.fits' %velscale_ratio)
                #print(j,file)
            else:
                templates = np.empty((sspNew.size, len(xshooter)))
                print('FWHM_gal=', FWHM_gal)
                print('FWHM_tem=', FWHM_tem)

                if FWHM_gal > FWHM_tem:
                    FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
                    sigma = FWHM_dif / 2.355 / h2  # Sigma difference in pixels
                    for j, file in enumerate(xshooter):
                        print(j, file)
                        ssp, lamRange2, h2 = Xshooter(file)
                        ssp = ndimage.gaussian_filter1d(ssp, sigma)
                        sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                        velscale=velscale / velscale_ratio)
                        templates[:, j] = sspNew / np.median(
                            sspNew)  # Normalizes templates
                    fits.writeto(libary_dir+'/templates_vs%s.fits' %velscale_ratio, templates)
                elif FWHM_gal < FWHM_tem:
                    print("sigma<0, so we do not do any convolution")
                    for j, file in enumerate(xshooter):
                        print(j, file)
                        ssp, lamRange2, h2 = Xshooter(file)
                        sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                        velscale=velscale / velscale_ratio)
                        templates[:, j] = sspNew / np.median(
                            sspNew)  # Normalizes templates
                else:
                    print("sigma=0, so we do not do any convolution")
                    for j, file in enumerate(xshooter):
                        print(j, file)
                        ssp, lamRange2, h2 = Xshooter(file)
                        sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                        velscale=velscale / velscale_ratio)
                        templates[:, j] = sspNew / np.median(
                            sspNew)  # Normalizes templates

        elif templates_name == 'miles':
            templates = np.empty((sspNew.size, len(vazdekis)))
            print('FWHM_gal=', FWHM_gal)
            print('FWHM_tem=', FWHM_tem)
            if FWHM_gal > FWHM_tem:
                FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
                sigma = FWHM_dif / 2.355 / h2  # Sigma difference in pixels
                for j, file in enumerate(vazdekis):
                    ssp, lamRange2, h2 = miles(file)
                    ssp = ndimage.gaussian_filter1d(ssp, sigma)
                    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                    velscale=velscale / velscale_ratio)
                    templates[:, j] = sspNew / np.median(
                        sspNew)  # Normalizes templates
            elif FWHM_gal < FWHM_tem:
                print("sigma<0, so we do not do any convolution")
                for j, file in enumerate(vazdekis):
                    ssp, lamRange2, h2 = miles(file)
                    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                    velscale=velscale / velscale_ratio)
                    templates[:, j] = sspNew / np.median(
                        sspNew)  # Normalizes templates
            else:
                print("sigma=0, so we do not do any convolution")
                for j, file in enumerate(vazdekis):
                    ssp, lamRange2, h2 = miles(file)
                    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                    velscale=velscale / velscale_ratio)
                    templates[:, j] = sspNew / np.median(
                        sspNew)  # Normalizes templates

        elif templates_name == 'Munari05_subset':
            templates = np.empty((sspNew.size, len(Munari05)))
            wavelength = libary_dir + '/LAMBDA_R20.DAT'
            print('FWHM_gal=', FWHM_gal)
            print('FWHM_tem=', FWHM_tem)

            if FWHM_gal > FWHM_tem:
                FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
                sigma = FWHM_dif / 2.355 / h2  # Sigma difference in pixels
                for j, file in enumerate(Munari05):
                    ssp, lamRange2, h2 = Munari05_subset(file,wavelength)
                    ssp = ndimage.gaussian_filter1d(ssp, sigma)
                    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                    velscale=velscale / velscale_ratio)
                    templates[:, j] = sspNew / np.median(
                        sspNew)  # Normalizes templates
            elif FWHM_gal < FWHM_tem:
                print("sigma<0, so we do not do any convolution")
                for j, file in enumerate(Munari05):
                    ssp, lamRange2, h2 = Munari05_subset(file,wavelength)
                    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                    velscale=velscale / velscale_ratio)
                    templates[:, j] = sspNew / np.median(
                        sspNew)  # Normalizes templates
            else:
                print("sigma=0, so we do not do any convolution")
                for j, file in enumerate(Munari05):
                    ssp, lamRange2, h2 = Munari05_subset(file,wavelength)
                    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                    velscale=velscale / velscale_ratio)
                    templates[:, j] = sspNew / np.median(
                        sspNew)  # Normalizes templates

        elif templates_name == 'indo_US':
            templates = np.empty((sspNew.size, len(Indo_US)))
            print('FWHM_gal=', FWHM_gal)
            print('FWHM_tem=', FWHM_tem)
            if FWHM_gal > FWHM_tem:
                FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
                sigma = FWHM_dif / 2.355 / h2  # Sigma difference in pixels
                for j, file in enumerate(Indo_US):
                    ssp, lamRange2, h2 = indo_US(file)
                    ssp = ndimage.gaussian_filter1d(ssp, sigma)
                    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                    velscale=velscale / velscale_ratio)
                    templates[:, j] = sspNew / np.median(
                        sspNew)  # Normalizes templates
            elif FWHM_gal < FWHM_tem:
                print("sigma<0, so we do not do any convolution")
                for j, file in enumerate(Indo_US):
                    ssp, lamRange2, h2 = indo_US(file)
                    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                    velscale=velscale / velscale_ratio)
                    templates[:, j] = sspNew / np.median(
                        sspNew)  # Normalizes templates
            else:
                print("sigma=0, so we do not do any convolution")
                for j, file in enumerate(Indo_US):
                    ssp, lamRange2, h2 = indo_US(file)
                    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                    velscale=velscale / velscale_ratio)
                    templates[:, j] = sspNew / np.median(
                        sspNew)  # Normalizes templates
        else:
            print('need to provide global templates or provide template libary')


    c = 299792.458
    dv = (np.mean(logLam2[:velscale_ratio]) - logLam1[0])*c  # km/s

    # after de-redshift, the initial redshift is zero.
    goodPixels = util.determine_goodpixels(logLam1, lamRange2, 0)
    print(goodPixels)

    ind_min = find_nearest(np.exp(logLam1) / 10, wave_min)
    ind_max = find_nearest(np.exp(logLam1) / 10, wave_max)
    
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
    t = clock()
    if background_source_spectrum is not None:
        pp = ppxf(templates, galaxy, noise, velscale, start, plot=plot,
                  moments=2, goodpixels=mask,
                  degree=degree, vsyst=dv, velscale_ratio=velscale_ratio,
                  sky=background_source, lam=lam)
        plt.xlim(wave_min, wave_max)
    else:
        pp = ppxf(templates, galaxy, noise, velscale, start, plot=plot,
                  moments=2, goodpixels=mask,
                  degree=degree, vsyst=dv, velscale_ratio=velscale_ratio,
                  lam=lam)
        plt.xlim(wave_min, wave_max)
    print("Formal errors:")
    print("     dV    dsigma   dh3      dh4")
    print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

    print('Elapsed time in pPXF: %.2f s' % (clock() - t))
    return templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, \
           background_source


def ppxf_kinematics_getGlobal_lens_deredshift_library_test(library_dir,
                                                              degree,
                                                            spectrum_aperture,
                                                              wave_min, wave_max,
                                                                z,
                                                              noise, 
                                                              FWHM,
                                                              FWHM_tem,
                                                              templates_name,
                                                              velscale_ratio,
                                                              background_source_spectrum=None,
                                                            z_background_source=None,
                                                           global_template_lens=None, plot=False, 
                                                              spectrum_perpixel=None,
                                                             wavelength_goobers=False):

    '''
    This function returns the information of the ppxf fitting given the
    data spectrum (either single pixel or voronoi binning spectrum) and
    the background_source spectrum (I use background_source spectrum in the "sky" keyword in ppxf).
    The noise can be either single value (not accurate) or poisson noise (
    the accurate noise input).
    '''

    # Read a galaxy spectrum and define the wavelength range
    file = spectrum_aperture
    hdu = fits.open(file)
    if spectrum_perpixel is not None:
        gal_lin = spectrum_perpixel
    else:
        gal_lin = hdu[0].data
    h1 = hdu[0].header
    lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1'] - 1)])
    print('CRVAL1 is', h1['CRVAL1'])
    print('CDELT1 is', h1['CDELT1'])
    print('NAXIS1 is', h1['NAXIS1'], gal_lin.shape[0])
    FWHM_gal = FWHM

    z = z # Initial estimate of the galaxy redshift
    lamRange1 = lamRange1/(1+z) # Compute approximate restframe wavelength range
    FWHM_gal = FWHM_gal/(1+z)   # Adjust resolution in Angstrom
    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_lin)
    lam = np.exp(logLam1)
    # 11/19/23 - Truncating galaxy spectrum to wave_min and wave_max because it won't let it run while the galaxy spectrum extends to lower wavelengths than the templates
    # Even though I am masking out those wavelengths anyway
    print('Truncating galaxy spectrum to wave min and wave max')
    galaxy = galaxy[lam>wave_min*10]
    lam = lam[lam>wave_min*10]
    galaxy = galaxy[lam<wave_max*10]
    lam = lam[lam<wave_max*10]
    lamRange1 = [wave_min*10, wave_max*10]
    logLam1 = np.log(lam)
    #print("velscale of the data is", velscale)
    print('lamRange1 is', lamRange1)
    # de redshift
    
    # take lam_range temp
    lam_range_temp = [lamRange1[0]/1.02, lamRange1[1]*1.02]
    
    # create a noise
    if isinstance(noise,np.ndarray):
        noise = noise
    else:
        noise = np.full_like(galaxy, noise) # Assume constant noise per
        # pixel here

    if background_source_spectrum is not None:
        z_bs = z_background_source

        # CF's method...
        # Read a background_source spectrum and define the wavelength range
        file_q = background_source_spectrum
        hdu_q = fits.open(file_q) # open the fits file (if it's a fits file)
        h1_q = hdu_q[0].header # take header
        lamRange1_q = h1_q['CRVAL1'] + np.array( # wavelength range
            [0., h1_q['CDELT1'] * (h1_q['NAXIS1'] - 1)])
        background_source_lin = hdu_q[0].data # lin??? just the data?

        lamRange1_q = lamRange1_q/(1+z_bs) # Compute approximate restframe wavelength # shouldn't the background source template already be at restframe wavelength?

        # My method 7/13/22 - using Kinney 1996 galaxy templates given to me by Tommaso for my LinKS project (in dir KECK_KCWI.../kinney_1996_galaxy_templates/)
        # Read a background_source spectrum and define the wavelength range
        #background_template = np.loadtxt(background_source_spectrum, unpack=True)
        #lamRange1_q = np.array([background_template[0,0], background_template[0,-1]]) # wavelength range
        #background_source_lin = background_template[1,:] # spectrum

        # Compute the approximate wavelength redshifted by z_bs - z
        #z_ls = z_bs - z # redshift between lens and source
        #lamRange1_q = lamRange1_q*(1+z_ls) # Compute approximate restframe wavelength # shouldn't the background source template already be at restframe wavelength?

        background_source, logLam1_q, velscale_q = util.log_rebin(lamRange1_q, background_source_lin)
        background_source = background_source/np.median(background_source)  # Normalize spectrum to avoid numerical

    else:
        background_source = 1 # this is only for returning the value and does not have
        # any meaning
        print('no sky spectrum (i.e., no background_source)')


    # 11/17/23 split between Chih-Fan's method and the new ppxf sps_utils.py stuff
    if library_dir is not None: # I will only give it a library dir if I have it, which means I'm using Chih-Fan's code
        # Read the list of filenames from Xshooter library
        if templates_name == 'xshooter':
            xshooter = glob.glob(library_dir + '/*uvb.fits')
            #FWHM_tem = 0.43  # Xshooter spectra have a constant resolution FWHM of
            # 1/9200*3950 = 0.43
            ssp, lamRange2, h2 = Xshooter(xshooter[0]) # I'm using a consistent lamda range for the templates across the different libraries
            print('h2 =', h2)
            print('lamRange2 *uvb.fits', lamRange2)
        elif templates_name == None:
            pass
        else:
            print('either xshooter or miles or Munari05_subset')

        velscale_ratio = velscale_ratio  # adopts 2x higher spectral sampling for templates
        # than for galaxy
        sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                        velscale=velscale / velscale_ratio)
        print('lamRange2',lamRange2)
        print('logLam2', logLam2)
        
        if wavelength_goobers == True:
            wavelength_goobers = np.exp(logLam2) # I'm saving this to have the proper wavelength range for replotting later
            print(f'Wavelength goobers! {wavelength_goobers}')
            np.savetxt(f'{library_dir}/templates_vs{velscale_ratio}_wavelength_array.txt', wavelength_goobers, delimiter=',')
            print('velscale of the templates is ', velscale_temp)


        # create the templates
        if global_template_lens is not None:
            templates = global_template_lens
        else:
            if templates_name == 'xshooter':
                # checking the templates existing or not
                check_templates = Path(library_dir+'/templates_templates_test.fits'
                                      )
                if check_templates.is_file():
                    templates = fits.getdata(library_dir+'/templates_templates_test.fits')
                    print('get templates from '+
                          library_dir+'/templates_templates_test.fits')
                    #print(j,file)
                else:
                    print('Gotta make the templates')
                    templates = np.empty((sspNew.size, len(xshooter)))
                    print('FWHM_gal=', FWHM_gal)
                    print('FWHM_tem=', FWHM_tem)

                    if FWHM_gal > FWHM_tem:
                        FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
                        sigma = FWHM_dif / 2.355 / h2  # Sigma difference in pixels
                        for j, file in enumerate(xshooter):
                            print(j, file)
                            ssp, lamRange2, h2 = Xshooter(file)
                            ssp = ndimage.gaussian_filter1d(ssp, sigma)
                            sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                            velscale=velscale / velscale_ratio)
                            templates[:, j] = sspNew / np.median(
                                sspNew)  # Normalizes templates
                        fits.writeto(library_dir+'/templates_templates_test.fits', templates)
                    elif FWHM_gal < FWHM_tem:
                        print("sigma<0, so we do not do any convolution")
                        for j, file in enumerate(xshooter):
                            print(j, file)
                            ssp, lamRange2, h2 = Xshooter(file)
                            sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                            velscale=velscale / velscale_ratio)
                            templates[:, j] = sspNew / np.median(
                                sspNew)  # Normalizes templates
                    else:
                        print("sigma=0, so we do not do any convolution")
                        for j, file in enumerate(xshooter):
                            print(j, file)
                            ssp, lamRange2, h2 = Xshooter(file)
                            sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp,
                                                                            velscale=velscale / velscale_ratio)
                            templates[:, j] = sspNew / np.median(
                                sspNew)  # Normalizes templates

            else:
                print('need to provide global templates or provide template library')
        # get the template range of lambda
        lam_temp = np.exp(logLam2)
        
    # else use Michele's new ppxf sps stuff
    else:
        ppxf_dir = path.dirname(path.realpath(util.__file__))
        basename = f"spectra_{templates_name}_9.0.npz"
        filename = path.join(ppxf_dir, 'sps_models', basename)
        if not path.isfile(filename):
            url = "https://github.com/micappe/ppxf_data/blob/9e22598d357b6c02c5355b312073e1708eb3ded3/spectra_emiles_9.0.npz" #"https://github.com/micappe/ppxf_data/main/" + basename
            request.urlretrieve(url, filename)
        sps = sps_util.sps_lib(filename, velscale, FWHM_gal, wave_range=lam_range_temp)
        templates = sps.templates
        lam_temp = sps.lam_temp
    
    c = 299792.458
    #dv = (np.mean(logLam2[:velscale_ratio]) - logLam1[0])*c  # km/s # this is unnecessary when I give it lam_temp

    # after de-redshift, the initial redshift is zero.
    goodPixels = util.determine_goodpixels(logLam1, lam_range_temp, 0)
    print(goodPixels)

    ind_min = find_nearest(np.exp(logLam1) / 10, wave_min)
    ind_max = find_nearest(np.exp(logLam1) / 10, wave_max)
    
    mask=goodPixels[goodPixels<ind_max]
    mask = mask[mask>ind_min]
    boolen = ~((2956 < mask) & (mask < 2983))  # mask the Mg II
    mask = mask[boolen]
    boolen = ~((2983 < mask) & (mask < 3001))  # mask the Mg II
    mask = mask[boolen]
    
    print('lam', lam.shape)
    print('lam_temp', lam_temp.shape)
    print(lam_temp)
    print('templates', templates.shape)
    print()
    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #
    vel = c*np.log(1 + 0)   # eq.(8) of Cappellari (2017)
    start = [vel, 250.]  # (km/s), starting guess for [V, sigma]
    t = clock()
    if background_source_spectrum is not None:
        pp = ppxf(templates, galaxy, noise, velscale, start, plot=plot,
                  moments=2, goodpixels=mask,
                  degree=degree, #vsyst=dv, 
                  velscale_ratio=velscale_ratio,
                  sky=background_source, lam=lam, lam_temp=lam_temp) # 11/17/23 added lam_temp
        plt.xlim(wave_min, wave_max)
    else:
        pp = ppxf(templates, galaxy, noise, velscale, start, plot=plot,
                  moments=2, goodpixels=mask,
                  degree=degree, #vsyst=dv, 
                  velscale_ratio=velscale_ratio,
                  lam=lam, lam_temp=lam_temp)
        plt.xlim(wave_min, wave_max)
    print("Formal errors:")
    print("     dV    dsigma   dh3      dh4")
    print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

    print('Elapsed time in pPXF: %.2f s' % (clock() - t))
    return templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, \
           background_source



def visualization(hdu):
    '''
    :param hdu: the fits file of the data
    :return: no return. This function is just for visualization
    '''
    norm = simple_norm(np.nansum(hdu[0].data, axis=0), 'sqrt')
    plt.imshow(np.nansum(hdu[0].data, axis=0), origin="lower", norm=norm)
    plt.title('KCWI data')
    plt.colorbar(label='flux')
    #plt.show()
    #plt.pause(3)


def get_datacube(hdu, lens_center_x, lens_center_y, radius_in_pixels):
    '''

    :param hdu: the fits file of the entir mosaic data
    :param lens_center_x: x coordinate of the lens center
    :param lens_center_y: y coordinate of the lens center
    :param radius_in_pixels: the size where we want to extract the kinematics
    :return: new fits file with smaller size
    '''
    r = radius_in_pixels
    data = hdu[0].data
    data_crop = data[:, lens_center_y - r-1:lens_center_y +
                                                          r, lens_center_x
                                                               - r -1:
                                                             lens_center_x + r]
    new_hdu = hdu.copy()
    new_hdu[0].data = data_crop
    visualization(new_hdu)

    return new_hdu


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




def find_nearest(array, value):
    '''
    :param array: wavelength array
    :param value: wavelength that we want to get the index
    :return: the index of the wavelength
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def SN_CaHK(ind_min, ind_max, data_no_background_source, noise_cube, T_exp):
    # first, I need to estimate the flux/AA
    flux_per_half_AA = np.nanmedian(data_no_background_source[ind_min:ind_max, :, :],
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
    sigma_per_half_pixel = np.std(noise_cube[ind_min:ind_max,:,:], axis=0)
    sigma = np.sqrt(2) * sigma_per_half_pixel
    # some weired pattern so we find average in the black region around 36, 36
    #sigma_mean = np.mean(sigma [36-6:36+5, 36-6:36+5])
    #sigma = np.ones(sigma.shape)*sigma_mean
    plt.imshow(sigma,origin='lower')
    plt.show()

    # then, estimate the poisson noise
    sigma_poisson = poisson_noise(T_exp, flux_per_AA, sigma, per_second=True)
    plt.imshow(sigma_poisson,origin="lower")
    plt.title('poisson noise')
    plt.colorbar()
    plt.show()

    SN_per_AA = flux_per_AA / sigma_poisson
    plt.imshow(SN_per_AA, origin="lower")
    plt.title('S/N ratio')
    plt.colorbar()
    plt.show()
    return SN_per_AA, flux_per_AA, sigma_poisson



def remove_background_source_from_galaxy_deredshift(libary_dir, degree, spectrum_aperture,
                                         wave_min,
                                         wave_max,
                                         velscale_ratio,
                                         background_source_spectrum, 
                                            z, 
                                            z_background_source,
                                            noise,
                                         templates_name,
                                         FWHM, FWHM_tem,
                                         global_temp, plot=False,
                                         spectrum_perpixel=None):
    '''
    this function output "the data without background_source" and "the noise spectrum".
    - data without background_source: data minus the best fit of the background_source light
    - noise spectrum: data minus the pp.bestfit
    '''

    templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, background_source =         \
        ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(libary_dir=libary_dir,
                                                          degree=degree,
                                                          spectrum_aperture=spectrum_aperture,
                                                          wave_min=wave_min,
                                                          wave_max=wave_max,
                                                          velscale_ratio=velscale_ratio,
                                                          background_source_spectrum = background_source_spectrum,
                                                          z=z,
                                                          z_background_source=z_background_source,
                                                          noise=noise,
                                                          templates_name=templates_name,
                                                          FWHM=FWHM,
                                                          FWHM_tem=FWHM_tem,
                                                          global_template_lens=global_temp,
                                                          plot=plot,
                                                          spectrum_perpixel=spectrum_perpixel)
    if plot is not False:
        plt.show()
    else:
        plt.clf()

    best_sky = background_source * pp.weights[-1]
    if spectrum_perpixel is not None:
        gal_lin = spectrum_perpixel
    else:
        gal_lin = fits.getdata(spectrum_aperture)

    log_axis_sky = np.exp(logLam1)
    lin_axis_sky = np.linspace(lamRange1[0], lamRange1[1], gal_lin.size)

    sky_lin = de_log_rebin(log_axis_sky, best_sky, lin_axis_sky)
    best_lin = de_log_rebin(log_axis_sky, pp.bestfit, lin_axis_sky)

    if plot is not False:
        plt.figure(figsize=(8,8))
        plt.plot(lin_axis_sky/10, gal_lin, 'k-', label='data')
        plt.plot(lin_axis_sky/10, best_lin, 'r-', label='best model ('
                                                     'lens+background_source)')
        plt.plot(lin_axis_sky/10, gal_lin - sky_lin, 'm-',
                 label='remove background_source from data')
        plt.plot(lin_axis_sky/10, sky_lin, 'c-',label='best background_source model',
                zorder=10)
        plt.plot(lin_axis_sky/10, gal_lin - best_lin, 'g-',
                 label='noise (data - best model)')
        plt.legend()
        plt.xlabel('wavelength (nm)')
        plt.ylabel('relative flux')
        plt.xlim(wave_min,wave_max)
        #plt.autoscale(axis='y')#,tight=True) # this is what CF originally did, not showing for me
        plt.ylim(-0.01, 0.07)
        plt.show()
    else:
        pass

    gal_lin_nosky = gal_lin - sky_lin
    noise = gal_lin - best_lin
    ## replance ana with 0
    gal_lin_nosky[np.isnan(gal_lin_nosky)]=0
    noise[np.isnan(noise)] = 0
    return gal_lin_nosky, noise, sky_lin


#import pyregion
#def getMaskInFitsFromDS9reg(input,nx):
#    r = pyregion.open(input)
#    mask = r.get_mask(shape=(nx, nx))
#    return mask
# 11/17/23 - Shawn, taken from kcwi_util2.py, changed from above
import pyregion
def getMaskInFitsFromDS9reg(input,nx,hdu):
    r = pyregion.open(input)
    mask = r.get_mask(shape=(nx, nx),hdu=hdu)
    return mask



def voronoi_binning(targetSN, dir, obj_name):
    """
    Usage example for the procedure VORONOI_2D_BINNING.

    It is assumed below that the file voronoi_2d_binning_example.txt
    resides in the current directory. Here columns 1-4 of the text file
    contain respectively the x, y coordinates of each SAURON lens
    and the corresponding Signal and Noise.

    """
    #dir="/Users/Geoff/drizzlepac_test/drizzle_file_KCWI_J1306/"
    #name="KCWI_J1306_icubes_mosaic_0.1457"
    #file_dir = path.dirname(path.realpath(vorbin.__file__))  # path of vorbin
    x, y, signal, noise = np.loadtxt(
        dir+'voronoi_2d_binning_'+obj_name+'_input.txt').T

    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    #
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
        x, y, signal, noise, targetSN, plot=1, quiet=0)

    # Save to a text file the initial coordinates of each pixel together
    # with the corresponding bin number computed by this procedure.
    # binNum uniquely specifies the bins and for this reason it is the only
    # number required for any subsequent calculation on the bins.
    #
    np.savetxt(dir+'voronoi_2d_binning_'+obj_name+'_output.txt', \
                      np.column_stack([x, y,binNum]),
               fmt=b'%10.6f %10.6f %8i')

    #x, y, signal, noise = np.loadtxt(
    #    dir+'voronoi_2d_binning_'+name+'_input.txt').T
    #x = (x - int(np.mean(x)))*0.1457
    #y = (y - int(np.mean(y)))*0.1457

    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    #
    #binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
    #    x, y, signal, noise, targetSN, plot=1, quiet=0)

    # Save to a text file the initial coordinates of each pixel together
    # with the corresponding bin number computed by this procedure.
    # binNum uniquely specifies the bins and for this reason it is the only
    # number required for any subsequent calculation on the bins.


def fitting_SN(x0, mask, flux_per_AA, noise):
    amplitude, r_eff, n, x_0, y_0, ellip, theta = x0
    if amplitude<0 or r_eff <0 or n<0.3 or ellip>0.4:
        chi2 = 10**10
    else:
        x, y = np.meshgrid(np.arange(flux_per_AA.shape[0]), np.arange(
            flux_per_AA.shape[1]))
        mod = Sersic2D(amplitude=amplitude, r_eff=r_eff, n=n, x_0=x_0, y_0=y_0,
                       ellip=ellip, theta=theta)
        img = mod(x, y)
        chi2 = np.sum(((flux_per_AA - img) / noise * mask) ** 2)
        plt.imshow((flux_per_AA - img) / noise, origin='lower')
    #print(x0)
    print(chi2)
    return chi2

def fitting_SN_forshow(x0, mask, flux_per_AA, sigma_poisson):
    amplitude, r_eff, n, x_0, y_0, ellip, theta = x0
    x, y = np.meshgrid(np.arange(flux_per_AA.shape[0]), np.arange(
        flux_per_AA.shape[1]))
    mod = Sersic2D(amplitude=amplitude, r_eff=r_eff, n=n, x_0=x_0, y_0=y_0,
                   ellip=ellip, theta=theta)
    img = mod(x, y)

    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(flux_per_AA, origin="lower")
    axarr[0].set_title('flux per AA')
    axarr[1].imshow(img, origin="lower")
    axarr[1].set_title('model')
    axarr[2].imshow((flux_per_AA - img) / sigma_poisson * mask, origin="lower")
    axarr[2].set_title('residuals')
    axarr[3].imshow(img / sigma_poisson, origin="lower")
    axarr[3].set_title('S/N model')
    plt.show()
    return img / sigma_poisson, (flux_per_AA - img) / sigma_poisson * mask


def poisson_noise(T_exp, gal_lin, std_bk_noise, per_second=False):
    '''
    This means that the pixel uncertainty of pixel i (sigma_i) is obtained
    from the science image intensity pixel i (d_i) by:
    sigma_i^2 = scale * (d_i)^power + const
    The first term represents noise from the astrophysical source, and the
    second term is background noise (including read noise etc.).
    When power=1 and scale=1 with d_i in counts, the astrophysical source noise
    (=1*d_i^1=d_i) is Poisson. Suyu 2012 and Suyu et al. 2013a have somels

    description of this.

    To construct the weight map using the esource_noise_model:
    -- set power=1
    -- obtain const by estimating the variance of the background (i.e., const = sigma_bkgd^2 from an empty part of of the science image).
    -- the scale is 1 if d_i is in counts, but otherwise it needs to account for exposure time if d_i is in counts per second.

    Since the unit of the KCWI data is flux/AA (see fits header),
    I need to compute scale with appropriate multiplications/divisions of
    the exposure time (T_exp).  In this case, scale should be 1/texp so that
    the units are in counts/sec for sigma_i^2 (since d_i needs to be
    multiplied by texp to get to counts for Poisson noise estimation,
    but then divided by texp^2 to get to counts/sec).

    :param T_exp: the total exposure time of the dataset
    :param gal_lin: input data
    :param bk_noise: standard deviation of the background noise
    :param per_second: set True if it is in the unit of counts/second
    :return: poisson noise
    '''

    const = std_bk_noise**2
    if per_second:
        scale= 1/T_exp
        sigma2 = scale * (gal_lin) + const
    else:
        scale = 1.
        sigma2 = scale * (gal_lin) + const

    if (sigma2<0).any():
        sigma2[sigma2 < 0] = const

    if np.isnan(sigma2).any():
        sigma2[np.isnan(sigma2)] = const

    poisson_noise = np.sqrt(sigma2)

    return poisson_noise




def select_region(dir, origin_imaging_data_perAA, SN_per_AA, SN_x_center,
                  SN_y_center, radius_in_pixels,
                  max_radius,
                  target_SN, obj_name):
    
    xx = np.arange(radius_in_pixels * 2 + 1)
    yy = np.arange(radius_in_pixels * 2 + 1)
    xx, yy = np.meshgrid(xx, yy)


    dist = np.sqrt((xx - SN_x_center) ** 2 + (yy - SN_y_center) ** 2)

    mask = (SN_per_AA > target_SN) & (dist < max_radius)

    xx_1D = xx[mask]
    yy_1D = yy[mask]
    SN_1D = SN_per_AA[mask]
    vor_input = np.vstack((xx_1D, yy_1D, SN_1D, np.ones(SN_1D.shape[0]))).T
    np.savetxt(dir + 'voronoi_2d_binning_' + obj_name + '_input.txt', vor_input,
               fmt=b'%10.6f %10.6f %10.6f %10.6f')
    #f, axarr = plt.subplots(1, 2)
    #axarr[0].imshow(yy, origin="lower")
    #axarr[0].set_title('y_coordiate')
    #axarr[1].imshow(xx, origin="lower")
    #axarr[1].set_title('x_coordiate')
    #plt.show()
    norm = simple_norm(origin_imaging_data_perAA, 'sqrt')
    plt.imshow(mask, origin="lower", cmap='gray')
    plt.imshow(origin_imaging_data_perAA, origin="lower", norm=norm,alpha=0.9)
    plt.axis('off')
    plt.show()
    #f, axarr = plt.subplots(1, 2)
    #axarr[0].imshow(origin_imaging_data_perAA, origin="lower", norm=norm)
    #axarr[1].imshow(mask, origin="lower", cmap='gray', alpha=0.9)
    #axarr[0].set_title('y_coordiate')
    #
    #axarr[1].imshow(SN_per_AA_model*mask, origin="lower")  #
    #axarr[1].set_title('region selected for voronoi binning (S/N > %s)' %
    # target_SN)

    plt.imshow(mask, origin="lower", cmap='gray')
    plt.imshow(SN_per_AA, origin="lower", alpha=0.9)  #
    plt.title('region selected for voronoi binning (S/N > %s)' % target_SN)
    plt.axis('off')
    plt.colorbar()
    plt.show()


def get_voronoi_binning_data(mos_dir, kin_dir, mos_name, obj_name, nobs_name=None):
    # nobs_name is for targets where I had to remove the background source, so we used that cropped mosaic
    if nobs_name is not None:
        a = fits.getdata(mos_dir + nobs_name + '_crop.fits')
    else:
        a = fits.getdata(mos_dir + mos_name + '_crop.fits')

    output=np.loadtxt(kin_dir +'voronoi_2d_binning_' + obj_name + '_output.txt')

    b=np.zeros((int(np.max(output.T[2]))+1,a.shape[0])) #construct the binning
    #  data
    check = np.zeros(a[0, :, :].shape)
    for i in range(output.shape[0]):
        print(i)
        wx = int(output[i][0])
        wy = int(output[i][1])
        num = int(output[i][2])
        b[num]=b[num]+a[:,wy,wx]
        check[wy, wx] = num+1

    fits.writeto(kin_dir +'voronoi_binning_' + obj_name + '_data.fits', b, overwrite=True)
    print("Number of bins =", b.shape[0])
    plt.imshow(check, origin="lower", cmap='sauron')
    plt.colorbar()
    #for (j, i), label in np.ndenumerate(check):
    #    plt.text(i, j, label, ha='center', va='center')
    plt.show()



def fitting_deredshift(libary_dir,
                       degree,
                       spectrum_aperture,
                       wave_min,
                       wave_max,
                       velscale_ratio,
                       z,
                       noise,
                       templates_name,
                       FWHM,
                       FWHM_tem,
                       global_temp,
                       T_exp,
                       background_source_spectrum=None,
                       plot=False,
                       spectrum_perpixel=None):
    if background_source_spectrum == None:

        templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, background_source =\
        ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(
            libary_dir=libary_dir,
            degree=degree,
            spectrum_aperture=spectrum_aperture,
            wave_min=wave_min,
            wave_max=wave_max,
            velscale_ratio=velscale_ratio,
            z=z,
            noise=noise,
            templates_name=templates_name,
            FWHM=FWHM,
            FWHM_tem=FWHM_tem,
            global_template_lens=global_temp,
            plot=plot,
            spectrum_perpixel=spectrum_perpixel)
        if plot is True:
            plt.show()
        else:
            plt.clf()

        if spectrum_perpixel is not None:
            gal_lin = spectrum_perpixel
        else:
            gal_lin = fits.getdata(spectrum_aperture)

        log_axis_sky = np.exp(logLam1)
        lin_axis_sky = np.linspace(lamRange1[0], lamRange1[1], gal_lin.size)

        best_lin = de_log_rebin(log_axis_sky, pp.bestfit, lin_axis_sky)

        noise = gal_lin - best_lin

        sigma_poisson = poisson_noise(T_exp, best_lin,
                                      np.std(noise[wave_min:wave_max]),
                                      per_second=True)

        templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, background_source = \
            ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(
                libary_dir=libary_dir,
                degree=degree, spectrum_aperture=spectrum_aperture,
                wave_min=wave_min,
                wave_max=wave_max,
                velscale_ratio=velscale_ratio,
                z=z,
                noise=sigma_poisson,
                templates_name=templates_name,
                FWHM=FWHM,
                FWHM_tem=FWHM_tem,
                global_template_lens=global_temp,
                plot=plot,
                spectrum_perpixel=spectrum_perpixel)
        if plot is not False:
            plt.show()
        else:
            plt.clf()


    else:
        templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, background_source =\
        ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(
            libary_dir=libary_dir,
            degree=degree,
            spectrum_aperture=spectrum_aperture,
            wave_min=wave_min,
            wave_max=wave_max,
            velscale_ratio=velscale_ratio,
            background_source_spectrum=background_source_spectrum,
            z=z,
            noise=noise,
            templates_name=templates_name,
            FWHM=FWHM,
            FWHM_tem=FWHM_tem,
            global_template_lens=global_temp,
            plot=plot,
            spectrum_perpixel=spectrum_perpixel)
        if plot is True:
            plt.show()
        else:
            plt.clf()

        best_sky = background_source * pp.weights[-1]
        if spectrum_perpixel is not None:
            gal_lin = spectrum_perpixel
        else:
            gal_lin = fits.getdata(spectrum_aperture)

        log_axis_sky = np.exp(logLam1)
        lin_axis_sky = np.linspace(lamRange1[0], lamRange1[1], gal_lin.size)

        sky_lin = de_log_rebin(log_axis_sky, best_sky, lin_axis_sky)
        best_lin = de_log_rebin(log_axis_sky, pp.bestfit, lin_axis_sky)

        noise = gal_lin - best_lin

        sigma_poisson = poisson_noise(T_exp, best_lin, np.std(noise[wave_min:wave_max]),
                                      per_second=True)

        if plot is not False:
            plt.plot(lin_axis_sky, gal_lin, 'k-', label='data')
            plt.plot(lin_axis_sky, best_lin, 'r-', label='best model ('
                                                         'lens+background_source)')
            plt.plot(lin_axis_sky, gal_lin - sky_lin, 'm-',
                     label='remove background_source from data')
            plt.plot(lin_axis_sky, sky_lin, 'c-',label='best background_source model')
            plt.plot(lin_axis_sky, gal_lin - best_lin, 'g-',
                     label='noise (data - best model)')
            plt.legend()
            plt.xlabel('wavelength (A)')
            plt.ylabel('relative flux')
            plt.show()
            plt.show()
        else:
            plt.clf()




        templates, pp, lamRange1, logLam1, lamRange2, logLam2, galaxy, background_source = \
            ppxf_kinematics_RXJ1131_getGlobal_lens_deredshift(
                libary_dir=libary_dir,
                degree=degree, spectrum_aperture=spectrum_aperture,
                wave_min=wave_min,
                wave_max=wave_max,
                velscale_ratio=velscale_ratio,
                background_source_spectrum=background_source_spectrum, z=z, noise=sigma_poisson,
                templates_name=templates_name,
                FWHM=FWHM,
                FWHM_tem=FWHM_tem,
                global_template_lens=global_temp,
                plot=plot,
                spectrum_perpixel=spectrum_perpixel)
        if plot is not False:
            plt.show()
        else:
            plt.clf()

    return pp


def get_velocity_dispersion_deredshift(degree,
                                       spectrum_aperture,
                                       voronoi_binning_data,
                                       velscale_ratio,
                                       z,
                                       noise,
                                       FWHM,
                                       FWHM_tem_xshooter,
                                       dir,
                                       obj_name,
                                       libary_dir,
                                       global_temp,
                                       wave_min,
                                       wave_max,
                                       T_exp,
                                       background_source_spectrum=None,
                                       VD_name=None,
                                       plot=False):
    measurements = np.zeros(shape=(0,5))
    N = voronoi_binning_data.shape[0]
    for i in range(N):
        spectrum_perpixel = voronoi_binning_data[i]
        pp = fitting_deredshift(libary_dir,
                                degree,
                                spectrum_aperture,
                                wave_min,
                                wave_max,
                                velscale_ratio,
                                z,
                                noise,
                                templates_name='xshooter',
                                FWHM=FWHM,
                                FWHM_tem=FWHM_tem_xshooter,
                                global_temp = global_temp,
                                T_exp=T_exp,
                                background_source_spectrum=background_source_spectrum,
                                plot=plot,
                                spectrum_perpixel=spectrum_perpixel)
        measurements = np.vstack((measurements, np.hstack( (pp.sol[:2],
                                                   (pp.error*np.sqrt(pp.chi2))[:2],
                                                              pp.chi2) 
                                                              )
                                 )
                                )

    if VD_name==None:
        np.savetxt(dir + obj_name + 'kinematics.txt', measurements, fmt='%1.4e')
    else:
        np.savetxt(dir + obj_name + 'kinematics.txt' % VD_name, measurements, fmt='%1.4e')
    
    return measurements # return V, VD, dV, dVD, chi2



def kinematics_map(dir, obj_name, radius_in_pixels):
    '''
    this code remap the kinematics measurements above into 2D array
    :return: 2D velocity dispersion, uncertainty of the velocity dispersion, velocity, and the uncertainty of the velocity
    '''

    measurements=np.loadtxt(dir + obj_name + 'kinematics.txt')
    # Vel, sigma, dv, dsigma
    output=np.loadtxt(dir +'voronoi_2d_binning_' + obj_name + '_output.txt')

    VD_array    =np.zeros(output.shape[0])
    noise_array =np.zeros(output.shape[0])
    V_array     =np.zeros(output.shape[0])
    dv_array    =np.zeros(output.shape[0])


    for i in range(output.shape[0]):
        num=int(output.T[2][i])
        results = measurements[num][1]
        sigma = measurements[num][3]
        v = measurements[num][0]
        dv = measurements[num][2]

        VD_array[i]=results
        noise_array[i]=sigma
        V_array[i]=v
        dv_array[i]=dv


    final=np.vstack((output.T, VD_array, noise_array, V_array, dv_array))

    dim = radius_in_pixels*2+1

    VD_2d=np.zeros((dim, dim))
    VD_2d[:]=np.nan
    for i in range(final.shape[1]):
        VD_2d[int(final[1][i])][int(final[0][i])]=final[3][i]

    sigma_2d=np.zeros((dim, dim))
    sigma_2d[:]=np.nan
    for i in range(final.shape[1]):
        sigma_2d[int(final[1][i])][int(final[0][i])]=final[4][i]


    V_2d=np.zeros((dim, dim))
    V_2d[:]=np.nan
    for i in range(final.shape[1]):
        V_2d[int(final[1][i])][int(final[0][i])]=final[5][i]

    dv_2d=np.zeros((dim, dim))
    dv_2d[:]=np.nan
    for i in range(final.shape[1]):
        dv_2d[int(final[1][i])][int(final[0][i])]=final[6][i]
    return VD_2d, sigma_2d, V_2d, dv_2d


def kinematics_map_systematicsdir(dir, obj_name, mosaic_name, radius_in_pixels):
    '''
    this code modifies CF's kinematics_map function to remap the kinematics measurements above into 2D array from my systematics scripts.
    :return: 2D velocity dispersion, uncertainty of the velocity dispersion, velocity, and the uncertainty of the velocity
    
    # for now, only gives the velocity dispersion terms!
    '''

    VD=np.genfromtxt(f'{dir}{obj_name}_VD_binned.txt',
                 delimiter=',')
    VD_covariance = np.genfromtxt(f'{dir}{obj_name}_covariance_matrix_VD.txt',
                                 delimiter=',')
    dVD = np.sqrt(np.diagonal(VD_covariance)) 
# diagonal is sigma**2
    
    
    V=np.genfromtxt(f'{dir}{name}_V_binned.txt',
                 delimiter=',')
    V_covariance = np.genfromtxt(f'{dir}{obj_name}_covariance_matrix_V.txt',
                                 delimiter=',')
    dV = np.sqrt(np.diagonal(VD_covariance)) # diagonal is sigma**2
    
    # Vel, sigma, dv, dsigma
    output=np.loadtxt(dir +'voronoi_2d_binning_' + obj_name + '_output.txt')

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