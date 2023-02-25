import numpy as np
from astropy.io import fits

def Xshooter(name):
    '''

    :param name: names of the template files
    :return: ssp=flux, lamRange2 = [min wavelength, max wavelength],
    h2 = delta wavelength per pixel in the unit of Angstrom
    '''
    hdu = fits.open(name)
    ssp = hdu[1].data['Flux']
    h2 = hdu[1].data['Wave'][2] - hdu[1].data['Wave'][1] # in the unit of nm
    lamRange2 = np.array([hdu[1].data['Wave'][0], hdu[1].data['Wave'][-1]])
    # in the unit of nm
    return ssp, lamRange2 * 10, h2 * 10

def miles(name):
    hdu = fits.open(name)
    ssp = hdu[0].data
    h2 = hdu[0].header
    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])
    return ssp, lamRange2, h2['CDELT1']

def Munari05_subset(name, wavelength):
    '''
    This subset are from Michele Capperlari
    '''
    ssp = np.loadtxt(name)
    wave = np.loadtxt(wavelength)
    h2 = wave[3] - wave[2]
    lamRange2 = np.array([wave[0], wave[-1]])
    return ssp, lamRange2, h2

def indo_US(name):
    '''
    indo_US templates
    :param name:
    :return:
    '''
    file = np.loadtxt(name)
    ssp = file[:,1]
    wave = file[:,0]
    h2 = wave[2]-wave[1]
    lamRange2 = np.array([wave[0], wave[-1]])
    return ssp, lamRange2, h2