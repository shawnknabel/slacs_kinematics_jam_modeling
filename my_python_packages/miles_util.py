###############################################################################
#
# Copyright (C) 2016-2022, Michele Cappellari
# E-mail: michele.cappellari_at_physics.ox.ac.uk
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
###############################################################################

# This file contains the 'miles' class with functions to construct
# a library of MILES templates and interpret and display the output
# of pPXF when using those templates as input.
#
# These procedures can be used as templates and can be easily
# adapted for use with alternative stellar population libraries.
# pPXF itself is designed to be independent of the adopted
# stellar population models.

from os import path
import glob, re

import numpy as np
from scipy import ndimage
from astropy.io import fits

from . import ppxf_util as util

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 27 November 2016
#   V1.0.1: Assume seven characters for the age field. MC, Oxford, 5 July 2017
#   V1.0.2: Robust string matching to deal with different MILES conventions.
#       MC, Oxford, 29 November 2017

def age_metal(filename):
    """
    Extract the age and metallicity from the name of a file of
    the MILES library of Single Stellar Population models as
    downloaded from http://miles.iac.es/ as of 2022

    This function relies on the MILES file containing a substring of the
    precise form like Zm0.40T00.0794, specifying the metallicity and age.

    :param filename: string possibly including full path
        (e.g. 'miles_library/Eun1.30Zm0.40T00.0794.fits')
    :return: age (Gyr), [M/H]

    """
    s = re.findall(r'Z[m|p][0-9]\.[0-9]{2}T[0-9]{2}\.[0-9]{4}', filename)[0]
    metal = s[:6]
    age = float(s[7:])
    if "Zm" in metal:
        metal = -float(metal[2:])
    elif "Zp" in metal:
        metal = float(metal[2:])

    return age, metal


###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Adapted from my procedure setup_spectral_library() in
#       ppxf_example_population_sdss(), to make it a stand-alone procedure.
#     - Read the characteristics of the spectra directly from the file names
#       without the need for the user to edit the procedure when changing the
#       set of models. Michele Cappellari, Oxford, 28 November 2016
#   V1.0.1: Check for files existence. MC, Oxford, 31 March 2017
#   V1.0.2: Included `max_age` optional keyword. MC, Oxford, 4 October 2017
#   V1.0.3: Included `metal` optional keyword. MC, Oxford, 29 November 2017
#   V1.0.4: Changed imports for pPXF as a package. MC, Oxford, 16 April 2018
#   V1.1.0: Replaced ``normalize`, `max_age`, `min_age` and 'metal` keywords
#       with `norm_range`, `age_range` and `metal_range`.
#       MC, Oxford, 23 November 2018
#  V1.2.0: Added .flux attribute to convert light weights into mass weights.
#       MC, Oxford, 16 July 2021
#  V1.3.0: New keyword ``wave_range``. MC, Oxford, 16 March 2022.

class miles:
    """
    This class is meant as an example that can be easily adapted by the users
    to deal with other spectral templates libraries, different IMFs or different
    chemical abundances.

    This code produces an array of logarithmically-binned templates by reading
    the spectra from the Single Stellar Population (SSP) `MILES <http://miles.iac.es/>`_ 
    library by `Vazdekis et al. (2016) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.463.3409V>`_.
    The code checks that the model spectra form a rectangular grid
    in age and metallicity and properly sorts them in both parameters.
    The code also returns the age and metallicity of each template
    by reading these parameters directly from the file names.
    The templates are broadened by a Gaussian with dispersion
    ``sigma_diff = np.sqrt(sigma_gal**2 - sigma_tem**2)``.

    Thie script is designed to use the files naming convention adopted by
    the MILES library, where SSP spectra file names have the form like below::
    
        ...Z[Metallicity]T[Age]...fits
        e.g. Eun1.30Zm0.40T00.0631_iPp0.00_baseFe_linear_FWHM_variable.fits

    Input Parameters
    ----------------

    pathname: 
        path with wildcards returning the list of files to use
        (e.g. ``miles_models/Eun1.30*.fits``). The files must form a Cartesian
        grid in age and metallicity and the procedure returns an error if
        they do not.
    velscale: 
        desired velocity scale for the output templates library in km/s 
        (e.g. 60). This is generally the same or an integer fraction of the 
        ``velscale`` of the galaxy spectrum used as input to ``ppxf``.
    FWHM_gal: 
        scalar or vector with the FWHM of the instrumental resolution of the 
        galaxy spectrum in Angstrom at every pixel of the stellar templates.
        
        - If ``FWHM_gal=None`` (default), no convolution is performed.

    Optional Keywords
    -----------------

    age_range: array_like with shape (2,)
        ``[age_min, age_max]`` optional age range (inclusive) in Gyr for the 
        MILES models. This can be useful e.g. to limit the age of the templates 
        to be younger than the age of the Universe at a given redshift.
    metal_range: array_like with shape (2,)
        ``[metal_min, metal_max]`` optional metallicity [M/H] range (inclusive) 
        for the MILES models (e.g.`` metal_range = [0, 10]`` to select only 
        the spectra with Solar metallicity and above).
    norm_range: array_like with shape (2,)
        A two-elements vector specifying the wavelength range in Angstrom 
        within which to compute the templates normalization 
        (e.g. ``norm_range=[5070, 5950]`` for the FWHM of the V-band).
        
        - When this keyword is set, the templates are normalized to
          ``np.mean(template[band]) = 1`` in the given wavelength range.
          
        - When this keyword is used, ``ppxf`` will output light weights, and
          ``mean_age_metal()`` will provide light-weighted stellar population
          quantities.
          
        - If ``norm_range=None`` (default), the templates are not normalized.

        - One can use the output attribute ``.flux`` to convert mass-weights
          into light-weights, without repeating the ``ppxf`` fit. However,
          when using regularization in ``ppxf`` the results will not be
          identical. In fact, enforcing smoothness to the light-weights is
          not quite the same as enforcing it to the mass-weights.
    wave_range: array_like with shape (2,)
        A two-elements vector specifying the wavelength range in Angstrom for
        which to extract the stellar templates. Restricting the wavelength
        range of the templates to the range of the galaxy data is useful to
        save some computational time. By default ``wave_range=[3541, 1e4]``

    Output Parameters
    -----------------

    Stored as attributes of the ``miles`` class:

    .age_grid: array_like with shape (n_ages, n_metals)
        Age in Gyr of every template.
    .flux: array_like with shape (n_ages, n_metals)
        If ``norm_range is not None`` then ``.flux`` contains the mean flux
        in each template spectrum within ``norm_range`` before normalization.

        When using the ``norm_range`` keyword, the weights returned by 
        ``ppxf`` represent light contributed by each SSP population template.
        One can then use this ``.flux`` attribute to convert the light weights
        into fractional masses as follows::

            pp = ppxf(...)                                  # Perform the ppxf fit
            light_weights = pp.weights[~gas_component]      # Exclude gas templates weights
            light_weights = light_weights.reshape(reg_dim)  # Reshape to a 2D matrix
            mass_weights = light_weights/miles.flux         # Divide by this attribute
            mass_weights /= mass_weights.sum()              # Normalize to sum=1

    .ln_lam_temp: array_like with shape (npixels,)
        Natural logarithm of the wavelength in Angstrom of every pixel.
    .metal_grid: array_like with shape (n_ages, n_metals)
        Metallicity [M/H] of every template.
    .n_ages: 
        Number of different ages.
    .n_metal: 
        Number of different metallicities.
    .templates: array_like with shape (npixels, n_ages, n_metals)
        Array with the spectral templates.

    """
    def __init__(self, pathname, velscale, FWHM_gal=None, FWHM_tem=2.51, age_range=None,
                 metal_range=None, norm_range=None, wave_range=[0, np.inf]):
        
        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        all = [age_metal(f) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        n_ages, n_metal = len(ages), len(metals)

        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the galaxy spectrum, to determine the
        # size needed for the array which will contain the template spectra.
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam = h2['CRVAL1'] + np.arange(h2['NAXIS1'])*h2['CDELT1']
        lam_range_temp = lam[[0, -1]]
        ssp_new, ln_lam_temp = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[:2]

        if norm_range is not None:
            norm_range = np.log(norm_range)
            band = (norm_range[0] <= ln_lam_temp) & (ln_lam_temp <= norm_range[1])

        templates = np.empty((ssp_new.size, n_ages, n_metal))
        age_grid, metal_grid, flux = np.empty((3, n_ages, n_metal))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        if FWHM_gal is not None:
            FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
            sigma = FWHM_dif/2.355/h2['CDELT1']   # Sigma difference in pixels

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for j, age in enumerate(ages):
            for k, met in enumerate(metals):
                p = all.index((age, met))
                hdu = fits.open(files[p])
                ssp = hdu[0].data
                if FWHM_gal is not None:
                    if np.isscalar(FWHM_gal):
                        if sigma > 0.1:   # Skip convolution for nearly zero sigma
                            ssp = ndimage.gaussian_filter1d(ssp, sigma)
                    else:
                        ssp = util.gaussian_filter1d(ssp, sigma)  # convolution with variable sigma
                ssp_new = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[0]
                if norm_range is not None:
                    flux[j, k] = np.mean(ssp_new[band])
                    ssp_new /= flux[j, k]   # Normalize every spectrum
                templates[:, j, k] = ssp_new
                age_grid[j, k] = age
                metal_grid[j, k] = met

        if age_range is not None:
            w = (age_range[0] <= age_grid[:, 0]) & (age_grid[:, 0] <= age_range[1])
            templates = templates[:, w, :]
            age_grid = age_grid[w, :]
            metal_grid = metal_grid[w, :]
            flux = flux[w, :]
            n_ages, n_metal = age_grid.shape

        if metal_range is not None:
            w = (metal_range[0] <= metal_grid[0, :]) & (metal_grid[0, :] <= metal_range[1])
            templates = templates[:, :, w]
            age_grid = age_grid[:, w]
            metal_grid = metal_grid[:, w]
            flux = flux[:, w]
            n_ages, n_metal = age_grid.shape

        if norm_range is None:
            flux = np.median(templates[templates > 0])
            templates /= flux  # Normalize by a scalar

        self.templates_full = templates
        self.ln_lam_temp_full = ln_lam_temp
        self.lam_temp_full = np.exp(ln_lam_temp)
        if wave_range is not None:
            lam = np.exp(ln_lam_temp)
            good_lam = (lam >= wave_range[0]) & (lam <= wave_range[1])
            ln_lam_temp = ln_lam_temp[good_lam]
            templates = templates[good_lam, :, :]

        self.templates = templates
        self.ln_lam_temp = ln_lam_temp
        self.lam_temp = np.exp(ln_lam_temp)
        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.flux = flux


###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 1 December 2016
#   V1.0.1: Use path.realpath() to deal with symbolic links.
#       Thanks to Sam Vaughan (Oxford) for reporting problems.
#       MC, Garching, 11 January 2016
#   V1.0.2: Changed imports for pPXF as a package. MC, Oxford, 16 April 2018
#   V1.0.3: Removed dependency on cap_readcol. MC, Oxford, 10 May 2018

    def mass_to_light(self, weights, band="r", quiet=False):
        """
        Computes the M/L in a chosen band, given the weights produced
        in output by pPXF. A Salpeter IMF is assumed (slope=1.3).
        The returned M/L includes living stars and stellar remnants,
        but excludes the gas lost during stellar evolution.

        This procedure uses the photometric predictions
        from Vazdekis+12 and Ricciardelli+12
        http://adsabs.harvard.edu/abs/2012MNRAS.424..157V
        http://adsabs.harvard.edu/abs/2012MNRAS.424..172R
        I downloaded them from http://miles.iac.es/ in December 2016 and I
        included them in pPXF with permission.

        :param weights: pPXF output with dimensions weights[miles.n_ages, miles.n_metal]
        :param band: possible choices are "U", "B", "V", "R", "I", "J", "H", "K" for
            the Vega photometric system and "u", "g", "r", "i" for the SDSS AB system.
        :param quiet: set to True to suppress the printed output.
        :return: mass_to_light in the given band

        """
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
        sdss_bands = ["u", "g", "r", "i"]
        vega_sun_mag = [5.600, 5.441, 4.820, 4.459, 4.148, 3.711, 3.392, 3.334]
        sdss_sun_mag = [6.55, 5.12, 4.68, 4.57]  # values provided by Elena Ricciardelli

        ppxf_dir = path.dirname(path.realpath(util.__file__))

        if band in vega_bands:
            k = vega_bands.index(band)
            sun_mag = vega_sun_mag[k]
            file2 = ppxf_dir + "/miles_models/Vazdekis2012_ssp_phot_Padova00_UN_v10.0.txt"
        elif band in sdss_bands:
            k = sdss_bands.index(band)
            sun_mag = sdss_sun_mag[k]
            file2 = ppxf_dir + "/miles_models/Vazdekis2012_ssp_sdss_miuscat_UN1.30_v9.txt"
        else:
            raise ValueError("Unsupported photometric band")

        file1 = ppxf_dir + "/miles_models/Vazdekis2012_ssp_mass_Padova00_UN_baseFe_v10.0.txt"
        slope1, MH1, Age1, m_no_gas = np.loadtxt(file1, usecols=[1, 2, 3, 5]).T

        slope2, MH2, Age2, mag = np.loadtxt(file2, usecols=[1, 2, 3, 4 + k]).T

        # The following loop is a brute force, but very safe and general,
        # way of matching the photometric quantities to the SSP spectra.
        # It makes no assumption on the sorting and dimensions of the files
        mass_no_gas_grid = np.empty_like(weights)
        lum_grid = np.empty_like(weights)
        for j in range(self.n_ages):
            for k in range(self.n_metal):
                p1 = (np.abs(self.age_grid[j, k] - Age1) < 0.001) & \
                     (np.abs(self.metal_grid[j, k] - MH1) < 0.01) & \
                     (np.abs(1.30 - slope1) < 0.01)   # Salpeter IMF
                mass_no_gas_grid[j, k] = m_no_gas[p1]

                p2 = (np.abs(self.age_grid[j, k] - Age2) < 0.001) & \
                     (np.abs(self.metal_grid[j, k] - MH2) < 0.01) & \
                     (np.abs(1.30 - slope2) < 0.01)   # Salpeter IMF
                lum_grid[j, k] = 10**(-0.4*(mag[p2] - sun_mag))

        # This is eq.(2) in Cappellari+13
        # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
        mlpop = np.sum(weights*mass_no_gas_grid)/np.sum(weights*lum_grid)

        if not quiet:
            print('M/L_' + band + ': %#.4g' % mlpop)

        return mlpop


###############################################################################
    # MODIFICATION HISTORY:
    #   V1.0.0: Written. Michele Cappellari, Oxford, 15 March 2022

    def photometry_from_table(self, band="r"):
        """
        Returns the fluxes in ergs/(s cm^2 A) of each SSP spectral template in
        the requested band.

        This procedure uses the MILES population models http://miles.iac.es/
        photometric predictions from Vazdekis+12 and Ricciardelli+12
        http://adsabs.harvard.edu/abs/2012MNRAS.424..157V
        http://adsabs.harvard.edu/abs/2012MNRAS.424..172R they were downloaded
        in December 2016 below and are included in pPXF with permission

        :param band: possible choices are "U", "B", "V", "R", "I", "J", "H", "K" for
            the Vega photometric system and "u", "g", "r", "i" for the SDSS AB system.
        :return: fluxes for all templates in the given band, normalized in the same
            way as the spectral templates.

        """
        vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
        vega_lam_pivot = [3611, 4396, 5511, 6582, 8034, 12393, 16495, 21638]     # Willmer+18
        vega_fluxes = np.array([422, 640, 355, 210, 115, 32.1, 11.6, 4.5])/1e11  # erg/(s cm^2 A)

        sdss_bands = ["u", "g", "r", "i"]
        sdss_lam_pivot = [3556, 4702, 6176, 7490]   # Willmer+18

        ppxf_dir = path.dirname(path.realpath(util.__file__))

        if band in vega_bands:
            b = vega_bands.index(band)
            file2 = ppxf_dir + "/miles_models/Vazdekis2012_ssp_phot_Padova00_UN_v10.0.txt"
        elif band in sdss_bands:
            b = sdss_bands.index(band)
            file2 = ppxf_dir + "/miles_models/Vazdekis2012_ssp_sdss_miuscat_UN1.30_v9.txt"
        else:
            raise ValueError("Unsupported photometric band")

        slope, MH, age, mag = np.loadtxt(file2, usecols=[1, 2, 3, 4 + b]).T

        # The following loop is a brute force, but very safe and general,
        # way of matching the photometric quantities to the SSP spectra.
        # It makes no assumption on the sorting and dimensions of the files
        mag_grid = np.empty((self.n_ages, self.n_metal))
        for j in range(self.n_ages):
            for k in range(self.n_metal):
                p2 = (np.abs(self.age_grid[j, k] - age) < 0.001) & \
                     (np.abs(self.metal_grid[j, k] - MH) < 0.01) & \
                     (np.abs(1.30 - slope) < 0.01)  # Salpeter IMF
                mag_grid[j, k] = mag[p2]

        if band in vega_bands:
            f_lam_grid = vega_fluxes[b]*10**(-0.4*mag_grid)
            lam = vega_lam_pivot[b]
        elif band in sdss_bands:  # AB system
            c = 2.99792458e18                               # A/s
            lam = sdss_lam_pivot[b]
            f_nu_grid = 10**(-0.4*(mag_grid + 48.60))       # erg/(s cm^2 Hz)
            f_lam_grid = f_nu_grid*c/lam**2                 # erg/(s cm^2 A)

        return lam, f_lam_grid*3.14e6/self.flux  # normalized like the spectra


###############################################################################
    # MODIFICATION HISTORY:
    #   V1.0.0: Written. Michele Cappellari, Oxford, 15 March 2022

    def photometry_from_spectra(self, band="r", redshift=0):
        """
        Returns the fluxes in ergs/(s cm^2 A) of each SSP spectral template in
        the requested band. By design these fluxes are consistent with the
        fluxes in the spectral templates as requested by pPXF.

        :param band: possible choices are "NUV" (GALEX), ["u", "g", "r", "i"]
            (SDSS) and ["J", "H", "K"] (2MASS). Additional filters can be
            easily included in the file "filters.txt" below.
        :param redshift: approximate redshift of the galaxy under study for
            which the photometric fluxes were measured.
        :return: fluxes for all templates in the given band, normalized in the
            same way as the spectral templates.

        IMPORTANT: Some of the templates have no NIR spectrum at age < 1Gyr.
            See Vazdekis et al. (2016, https://ui.adsabs.harvard.edu/abs/2016MNRAS.463.3409V)
            for details on the caveats.

        NOTE: See the documentation of the ``pPXF`` keyword ``phot`` for an
            explanation of the formulas in this procedure.

        """
        ppxf_dir = path.dirname(path.realpath(util.__file__))

        if band == 'NUV':   # GALEX
            lstart, ltot = 2001, 67  # (1) lines to skip (2) lines to read
        elif band == 'u':   # SDSS
            lstart, ltot = 279, 47
        elif band == 'g':   # SDSS
            lstart, ltot = 327, 89
        elif band == 'r':   # SDSS
            lstart, ltot = 417, 75
        elif band == 'i':   # SDSS
            lstart, ltot = 493, 89
        elif band == 'z':   # SDSS
            lstart, ltot = 583, 141
        elif band == 'J':   # 2MASS
            lstart, ltot = 725, 109
        elif band == 'H':   # 2MASS
            lstart, ltot = 835, 58
        elif band == 'K':   # 2MASS
            lstart, ltot = 894, 78
        else:
            raise ValueError("Unsupported photometric band")

        # The spectra are logarithmically sampled and I
        # integrate in the logarithm using the identity
        # Integrate[f[lam], lam] = Integrate[lam*f[lam], ln_lam]
        lam_resp, resp = np.loadtxt(ppxf_dir + "/miles_models/filters.txt", skiprows=lstart, max_rows=ltot).T
        lam_resp = lam_resp/(1 + redshift)
        lam = self.lam_temp_full
        lam_in_fwhm = lam_resp[resp > 0.5*np.max(resp)]
        l1, l2 = np.min(lam_in_fwhm), np.max(lam_in_fwhm)   # I want FWHM fully covered
        assert np.all((l1 >= lam[0]) & (l2 <= lam[-1])), \
            f"The {band} filter falls outside the templates wavelength range"
        fil = np.interp(lam, lam_resp, resp, left=0, right=0)
        filam2 = fil*lam**2
        filam3 = filam2*lam
        int1 = filam2.sum()
        int2 = filam3.sum()
        int3 = (self.templates_full.T*filam2).T.sum(0)
        int4 = (self.templates_full.T*filam3).T.sum(0)
        flux_grid = int3/int1
        with np.errstate(invalid='ignore'):
            lam_grid = np.where(int3 > 0, int4/int3, int2/int1)

        return lam_grid, flux_grid


###############################################################################

    def plot(self, weights, nodots=False, colorbar=True, **kwargs):

        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        xgrid = np.log10(self.age_grid) + 9
        ygrid = self.metal_grid
        util.plot_weights_2d(xgrid, ygrid, weights,
                             nodots=nodots, colorbar=colorbar, **kwargs)


##############################################################################

    def mean_age_metal(self, weights, quiet=False):

        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        lg_age_grid = np.log10(self.age_grid) + 9
        metal_grid = self.metal_grid

        # These are eq.(1) and (2) in McDermid+15
        # http://adsabs.harvard.edu/abs/2015MNRAS.448.3484M
        mean_lg_age = np.sum(weights*lg_age_grid)/np.sum(weights)
        mean_metal = np.sum(weights*metal_grid)/np.sum(weights)

        if not quiet:
            print('Weighted <lg_age> [yr]: %#.3g' % mean_lg_age)
            print('Weighted <[M/H]>: %#.3g' % mean_metal)

        return mean_lg_age, mean_metal


##############################################################################
