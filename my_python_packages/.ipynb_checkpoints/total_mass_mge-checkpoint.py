# mass profile as total mass with other pieces

################################################################

# import general libraries and modules
import numpy as np
np.set_printoptions(threshold=10000)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 6)
plt.switch_backend('agg')
import pandas as pd
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings( "ignore", module = "plotbin\..*" )

# astronomy/scipy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
import astropy.units as u
import astropy.constants as constants

# mge fit
import mgefit
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.mge_print_contours import mge_print_contours

# my functions
import sys
sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages")


class total_mass_mge:
    
    '''
    Class purpose
    ----------------
    
    Return a mass profile designated by input parameters for use in JAM modeling. Uses MGE and JAM modules.
    
    Input Parameters
    ----------------
    
    surf_lum: array_like with shape (nlum)
        peak surface brightness of each MGE component of the surface brightness profile
        
    sigma_lum: array_like with shape (nlum)
        sigma of MGE component of the surface brightness profile
        
    qobs_lum: array_like with shape (nlum)
        axis ratio of each MGE component of the surface brightness profile
    
    model: str
        tells which total mass profile model to use (usually "power_law")
    
    qobs_eff: float
        effective axis ratio of observed 2d surface brightness profile, measured from MGE isophotes
        
    reff: float
        half-light radius of observed 2d surface brightness profile, measured from MGE isophotes
        
    break_factor: float
        factor of the effective radius (reff) at which to truncate the mass profile (should be much higher than radius of kinematic data)
        
    zlens, zsource: float
        redshift of lens and background source, needed for surface mass calculation (distances)
        
    cosmo: astropy.cosmology instance
        describes the cosmology used to get the distance for teh surface mass calculation
        
    gamma: float
        pure power law slope for mass profile, parameter fit in MCMC
        
    f_dm: float
        fraction of dark matter, not needed for power law
        
    theta_E: float
        einstein radius, in arcsec, parameter fit in MCMC
        
    k_mst: float
        scale of mass sheet parameter lambda_int in range(min, max) allowed by the mass profile, parameter fit in MCMC
        
    a_mst: float
        scale of mass sheet truncation in units of reff, parameter fit in MCMC
        
    lambda_int: float
        lambda_int, will override k_mst if specified
        
    Optional Keywords
    --------------
    
    ngauss: int
        number of gaussians for 1d MGE of surface mass profile
    
    inner_slope, outer_slope: int
        constraints to help MGE get itself right at the very center and very end of the profile
        
    quiet: 0 or 1
        makes the output quieter
        
    plot: boolean
        plot things in MGE
    
    skip_mge: boolean
        if true, doesn't do the mge fit, useful just to return the mass profile to look at it
        
    Output Parameters
    --------------
    
    Stored as attributes of total_mass_mge class
    
    .surf_pot
        peak surface mass density of MGE components
    
    .sigma_pot
        sigma of surface mass density MGE components
    
    .qobs_pot
        axis ratio of surface mass density components (2D projection)

    '''
    
    # initialize with self and inputs
    def __init__(self, 
                 surf_lum, sigma_lum, qobs_lum, 
                 model, qobs_eff, reff, break_factor, zlens, zsource, cosmo,
                 gamma, f_dm, theta_E, k_mst, a_mst, lambda_int=None, 
                 ngauss=30, inner_slope=2, outer_slope=3, 
                 quiet=1, plot=False, skip_mge=False):
        
        # luminosity MGE decomponsition
        self.surf_lum = surf_lum
        self.sigma_lum = sigma_lum
        self.qobs_lum = qobs_lum
        self.nlum = surf_lum.shape[0]
        # options/specifics of the object
        self.model = model
        self.qobs_eff = qobs_eff
        self.reff = reff
        self.break_factor = break_factor
        self.rbreak = reff*break_factor
        self.zlens = zlens
        self.zsource = zsource
        # parameters that describe mass profile
        self.gamma = gamma
        self.f_dm = f_dm
        self.theta_E = theta_E
        self.k_mst = k_mst
        self.lambda_int = lambda_int
        self.a_mst = a_mst
        
        #### Check possible input errors
        
        # value of c^2 / 4 pi G
        c2_4piG = (constants.c **2 / constants.G / 4 / np.pi).to('solMass/pc')        
        # profile set up
        # The fit is performed in log spaced radii from 1" to 10*rbreak
        n = 1000     # Number of values to sample the gNFW profile for the MGE fit
        self.r = np.geomspace(0.01, self.rbreak, n)   # logarithmically spaced radii in arcsec
        
        # fit the initial mass model
        if model=='power_law':
            self.power_law()
        
        # transform by mass sheet
        self.mass_sheet_transform()
        
        if skip_mge == False:
            # get surface mass density by dividing by sigma crit
            self.convergence_to_surf_mass_density()

            # get 1d mge profile
            m = mge_fit_1d(self.r, self.surf_mass_density, ngauss=ngauss, inner_slope=inner_slope, outer_slope=outer_slope, quiet=quiet, plot=plot) # this creates a circular gaussian with sigma=sigma_x (i.e. along the major axis)
            surf_pot_tot, sigma_pot = m.sol           # total counts of gaussians Msol/(pc*2/arcsec**2)
            # Normalize by dividing by (2 * np.pi * sigma_pot**2 * q)
            self.surf_pot = surf_pot_tot / (2 * np.pi * sigma_pot**2 * self.qobs_eff) # peak surface density
            self.sigma_pot = sigma_pot
            self.qobs_pot = np.ones_like(self.surf_pot)*self.qobs_eff   # Multiply by q to convert to elliptical Gaussians where sigma is along the major axis...    

    def power_law (self):
        
        """
        Return convergence profile of pure power law profile from parameters
        """
        
        # fit to power law mass density (convergence) profile
        self.kappa_power_law = (3 - self.gamma) / 2 * (self.theta_E/self.r)**(self.gamma-1)
        
    def mass_sheet_transform (self):
        
        '''
        Return transformed convergence profile
        kappa is the convergence profile (surface mass density/critical surface density).
        MST scales by lambda and adds the "infinite" sheet
        kappa_s is the mass sheet
        rs_mst is a "turnover" radius [0,1] (multiplicative factor of rbreak) where it goes to 0, so that it is physical.
        kappa_s = theta_s**2 / (theta_E**2 + theta_s**2)
        Figure 12 from Shajib2023 https://arxiv.org/pdf/2301.02656.pdf
        11/13/23 now, lambda_int will be parameterized as a value k_mst [0,1] that will be transformed into a parameter space allowed by the rest of the model
        '''

        r_s = self.a_mst * self.reff
        kappa_s = r_s**2/(self.r**2 + r_s**2)
        
        if self.lambda_int==None:
            # find the maximum lambda_int possible given the model
            lambda_int_min = 0.8
            lambda_int_max = 1.2
            lambda_ints = np.linspace(1.0,1.2,1000)
            for i, test in enumerate(lambda_ints):
                kappa_bounds = self.kappa_power_law * test + (1 - test) * kappa_s
                if any(kappa_bounds<0):
                    lambda_int_max = lambda_ints[i-1]
                    break

            # calculate surface mass density with mass sheet transform
            self.lambda_int = lambda_int_min + (lambda_int_max - lambda_int_min) * self.k_mst # lambda_int is a value [0,1] so lambda_internal will be between [0.8, lambda_int_max]
        
        # transform using mass sheet
        mass_sheet = (1 - self.lambda_int) * kappa_s
        self.kappa_int = self.lambda_int * self.kappa_power_law + mass_sheet

        if any(self.kappa_int<0):
            print('Somehow, we have negative mass even though we set it up not to.')
            self.lambda_int=0
            
    def convergence_to_surf_mass_density(self):
        # Go from convergence to surface mass density with critical surface density
        # get distances
        DL = cosmo.angular_diameter_distance(self.zlens).to('pc')
        DS = cosmo.angular_diameter_distance(self.zsource).to('pc')
        DLS = cosmo.angular_diameter_distance_z1z2(self.zlens, self.zsource).to('pc')
        # calculate critical surface density
        sigma_crit = c2_4piG * DS / DL / DLS
        self.surf_mass_density = self.kappa_int / sigma_crit.value
        
    
    
        