# # 02/09/24 - Revising the spherical geometry git to be to the 2D data instead of the 1D.
# ## Just give the bin R=sqrt(x**2+y**2)
# # 02/08/24 - Preparing all 10 models (5 PL, 5 MFL, 3 const, 2 OM) to be able to run in a script.
# # 02/07/24 - Added spherical modeling and OM anisotropy
# # 02/02/24 - Modified so that I can show the differences between cyl/sph and MFL/PL
# # 02/01/24 - This notebook was copied from 011924_jam_mass_profile_testing.ipynb and will be used to measure the enclossed mass within the einstein radius of mass-follows-light models
# ______________
# # 01/19/24 - This notebook tests my mass profile class "total_mass_mge" in e.g.
# # 01/29/24 - Added looking at the mass profile and brightness profile at Michele's suggestion, using the mass-follows-light to compare with powerlaw
# # 01/30/24 - Added Michele's power law code to see if I can reproduce it.
# ## Shawn wrote this. I've compiled this to be ready for Michele to test.

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

# import general libraries and modules
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 6)
import pandas as pd
import dill as pickle

# astronomy/scipy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
import astropy.units as u
import astropy.constants as constants

# ppxf/capfit
from ppxf.capfit import capfit

# mge fit
import mgefit
from mgefit.mge_fit_1d import mge_fit_1d

# jampy
from jampy.jam_axi_proj import jam_axi_proj
from jampy.jam_sph_proj import jam_sph_proj
from jampy.mge_half_light_isophote import mge_half_light_isophote
from jampy.mge_half_light_isophote import mge_half_light_radius

# plotbin
from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield
from plotbin.sauron_colormap import register_sauron_colormap
register_sauron_colormap()


# my functions
import sys
sys.path.append("/home/shawnknabel/Documents/slacs_kinematics/my_python_packages")
import slacs_mge_jampy

###############################################################################################################################
###############################################################################################################################

################################################################
# some needed information
kcwi_scale = 0.1457  # arcsec/pixel

# value of c^2 / 4 pi G
c2_4piG = (constants.c **2 / constants.G / 4 / np.pi).to('solMass/pc')


# In[2]:


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


# In[3]:


#################################################
# directories and tables

date_of_kin = '2023-02-28_2'
# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
hst_dir = '/data/raw_data/HST_SLACS_ACS/kcwi_kinematics_lenses/'
tables_dir = f'{data_dir}tables/'
mosaics_dir = f'{data_dir}mosaics/'
kinematics_full_dir = f'{data_dir}kinematics/'
kinematics_dir =f'{kinematics_full_dir}{date_of_kin}/'
jam_output_dir = f'{data_dir}jam_outputs/2024_02_09_jam_milestone_outputs/'
milestone_dir = f'{data_dir}milestone23_data/'

paper_table = pd.read_csv(f'{tables_dir}paper_table_100223.csv')
slacs_ix_table = pd.read_csv(f'{tables_dir}slacs_ix_table3.csv')
zs = paper_table['zlens']
zlenses = slacs_ix_table['z_lens']
zsources = slacs_ix_table['z_src']
# get the revised KCWI sigmapsf
sigmapsf_table = pd.read_csv(f'{tables_dir}kcwi_sigmapsf_estimates.csv')
lens_models = pd.read_csv(f'{tables_dir}lens_models_table_chinyi.csv')


# In[4]:


lens_models_chinyi = pd.read_csv(f'{tables_dir}lens_models_table_chinyi.csv')
lens_models_chinyi_sys = pd.read_csv(f'{tables_dir}lens_models_table_chinyi_with_sys.csv')
lens_models_anowar = pd.read_csv(f'{tables_dir}lens_models_table_anowar.csv')

lens_models_chinyi_sys.rename(columns={'dgamma':'gamma_err',
                                       'dgamma_sys':'gamma_sys',
                                      },inplace=True)


lens_models_chinyi_sys['dgamma'] = np.sqrt( lens_models_chinyi_sys['gamma_err']**2 + lens_models_chinyi_sys['gamma_sys']**2 )
lens_models_chinyi_sys.loc[9, 'dgamma'] = np.sqrt( lens_models_chinyi_sys.loc[9, 'gamma_err']**2 + np.nanmean(lens_models_chinyi_sys['gamma_sys'])**2)
lens_models = lens_models_chinyi_sys


###############################################################################################################################
###############################################################################################################################

# # Functions needed to get the data, etc

# In[5]:


###############################################################################

# class to collect and save all the attributes I need for jampy
class jampy_details:
    
    def __init__(details, surf_density, mge_sigma, q, kcwi_sigmapsf, Vrms_bin, dVrms_bin, V_bin, dV_bin, xbin_phot, ybin_phot, reff):
        details.surf_density=surf_density 
        details.mge_sigma=mge_sigma
        details.q=q 
        details.kcwi_sigmapst=kcwi_sigmapsf 
        details.Vrms_bin=Vrms_bin 
        details.dVrms_bind=dVrms_bin
        details.V_bin=V_bin 
        details.dV_bin=dV_bin 
        details.xbin_phot=xbin_phot 
        details.ybin_phot=ybin_phot
        details.reff=reff
        
        
###############################################################################


def prepare_to_jam(obj_name, file_dir, SN):

    # take the surface density, etc from mge saved parameters
    with open(f'{file_dir}{obj_name}_{SN}_details_for_jampy.pkl', 'rb') as f:
        tommy_pickles = pickle.load(f)
        
    surf = tommy_pickles.surf_density
    sigma = tommy_pickles.mge_sigma
    qObs = tommy_pickles.q
    #kcwi_sigmapsf = tommy_pickles.kcwi_sigmapst # mistake in name  
    try:
        kcwi_sigmapsf = tommy_pickles.kcwi_sigmapsf
    except:
        kcwi_sigmapsf = tommy_pickles.kcwi_sigmapst
    Vrms_bin = tommy_pickles.Vrms_bin
    try:
        dVrms_bin = tommy_pickles.dVrms_bin 
    except:
        dVrms_bin = tommy_pickles.dVrms_bind # mistake in name
    V_bin = tommy_pickles.V_bin
    dV_bin = tommy_pickles.dV_bin
    xbin_phot = tommy_pickles.xbin_phot
    ybin_phot = tommy_pickles.ybin_phot
    reff = tommy_pickles.reff
    
    return (surf, sigma, qObs, kcwi_sigmapsf, Vrms_bin, dVrms_bin, V_bin, dV_bin, xbin_phot, ybin_phot, reff)

def calculate_1d_gaussian(r, amp, sigma):
    return amp * np.exp(-1/2 * r**2 / sigma**2)

###############################################################################################################################
###############################################################################################################################

# # Michele's power law code

# In[6]:


###############################################################################

def total_mass_mge_cap(gamma, rbreak):
    """
    Returns the MGE parameters for a generalized NFW dark halo profile
    https://ui.adsabs.harvard.edu/abs/2001ApJ...555..504W
    - gamma is the inner logarithmic slope (gamma = -1 for NFW)
    - rbreak is the break radius in arcsec

    """
    # The fit is performed in log spaced radii from 0.01" to 10*rbreak
    n = 1000#300     # Number of values to sample the gNFW profile for the MGE fit
    r = np.geomspace(0.01, rbreak, n)   # logarithmically spaced radii in arcsec
    rho = (r/rbreak)**gamma * (0.5 + 0.5*r/rbreak)**(-gamma - 3)  # rho=1 at r=rbreak
    ngauss=30#15
    m = mge_fit_1d(r, rho, ngauss=ngauss, inner_slope=2, outer_slope=3, quiet=1, plot=1)

    surf_dm, sigma_dm = m.sol           # Peak surface density and sigma
    #surf_dm /= np.sqrt(2*np.pi*sigma_dm**2)
    qobs_dm = np.ones_like(surf_dm)     # Assume spherical dark halo

    return surf_dm, sigma_dm, qobs_dm

###############################################################################

def jam_resid(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, 
              distance=None, align=None, goodbins=None,
              xbin=None, ybin=None, sigmapsf=None, normpsf=None, 
              rms=None, erms=None, pixsize=None, plot=True, return_mge=False):

    q, ratio, gamma = pars

    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    beta = np.full_like(qobs_lum, 1 - ratio**2)   # assume constant anisotropy

    rbreak = 50 # arcsec 20*reff
    mbh = 0                     # Ignore the central black hole
    surf_pot, sigma_pot, qobs_pot = total_mass_mge_cap(gamma, rbreak)

    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=None)
    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = np.sum(resid**2)
    
    if return_mge==True:
        
        # return the mass mge multiplied by the ml from jam
        return surf_pot*jam.ml, sigma_pot, qobs_pot, chi2
    
    else:
        return resid   # ln(likelihood) + cost.



# In[7]:


def calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius, plot=False):
    
    # circularize mass
    pc = distance*np.pi/0.648 # constant factor arcsec -> pc
    sigma = sigma_pot*np.sqrt(qobs_pot) # circularize while conserving mass
    mass = 2*np.pi*surf_pot*(sigma*pc)**2 # keep peak surface brightnesss of each gaussian
    mass_tot = np.sum(mass)
    
    # radii to sample
    nrad = 50
    rmin = 0.01 #np.min(sigma)
    rad = np.geomspace(rmin, radius, nrad) # arcsec
    
    # enclosed mass is integral
    mass_enclosed = (mass*(1 - np.exp(-0.5*(rad[:, None]/sigma)**2))).sum(1)[-1]
    
    # slope is average over interval
    mass_profile_1d = np.zeros_like(rad, dtype=float)
    for i in range(len(surf_pot)):
        amp = surf_pot[i]
        sigma = sigma_pot[i]
        q = qobs_pot[i]
        zz = calculate_1d_gaussian(rad, amp, sigma)
        mass_profile_1d += zz
    # diff in log space minus 1 (because we're in surface mass density    
    gamma_avg = (np.log(mass_profile_1d[-1])-np.log(mass_profile_1d[0]))/(np.log(rad[-1])-np.log(rad[0]))-1
    
    # add up the mass profile
    r = np.geomspace(np.min(sigma_pot), np.max(sigma_pot), nrad) # arcsec
    # slope is average over interval
    mass_profile_1d = np.zeros_like(r, dtype=float)
    for i in range(len(surf_pot)):
        amp = surf_pot[i]
        sigma = sigma_pot[i]
        q = qobs_pot[i]
        zz = calculate_1d_gaussian(r, amp, sigma)
        mass_profile_1d += zz
        
    if plot==True:
        plt.loglog(r, mass_profile_1d)
        plt.pause(1)
        
    return mass_enclosed, gamma_avg, r, mass_profile_1d

###############################################################################################################################
###############################################################################################################################

# # PL, constant anisotropy

# In[31]:


# anis_ratio, gamma_fit, mass_enclosed, gamma

# axisymmetric sph
pl_const_axisph_solutions = np.ones((len(obj_names), 8))
pl_const_axisph_cov = np.ones((len(obj_names), 3, 3))
pl_const_axisph_mass_profiles_r = np.ones((len(obj_names), 50))
pl_const_axisph_mass_profiles = np.ones((len(obj_names), 50))

# axis cyl
pl_const_axicyl_solutions = np.ones((len(obj_names), 8))
pl_const_axicyl_cov = np.ones((len(obj_names), 3, 3))
pl_const_axicyl_mass_profiles_r = np.ones((len(obj_names), 50))
pl_const_axicyl_mass_profiles = np.ones((len(obj_names), 50))


# ## Axisymmetric Cylindrical PL

# In[28]:


for i in range(len(obj_names)):

    obj_name = obj_names[i]
    obj_abbr = obj_name[4:9]
    
    print('######################')
    print()
    print(obj_name)
    print()
    
    SN = 15
    mass_model='power_law'
    anisotropy='const'
    geometry='axi'
    align='cyl'
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    theta_E = lens_models[lens_models['obj_name']==obj_name]['theta_E'].to_numpy()[0]
    cosmo = cosmo
    distance = cosmo.angular_diameter_distance(zlens).to_numpy()[0]
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    
    if obj_abbr=='J0330':
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
        
    surf_lum, sigma_lum, qobs_lum, wrong_psf, Vrms, dVrms, V_bin, dV_bin, xbin, ybin, reff = prepare_to_jam(obj_name, file_dir, SN)

    goodbins = np.isfinite(Vrms/dVrms)
    
    # cut out the bad bins
    Vrms = Vrms[goodbins]
    dVrms = dVrms[goodbins]
    xbin = xbin[goodbins]
    ybin = ybin[goodbins]
    
    # Starting guesses
    q0 = np.median(qobs_lum)
    ratio0 = 1          # Anisotropy ratio sigma_z/sigma_R
    gamma0 = -2         # Total mass logarithmic slope rho = r^gamma
    normpsf= 1
    
    qmin = np.min(qobs_lum)
    p0 = [q0, ratio0, gamma0]
    bounds = [[0.051, 0.01, -2.6], [qmin, 1, -1.4]]
    goodbins = np.isfinite(xbin)  # Here I fit all bins

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum, 'qobs_lum': qobs_lum,
              'distance': distance, 'xbin': xbin, 'ybin': ybin, 'sigmapsf': sigmapsf,
              'normpsf': normpsf, 'rms': Vrms, 'erms': dVrms, 'pixsize': kcwi_scale,#pixsize,
              'goodbins': goodbins, 'align':align, 'plot': 0, 'return_mge': False}

    sol = capfit(jam_resid, p0, kwargs=kwargs, bounds=bounds, verbose=2, fixed=[0, 0, 0])
    if sol.success==True:
        print()
        print('Success!')
        print()
    else:
        print()
        print('Fit failed to meet convergence criteria.')
        print()
        
    # bestfit paramters
    q = sol.x[0]
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    anis_ratio = sol.x[1]
    gamma_fit = sol.x[2]
    print('Best fit,', anis_ratio, gamma_fit)
    
    # covariance matrix of free parameters
    cov = sol.cov
    # qerr doesn't matter but is the first index
    anis_ratio_err = np.sqrt(sol.cov.diagonal()[1])
    gamma_err = np.sqrt(sol.cov.diagonal()[2])
    
    # plot the best fit
    kwargs['plot'] = 1
    kwargs['return_mge'] = True
    surf_pot, sigma_pot, qobs_pot, chi2 = jam_resid(sol.x, **kwargs)
    plt.savefig(f'{jam_output_dir}{obj_name}_pl_const_axicyl_fit.png')
    plt.pause(1)
    plt.clf()
    
    # take chi2/dof
    #dof = len(Vrms)
    #chi2_red = chi2/dof
    
    # calculate the enclosed mass
    mass_enclosed, gamma_avg, profile_radii, profile_1d \
                = calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius=theta_E, plot=True)
    
    print('Mass enclosed, avg gamma', mass_enclosed, gamma_avg)
    
    pl_const_axicyl_solutions[i] = np.array([inc, 
                                             anis_ratio, anis_ratio_err, 
                                             gamma_fit, gamma_err, 
                                             gamma_avg, mass_enclosed, 
                                             chi2])
    pl_const_axicyl_cov[i] = cov
    pl_const_axicyl_mass_profiles_r[i] = profile_radii
    pl_const_axicyl_mass_profiles[i] = profile_1d


# In[29]:


# save with header
sol_header = '#### 14 objects - inc, anis_ratio, anis_ratio_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2'
np.savetxt(f'{jam_output_dir}jam_pl_const_axicyl_solutions_2024_02_09.txt', pl_const_axicyl_solutions, header=sol_header) 
np.savetxt(f'{jam_output_dir}jam_pl_const_axicyl_covariances_2024_02_09.txt', pl_const_axicyl_cov.reshape(14,9), header='covariance matrix has been reshaped to 2D, reshape back to (14,3,3)') 

profile_header = '#### radius, surface mass density (Msol/pc2), has been reshaped to 2D, reshape back to (14,50,2)'
pl_const_axicyl_mass_profile_joined = np.dstack((pl_const_axicyl_mass_profiles_r, pl_const_axicyl_mass_profiles)).reshape(14, 100)
np.savetxt(f'{jam_output_dir}jam_pl_const_axicyl_profiles_2024_02_09.txt', pl_const_axicyl_mass_profile_joined, header=profile_header) 


###############################################################################################################################
###############################################################################################################################

# ## Axisymmetric Spherical PL

# In[ ]:


for i in range(len(obj_names)):

    obj_name = obj_names[i]
    obj_abbr = obj_name[4:9]
    
    print('######################')
    print()
    print(obj_name)
    print()
    
    # Model
    SN = 15
    mass_model='power_law'
    anisotropy='const'
    geometry='axi'
    align='sph'
    
    # object attributes
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    theta_E = lens_models[lens_models['obj_name']==obj_name]['theta_E'].to_numpy()[0]
    cosmo = cosmo
    distance = cosmo.angular_diameter_distance(zlens).to_numpy()[0]
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    
    if obj_abbr=='J0330':
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
        
    surf_lum, sigma_lum, qobs_lum, wrong_psf, Vrms, dVrms, V_bin, dV_bin, xbin, ybin, reff = prepare_to_jam(obj_name, file_dir, SN)
    
    goodbins = np.isfinite(Vrms/dVrms)
    
    # cut out the bad bins
    Vrms = Vrms[goodbins]
    dVrms = dVrms[goodbins]
    xbin = xbin[goodbins]
    ybin = ybin[goodbins]
    
    # Starting guesses
    q0 = np.median(qobs_lum)
    ratio0 = 1          # Anisotropy ratio sigma_z/sigma_R
    gamma0 = -2         # Total mass logarithmic slope rho = r^gamma
    normpsf= 1
    
    qmin = np.min(qobs_lum)
    p0 = [q0, ratio0, gamma0]
    bounds = [[0.051, 0.01, -2.6], [qmin, 2.0, -1.4]]
    goodbins = np.isfinite(xbin)  # Here I fit all bins

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum, 'qobs_lum': qobs_lum,
              'distance': distance, 'xbin': xbin, 'ybin': ybin, 'sigmapsf': sigmapsf,
              'normpsf': normpsf, 'rms': Vrms, 'erms': dVrms, 'pixsize': kcwi_scale,#pixsize,
              'goodbins': goodbins, 'align':align, 'plot': 0, 'return_mge': False}

    sol = capfit(jam_resid, p0, kwargs=kwargs, bounds=bounds, verbose=2, fixed=[0, 0, 0])
    if sol.success==True:
        print()
        print('Success!')
        print()
    else:
        print()
        print('Fit failed to meet convergence criteria.')
        print()
        
    # bestfit paramters
    q = sol.x[0]
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    anis_ratio = sol.x[1]
    gamma_fit = sol.x[2]
    print('Best fit,', anis_ratio, gamma_fit)
    
    # covariance matrix of free parameters
    cov = sol.cov
    # qerr doesn't matter but is the first index
    anis_ratio_err = np.sqrt(sol.cov.diagonal()[1])
    gamma_err = np.sqrt(sol.cov.diagonal()[2])
    
    
    # plot the best fit
    kwargs['plot'] = 1
    kwargs['return_mge'] = True
    surf_pot, sigma_pot, qobs_pot, chi2 = jam_resid(sol.x, **kwargs)
    plt.savefig(f'{jam_output_dir}{obj_name}_pl_const_axisph_fit.png')
    plt.pause(1)
    
    # take chi2/dof
    #dof = len(Vrms)
    #chi2_red = chi2/dof
    
    # calculate the enclosed mass
    mass_enclosed, gamma_avg, profile_radii, profile_1d \
                = calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius=theta_E, plot=True)
    
    print('Mass enclosed, avg gamma', mass_enclosed, gamma_avg)
    
    pl_const_axisph_solutions[i] = np.array([inc, 
                                             anis_ratio, anis_ratio_err, 
                                             gamma_fit, gamma_err, 
                                             gamma_avg, mass_enclosed, 
                                             chi2])
    pl_const_axisph_cov[i] = cov
    pl_const_axisph_mass_profiles_r[i] = profile_radii
    pl_const_axisph_mass_profiles[i] = profile_1d


# In[31]:


# save with header
sol_header = '#### 14 objects - inc, anis_ratio, anis_ratio_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2'
np.savetxt(f'{jam_output_dir}jam_pl_const_axisph_solutions_2024_02_09.txt', pl_const_axisph_solutions, header=sol_header) 
np.savetxt(f'{jam_output_dir}jam_pl_const_axisph_covariances_2024_02_09.txt', pl_const_axisph_cov.reshape(14,9), header='covariance matrix has been reshaped to 2D, reshape back to (14,3,3)') 

pl_const_axisph_mass_profile_joined = np.dstack((pl_const_axisph_mass_profiles_r, pl_const_axisph_mass_profiles)).reshape(14, 100)
np.savetxt(f'{jam_output_dir}jam_pl_const_axisph_profiles_2024_02_09.txt', pl_const_axisph_mass_profile_joined, header=profile_header) 


###############################################################################################################################
###############################################################################################################################


# ## Spherical Geometry PL

# In[32]:


# spherical models require my 1D kinematics
# as in e.g. data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/milestone23_data/SDSSJ0029-0055/SDSSJ0029-0055_kinmap.csv
# 02/09/2024 WRONG as of now. Use the kinematic map. Convert bin x and y to r

# In[8]:


###############################################################################

def jam_resid_sph(pars, surf_lum=None, sigma_lum=None,
                  distance=None, rbin=None,
                   sigmapsf=None, normpsf=None, 
                  rms=None, erms=None, pixsize=None, plot=True, return_mge=False):

    ratio, gamma = pars

    beta = np.full_like(surf_lum, 1 - ratio**2)   # assume constant anisotropy

    #rbreak = 20e3              # Adopt fixed halo break radius of 20 kpc (much larger than the data)
    #pc = distance*np.pi/0.648   # Constant factor to convert arcsec --> pc
    #rbreak /= pc
    # Convert the break radius from pc --> arcsec
    rbreak = 50 #20*reff
    mbh = 0                     # Ignore the central black hole
    surf_pot, sigma_pot, qobs_pot = total_mass_mge_cap(gamma, rbreak)

    jam = jam_sph_proj(surf_lum, sigma_lum, surf_pot, sigma_pot,
                       mbh, distance, rbin,
                       sigmapsf=sigmapsf, normpsf=normpsf,
                      beta=beta, data=rms, errors=erms, 
                       plot=plot, pixsize=pixsize, quiet=1, ml=None)
    
    resid = (rms - jam.model)/erms
    chi2 = np.sum(resid**2)
    
    if return_mge==True:
        
        # return the mass mge multiplied by the ml from jam
        return surf_pot*jam.ml, sigma_pot, qobs_pot, chi2
    
    else:
        return resid   # ln(likelihood) + cost.


# In[9]:


pl_const_sphsph_solutions = np.ones((len(obj_names), 8))
pl_const_sphsph_cov = np.ones((len(obj_names), 2, 2))
pl_const_sphsph_mass_profiles_r = np.ones((len(obj_names), 50))
pl_const_sphsph_mass_profiles = np.ones((len(obj_names), 50))


# In[13]:


for i in range(len(obj_names)):

    obj_name = obj_names[i]
    obj_abbr = obj_name[4:9]
    
    print('######################')
    print()
    print(obj_name)
    print()
    
    SN = 15
    mass_model='power_law'
    anisotropy='const'
    geometry='sph'
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    theta_E = lens_models[lens_models['obj_name']==obj_name]['theta_E'].to_numpy()[0]
    cosmo = cosmo
    distance = cosmo.angular_diameter_distance(zlens).to_numpy()[0]
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    
    # load in the 1D kinematics
    #Vrms_1d = pd.read_csv(f'{milestone_dir}{obj_name}/{obj_name}_kinmap.csv')
    #Vrms = Vrms_1d['binned_Vrms'].to_numpy()
    #rbin = np.mean(Vrms_1d[['bin_inner_edge','bin_outer_edge']].to_numpy(), axis=1)
    #Vrms_cov = np.loadtxt(f'{milestone_dir}{obj_name}/{obj_name}_kinmap_cov.csv', delimiter=',', dtype=float)
    #dVrms = np.sqrt(Vrms_cov.diagonal())
    
    if obj_abbr=='J0330':
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
        
    #surf_lum, sigma_lum, _, _, _, _, _, _, _, _, reff = prepare_to_jam(obj_name, file_dir, SN)
    
    # 02/09/24 - We're still doing this in 2D, just take the bin x and y to get r
    surf_lum, sigma_lum, qobs_lum, wrong_psf, Vrms, dVrms, V_bin, dV_bin, xbin, ybin, reff = prepare_to_jam(obj_name, file_dir, SN)
    
    goodbins = np.isfinite(Vrms/dVrms)
    
    # cut out the bad bins
    Vrms = Vrms[goodbins]
    dVrms = dVrms[goodbins]
    xbin = xbin[goodbins]
    ybin = ybin[goodbins]
    
    # get radius of bin centers
    rbin = np.sqrt(xbin**2 + ybin**2)
        
    # sort by bin radius
    sort = np.argsort(rbin)
    rbin = rbin[sort]
    Vrms = Vrms[sort]
    dVrms = dVrms[sort]
    
    # Starting guesses
    ratio0 = 1          # Anisotropy ratio sigma_z/sigma_R
    gamma0 = -2         # Total mass logarithmic slope rho = r^gamma
    
    normpsf= 1  
    p0 = [ratio0, gamma0]
    bounds = [[0.01, -2.6], [2.0, -1.4]]

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum,
              'distance': distance, 'rbin':rbin, 
              'sigmapsf': sigmapsf, 'normpsf': normpsf, 
              'rms': Vrms, 'erms': dVrms, 
              'pixsize': kcwi_scale,#pixsize,
              'plot': 0, 'return_mge': False}

    sol = capfit(jam_resid_sph, p0, kwargs=kwargs, bounds=bounds, verbose=2, fixed=[0, 0])
    if sol.success==True:
        print()
        print('Success!')
        print()
    else:
        print()
        print('Fit failed to meet convergence criteria.')
        print()
    
    # bestfit paramters
    anis_ratio = sol.x[0]
    gamma_fit = sol.x[1]
    print('Best fit,', anis_ratio, gamma_fit)
    
    # covariance matrix of free parameters
    cov = sol.cov
    anis_ratio_err = np.sqrt(sol.cov.diagonal()[0])
    gamma_err = np.sqrt(sol.cov.diagonal()[1])
    
    # plot the bestfit
    kwargs['plot'] = 1
    kwargs['return_mge'] = True
    surf_pot, sigma_pot, qobs_pot, chi2 = jam_resid_sph(sol.x, **kwargs)
    plt.savefig(f'{jam_output_dir}{obj_name}_pl_const_sphsph_fit.png')
    plt.pause(1)
    
    # take chi2/dof
    #dof = len(Vrms)
    #chi2_red = chi2/dof
    
    # calculate the enclosed mass
    mass_enclosed, gamma_avg, profile_radii, profile_1d = calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius=theta_E, plot=True)
    
    print('Mass enclosed, avg gamma', mass_enclosed, gamma_avg)
    
    inc=np.nan
    
    pl_const_sphsph_solutions[i] = np.array([inc, anis_ratio, anis_ratio_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2])
    pl_const_sphsph_cov[i] = cov
    pl_const_sphsph_mass_profiles_r[i] = profile_radii
    pl_const_sphsph_mass_profiles[i] = profile_1d


# In[16]:


# save with header
sol_header = '#### 14 objects - inc, anis_ratio, anis_ratio_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2'
np.savetxt(f'{jam_output_dir}jam_pl_const_sphsph_solutions_2024_02_09.txt', pl_const_sphsph_solutions, header=sol_header) 
np.savetxt(f'{jam_output_dir}jam_pl_const_sphsph_covariances_2024_02_09.txt', pl_const_sphsph_cov.reshape(14,4), header='covariance matrix has been reshaped to 2D, reshape back to (14,3,3)') 

pl_const_sphsph_mass_profile_joined = np.dstack((pl_const_sphsph_mass_profiles_r, pl_const_sphsph_mass_profiles)).reshape(14, 100)
np.savetxt(f'{jam_output_dir}jam_pl_const_sphsph_profiles_2024_02_09.txt', pl_const_sphsph_mass_profile_joined, header=profile_header) 


# In[ ]:

###############################################################################################################################
###############################################################################################################################


# # PL, Osipkov-Merrit

# In[23]:


###############################################################################

def jam_resid_sph_OM(pars, surf_lum=None, sigma_lum=None,
                      distance=None, rbin=None,
                       sigmapsf=None, normpsf=None, 
                      rms=None, erms=None, pixsize=None, plot=True, return_mge=False):
    
    # OM uses anisotropy radius instead of the ratio
    a_ani, gamma = pars
    rani = reff * a_ani

    rbreak = 50 #20*reff
    mbh = 0                     # Ignore the central black hole
    surf_pot, sigma_pot, qobs_pot = total_mass_mge_cap(gamma, rbreak)

    jam = jam_sph_proj(surf_lum, sigma_lum, surf_pot, sigma_pot,
                       mbh, distance, rbin,
                       sigmapsf=sigmapsf, normpsf=normpsf,
                       rani=rani, data=rms, errors=erms, 
                       plot=plot, pixsize=pixsize, quiet=1, ml=None)
    
    resid = (rms - jam.model)/erms
    chi2 = np.sum(resid**2)
    
    if return_mge==True:
        
        # return the mass mge multiplied by the ml from jam
        return surf_pot*jam.ml, sigma_pot, qobs_pot, chi2
    
    else:
        return resid   # ln(likelihood) + cost.


# In[24]:


###############################################################################

def jam_resid_axi_OM(pars, surf_lum=None, sigma_lum=None, qobs_lum=None, 
                      distance=None, align=None, goodbins=None,
                      xbin=None, ybin=None, sigmapsf=None, normpsf=None, 
                      rms=None, erms=None, pixsize=None, plot=True, return_mge=False):
    
    #### OM should only be done with spherical alignment
    
    q, a_ani, gamma = pars
    rani = a_ani * reff
    
    # Sample inclination using min(q), with Equation (14) of Cappellari (2008)
    qmin = np.min(qobs_lum)
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    beta= [rani, 0, 1, 2]

    rbreak = 50 #20*reff
    mbh = 0                     # Ignore the central black hole
    surf_pot, sigma_pot, qobs_pot = total_mass_mge_cap(gamma, rbreak)

    jam = jam_axi_proj(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                       inc, mbh, distance, xbin, ybin, plot=plot, pixsize=pixsize, quiet=1,
                       sigmapsf=sigmapsf, normpsf=normpsf, goodbins=goodbins, align=align,
                       beta=beta, data=rms, errors=erms, ml=None, logistic=True)
    resid = (rms[goodbins] - jam.model[goodbins])/erms[goodbins]
    chi2 = np.sum(resid**2)
    
    if return_mge==True:
        
        # return the mass mge multiplied by the ml from jam
        return surf_pot*jam.ml, sigma_pot, qobs_pot, chi2
    
    else:
        return resid   # ln(likelihood) + cost.


# In[25]:


# will need the fit parameters, average slopes, enclosed masses, and uncertainties on fit parameters

# np.array([inc, a_ani, a_ani_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2_red])
pl_om_sphsph_solutions = np.ones((len(obj_names), 8))
pl_om_sphsph_cov = np.ones((len(obj_names), 2, 2))
pl_om_sphsph_mass_profiles_r = np.ones((len(obj_names), 50))
pl_om_sphsph_mass_profiles = np.ones((len(obj_names), 50))

# np.array([inc, a_ani, a_ani_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2_red])
pl_om_axisph_solutions = np.ones((len(obj_names), 8))
pl_om_axisph_cov = np.ones((len(obj_names), 3, 3))
pl_om_axisph_mass_profiles_r = np.ones((len(obj_names), 50))
pl_om_axisph_mass_profiles = np.ones((len(obj_names), 50))


# ## OM PL spherical geometry

# In[27]:


for i in range(len(obj_names)):

    obj_name = obj_names[i]
    obj_abbr = obj_name[4:9]
    
    print('######################')
    print()
    print(obj_name)
    print()
    
    SN = 15
    mass_model='power_law'
    anisotropy='om'
    geometry='sph'
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    theta_E = lens_models[lens_models['obj_name']==obj_name]['theta_E'].to_numpy()[0]
    cosmo = cosmo
    distance = cosmo.angular_diameter_distance(zlens).to_numpy()[0]
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    
    if obj_abbr=='J0330':
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
        
    #surf_lum, sigma_lum, _, _, _, _, _, _, _, _, reff = prepare_to_jam(obj_name, file_dir, SN)
    # 02/09/24 - We're still doing this in 2D, just take the bin x and y to get r
    surf_lum, sigma_lum, qobs_lum, wrong_psf, Vrms, dVrms, V_bin, dV_bin, xbin, ybin, reff = prepare_to_jam(obj_name, file_dir, SN)
    
    goodbins = np.isfinite(Vrms/dVrms)
    
    # cut out the bad bins
    Vrms = Vrms[goodbins]
    dVrms = dVrms[goodbins]
    xbin = xbin[goodbins]
    ybin = ybin[goodbins]
    
    # get radius of bin centers
    rbin = np.sqrt(xbin**2 + ybin**2)
    
    # sort by bin radius
    sort = np.argsort(rbin)
    rbin = rbin[sort]
    Vrms = Vrms[sort]
    dVrms = dVrms[sort]
    
    # Starting guesses
    ani0 = 0.5          # Anisotropy ratio sigma_z/sigma_R
    gamma0 = -2         # Total mass logarithmic slope rho = r^gamma
    
    normpsf= 1  
    p0 = [ani0, gamma0]
    bounds = [[0.01, -2.6], [2.0, -1.4]]

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum,
              'distance': distance, 'rbin':rbin, 
              'sigmapsf': sigmapsf, 'normpsf': normpsf, 
              'rms': Vrms, 'erms': dVrms, 
              'pixsize': kcwi_scale,#pixsize,
              'plot': 0, 'return_mge': False}
    
    # do the fit, take the solutions and covariance matrix
    sol = capfit(jam_resid_sph_OM, p0, kwargs=kwargs, bounds=bounds, verbose=2, fixed=[0, 0])
    if sol.success==True:
        print()
        print('Success!')
        print()
    else:
        print()
        print('Fit failed to meet convergence criteria.')
        print()
    # bestfit paramters
    a_ani = sol.x[0]
    gamma_fit = sol.x[1]
    # covariance matrix of free parameters
    cov = sol.cov
    a_ani_err = np.sqrt(sol.cov.diagonal()[0])
    gamma_err = np.sqrt(sol.cov.diagonal()[1])                    
    
    print('Best fit,', anis_ratio, gamma_fit)
    
    kwargs['plot'] = 1
    kwargs['return_mge'] = True
    surf_pot, sigma_pot, qobs_pot, chi2 = jam_resid_sph_OM(sol.x, **kwargs)
    plt.savefig(f'{jam_output_dir}{obj_name}_pl_om_sphsph_fit.png')
    plt.pause(1)
    
    # take chi2/dof
    #dof = len(Vrms)
    #chi2_red = chi2/dof
    
    # calculate the enclosed mass
    mass_enclosed, gamma_avg, profile_radii, profile_1d = calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius=theta_E, plot=True)
    
    print('Mass enclosed, avg gamma', mass_enclosed, gamma_avg)
    
    inc=np.nan
    
    pl_om_sphsph_solutions[i] = np.array([inc, a_ani, a_ani_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2])
    pl_om_sphsph_cov[i] = cov
    pl_om_sphsph_mass_profiles_r[i] = profile_radii
    pl_om_sphsph_mass_profiles[i] = profile_1d


# In[29]:


# save with header
sol_header = '#### 14 objects - inc, anis_radius, anis_radius_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2'
np.savetxt(f'{jam_output_dir}jam_pl_om_sphsph_solutions_2024_02_09.txt', pl_om_sphsph_solutions, header=sol_header) 
np.savetxt(f'{jam_output_dir}jam_pl_om_sphsph_covariances_2024_02_09.txt', pl_om_sphsph_cov.reshape(14,4), header='covariance matrix has been reshaped to 2D, reshape back to (14,2,2)') 
 
pl_om_sphsph_mass_profile_joined = np.dstack((pl_om_sphsph_mass_profiles_r, pl_om_sphsph_mass_profiles)).reshape(14, 100)
np.savetxt(f'{jam_output_dir}jam_pl_om_sphsph_profiles_2024_02_09.txt', pl_om_sphsph_mass_profile_joined, header=profile_header) 


# ## OM PL axisymmetric geometry

# In[66]:


for i in range(len(obj_names)):

    obj_name = obj_names[i]
    obj_abbr = obj_name[4:9]
    
    print('######################')
    print()
    print(obj_name)
    print()
    
    SN = 15
    mass_model='power_law'
    anisotropy='om'
    geometry='axi'
    align='sph'
    zlens= zlenses[slacs_ix_table['Name']==obj_name]
    zsource = zsources[slacs_ix_table['Name']==obj_name]
    sigmapsf = sigmapsf_table[sigmapsf_table['obj_name']==obj_name]['kcwi_sigmapsf'].to_numpy()[0]
    theta_E = lens_models[lens_models['obj_name']==obj_name]['theta_E'].to_numpy()[0]
    cosmo = cosmo
    distance = cosmo.angular_diameter_distance(zlens).to_numpy()[0]
    fast_slow = paper_table.loc[0, 'class_for_JAM_models']
    
    if obj_abbr=='J0330':
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_final_kinematics/no_g/'
    else:
        file_dir = f'{kinematics_dir}{obj_name}/target_sn_{SN}/{obj_name}_{SN}_marginalized_gnog_final_kinematics/'
        
    surf_lum, sigma_lum, qobs_lum, wrong_psf, Vrms, dVrms, V_bin, dV_bin, xbin, ybin, reff = prepare_to_jam(obj_name, file_dir, SN)
    
    goodbins = np.isfinite(Vrms/dVrms)
    
    # cut out the bad bins
    Vrms = Vrms[goodbins]
    dVrms = dVrms[goodbins]
    xbin = xbin[goodbins]
    ybin = ybin[goodbins]
    
    # Starting guesses
    q0 = np.median(qobs_lum)
    ani0 = 0.5          # Anisotropy ratio sigma_z/sigma_R
    gamma0 = -2         # Total mass logarithmic slope rho = r^gamma
    normpsf= 1
    
    
    qmin = np.min(qobs_lum)
    p0 = [q0, ani0, gamma0]
    bounds = [[0.051, 0.01, -2.6], [qmin, 2.0, -1.4]]
    goodbins = np.isfinite(xbin)  # Here I fit all bins

    # These parameters are passed to JAM
    kwargs = {'surf_lum': surf_lum, 'sigma_lum': sigma_lum, 'qobs_lum': qobs_lum,
              'distance': distance, 'xbin': xbin, 'ybin': ybin, 'sigmapsf': sigmapsf,
              'normpsf': normpsf, 'rms': Vrms, 'erms': dVrms, 'pixsize': kcwi_scale,#pixsize,
              'goodbins': goodbins, 'align':align, 'plot': 0, 'return_mge': False}

    sol = capfit(jam_resid_axi_OM, p0, kwargs=kwargs, bounds=bounds, verbose=2, fixed=[0, 0, 0])
    # bestfit paramters
    q = sol.x[0]
    inc = np.degrees(np.arctan2(np.sqrt(1 - qmin**2), np.sqrt(qmin**2 - q**2)))
    a_ani = sol.x[1]
    gamma_fit = sol.x[2]
    # covariance matrix of free parameters
    cov = sol.cov
    q_err = np.sqrt(sol.cov.diagonal()[0])
    a_ani_err = np.sqrt(sol.cov.diagonal()[1])
    gamma_err = np.sqrt(sol.cov.diagonal()[2])   
    
    print('Best fit,', anis_ratio, gamma_fit)
    
    kwargs['plot'] = 1
    kwargs['return_mge'] = True
    surf_pot, sigma_pot, qobs_pot, chi2 = jam_resid_axi_OM(sol.x, **kwargs)
    plt.savefig(f'{jam_output_dir}{obj_name}_pl_om_axisph_fit.png')
    plt.pause(1)
    
    # take chi2/dof
    #dof = len(Vrms)
    #chi2_red = chi2/dof
    
    # calculate the enclosed mass
    mass_enclosed, gamma_avg, profile_radii, profile_1d = calculate_mass_and_slope_enclosed(surf_pot, sigma_pot, qobs_pot, distance, radius=theta_E, plot=True)
    
    print('Mass enclosed, avg gamma', mass_enclosed, gamma_avg)
    
    pl_om_axisph_solutions[i] = np.array([inc, a_ani, a_ani_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2])
    pl_om_axisph_cov[i] = cov
    pl_om_axisph_mass_profiles_r[i] = profile_radii
    pl_om_axisph_mass_profiles[i] = profile_1d


# In[67]:


# save with header
sol_header = '#### 14 objects - inc, anis_radius, anis_radius_err, gamma_fit, gamma_err, gamma_avg, mass_enclosed, chi2'
np.savetxt(f'{jam_output_dir}jam_pl_om_axisph_solutions_2024_02_09.txt', pl_om_axisph_solutions, header=sol_header) 
np.savetxt(f'{jam_output_dir}jam_pl_om_axisph_covariances_2024_02_09.txt', pl_om_axisph_cov.reshape(14,9), header='covariance matrix has been reshaped to 2D, reshape back to (14,2,2)') 
 
pl_om_axisph_mass_profile_joined = np.dstack((pl_om_axisph_mass_profiles_r, pl_om_axisph_mass_profiles)).reshape(14, 100)
np.savetxt(f'{jam_output_dir}jam_pl_om_axisph_profiles_2024_02_09.txt', pl_om_axisph_mass_profile_joined, header=profile_header) 


###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

print('###############################################################################################################################')
print('###############################################################################################################################')
print()
print("Job's finished.")
print()