# visualize the 2d kinematics after having run systematics and collective covariance matrices

# 
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({"figure.figsize" : (8, 6)})
from scipy.optimize import curve_fit
import pathlib # to create directory
from datetime import date
today = date.today().strftime('%d%m%y')
from time import perf_counter as timer
# register first tick
tick = timer()
from ppxf.kcwi_util import register_sauron_colormap
register_sauron_colormap()

#################################################
# date and number of initial kinematics run e.g. 2023-02-28_2
date_of_kin = '2023-02-28_2'

# command line arguments to select obj_names to be used
import sys
obj_index = np.array(sys.argv[1:], dtype=int)

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

# select the active object names to run from command line
active_objects = []
for i in obj_index:
    active_objects.append(obj_names[i])

obj_names = active_objects
    
print(f'Active objects are {obj_names}')
print()

#################################################

def kinematics_map_systematics(dir, obj_name, model_name, vorbin_output, mask, radius_in_pixels):
    '''
    this code modifies CF's kinematics_map function to remap the kinematics measurements above into 2D array from my systematics scripts.
    :return: 2D velocity dispersion, uncertainty of the velocity dispersion, velocity, and the uncertainty of the velocity
    # also Vrms
    '''
    # VD
    VD=np.genfromtxt(f'{dir}{model_name}_VD_binned.txt',
                 delimiter=',')
    VD_covariance = np.genfromtxt(f'{dir}{model_name}_VD_covariance_matrix.txt',
                                 delimiter=',')
    dVD = np.sqrt(np.diagonal(VD_covariance)) 
    print(VD_covariance[-4:])
    print('cov matrix len, ', len(VD_covariance))
    print('dVD len, ', len(dVD))

    # diagonal is sigma**2
    # V
    V=np.genfromtxt(f'{dir}{model_name}_V_binned.txt',
                 delimiter=',')
    V_covariance = np.genfromtxt(f'{dir}{model_name}_V_covariance_matrix.txt',
                                 delimiter=',')
    dV = np.sqrt(np.diagonal(V_covariance)) # diagonal is sigma**2
    
    # Vrms
    Vrms=np.genfromtxt(f'{dir}{model_name}_Vrms_binned.txt',
                 delimiter=',')
    Vrms_covariance = np.genfromtxt(f'{dir}{model_name}_Vrms_covariance_matrix.txt',
                                 delimiter=',')
    dVrms = np.sqrt(np.diagonal(Vrms_covariance)) # diagonal is sigma**2
    
    # apply mask
    VD_m, dVD_m, V_m, dV_m, Vrms_m, dVrms_m, masked_bins = apply_mask (VD, dVD, V, dV, Vrms, dVrms, mask, dir)

    # create new arrays for number of pixels?
    VD_array    =np.zeros(vorbin_output.shape[0])
    dVD_array   =np.zeros(vorbin_output.shape[0])
    V_array     =np.zeros(vorbin_output.shape[0])
    dV_array    =np.zeros(vorbin_output.shape[0])
    Vrms_array     =np.zeros(vorbin_output.shape[0])
    dVrms_array    =np.zeros(vorbin_output.shape[0])
    # masked
    VD_m_array    =np.zeros(vorbin_output.shape[0])
    dVD_m_array   =np.zeros(vorbin_output.shape[0])
    V_m_array     =np.zeros(vorbin_output.shape[0])
    dV_m_array    =np.zeros(vorbin_output.shape[0])
    Vrms_m_array     =np.zeros(vorbin_output.shape[0])
    dVrms_m_array    =np.zeros(vorbin_output.shape[0])
    
    for i in range(vorbin_output.shape[0]):
        num=int(vorbin_output.T[2][i])
        VD_array[i] = VD[num]
        dVD_array[i] = dVD[num]
        V_array[i] = V[num]
        dV_array[i] = dV[num]
        Vrms_array[i] = V[num]
        dVrms_array[i] = dV[num]
        # masked
        VD_m_array[i] = VD_m[num]
        dVD_m_array[i] = dVD_m[num]
        V_m_array[i] = V_m[num]
        dV_m_array[i] = dV_m[num]
        Vrms_m_array[i] = Vrms_m[num]
        dVrms_m_array[i] = dVrms_m[num]
    
    final=np.vstack((vorbin_output.T, VD_array, dVD_array, V_array, dV_array, Vrms_array, dVrms_array, \
                        VD_m_array, dVD_m_array, V_m_array, dV_m_array, Vrms_m_array, dVrms_m_array))
    
    dim = radius_in_pixels*2+1
    
    #########
    # unmasked
    # VD
    VD_2d=np.zeros((dim, dim))
    VD_2d[:]=np.nan
    for i in range(final.shape[1]):
        VD_2d[int(final[1][i])][int(final[0][i])]=final[3][i]
        
    dVD_2d=np.zeros((dim, dim))
    dVD_2d[:]=np.nan
    for i in range(final.shape[1]):
        dVD_2d[int(final[1][i])][int(final[0][i])]=final[4][i]

    # V
    V_2d=np.zeros((dim, dim))
    V_2d[:]=np.nan
    for i in range(final.shape[1]):
        V_2d[int(final[1][i])][int(final[0][i])]=final[5][i]
    dV_2d=np.zeros((dim, dim))
    dV_2d[:]=np.nan
    for i in range(final.shape[1]):
        dV_2d[int(final[1][i])][int(final[0][i])]=final[6][i]
    # Vrms
    Vrms_2d=np.zeros((dim, dim))
    Vrms_2d[:]=np.nan
    for i in range(final.shape[1]):
        Vrms_2d[int(final[1][i])][int(final[0][i])]=final[7][i]
    dVrms_2d=np.zeros((dim, dim))
    dVrms_2d[:]=np.nan
    for i in range(final.shape[1]):
        dVrms_2d[int(final[1][i])][int(final[0][i])]=final[8][i]
        
    #########
    # unmasked
    # VD
    VD_m_2d=np.zeros((dim, dim))
    VD_m_2d[:]=np.nan
    for i in range(final.shape[1]):
        VD_m_2d[int(final[1][i])][int(final[0][i])]=final[9][i]
        
    dVD_m_2d=np.zeros((dim, dim))
    dVD_m_2d[:]=np.nan
    for i in range(final.shape[1]):
        dVD_m_2d[int(final[1][i])][int(final[0][i])]=final[10][i]
    # V
    V_m_2d=np.zeros((dim, dim))
    V_m_2d[:]=np.nan
    for i in range(final.shape[1]):
        V_m_2d[int(final[1][i])][int(final[0][i])]=final[11][i]
    dV_m_2d=np.zeros((dim, dim))
    dV_m_2d[:]=np.nan
    for i in range(final.shape[1]):
        dV_m_2d[int(final[1][i])][int(final[0][i])]=final[12][i]
    # Vrms
    Vrms_m_2d=np.zeros((dim, dim))
    Vrms_m_2d[:]=np.nan
    for i in range(final.shape[1]):
        Vrms_m_2d[int(final[1][i])][int(final[0][i])]=final[13][i]
    dVrms_m_2d=np.zeros((dim, dim))
    dVrms_m_2d[:]=np.nan
    for i in range(final.shape[1]):
        dVrms_m_2d[int(final[1][i])][int(final[0][i])]=final[14][i]
            
    if np.array_equiv(VD_2d, VD_m_2d)==False: # why doesn't this work?
        print('Masking this one... bins: ', masked_bins)
        print()
        # plot the masked bins
        # pixel scale is 0.147 arcsec/pixel, set x and y ticks on the plot to be in arcsec instead of pixels
        pixel_scale = 0.147
        ticks = np.arange(7)
        ticks_pix = ticks/pixel_scale
        ticklabels= np.arange(-3, 4)
        # VD
        plt.figure()
        plt.imshow(VD_2d - VD_m_2d,origin='lower',cmap='sauron')
        cbar1 = plt.colorbar()
        cbar1.set_label(r'$\sigma$ [km/s]')
        plt.title(f"{obj_name} velocity dispersion")
        plt.xticks(ticks_pix, labels=ticklabels)
        plt.yticks(ticks_pix, labels=ticklabels)
        plt.savefig(f'{dir}{model_name}_VD_masked_bins.png')
        #plt.savefig(f'{dir}{model_name}_VD_masked_bins.pdf')
        plt.pause(1)
        plt.clf()    
        # V
        plt.figure()
        plt.imshow(V_2d - V_m_2d,origin='lower',cmap='sauron')
        cbar1 = plt.colorbar()
        cbar1.set_label(r'V [km/s]')
        plt.title(f"{obj_name} velocity")
        plt.xticks(ticks_pix, labels=ticklabels)
        plt.yticks(ticks_pix, labels=ticklabels)
        plt.xlabel('arcsec')
        plt.ylabel('arcsec')
        plt.savefig(f'{dir}{model_name}_V_masked_bins.png')
        #plt.savefig(f'{dir}{model_name}_V_masked_bins.pdf')
        plt.pause(1)
        plt.clf()    
        # Vrms
        plt.figure()
        plt.imshow(Vrms_2d - Vrms_m_2d,origin='lower',cmap='sauron')
        cbar1 = plt.colorbar()
        cbar1.set_label(r'$V_{rms}$ [km/s]')
        plt.title(f"{obj_name} rms velocity")
        plt.xticks(ticks_pix, labels=ticklabels)
        plt.yticks(ticks_pix, labels=ticklabels)
        plt.xlabel('arcsec')
        plt.ylabel('arcsec')
        plt.savefig(f'{dir}{model_name}_Vrms_masked_bins.png')
        # plt.savefig(f'{dir}{model_name}_Vrms_masked_bins.pdf')
        plt.pause(1)
        plt.clf() 
    
    return VD_2d, dVD_2d, V_2d, dV_2d, Vrms_2d, dVrms_2d, \
                VD_m_2d, dVD_m_2d, V_m_2d, dV_m_2d, Vrms_m_2d, dVrms_m_2d

def apply_mask (VD, dVD, V, dV, Vrms, dVrms, mask, directory):
    
    masked_bins = np.argwhere(mask==0)
    
    ''' _m is "masked" '''
    VD_m = VD * mask
    dVD_m = dVD * mask
    V_m = V * mask
    dV_m = dV * mask
    Vrms_m = Vrms * mask
    dVrms_m = dVrms * mask
    
    return VD_m, dVD_m, V_m, dV_m, Vrms_m, dVrms_m, masked_bins

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

###################################################################################################################################

#------------------------------------------------------------------------------
# Directories and files

# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
tables_dir = f'{data_dir}tables/'
mosaics_dir = f'{data_dir}mosaics/'
kinematics_full_dir = f'{data_dir}kinematics/'
kinematics_dir =f'{kinematics_full_dir}{date_of_kin}/'
print(f'Outputs will be in {kinematics_dir}')
print()

#------------------------------------------------------------------------------
# Kinematics systematics choices

# target SN for voronoi binning
vorbin_SN_targets = np.array([10, 15, 20])

# include G band or no
g_bands = ['no_g','g_band']


#########################################################################
### loop through the objects

radius_in_pixels=21

# pixel scale is 0.147 arcsec/pixel, set x and y ticks on the plot to be in arcsec instead of pixels
pixel_scale = 0.147
ticks = np.arange(7)
ticks_pix = ticks/pixel_scale
ticklabels= np.arange(-3, 4)

# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'

for obj_name in obj_names:
    
    print('#####################################################')
    print('#####################################################')
    print()
    print(f'Beginning final kinematics visualization and plotting script for object {obj_name}.')
    print()
    
    # set abbreviation for directories
    obj_abbr = obj_name[4:9] # e.g. J0029
    
    #------------------------------------------------------------------------------
    # object-specific directories
    
    mos_dir = f'{mosaics_dir}{obj_name}/' 
    kin_dir = f'{kinematics_dir}{obj_name}/'
    # create kin_dir if not exists
    #Path(f'{kin_dir}').mkdir(parents=True, exist_ok=True)
    
    #KCWI mosaic datacube
    mos_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'
    
    for vorbin_SN_target in vorbin_SN_targets:
        target_dir = f'{kin_dir}target_sn_{vorbin_SN_target}/'
        # create kin_dir if not exists
        #Path(f'{target_dir}').mkdir(parents=True, exist_ok=True)
        
        print('################################################')
        print('################################################')
        print()
        print('Working with S/N target ', vorbin_SN_target)
        print()
        print('################################################')
        print('################################################')

        '''
        Step 0: input the necessary information of the datacube
        '''

        # point to final kinematics directory for the vorbin sn target
        fin_kin_dir = f'{target_dir}{obj_name}_{vorbin_SN_target}_final_kinematics/'
        #Path(syst_dir).mkdir(parents=True, exist_ok=True) 

        ## import voronoi binning data
        voronoi_binning_data = fits.getdata(target_dir +'voronoi_binning_' + obj_name + '_data.fits')
        # voronoi binning "output"
        vorbin_output = np.loadtxt(target_dir +'voronoi_2d_binning_' + obj_name + '_output.txt') # this comes from a different directory
        N = voronoi_binning_data.shape[0] # for saving the systematics measurements

        '''
        Step 8: systematics tests
        '''

        # iterate over G-band and not G-band
        for g_band in g_bands:
            
            if (obj_abbr=='J0330') & (g_band=='g_band'):
                print('J0330 does not get G band')
                continue

            print()
            print('Working with ', g_band)
            print()

            # point to G-band and no_G directories
            g_dir = f'{fin_kin_dir}{g_band}/'
            #Path(g_dir).mkdir(parents=True, exist_ok=True)
            
            # name for file saving
            model_name = f'{obj_name}_{vorbin_SN_target}_{g_band}'
            
            ##################################################
            # take mask
            mask = np.genfromtxt(f'{g_dir}{model_name}_Vrms_bin_mask.txt')# I accidentally saved them as Vrms, but they should be right

            ##################################################
            # collect the kinematics info from the final kinematics covariance script
            # take the masked versions
            _, _, _, _, _, _, \
                VD_2d, dVD_2d, V_2d, dV_2d, Vrms_2d, dVrms_2d, = kinematics_map_systematics(g_dir, obj_name, model_name, vorbin_output, mask, radius_in_pixels)
            np.savetxt(f'{g_dir}{model_name}_VD_2d.txt', VD_2d, delimiter=',')
            np.savetxt(f'{g_dir}{model_name}_dVD_2d.txt', dVD_2d, delimiter=',')
            np.savetxt(f'{g_dir}{model_name}_V_2d.txt', V_2d, delimiter=',')
            np.savetxt(f'{g_dir}{model_name}_dV_2d.txt', dV_2d, delimiter=',')
            np.savetxt(f'{g_dir}{model_name}_Vrms_2d.txt', Vrms_2d, delimiter=',')
            np.savetxt(f'{g_dir}{model_name}_dVrms_2d.txt', dVrms_2d, delimiter=',')
            
            print()
            print(f'{obj_name} {model_name} has completed')
            print('###########################################################')
        
        ################################################################
        # plot and
        # show the difference between the two
        
        print()
        print('Plotting....')
        print()
        
        # g
        g_VD_2d = np.genfromtxt(f'{fin_kin_dir}g_band/{obj_name}_{vorbin_SN_target}_g_band_VD_2d.txt', delimiter=',')
        g_dVD_2d = np.genfromtxt(f'{fin_kin_dir}g_band/{obj_name}_{vorbin_SN_target}_g_band_dVD_2d.txt', delimiter=',')
        g_V_2d = np.genfromtxt(f'{fin_kin_dir}g_band/{obj_name}_{vorbin_SN_target}_g_band_V_2d.txt', delimiter=',')
        g_dV_2d = np.genfromtxt(f'{fin_kin_dir}g_band/{obj_name}_{vorbin_SN_target}_g_band_dV_2d.txt', delimiter=',')
        g_Vrms_2d = np.genfromtxt(f'{fin_kin_dir}g_band/{obj_name}_{vorbin_SN_target}_g_band_Vrms_2d.txt', delimiter=',')
        g_dVrms_2d = np.genfromtxt(f'{fin_kin_dir}g_band/{obj_name}_{vorbin_SN_target}_g_band_dVrms_2d.txt', delimiter=',')
        # nog
        nog_VD_2d = np.genfromtxt(f'{fin_kin_dir}no_g/{obj_name}_{vorbin_SN_target}_no_g_VD_2d.txt', delimiter=',')
        nog_dVD_2d = np.genfromtxt(f'{fin_kin_dir}no_g/{obj_name}_{vorbin_SN_target}_no_g_dVD_2d.txt', delimiter=',')
        nog_V_2d = np.genfromtxt(f'{fin_kin_dir}no_g/{obj_name}_{vorbin_SN_target}_no_g_V_2d.txt', delimiter=',')
        nog_dV_2d = np.genfromtxt(f'{fin_kin_dir}no_g/{obj_name}_{vorbin_SN_target}_no_g_dV_2d.txt', delimiter=',')
        nog_Vrms_2d = np.genfromtxt(f'{fin_kin_dir}no_g/{obj_name}_{vorbin_SN_target}_no_g_Vrms_2d.txt', delimiter=',')
        nog_dVrms_2d = np.genfromtxt(f'{fin_kin_dir}no_g/{obj_name}_{vorbin_SN_target}_no_g_dVrms_2d.txt', delimiter=',')
        
        # get the velocity ranges so they plot with the same scaling
        # VD
        VD_min = np.nanmin((np.concatenate((np.ravel(g_VD_2d), np.ravel(nog_VD_2d)))))
        dVD_min = np.nanmin((np.concatenate((np.ravel(g_dVD_2d), np.ravel(nog_dVD_2d)))))
        VD_max = np.nanmax((np.concatenate((np.ravel(g_VD_2d), np.ravel(nog_VD_2d)))))
        dVD_max = np.nanmax((np.concatenate((np.ravel(g_dVD_2d), np.ravel(nog_dVD_2d)))))
        # V
        V_min = np.nanmin((np.concatenate((np.ravel(g_V_2d), np.ravel(nog_V_2d)))))
        dV_min = np.nanmin((np.concatenate((np.ravel(g_dV_2d), np.ravel(nog_dV_2d)))))
        V_max = np.nanmax((np.concatenate((np.ravel(g_V_2d), np.ravel(nog_V_2d)))))
        dV_max = np.nanmax((np.concatenate((np.ravel(g_dV_2d), np.ravel(nog_dV_2d)))))
        # Vrms
        Vrms_min = np.nanmin((np.concatenate((np.ravel(g_Vrms_2d), np.ravel(nog_Vrms_2d)))))
        dVrms_min = np.nanmin((np.concatenate((np.ravel(g_dVrms_2d), np.ravel(nog_dVrms_2d)))))
        Vrms_max = np.nanmax((np.concatenate((np.ravel(g_Vrms_2d), np.ravel(nog_Vrms_2d)))))
        dVrms_max = np.nanmax((np.concatenate((np.ravel(g_dVrms_2d), np.ravel(nog_dVrms_2d)))))
        
        ##################################################
        ##################################################
        # plot each (subtraction and putting them on the same scale)
        
        for g_band in g_bands:
            if g_band=='g_band':
                if (obj_abbr=='J0330') & (g_band=='g_band'):
                    print('J0330 does not get G band')
                    continue
                # point to G-band and no_G directories
                g_dir = f'{fin_kin_dir}{g_band}/'
                model_name = f'{obj_name}_{vorbin_SN_target}_{g_band}'
                g_band_name = 'with G-band'
                # kinematics
                VD_2d = g_VD_2d
                dVD_2d = g_dVD_2d
                V_2d = g_V_2d
                dV_2d = g_dV_2d
                Vrms_2d = g_Vrms_2d
                dVrms_2d = g_dVrms_2d 
            else:
                # point to G-band and no_G directories
                g_dir = f'{fin_kin_dir}{g_band}/'
                model_name = f'{obj_name}_{vorbin_SN_target}_{g_band}'
                g_band_name = 'without G-band'
                # kinematics
                VD_2d = nog_VD_2d
                dVD_2d = nog_dVD_2d
                V_2d = nog_V_2d
                dV_2d = nog_dV_2d
                Vrms_2d = nog_Vrms_2d
                dVrms_2d = nog_dVrms_2d 
                
            # velocity dispersion and error
            plt.figure()
            plt.imshow(VD_2d,origin='lower',cmap='sauron', vmin=VD_min, vmax=VD_max)
            cbar1 = plt.colorbar()
            cbar1.set_label(r'$\sigma$ [km/s]')
            plt.title(f"{obj_name} {g_band_name} velocity dispersion")
            plt.xticks(ticks_pix, labels=ticklabels)
            plt.yticks(ticks_pix, labels=ticklabels)
            plt.xlabel('arcsec')
            plt.ylabel('arcsec')
            plt.savefig(f'{g_dir}{model_name}_VD_map.png')
            plt.savefig(f'{g_dir}{model_name}_VD_map.pdf')
            plt.pause(1)
            plt.clf()

            plt.figure()
            plt.imshow(dVD_2d, origin='lower', cmap='sauron', vmin=dVD_min, vmax=dVD_max)
            cbar2 = plt.colorbar()
            cbar2.set_label(r'd$\sigma$ [km/s]')
            plt.title(f"{obj_name} {g_band_name} velocity dispersion error")
            plt.xticks(ticks_pix, labels=ticklabels)
            plt.yticks(ticks_pix, labels=ticklabels)
            plt.xlabel('arcsec')
            plt.ylabel('arcsec')
            plt.savefig(f'{g_dir}{model_name}_dVD_map.png')
            plt.savefig(f'{g_dir}{model_name}_dVD_map.pdf')

            plt.pause(1)
            plt.clf()

            # velocity and error
            #
            plt.figure()
            plt.imshow(V_2d,origin='lower',cmap='sauron', vmin=V_min, vmax=V_max)
            cbar3 = plt.colorbar()
            cbar3.set_label(r'V [km/s]')
            plt.title(f"{obj_name} {g_band_name} velocity")
            plt.xticks(ticks_pix, labels=ticklabels)
            plt.yticks(ticks_pix, labels=ticklabels)
            plt.xlabel('arcsec')
            plt.ylabel('arcsec')
            plt.savefig(f'{g_dir}{model_name}_V_map.png')
            plt.savefig(f'{g_dir}{model_name}_V_map.pdf')
            plt.pause(1)
            plt.clf()

            plt.figure()
            plt.imshow(dV_2d, origin='lower', cmap='sauron', vmin=dV_min, vmax=dV_max)
            cbar2 = plt.colorbar()
            cbar2.set_label(r'dV [km/s]')
            plt.title(f"{obj_name} {g_band_name} velocity error")
            plt.xticks(ticks_pix, labels=ticklabels)
            plt.yticks(ticks_pix, labels=ticklabels)
            plt.xlabel('arcsec')
            plt.ylabel('arcsec')
            plt.savefig(f'{g_dir}{model_name}_dV_map.png')
            plt.savefig(f'{g_dir}{model_name}_dV_map.pdf')
            plt.pause(1)
            plt.clf()
            
            # rms velocity and error
            #
            plt.figure()
            plt.imshow(Vrms_2d,origin='lower',cmap='sauron', vmin=Vrms_min, vmax=Vrms_max)
            cbar3 = plt.colorbar()
            cbar3.set_label(r'V$_{rms}$ [km/s]')
            plt.title(f"{obj_name} {g_band_name} RMS velocity")
            plt.xticks(ticks_pix, labels=ticklabels)
            plt.yticks(ticks_pix, labels=ticklabels)
            plt.xlabel('arcsec')
            plt.ylabel('arcsec')
            plt.savefig(f'{g_dir}{model_name}_Vrms_map.png')
            plt.savefig(f'{g_dir}{model_name}_Vrms_map.pdf')
            plt.pause(1)
            plt.clf()

            plt.figure()
            plt.imshow(dVrms_2d, origin='lower', cmap='sauron', vmin=dVrms_min, vmax=dVrms_max)
            cbar2 = plt.colorbar()
            cbar2.set_label(r'$dV{_rms}$ [km/s]')
            plt.title(f"{obj_name} {g_band_name} RMS velocity error")
            plt.xticks(ticks_pix, labels=ticklabels)
            plt.yticks(ticks_pix, labels=ticklabels)
            plt.xlabel('arcsec')
            plt.ylabel('arcsec')
            plt.savefig(f'{g_dir}{model_name}_dVrms_map.png')
            plt.savefig(f'{g_dir}{model_name}_dVrms_map.pdf')
            plt.pause(1)
            plt.clf()
            
        
        print('###########################################')
        print()
        print('Looking at difference between g and no g')
        print()

        
        # show including g - not including g
        gnog_VD = g_VD_2d - nog_VD_2d
        gnog_dVD = g_dVD_2d - nog_dVD_2d
        gnog_V = g_V_2d - nog_V_2d
        gnog_dV = g_dV_2d - nog_dV_2d
        gnog_Vrms = g_Vrms_2d - nog_Vrms_2d
        gnog_dVrms = g_dVrms_2d - nog_dVrms_2d
        
        
        # find velocity range for plotting each as the largest and smallest
        # VD
        VD_min = np.nanmin(gnog_VD)#(np.concatenate((np.ravel(g_VD_2d), np.ravel(nog_VD_2d)))))
        dVD_min = np.nanmin(gnog_dVD)#((np.concatenate((np.ravel(g_dVD_2d), np.ravel(nog_dVD_2d)))))
        VD_max = np.nanmax(gnog_VD)#((np.concatenate((np.ravel(g_VD_2d), np.ravel(nog_VD_2d)))))
        dVD_max = np.nanmax(gnog_dVD)#((np.concatenate((np.ravel(g_dVD_2d), np.ravel(nog_dVD_2d)))))
        # V
        V_min = np.nanmin(gnog_V)#((np.concatenate((np.ravel(g_V_2d), np.ravel(nog_V_2d)))))
        dV_min = np.nanmin(gnog_dV)#((np.concatenate((np.ravel(g_dV_2d), np.ravel(nog_dV_2d)))))
        V_max = np.nanmax(gnog_V)#((np.concatenate((np.ravel(g_V_2d), np.ravel(nog_V_2d)))))
        dV_max = np.nanmax(gnog_dV)#((np.concatenate((np.ravel(g_dV_2d), np.ravel(nog_dV_2d)))))
        # Vrms
        Vrms_min = np.nanmin(gnog_Vrms)#((np.concatenate((np.ravel(g_Vrms_2d), np.ravel(nog_Vrms_2d)))))
        dVrms_min = np.nanmin(gnog_dVrms)#((np.concatenate((np.ravel(g_dVrms_2d), np.ravel(nog_dVrms_2d)))))
        Vrms_max = np.nanmax(gnog_Vrms)#((np.concatenate((np.ravel(g_Vrms_2d), np.ravel(nog_Vrms_2d)))))
        dVrms_max = np.nanmax(gnog_dVrms)#((np.concatenate((np.ravel(g_dVrms_2d), np.ravel(nog_dVrms_2d)))))
        
        # velocity
        # mean is a final correction to the bulk velocity
        #mean = np.nanmean(V_2d)
        #print(f'Mean velocity: {mean}')
        #V_2d = V_2d-mean
        # velocity_range
        #vel_range = np.nanmax(np.abs(V_2d))

        # velocity dispersion
        print('VD')
        # normalize plotting colorbar
        vmin = VD_min
        vmax = VD_max
        print('Colorbar... vmin and vmax ', vmin, vmax)
        if vmin < 0:
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
            cmap = 'seismic'
        elif vmin >= 0:
            norm = None
            cmap = 'Reds'
        else:
            norm=None
            cmap= 'Reds'
            print('Colorbar problem... vmin and vmax ', vmin, vmax)
            
        plt.figure()
        p=plt.imshow(gnog_VD,origin='lower', cmap=cmap, norm=norm)
        cbar1 = plt.colorbar(p)
        cbar1.set_label(r'$\sigma$ [km/s]')
        plt.title(f"{obj_name} VD G difference")
        plt.xticks(ticks_pix, labels=ticklabels)
        plt.yticks(ticks_pix, labels=ticklabels)
        plt.xlabel('arcsec')
        plt.ylabel('arcsec')
        plt.savefig(f'{fin_kin_dir}{model_name}_VD_gdifference_map.png')
        plt.savefig(f'{fin_kin_dir}{model_name}_VD_gdifference_map.pdf')
        plt.pause(1)
        plt.clf()
        # dVD
        # normalize plotting colorbar
        vmin = dVD_min
        vmax = dVD_max
        print('Colorbar... vmin and vmax ', vmin, vmax)
        if vmin < 0:
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
            cmap = 'seismic'
        elif vmin >= 0:
            norm = None
            cmap = 'Reds'
        else:
            norm=None
            cmap= 'Reds'
            print('Colorbar problem... vmin and vmax ', vmin, vmax)
        plt.figure()
        p=plt.imshow(gnog_dVD, origin='lower', cmap=cmap, norm=norm)
        cbar2 = plt.colorbar(p)
        cbar2.set_label(r'd$\sigma$ [km/s]')
        plt.title(f"{obj_name} VD error G difference")
        plt.xticks(ticks_pix, labels=ticklabels)
        plt.yticks(ticks_pix, labels=ticklabels)
        plt.xlabel('arcsec')
        plt.ylabel('arcsec')
        plt.savefig(f'{fin_kin_dir}{model_name}_dVD_gdifference_map.png')
        plt.savefig(f'{fin_kin_dir}{model_name}_dVD_gdifference_map.pdf')

        plt.pause(1)
        plt.clf()

        # velocity 
        print('V')
        # V
        # normalize plotting colorbar
        vmin = V_min
        vmax = V_max
        print('Colorbar... vmin and vmax ', vmin, vmax)
        if vmin < 0:
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
            cmap = 'seismic'
        elif vmin >= 0:
            norm = None
            cmap = 'Reds'
        else:
            norm=None
            cmap= 'Reds'
            print('Colorbar problem... vmin and vmax ', vmin, vmax)
        plt.figure()
        p=plt.imshow(gnog_V,origin='lower', cmap=cmap, norm=norm)
        cbar3 = plt.colorbar(p)
        cbar3.set_label(r'V [km/s]')
        plt.title(f"{obj_name} V G difference")
        plt.xticks(ticks_pix, labels=ticklabels)
        plt.yticks(ticks_pix, labels=ticklabels)
        plt.xlabel('arcsec')
        plt.ylabel('arcsec')
        plt.savefig(f'{fin_kin_dir}{model_name}_V_gdifference_map.png')
        plt.savefig(f'{fin_kin_dir}{model_name}_V_gdifference_map.pdf')
        plt.pause(1)
        plt.clf()
        
        # dV
        # normalize plotting colorbar
        vmin = dV_min
        vmax = dV_max
        print('Colorbar... vmin and vmax ', vmin, vmax)
        if vmin < 0:
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
            cmap = 'seismic'
        elif vmin >= 0:
            norm = None
            cmap = 'Reds'
        else:
            norm=None
            cmap= 'Reds'
            print('Colorbar problem... vmin and vmax ', vmin, vmax)
        plt.figure()
        p=plt.imshow(gnog_dV,origin='lower', cmap=cmap, norm=norm)
        cbar2 = plt.colorbar(p)
        cbar2.set_label(r'dV [km/s]')
        plt.title(f"{obj_name} V error G difference")
        plt.xticks(ticks_pix, labels=ticklabels)
        plt.yticks(ticks_pix, labels=ticklabels)
        plt.xlabel('arcsec')
        plt.ylabel('arcsec')
        plt.savefig(f'{fin_kin_dir}{model_name}_dV_gdifference_map.png')
        plt.savefig(f'{fin_kin_dir}{model_name}_dV_gdifference_map.pdf')
        plt.pause(1)
        plt.clf()
            
        # rms velocity and error
        print('Vrms')
        # Vrms# normalize plotting colorbar
        vmin = Vrms_min
        vmax = Vrms_max
        print('Colorbar... vmin and vmax ', vmin, vmax)
        if vmin < 0:
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
            cmap = 'seismic'
        elif vmin >= 0:
            norm = None
            cmap = 'Reds'
        else:
            norm=None
            cmap= 'Reds'
            print('Colorbar problem... vmin and vmax ', vmin, vmax)
        plt.figure()
        p=plt.imshow(gnog_Vrms,origin='lower', cmap=cmap, norm=norm)
        cbar3 = plt.colorbar(p)
        cbar3.set_label(r'$V_{rms}$ [km/s]')
        plt.title(f"{obj_name} RMS velocity G difference")
        plt.xticks(ticks_pix, labels=ticklabels)
        plt.yticks(ticks_pix, labels=ticklabels)
        plt.xlabel('arcsec')
        plt.ylabel('arcsec')
        plt.savefig(f'{fin_kin_dir}{model_name}_Vrms_gdifference_map.png')
        plt.savefig(f'{fin_kin_dir}{model_name}_Vrms_gdifference_map.pdf')
        plt.pause(1)
        plt.clf()
        
        # dVrms
        # normalize plotting colorbar
        vmin = dVrms_min
        vmax = dVrms_max
        print('Colorbar... vmin and vmax ', vmin, vmax)
        if vmin < 0:
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
            cmap = 'seismic'
        elif vmin >= 0:
            norm = None
            cmap = 'Reds'
        else:
            norm=None
            cmap= 'Reds'
            print('Colorbar problem... vmin and vmax ', vmin, vmax)
        plt.figure()
        p=plt.imshow(gnog_dVD,origin='lower', cmap=cmap, norm=norm)
        cbar2 = plt.colorbar(p)
        cbar2.set_label(r'd$Vrms$ [km/s]')
        plt.title(f"{obj_name} Vrms error G difference")
        plt.xticks(ticks_pix, labels=ticklabels)
        plt.yticks(ticks_pix, labels=ticklabels)
        plt.xlabel('arcsec')
        plt.ylabel('arcsec')
        plt.savefig(f'{fin_kin_dir}{model_name}_dVrms_gdifference_map.pdf')
        plt.pause(1)
        plt.clf()


        
    print('###########################################################')
    print()
    print(f'{obj_name} has completed')
    print()
    
print()
print('Work complete.')
print('###########################################################')
print('###########################################################')

print('Done.')
