# visualize the 2d kinematics after having run systematics and collective covariance matrices

# 
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 14})
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

def kinematics_map_systematics(save_dir, name, mosaic_name, radius_in_pixels):
    '''
    this code modifies CF's kinematics_map function to remap the kinematics measurements above into 2D array from my systematics scripts.
    :return: 2D velocity dispersion, uncertainty of the velocity dispersion, velocity, and the uncertainty of the velocity
    
    # for now, only gives the velocity dispersion terms!
    '''

    VD=np.genfromtxt(f'{save_dir}{name}_VD_binned.txt',
                 delimiter=',')
    VD_covariance = np.genfromtxt(f'{save_dir}{name}_covariance_matrix_VD.txt',
                                 delimiter=',')
    dVD = np.sqrt(np.diagonal(VD_covariance)) 
# diagonal is sigma**2
    
    
    V=np.genfromtxt(f'{save_dir}{name}_V_binned.txt',
                 delimiter=',')
    V_covariance = np.genfromtxt(f'{save_dir}{name}_covariance_matrix_V.txt',
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

#########################################################################
### loop through the objects

radius_in_pixels=21

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
    # mosaic name to get the write voronoi binning data in
    mosaic_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'
    # object directory
    dir = f'{data_dir}mosaics/{obj_name}/'
    # set final kinematics directory
    save_dir_1 = f'{dir}{obj_name}_final_kinematics/'
    save_dir_2 = f'{dir}{obj_name}_final_kinematics_template_subsets_310822/'
    
    VD_2d_1, dVD_2d_1, V_2d_1, dV_2d_1 = kinematics_map_systematics(save_dir_1, obj_name, mosaic_name, radius_in_pixels)
    VD_2d_2, dVD_2d_2, V_2d_2, dV_2d_2 = kinematics_map_systematics(save_dir_2, obj_name, mosaic_name, radius_in_pixels)
    
    # velocity
    # mean is a final correction to the bulk velocity
    mean_1 = np.nanmean(V_2d_1)
    #print(f'Mean velocity: {mean_1}')
    V_2d_1 = V_2d_1-mean_1
    
    # velocity
    # mean is a final correction to the bulk velocity
    mean_2 = np.nanmean(V_2d_2)
    #print(f'Mean velocity: {mean_1}')
    V_2d_2 = V_2d_2-mean_2
    
    # subtract to get differences
    VD_2d = VD_2d_2 - VD_2d_1
    dVD_2d = dVD_2d_2 - dVD_2d_1
    V_2d = V_2d_2 - V_2d_1
    dV_2d = dV_2d_2 - dV_2d_1
    
    
    ##################################################
    # plot each
    
    # pixel scale is 0.147 arcsec/pixel, set x and y ticks on the plot to be in arcsec instead of pixels
    pixel_scale = 0.147
    ticks = np.arange(7)
    ticks_pix = ticks/pixel_scale
    ticklabels= np.arange(-3, 4)
    
    
    # velocity_range
    vel_range = np.nanmax(np.abs(V_2d))
    
    
    # velocity dispersion and error
    plt.figure()
    plt.imshow(VD_2d,origin='lower',cmap='sauron')
    cbar1 = plt.colorbar()
    cbar1.set_label(r'$\sigma$ [km/s]')
    plt.title(f"{obj_name} velocity dispersion (subsets - T ranges)")
    plt.xticks(ticks_pix, labels=ticklabels)
    plt.yticks(ticks_pix, labels=ticklabels)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.savefig(save_dir_2 + obj_name + '_VD_difference_tempsubsets_minus_Tranges_map.png')
    plt.savefig(save_dir_2 + obj_name + '_VD_difference_tempsubsets_minus_Tranges_map.pdf')
    plt.pause(1)
    plt.clf()

    plt.figure()
    plt.imshow(dVD_2d, origin='lower', cmap='sauron')#,vmin=0, vmax=40)
    cbar2 = plt.colorbar()
    cbar2.set_label(r'd$\sigma$ [km/s]')
    plt.title(f"{obj_name} velocity dispersion error (subsets - T ranges)")
    plt.xticks(ticks_pix, labels=ticklabels)
    plt.yticks(ticks_pix, labels=ticklabels)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.savefig(save_dir_2 + obj_name + '_dVD_difference_tempsubsets_minus_Tranges_map.png')
    plt.savefig(save_dir_2 + obj_name + '_dVD_difference_tempsubsets_minus_Tranges_map.pdf')
    
    plt.pause(1)
    plt.clf()
    
    # velocity and error
    #
    plt.figure()
    plt.imshow(V_2d,origin='lower',cmap='sauron',vmin=-vel_range-5, vmax=vel_range+5)
    cbar3 = plt.colorbar()
    cbar3.set_label(r'V [km/s]')
    plt.title(f"{obj_name} velocity (subsets - T ranges)")
    plt.xticks(ticks_pix, labels=ticklabels)
    plt.yticks(ticks_pix, labels=ticklabels)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.savefig(save_dir_2 + obj_name + '_V_difference_tempsubsets_minus_Tranges_map.png')
    plt.savefig(save_dir_2 + obj_name + '_V_difference_tempsubsets_minus_Tranges_map.pdf')
    plt.pause(1)
    plt.clf()
    
    plt.figure()
    plt.imshow(dVD_2d, origin='lower', cmap='sauron')#,vmin=0, vmax=40)
    cbar2 = plt.colorbar()
    cbar2.set_label(r'dV [km/s]')
    plt.title(f"{obj_name} velocity error (subsets - T ranges)")
    plt.xticks(ticks_pix, labels=ticklabels)
    plt.yticks(ticks_pix, labels=ticklabels)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.savefig(save_dir_2 + obj_name + '_dV_difference_tempsubsets_minus_Tranges_map.png')
    plt.savefig(save_dir_2 + obj_name + '_dV_difference_tempsubsets_minus_Tranges_map.pdf')
    plt.pause(1)
    plt.clf()
    
    print()
    print(f'{obj_name} has completed')
    print('###########################################################')
    
print()
print('###########################################################')
print('###########################################################')

print('Done.')
