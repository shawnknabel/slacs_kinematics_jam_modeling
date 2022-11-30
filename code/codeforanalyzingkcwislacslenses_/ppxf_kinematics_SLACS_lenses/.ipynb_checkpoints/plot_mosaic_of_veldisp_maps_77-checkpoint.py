# plot the cropped kcwi datacubes

# 

from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 10})
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
from astropy.visualization import simple_norm

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

def kinematics_map_veldisp(dir, name, mosaic_name, radius_in_pixels):
    '''
    this code modifies CF's kinematics_map function to remap the kinematics measurements above into 2D array from my systematics scripts.
    :return: 2D velocity dispersion, uncertainty of the velocity dispersion, velocity, and the uncertainty of the velocity
    
    # for now, only gives the velocity dispersion terms!
    '''

    VD=np.genfromtxt(f'{dir}{name}_final_kinematics/{name}_VD_binned.txt',
                 delimiter=',')
    
    # Vel, sigma, dv, dsigma
    output=np.loadtxt(dir +'voronoi_2d_binning_' + mosaic_name + '_output.txt')
    
    VD_array    =np.zeros(output.shape[0])

    for i in range(output.shape[0]):
        num=int(output.T[2][i])
        VD_array[i] = VD[num]

    final=np.vstack((output.T, VD_array))#, dVD_array, V_array, dV_array))
    
    dim = radius_in_pixels*2+1

    VD_2d=np.zeros((dim, dim))
    VD_2d[:]=np.nan
    for i in range(final.shape[1]):
        VD_2d[int(final[1][i])][int(final[0][i])]=final[3][i]
        
    del VD
    
    return VD_2d

#########################################################################

radius_in_pixels=21

# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'

# set up mosaic
axes = '''
        AABBCCDDEEFFGG
        AABBCCDDEEFFGG
        HHIIJJKKLLMMNN
        HHIIJJKKLLMMNN
        '''

alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

fig, axs = plt.subplot_mosaic(axes,gridspec_kw={"hspace":-0.3,
                                               "wspace":0.0},
                             figsize=((14,6)))

# pixel scale is 0.147 arcsec/pixel, set x and y ticks on the plot to be in arcsec instead of pixels
pixel_scale = 0.147
ticks = np.arange(0,7)
ticks_pix = ticks/pixel_scale+1
ticklabels= np.arange(-3, 4)

for i in range(len(obj_names)):
    
    obj_name = obj_names[i]
    letter = alphabet[i]
    
    # set abbreviation for directories
    obj_abbr = obj_name[4:9] # e.g. J0029
    # mosaic name to get the write voronoi binning data in
    mosaic_name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'
    # object directory
    dir = f'{data_dir}mosaics/{obj_name}/'
    # set final kinematics directory
    save_dir = f'{dir}{obj_name}_final_kinematics/'
    
    VD_2d = kinematics_map_veldisp(dir, obj_name, mosaic_name, radius_in_pixels)
    
    
    ##################################################
    # plot VD_2d
    
    # pixel scale is 0.147 arcsec/pixel, set x and y ticks on the plot to be in arcsec instead of pixels
    pixel_scale = 0.147
    ticks = np.arange(7)
    ticks_pix = ticks/pixel_scale
    ticklabels= np.arange(-3, 4)
    vmin = np.min(VD_2d)

    # plot
    plot = axs[letter].imshow(VD_2d,origin='lower',cmap='sauron')
    cbar = plt.colorbar(plot, ax=axs[letter],
                       location='bottom',
                       shrink=1.0,
                       pad=0.0
                       )
    cbar.set_label(r'$\sigma$ [km/s]', fontsize=10, labelpad=0.0)
    cbar.ax.tick_params(rotation=0, labelsize='small', pad=1.5, direction ='in')
    #cbar.ax.locator_params(nbins=4)
    if i == 8:
        cbar.ax.set_xticks(np.linspace(225,300,4))
    elif i == 6:
        cbar.ax.set_xticks(np.linspace(175,250,4))
    # remove lowest label if it's too close to vmin
    #cbar_ticks = cbar.ax.xaxis.get_major_ticks()
    #low_tick = cbar_ticks[0]
    #print(low_tick)
    #if low_tick - vmin < 10:
    #    print(low_tick - vmin)
    #    low_tick.set_visible(False)      
    axs[letter].set_xticks(ticks_pix, ticklabels, fontsize='small')
    axs[letter].set_yticks(ticks_pix, ticklabels, fontsize='small')
    yticks = axs[letter].yaxis.get_major_ticks()
    xticks = axs[letter].xaxis.get_major_ticks()
    yticks[0].label.set_visible(False)
    yticks[-1].label.set_visible(False)
    xticks[0].label.set_visible(False)
    #yticks[-1].label.set_visible(False)
    #xticks[-1].label.set_visible(False)
    axs[letter].tick_params('x', direction='in', pad=-14, colors='k')#, labelleft=False)
    axs[letter].tick_params('y', direction='in', pad=-14, colors='k')
    axs[letter].text(5,39,obj_name,fontsize=10)#,font='Helvetica')
    #plt.colorbar(label='flux') # no colorbar for paper plots
    
    #plt.savefig(f'{dir}{obj_name}_cropped_mosaic.png')
    #plt.savefig(f'{dir}{obj_name}_cropped_mosaic.pdf')
    
    #plt.show()
    
    print()

#plt.tight_layout()

plt.savefig(f'{data_dir}mosaics/slacs_kcwi_kinmaps_7x2.png')
plt.savefig(f'{data_dir}mosaics/slacs_kcwi_kinmaps_7x2.pdf')

plt.show()