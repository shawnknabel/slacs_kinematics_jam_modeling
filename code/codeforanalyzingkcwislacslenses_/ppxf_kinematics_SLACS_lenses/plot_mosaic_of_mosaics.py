# plot the cropped kcwi datacubes

# 
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({"figure.figsize" : (8, 6)})
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


# data directory
data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'

# set up mosaic
axes = '''
        AABBCC
        AABBCC
        DDEEFF
        DDEEFF
        GGHHII
        GGHHII
        JJKKLL
        JJKKLL
        .MMNN.
        .MMNN.
        '''

alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

fig, axs = plt.subplot_mosaic(axes,gridspec_kw={"hspace":0.1,
                                               "wspace":0.0},
                             figsize=((12,20)))

# pixel scale is 0.147 arcsec/pixel, set x and y ticks on the plot to be in arcsec instead of pixels
pixel_scale = 0.147
ticks = np.arange(0,7)
ticks_pix = ticks/pixel_scale+1
ticklabels= np.arange(-3, 4)

for i in range(len(obj_names)):
    
    obj_name = obj_names[i]
    letter = alphabet[i]
    
    print('#####################################################')
    print('#####################################################')
    print()
    print(f'Plotting datacube for object {obj_name}.')
    print()
    
    # set abbreviation for directories
    obj_abbr = obj_name[4:9] # e.g. J0029
    # object directory
    dir = f'{data_dir}mosaics/{obj_name}/'
    
    # open file and get data
    file = f'{dir}KCWI_{obj_abbr}_icubes_mosaic_0.1457_crop.fits'
    data = fits.open(file)[0].data
    
    # normalize data
    norm = simple_norm(np.nansum(data, axis=0), 'sqrt')
    
    # plot
    axs[letter].imshow(np.nansum(data, axis=0), origin="lower", norm=norm)
    axs[letter].set_xticks(ticks_pix, ticklabels, fontsize=8)
    axs[letter].set_yticks(ticks_pix, ticklabels, fontsize=8)
    yticks = axs[letter].yaxis.get_major_ticks()
    xticks = axs[letter].xaxis.get_major_ticks()
    yticks[0].label.set_visible(False)
    xticks[0].label.set_visible(False)
    yticks[-1].label.set_visible(False)
    xticks[-1].label.set_visible(False)
    axs[letter].tick_params('x', direction='in', pad=-14, colors='white')#, labelleft=False)
    axs[letter].tick_params('y', direction='in', pad=-14, colors='white')
    axs[letter].text(4,40,obj_name)
    #plt.colorbar(label='flux') # no colorbar for paper plots
    
    #plt.savefig(f'{dir}{obj_name}_cropped_mosaic.png')
    #plt.savefig(f'{dir}{obj_name}_cropped_mosaic.pdf')
    
    #plt.show()
    
    print()
    
plt.tight_layout()

plt.savefig(f'{data_dir}mosaics/mosaic_of_mosaics.png')
plt.savefig(f'{data_dir}mosaics/mosaic_of_mosaics.pdf')
