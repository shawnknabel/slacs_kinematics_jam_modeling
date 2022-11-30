# plot the cropped kcwi datacubes

# 
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patch
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({"figure.figsize" : (8, 6)})
from astropy.visualization import simple_norm
import pickle


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
        AA
        BB
        CC
        DD
        EE
        FF
        GG
        HH
        II
        JJ
        KK
        LL
        MM
        NN
        '''

alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

fig, axs = plt.subplot_mosaic(axes,gridspec_kw={"hspace":0.0,
                                               "wspace":0.0},
                             figsize=((12,16)),
                             sharex=True
                             )

for i in range(len(obj_names)):
    
    obj_name = obj_names[i]
    letter = alphabet[i]
    plt.axes(axs[letter])
    
    print('#####################################################')
    print('#####################################################')
    print()
    print(f'Plotting global template spectrum for object {obj_name}.')
    print()
    
    # set abbreviation for directories
    obj_abbr = obj_name[4:9] # e.g. J0029
    # object directory
    dir = f'{data_dir}mosaics/{obj_name}/'
    syst_dir = f'{dir}{obj_name}_systematics/'
    model = 'l0a1d1w1'
    
    file = open(f'{syst_dir}{model}/global_template_spectrum_fit.pkl','rb') 
    picle = pickle.load(file)
    #print(f'Velocity {picle.sol[0]}, Velocity dispersion {picle.sol[1]}')
    picle.plot()
    plt.xlim(0.32, 0.43)
    if obj_abbr == 'J0330':
        #plt.xlim(0.32, 0.41)
        plt.ylim(-0.0001, 0.07)
        rect = patch.Rectangle([0.409,-0.0001],width=0.025, height=0.1001,
                               facecolor='lightgrey',hatch='/',alpha=1.0,
                              )
        plt.gca().add_patch(rect)
    plt.ylabel(None)
    plt.tick_params(left = False, right = False , labelleft = False )
    
    # plot
    #yticks = axs[letter].yaxis.get_major_ticks()
    #xticks = axs[letter].xaxis.get_major_ticks()
    #yticks[0].label.set_visible(False)
    #xticks[0].label.set_visible(False)
    #yticks[-1].label.set_visible(False)
    #xticks[-1].label.set_visible(False)
    #axs[letter].tick_params('x', direction='in', pad=-14, colors='k')#, labelleft=False)
    #axs[letter].tick_params('y', direction='in', pad=-14, colors='k')
    axs[letter].text(0.01, 0.8, obj_name, transform=axs[letter].transAxes)
    #plt.colorbar(label='flux') # no colorbar for paper plots
    
    #plt.savefig(f'{dir}{obj_name}_cropped_mosaic.png')
    #plt.savefig(f'{dir}{obj_name}_cropped_mosaic.pdf')
    
    #plt.show()
    file.close()
    
    print()

plt.xticks(ticks=[0.32,0.34,0.36,0.38,0.40,0.42],labels=['3200','3400','3600','3800','4000','4200'])
plt.xlabel(r'$\lambda_{rest}$ ($\AA$)')
plt.tight_layout()

plt.savefig(f'{data_dir}mosaics/mosaic_of_global_temp_spectra.png')
plt.savefig(f'{data_dir}mosaics/mosaic_of_global_temp_spectra.pdf')
