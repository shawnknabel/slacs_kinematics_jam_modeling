import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({"figure.figsize" : (8, 6)})
from astropy.io import fits
from pafit.fit_kinematic_pa import fit_kinematic_pa

#################################################
# objects
obj_names = [#'SDSSJ0029-0055',
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

slit_width=0.5

##################################################
# functions


def calc_minmaxmean_R (bin_arrays, num_bins):
    
    bin_R_mins = np.zeros(num_bins)
    bin_R_maxs = np.zeros(num_bins)
    bin_R_means = np.zeros(num_bins)
    
    for i in range(num_bins):
        bin_pixels = bin_arrays[bin_arrays[:,2]==i]
        bin_xx = bin_pixels[:,0] - 21 # subtract 21 pixels, center don't know which is x and which is y, but doesn't matter for R
        bin_yy = bin_pixels[:,1] - 21
        bin_R = np.zeros(len(bin_pixels))
        for j in range(len(bin_pixels)):
            bin_R[j] = np.sqrt(bin_xx[j]**2 + bin_yy[j]**2)
        bin_R_mins[i] = np.min(bin_R)*0.1457
        bin_R_maxs[i] = np.max(bin_R)*0.1457
        bin_R_means[i] = np.mean(bin_R)*0.1457
        
    return bin_R_mins, bin_R_maxs, bin_R_means



def calc_rotate_minmaxmean_bins (bin_arrays, num_bins, pos_angle):
    
    '''
    calculate rotation of x and y coordinates for each pixel and bin with min mean and max for velocity along maj axisa
    
    pos_angle is maj_axis position angle measured from x axis
    
    '''
    
    theta = np.radians(pos_angle) # measure from x axis
    
    bin_yy_means = np.zeros(num_bins)
    bin_xx_mins = np.zeros(num_bins)
    bin_xx_maxs = np.zeros(num_bins)
    bin_xx_means = np.zeros(num_bins)
    
    for i in range(num_bins):
        # get bins and x, y for each pixel
        bin_pixels = bin_arrays[bin_arrays[:,2]==i]
        bin_x = bin_pixels[:,0] - 21 # subtract 21 pixel
        bin_y = bin_pixels[:,1] - 21
        # set up arrays for bin xx, yy (rotational transform by theta so that xx axis is the major axis)
        bin_xx = np.zeros(len(bin_pixels))
        bin_yy = np.zeros(len(bin_pixels))
        # loop through bin pixels
        for j in range(len(bin_pixels)):
            bin_xx[j] = bin_x[j] * np.cos(theta) + bin_y[j] * np.sin(theta)
            bin_yy[j] = - bin_x[j] * np.sin(theta) + bin_y[j] * np.cos(theta)
        
        bin_yy_means[i] = np.mean(bin_yy)*0.1457 # arcsec distance from maj axis
        
        bin_xx_mins[i] = np.min(bin_xx)*0.1457
        bin_xx_maxs[i] = np.max(bin_xx)*0.1457
        bin_xx_means[i] = np.mean(bin_xx)*0.1457
        
    return bin_yy_means, bin_xx_mins, bin_xx_maxs, bin_xx_means



# get bin centers
def get_bin_centers (bin_arrays, num_bins):
    bin_y_means = np.zeros(num_bins)
    bin_x_means = np.zeros(num_bins)
    
    for i in range(num_bins):
        # get bins and x, y for each pixel
        bin_pixels = bin_arrays[bin_arrays[:,2]==i]
        bin_x = bin_pixels[:,0] - 21 # subtract 21 pixel
        bin_y = bin_pixels[:,1] - 21
        # calculate mean x and y
        mean_x = np.mean(bin_x)
        mean_y = np.mean(bin_y)
        # update array
        bin_x_means[i] = mean_x
        bin_y_means[i] = mean_y
        
    return bin_x_means, bin_y_means
        



def calc_1d_kinematics_reff (obj_name, slit_width=0.5):
    
    obj_abbr = obj_name[4:9] # e.g. J0029
    
    #######################################
    # data directory
    data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
    # object directory
    dir = f'{data_dir}mosaics/{obj_name}/'
    #KCWI mosaic datacube
    name = f'KCWI_{obj_abbr}_icubes_mosaic_0.1457'
    save_dir = f'{dir}{obj_name}_systematics/'
    
    #######################################
    ## import voronoi binning data
    voronoi_binning_data = fits.getdata(dir +'voronoi_binning_' + name + '_data.fits')
    vorbin_pixels = np.genfromtxt(f'{dir}voronoi_2d_binning_{name}_output.txt',
                     delimiter='')
    # sort the voronoi bin pixel data by bin
    vorbin_pixels = vorbin_pixels[vorbin_pixels[:,2].argsort()]
    
    #######################################
    # get velocity dispersion data
    VD=np.genfromtxt(f'{dir}{obj_name}_final_kinematics/{obj_name}_VD_binned.txt',
                 delimiter=',')
    VD_cov = np.genfromtxt(f'{dir}{obj_name}_final_kinematics/{obj_name}_covariance_matrix_VD.txt',
                     delimiter=',')
    dVD = np.sqrt(np.diagonal(VD_cov)) # to have error bars
    
    #######################################
    # get velocity data 
    V=np.genfromtxt(f'{dir}{obj_name}_final_kinematics/{obj_name}_V_binned.txt',
                 delimiter=',')
    V = V-np.mean(V)
    V_cov = np.genfromtxt(f'{dir}{obj_name}_final_kinematics/{obj_name}_covariance_matrix_V.txt',
                     delimiter=',')
    dV = np.sqrt(np.diagonal(V_cov)) # to have error bars
    
    #######################################
    # get reff
    slacs_table = np.genfromtxt(f'{data_dir}slacs_tableA1.txt', delimiter='', dtype='U10')
    slacs_table_name = obj_name[4:]
    slacs_reffs = slacs_table[:,7].astype(float)
    reff = slacs_reffs[slacs_table[:,0]==slacs_table_name]
    
    #######################################
    # calculate velocity dispersion curve
    
    Rmins, Rmaxs, Rmeans = calc_minmaxmean_R (vorbin_pixels, len(voronoi_binning_data))
    
    plt.figure(figsize=(12,4))
    plt.errorbar(Rmeans/reff, VD, 
                 xerr=(Rmeans-Rmins,Rmaxs-Rmeans)/reff,
                 yerr=dVD,
                 c='k',
                 marker='o', 
                 linestyle='None')
    plt.title(f'{obj_name} velocity dispersion profile',fontsize=14)
    plt.xlabel(r'R/$R_{eff}$',fontsize=14)
    plt.xlim(0,2.0)
    plt.ylabel(r'$\sigma$ (km/s)',fontsize=14)
    plt.savefig(f'{save_dir}{obj_name}_1d_VD_profile_reff.png')
    plt.savefig(f'{save_dir}{obj_name}_1d_VD_profile_reff.pdf')
    plt.show()
    
    #######################################
    # calculate velocity curve
    
    xbin, ybin = get_bin_centers (vorbin_pixels, len(voronoi_binning_data))
    kin_pa, kin_pa_error, vel_offset = fit_kinematic_pa(xbin, ybin, V)
    plt.savefig(f'{save_dir}{obj_name}_kinematic_pa_fit_reff.png')
    plt.savefig(f'{save_dir}{obj_name}_kinematic_pa_fit_reff.pdf')
    
    # kin_pa is the rotational axis as measured from the x axis. We want kin_pa - 90
    maj_ax_pa = kin_pa - 90
    
    # rotate bins
    yy_means, xx_mins, xx_maxs, xx_means = calc_rotate_minmaxmean_bins(vorbin_pixels, len(voronoi_binning_data), maj_ax_pa)
    
    # take bins within slit width
    in_slit = np.abs(yy_means) < slit_width/2
    xx_mins_slit = xx_mins[in_slit]
    xx_maxs_slit = xx_maxs[in_slit]
    xx_means_slit = xx_means[in_slit]
    yy_means_slit = yy_means[in_slit]
    
    # plot
    plt.figure(figsize=(8,8))
    plt.errorbar(xx_means_slit, yy_means_slit,
                 xerr=(xx_means_slit-xx_mins_slit,xx_maxs_slit-xx_means_slit),
                 #yerr=(yy_means_slit-yy_mins_slit,yy_maxs_slit-yy_means_slit),
                 marker='o', linestyle='None')
    plt.scatter(xx_means, yy_means, c='k')
    plt.ylim(-3.0,3.0)
    plt.xlim(-3.0,3.0)
    
    # take velocities with slit condition
    V_slit = V[in_slit]
    dV_slit = dV[in_slit]
    
    # ensure that the positive V is on the right side of the xx axis
    flip_means = np.mean(V_slit[xx_means_slit<0]) > (np.mean(V_slit[xx_means_slit>0]))
    flip_ends = V_slit[np.argmin(xx_means_slit)] > V_slit[np.argmax(xx_means_slit)]
    if flip_means & flip_ends:
        V_slit = - V_slit
    if obj_name == 'SDSSJ2303+1422': # I can't get this one to work right
        V_slit = - V_slit
    
    # plot
    plt.figure(figsize=(12,4))
    plt.errorbar(xx_means_slit/reff, V_slit, 
                 xerr=(xx_means_slit-xx_mins_slit,xx_maxs_slit-xx_means_slit)/reff,
                 yerr=dV_slit,
                 c='k',
                 marker='o', 
                 linestyle='None')
    plt.title(f'{obj_name} velocity profile',fontsize=14)
    plt.xlabel(r'R/$R_{eff}$',fontsize=14)
    plt.xlim(-2.0,2.0)
    plt.ylabel(r'V (km/s)',fontsize=14)
    plt.savefig(f'{save_dir}{obj_name}_1d_V_profile_reff.png')
    plt.savefig(f'{save_dir}{obj_name}_1d_V_profile_reff.pdf')
    plt.show()
    
###################################################

# run the functions

# set up mosaic
axes = '''
        AA
        BB
        CC
        DD #################### this isn't ready yeat
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

for obj_name in obj_names:
    if obj_name == 'SDSSJ1306+0600':
        continue
    else:
        print('#########################################################')
        print('#########################################################')
        print()
        print(f'Calculating 1-D kinematics of {obj_name} with effective radius')
        print()
        calc_1d_kinematics_reff (obj_name, slit_width)
        print()