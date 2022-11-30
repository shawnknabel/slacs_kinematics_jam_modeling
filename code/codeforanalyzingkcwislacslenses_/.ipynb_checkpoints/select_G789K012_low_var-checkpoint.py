###
# Makes the low-temp systematics check for templates
###

import numpy as np

# directories

data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
xshooter_dir = f'{data_dir}xshooter_lib/'
libary_dir_xshooter = f'{xshooter_dir}all_dr2_fits/'
temp_selections = f'{xshooter_dir}temperature_range_selections/'

# select the lower temperature range... this looks silly but it just gets the labeling correct
a=np.genfromtxt(f'{temp_selections}T4300-5000.csv').astype(int).astype(str)
for i in range(len(a)):
    if len(a[i]) < 3:
        a[i] = '0' + a[i]
    if len(a[i]) < 3:
        a[i] = '0' + a[i]
        
os.chdir(libary_dir_xshooter)

import glob
## remove 298, 321 as it is only has NOT slit-loss corrected version
## remove 754 as it is "NOT slit-loss corrected version" of 755.
## remove 0042 as its "not slit-loss corrected version of 0089
import os
import shutil
for i in range(len(a)):
   file = glob.glob(f'*{a[i]}_uvb.fits')
   if file ==[]:
      print('no slit-loss corrected=', a[i])
   else:
       shutil.copyfile(file[0],
                       f'{xshooter_dir}all_dr2_fits_G789K012_lo_var/{file[0]}')
       print(glob.glob(f'{file[0]}'))



################################
