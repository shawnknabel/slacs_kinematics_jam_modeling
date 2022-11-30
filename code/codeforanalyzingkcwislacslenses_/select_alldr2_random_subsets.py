'''
This script takes Xshooter DR2 three subsets of uvb templates of length 100
'''


import numpy as np
import glob
import os
import shutil

# directories

data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
xshooter_dir = f'{data_dir}xshooter_lib/'
library_dir_xshooter = f'{xshooter_dir}all_dr2_fits/'
rand_selections = f'{xshooter_dir}random_subset_selections/'

os.chdir(library_dir_xshooter)

# read in all template filenames
all_templates = os.listdir()
# select only uvb
uvb_templates = glob.glob('*_uvb*')
# select subsets
a1 = np.random.choice(uvb_templates, 100, replace=False)
a2 = np.random.choice(uvb_templates, 100, replace=False)
a3 = np.random.choice(uvb_templates, 100, replace=False)

b = 1
for a in [a1, a2, a3]:
    os.makedirs(f'{rand_selections}all_dr2_fits_rand_subset_{b}/', exist_ok=True)
    for i in range(len(a)):
        shutil.copyfile(a[i],
                       rand_selections +
                       f'all_dr2_fits_rand_subset_{b}/' +
                        a[i])
    b = b+1
