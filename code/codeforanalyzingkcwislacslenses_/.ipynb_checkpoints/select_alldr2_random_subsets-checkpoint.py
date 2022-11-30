'''
This script takes Chih-Fan's selection of G789K012 stars and makes three subsets of them of length 100
'''


import numpy as np


a=['077',
   '084',
   '092',
   '094',
   '097',
   '110',
   '122',
   '138',
   '139',
   '151',
   '165',
   '171',
   '197',
   '207',
   '233',
   '247',
   '272',
   '274',
   '298',
   '299',
   '304',
   '321',
   '325',
   '339',
   '347',
   '349',
   '351',
   '352',
   '354',
   '355',
   '363',
   '372',
   '377',
   '388',
   '389',
   '395',
   '400',
   '401',
   '408',
   '409',
   '416',
   '417',
   '419',
   '426',
   '435',
   '436',
   '438',
   '441',
   '442',
   '444',
   '446',
   '453',
   '454',
   '466',
   '471',
   '473',
   '479',
   '481',
   '482',
   '491',
   '496',
   '500',
   '501',
   '502',
   '526',
   '536',
   '537',
   '548',
   '549',
   '552',
   '559',
   '563',
   '565',
   '567',
   '570',
   '581',
   '583',
   '593',
   '594',
   '598',
   '600',
   '601',
   '610',
   '612',
   '618',
   '620',
   '621',
   '627',
   '634',
   '637',
   '641',
   '652',
   '658',
   '661',
   '669',
   '670',
   '681',
   '700',
   '701',
   '703',
   '704',
   '705',
   '706',
   '713',
   '714',
   '715',
   '716',
   '717',
   '733',
   '735',
   '738',
   '741',
   '742',
   '750',
   '752',
   '762',
   '764',
   '768',
   '769',
   '772',
   '787',
   '788',
   '824',
   '829',
   '835',
   '839',
   '840',
   '841',
   '858',
   '866',
   '871',
   '883',
   '897',
   '902',
]

a1 = np.random.choice(a, 100)
a2 = np.random.choice(a, 100)
a3 = np.random.choice(a, 100)

import glob
## remove 298, 321 as it is only has NOT slit-loss corrected version
## remove 754 as it is "NOT slit-loss corrected version" of 755.
## remove 0042 as its "not slit-loss corrected version of 0089
import os
import shutil

# directories

data_dir = '/data/raw_data/KECK_KCWI_SLACS_kinematics_shawn/'
xshooter_dir = f'{data_dir}xshooter_lib/'
library_dir_xshooter = f'{xshooter_dir}all_dr2_fits/'
rand_selections = f'{xshooter_dir}random_subset_selections/'

os.chdir(libary_dir_xshooter)

b = 1
for a in [a1, a2, a3]:
    os.makedirs(f'{rand_selections}all_dr2_fits_G789K012_rand_subset_{b}/', exist_ok=True)
    for i in range(len(a)):
        if glob.glob('*' + a[i] + '_uvb*') ==[]:
            print('no slit-loss corrected=', a[i])
        else:
            shutil.copyfile(glob.glob('*' + a[i] + '_uvb*')[0],
                                       rand_selections +
                                       f'all_dr2_fits_G789K012_rand_subset_{b}/' +
                                        glob.glob('*' + a[i] + '_uvb*')[0])
            print(glob.glob('*' + a[i] + '_uvb*')[0])
    b = b+1
