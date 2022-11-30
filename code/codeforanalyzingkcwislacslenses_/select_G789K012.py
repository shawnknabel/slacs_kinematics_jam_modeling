'''
This script takes Chih-Fan's selection of G789K012 stars and makes three subsets of them
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

import glob
## remove 298, 321 as it is only has NOT slit-loss corrected version
## remove 754 as it is "NOT slit-loss corrected version" of 755.
## remove 0042 as its "not slit-loss corrected version of 0089
import os
import shutil
for i in range(len(a)):
   if glob.glob('*' + a[i] + '_uvb*') ==[]:
      print('no slit-loss corrected=', a[i])
   else:
       shutil.copyfile(glob.glob('*' + a[i] + '_uvb*')[0],
                       '/Users/Geoff/anaconda3/envs/py39/lib/python3.9/site-packages'
                       '/ppxf/all_dr2_fits_G789K012/' +
                       glob.glob('*' + a[i] + '_uvb*')[0])
       print(glob.glob('*' + a[i] + '_uvb*')[0])



################################

libary_dir_xshooter = '/Users/Geoff/anaconda3/envs/py39/lib/python3.9/site' \
                      '-packages/ppxf/all_dr2_fits/'

import glob
xshooter = glob.glob(libary_dir_xshooter + '/*uvb.fits')

import numpy as np
import pandas as pd
df = pd.DataFrame(pd.read_excel(
	'/Users/Geoff/anaconda3/envs/py39/lib/python3.9/site-packages/ppxf/all_dr2_fits/t_eff.xlsx'))
info=df.to_numpy()
col = 1
data=info[np.argsort(info[:,col])]

number = np.zeros(len(xshooter))
for i in range(number.shape[0]):
	number[i] = int(xshooter[i][-12:-9])

temperture = np.zeros(len(xshooter))
for i in range(temperture.shape[0]):
	if (~(data.T[0] == number[i])).all():
		temperture[i] = 0
	else:
		temperture[i]= data.T[1][data.T[0] == number[i]]


stellar_type=np.stack((temperture,pp_weights_2700)).T
stellar_type_reorder = stellar_type[np.argsort(stellar_type[:,0])]

from matplotlib import pyplot as plt
plt.plot(stellar_type_reorder[:,0], stellar_type_reorder[:,1])
plt.xlabel('temperture (K)')
plt.ylabel('weight')
plt.show()

plt.hist(stellar_type_reorder[:,0], weights=stellar_type_reorder[:,1],bins=100)
plt.xlabel('temperture (K)')
plt.ylabel('weight')
plt.show()

model = np.zeros(fits.getdata(xshooter[0])['FLUX'].shape[0])
for i in range(len(xshooter)):
	if pp_weights_2700[i] >0:
		wavelength = fits.getdata(xshooter[i])['WAVE']*10
		flux = fits.getdata(xshooter[i])['FLUX']*pp_weights_2700[i]
		model = model+flux
		plt.plot(wavelength, flux)
plt.plot(np.exp(logLam2_xshooter), global_temp_xshooter/np.mean(global_temp_xshooter))
plt.plot(wavelength, model/np.mean(model))
plt.show()
