#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:00:24 2017

@author: icgguest
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import cosmology
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy import units as u
from astropy.table import Table, Column
plt.close('all')
mpl5 = fits.open('MaNGA_targets_extNSA_tiled_ancillary(1).fits')

#Import the data needed from the fits files
mpl5_data = mpl5[1].data

manga_target_1 = mpl5_data.field('MANGA_TARGET1')
primary_z_min = mpl5_data.field('ZMIN')
primary_z_max = mpl5_data.field('ZMAX')
secondary_z_min = mpl5_data.field('SZMIN')
secondary_z_maz = mpl5_data.field('SZMAX')
primary_plus_z_min = mpl5_data.field('EZMIN')
primary_plus_z_max = mpl5_data.field('EZMAX')
#Select where the primary, secondary and colour enhanced targets are in these
#arrays
primary = np.where((manga_target_1 & 1024) !=0)[0]
colour_enhanced  = np.where((manga_target_1 & 4096) !=0)[0]
primary_plus = np.append(primary,colour_enhanced)
secondary = np.where((manga_target_1 & 2048) !=0)[0]
#Then we select their corresponding max and min redshift values
sample_z_max_pplus = np.append(primary_plus_z_max[primary],primary_plus_z_max[colour_enhanced])
sample_z_min_pplus  = np.append(primary_plus_z_min[primary],primary_plus_z_min[colour_enhanced])
sample_z_max_sec = secondary_z_maz[secondary]
sample_z_min_sec = secondary_z_min[secondary]
#Calculate the max and min comoving volume
Vmax = cosmo.comoving_volume(sample_z_max_pplus)
Vmin = cosmo.comoving_volume(sample_z_min_pplus)
Vmax_sec = cosmo.comoving_volume(sample_z_max_sec)
Vmin_sec = cosmo.comoving_volume(sample_z_min_sec)
#Then calculate the weights
V_weight = 1/np.array(Vmax-Vmin)
V_weight_sec = 1/(np.array(Vmax_sec-Vmin_sec)*0.769)
V_weight = np.append(V_weight,V_weight_sec)
#Mulityply the weights by a fiducial volume
weightsash = 1e6*V_weight
index_full_sample = np.append(np.append(primary,colour_enhanced),secondary)
mass_sec = mpl5_data.field('NSA_ELPETRO_MASS')[index_full_sample]
ancil_weights_sec = mpl5_data.field('ESWEIGHT')[index_full_sample]
my_weights_sec = weightsash

figtest, ax = plt.subplots()
area_weight = np.sum(ancil_weights_sec)/len(mass_sec)
my_area_weights = np.sum(ancil_weights_sec)/np.sum(my_weights_sec)
unweighted = ax.hist(mass_sec, weights = np.ones(len(mass_sec))*area_weight, histtype = 'step', color='r',label='Not Weighted')
weighted = ax.hist(mass_sec, weights = ancil_weights_sec, histtype = 'step', color = 'b',label='Weighted')
my_weights_hist = ax.hist(mass_sec, weights=my_weights_sec*my_area_weights, histtype ='step', color='k', label='My Weights')
ax.set_xlabel('$\log _{10}(\mathrm{mass})$', fontsize=16)
ax.set_ylabel('Frequency', fontsize = 14)
ax.legend()
figtest.tight_layout()
ax.set_title('All')
#figtest.savefig('/home/icgguest/Project/final figures/mass hist comparison with my weights only primary')






weight_full_sample = weightsash

whynottry = Table.read('MaNGA_targets_extNSA_tiled_ancillary(1).fits', format='fits')[index_full_sample]

mpl5_data = mpl5_data[index_full_sample]
wfs = Column(name = 'volume sample weights', data=weight_full_sample)
whynottry.add_columns([wfs])
whynottry.write('drpall-mpl5-volume-weights.fits', format='fits', overwrite=True)
