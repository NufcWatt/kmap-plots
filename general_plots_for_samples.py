#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:03:30 2017

@author: icgguest
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import cosmology
from marvin import config
from marvin.tools.maps import Maps
from marvin.utils.general.images import showImage
from marvin.utils.general.images import getImagesByList
import marvin.utils.plot.map as mapplot
import matplotlib.image as mpimg
config.setRelease('MPL-5')
config.mode = 'local'
config.download = True

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=71.8, Om0=0.273)
from astropy import units as u

#gz2sample uses WMAP cosmology of H0 = 71.8, Ωm =  0.273, ΩΛ = 0.727 https://arxiv.org/pdf/1308.3496v2.pdf




def open_fits_arrays(fits_file,variable_names,column_header_names,dictionary):
    #Adds values from a fits file to variable name(s) in a dictionary 
    for h in range(len(variable_names)):
        dictionary[variable_names[h]] = fits_file.field(column_header_names[h])
        
def remove_rows(array_name,array_deleting_from,value,condition,dictionary):
    #Removes rows of data from a in a defined arrays in a dictionary
    #by finding the indices of where the condition is true in the array 
    #deleting from
    if condition=='greater':
        where_to_remove = np.where(dictionary[array_deleting_from] > value)[0]
        
    if condition=='greater or equal to':
        where_to_remove = np.where(dictionary[array_deleting_from] >= value)[0]
    
    if condition=='less':
        where_to_remove = np.where(dictionary[array_deleting_from] < value)[0]
    
    if condition=='less or equal to':
        where_to_remove = np.where(dictionary[array_deleting_from] <= value)[0]
        
    if array_name == array_deleting_from:
        return dictionary[array_name]
    else:
        return np.delete(dictionary[array_name],where_to_remove,axis=None)
    

def number_of_votes_to_frac(start_dict,variable_names,num):
    #Divids each array in star_dict by num
    #Used for getting the vote fraction for each choice by dividing by
    #the number of classifications
    end_dict = {}
    for k in range(len(variable_names)):
        end_dict[variable_names[k]] = start_dict[variable_names[k]]/num
    return end_dict

def pair_matching(array1,array2):
    pair_arrays = np.zeros(len(array1))
    
    for k in range(len(array1)):
        if array1[k] < array2[k]:
            pair_arrays[k] = -np.sqrt(array1[k]**2 + array2[k]**2)
        else:
            pair_arrays[k] = np.sqrt(array1[k]**2 + array2[k]**2)
    return pair_arrays, np.unique(pair_arrays,return_index=True)

def kmaps_on_lambdar_ellipticity_plot(the_dict,index_array,axis):
    #This plots the stelvel maps for the indicies given in plateifu array
    #in the dictionary of your choice
    for muffin in range(len(index_array)):
        maps = Maps(plateifu=the_dict['plateifu'][index_array[muffin]])
        stelvel = maps['stellar_vel']
        stelvel_yeah = np.ma.array(stelvel.value,mask=stelvel.mask)
         #The maps are percentile clipped at 10% and 90% just like in marvin
         #The larger of the two is then set as the vmin and vmax in imshow to
         #centre the colourmap on zero
        cblow = np.percentile(stelvel_yeah.data[~stelvel_yeah.mask],10)
        cbup = np.percentile(stelvel_yeah.data[~stelvel_yeah.mask],90)
        cbmax = np.max(np.abs([cblow,cbup]))
        ellip = the_dict['ellipticity'][index_array[muffin]]
        lamb = the_dict['lambda r'][index_array[muffin]]
        if lamb != np.nan:
            axis.imshow(stelvel_yeah,origin='lower',extent=(ellip-0.05,ellip+0.05,lamb-0.05,lamb+0.05), cmap='RdBu_r',interpolation='nearest',vmin = -cbmax, vmax = cbmax)
            axis.plot(ellip,lamb,'ko',markersize=5)


def Galaxy_image_star_and_gas_maps(the_dict,index_array,savepath):
    #This makes a 3 panel subplot of the galaxy image, star velocity map and
    #gas velocity map with some metadata added about the galaxy
    #this is then saved in the filepath you give it with its file name being 
    #its plateifu number
    for cake in range(len(index_array)):
        maps = Maps(mangaid=the_dict['mangaid'][index_array[cake]])
        starvel = maps['stellar_vel']
        gas_vel = maps['emline_gvel_ha_6564']
        theimage = showImage(plateifu=the_dict['plateifu'][index_array[cake]],show_image=False,mode='remote')
        fige, axe = plt.subplots(1,3,figsize=(14,4))
        axe[0].imshow(theimage)
        axe[0].axis('off')
        for ax, map_ in zip(axe[1:],[starvel,gas_vel]):
            mapplot.plot(dapmap=map_,fig=fige,ax=ax)
        fige.tight_layout()
        fige.text(0.05,0.05,'Plateifu-'+str(the_dict['plateifu'][index_array[cake]]),ha='left',fontsize=10)
        fige.text(0.05,0.01,'$\lambda _R$:'+str(the_dict['lambda r'][index_array[cake]]),ha='left',fontsize=10)
        fige.text(0.15,0.05,'$\epsilon$:'+str(the_dict['ellipticity'][index_array[cake]]),ha='left',fontsize=10)
        fige.text(0.15,0.01,'$M_{\star}$:$10^{'+str(np.around(np.log10(the_dict['mass'][index_array[cake]]),decimals=3))+'}$$M_{\odot}$',ha='left',fontsize=10)
        plt.savefig(savepath+'Plateifu-'+str(the_dict['plateifu'][index_array[cake]]))
        plt.close()

plt.close('all')

#Opening the fits files
ancillary = fits.open('MaNGA_targets_extNSA_tiled_ancillary_lambda_r_GZ_plus_my_weights9.fits')
ancillary_data = ancillary[1].data
aggregated = fits.open('gzsamplezooMainSpeczMANGAMeta_mpl5weights2_lambda2.fits')
aggregated_data = aggregated[1].data
#Setting up the stars dictionary
variables_stars = ['regular rotation (stars)','kinematically distinct core (stars)','counter rotating core (stars)','kinematic twist (stars)','double peak (stars)','non rotating (stars)','disturbed (stars)','not enough data present cant tell (stars)']
column_headers_stars = ['Regular Rotation_2','Kinematically Distinct Core_2','Counter-Rotating Core_2','Kinematic Twist_2','Double Peak_2','Non-Rotating_2','Disturbed_2',"Not enough data present/ can't tell"]
#Setting up the gas dictionary
variables_gas = ['regular rotation (gas)','kinematically distinct core (gas)','counter rotating core (gas)','kinematic twist (gas)','double peak (gas)','non rotating (gas)','disturbed (gas)','not enough data present cant tell (gas)']
column_headers_gas = ['Regular Rotation_2a','Kinematically Distinct Core_2a','Counter-Rotating Core_2a','Kinematic Twist_2a','Double Peak_2a','Non-Rotating_2a','Disturbed_2a',"Not enough data present/ can't tell/ No gas detected"]
#Setting up for values to be appended onto both dictionaries
other_variables = ['volume weights','mass','vote fraction for features or disk','lambda r','ellipticity','petromag Mg','petromag Mr','FvS','no bulge','noticeable bulge','obvious bulge','dominant bulge','bar flag','no edge on fraction','bar fraction','mangaid','plateifu']
other_column_headers = ['ESWEIGHT','nsa_elpetro_mass','t01_smooth_or_features_a02_features_or_disk_debiased','lambda_r','ECOOELL','PETROMAG_MG','PETROMAG_MR','FvS','t05_bulge_prominence_a10_no_bulge_debiased ','t05_bulge_prominence_a11_just_noticeable_debiased ','t05_bulge_prominence_a12_obvious_debiased','t05_bulge_prominence_a13_dominant_debiased  ', 't03_bar_a06_bar_flag','t02_edgeon_a05_no_debiased','t03_bar_a06_bar_debiased','mangaid_2a','plateifu']
number_of_classifications = aggregated_data.field('Number of Classifications')
variabe_names = ['regular rotation','kinematically distinct core','counter rotating core', 'kinematic twist','double peak', 'non-rotating', 'disturbed',"not enoguh data present/can't tell"]
#Creating the dictionaries
variables_stars_all = np.append(variables_stars,other_variables)
variables_gas_all = np.append(variables_gas,other_variables)
stars_dict = {}
gas_dict = {}
open_fits_arrays(aggregated_data, variables_stars, column_headers_stars, stars_dict)
open_fits_arrays(aggregated_data,variables_gas,column_headers_gas,gas_dict)
stars_dict_frac = number_of_votes_to_frac(stars_dict,variables_stars,number_of_classifications)
gas_dict_frac = number_of_votes_to_frac(gas_dict,variables_gas,number_of_classifications)
#These dictionaries have mathcing array lengths
stars_dict_frac_2 = number_of_votes_to_frac(stars_dict,variables_stars,number_of_classifications)
gas_dict_frac_2 = number_of_votes_to_frac(gas_dict,variables_gas,number_of_classifications)
open_fits_arrays(aggregated_data, other_variables, other_column_headers, stars_dict_frac)
open_fits_arrays(aggregated_data, other_variables, other_column_headers, gas_dict_frac)
open_fits_arrays(aggregated_data, other_variables, other_column_headers, stars_dict_frac_2)
open_fits_arrays(aggregated_data, other_variables, other_column_headers, gas_dict_frac_2)
open_fits_arrays(aggregated_data, other_variables, other_column_headers, stars_dict)



fg, axh = plt.subplots(3,3,figsize=(15,15))
mosaic_array = np.zeros(9)
mosaic_array = mosaic_array.astype(str)
for g in range(len(mosaic_array)):
    try:
        if g==5:
            temp = np.where(gas_dict_frac[variables_gas_all[g]]==1)[0]
            mosaic_array[g] = gas_dict_frac['plateifu'][temp[2]]
        else:
            temp = np.where(gas_dict_frac[variables_gas_all[g]]==1)[0]
            mosaic_array[g] = gas_dict_frac['plateifu'][temp[0]]
    except IndexError:
        if g==2:
            temp = np.where(gas_dict_frac[variables_gas_all[g]]>0.6)[0]
            mosaic_array[g] = gas_dict_frac['plateifu'][temp[0]]
        else:
            temp = np.where(gas_dict_frac[variables_gas_all[g]]>0.6)[0]
            mosaic_array[g] = gas_dict_frac['plateifu'][temp[0]]
gh=0
for h in range(3):
    for hh in range(3):
        if h==2 & hh==2:
            break
        mm = Maps(plateifu=mosaic_array[gh])
        gas_map = mm['emline_gvel_ha_6564']
        gas_map_value= gas_map.value
        gas_map_mask = gas_map.mask
        
        gas_map_yeah = np.ma.array(gas_map_value,mask=gas_map_mask)
        cblow = np.percentile(gas_map_yeah.data[~gas_map_yeah.mask],10)
        cbup = np.percentile(gas_map_yeah.data[~gas_map_yeah.mask],90)
        cbmax = np.max(np.abs([cblow,cbup]))
        axh[h,hh].imshow(gas_map_yeah,origin='lower', cmap='RdBu_r',interpolation='nearest',vmin = -cbmax, vmax = cbmax)
        axh[h,hh].set_title(variabe_names[gh])
        axh[h,hh].axis('off')
        gh +=1
axh[2,2].axis('off')
fg.tight_layout()
    
      

for b in range(len(variables_gas_all)):
    stars_dict_frac[variables_stars_all[b]] = remove_rows(variables_stars_all[b],'not enough data present cant tell (stars)',0.5,'greater',stars_dict_frac)
    gas_dict_frac[variables_gas_all[b]] = remove_rows(variables_gas_all[b],'not enough data present cant tell (gas)',0.5,'greater',gas_dict_frac)

fig1, ax = plt.subplots(1,2,sharey=True, figsize=(12,6))
ax[0].hist2d(stars_dict_frac['regular rotation (stars)'],stars_dict_frac['vote fraction for features or disk'],bins=[4,5],cmap='Reds',weights=stars_dict_frac['volume weights'])
ax[1].hist2d(gas_dict_frac['regular rotation (gas)'],gas_dict_frac['vote fraction for features or disk'],bins=[4,5], cmap='Reds', weights=gas_dict_frac['volume weights'])
ax[0].set_xlim([0,1])
ax[0].set_ylim([0,1])
ax[0].set_xlabel('Vote fraction for regular rotation (stars)')
ax[1].set_xlabel('Vote fraction for regular rotation (gas)')
ax[0].set_ylabel('Vote fraction for features or disk')
fig1.tight_layout()
fig1.savefig('/home/icgguest/Project/final figures/hist2d_rotation_features')



fig2, ax = plt.subplots(3)
var1 = variables_stars_all[0]
var2 = variables_stars_all[10]
mass_low_reg = np.where(stars_dict_frac['mass'] < 10**10)[0]
mass_med_reg = np.zeros(len(stars_dict_frac['mass']))
for c in range(len(stars_dict_frac['mass'])):
    if 10**10 <= stars_dict_frac['mass'][c] < 10**10.3:
        mass_med_reg[c] = c+1
mass_med_reg = np.delete(mass_med_reg,np.where(mass_med_reg==0)[0],axis=None) -1
mass_med_reg = mass_med_reg.astype(int)
mass__high_reg = np.where(stars_dict_frac['mass'] >=10**10.3)[0]
ax[0].hist2d(stars_dict_frac[var1][mass_low_reg],stars_dict_frac[var2][mass_low_reg], bins=[5,5], cmap='Reds',weights=stars_dict_frac['volume weights'][mass_low_reg])
ax[1].hist2d(stars_dict_frac[var1][mass_med_reg],stars_dict_frac[var2][mass_med_reg], bins=[5,5], cmap='Reds',weights=stars_dict_frac['volume weights'][mass_med_reg])
ax[2].hist2d(stars_dict_frac[var1][mass__high_reg],stars_dict_frac[var2][mass__high_reg], bins=[5,5], cmap='Reds',weights=stars_dict_frac['volume weights'][mass__high_reg])
fig2.text(0.5, 0.04, 'vote fraction for Regular Rotation (stars)', ha='center')
fig2.text(0.04, 0.5, 'vote fraction for features or disk', va='center', rotation='vertical')
ax[0].set_title('Low mass ($M < 10^{9.3}M_{\odot}$)')
ax[1].set_title('Medium mass ($10^{9.3}M_{\odot} \leq M < 10^{10.4}M_{\odot}$)')
ax[2].set_title('High mass ($10^{10.3}M_{\odot} \leq M$)')



fig3, ax = plt.subplots(2,figsize = (9,12))
mass = (ancillary_data.field('NSA_ELPETRO_MASS'))
ancil_weights = ancillary_data.field('ESWEIGHT')
my_weights = ancillary_data.field('volume sample weights')
primary_index = np.where((ancillary_data.field('MANGA_TARGET1')&1024) != 0)[0]
mass_prim = mass[primary_index]
ancil_weights_prim = ancil_weights[primary_index]
my_weights_prim = my_weights[primary_index]

sec_index = np.where((ancillary_data.field('MANGA_TARGET1')&2048) != 0)[0]
ce_index = np.where((ancillary_data.field('MANGA_TARGET1')&4096)!=0)[0]
all_index = np.append(np.append(primary_index,ce_index),sec_index)
mass_sec = mass[all_index]
ancil_weights_sec = ancil_weights[all_index]
my_weights_sec = my_weights[all_index]




#no_ancil_weights = np.where(ancil_weights<-900)[0]
#ancil_weights = np.delete(ancil_weights,no_ancil_weights,axis=None)
#mass = np.delete(mass,no_ancil_weights,axis=None)
#my_weights = np.delete(my_weights,no_ancil_weights,axis=None)
#
#mass_mask = np.isnan(mass)
#mass = mass[~mass_mask]
#ancil_weights = ancil_weights[~mass_mask]
#my_weights = my_weights[~mass_mask]

ax[0].hist(mass, color=  'r')
ax[1].hist(mass, weights=ancil_weights,color =  'r')
ax[0].set_xlabel('$log_{10}(mass)$', fontsize = 16)
ax[1].set_xlabel('$log_{10}(mass)$', fontsize = 16)
ax[0].set_ylabel('frequency', fontsize = 14)
ax[1].set_ylabel('weighted frequency', fontsize =14)
fig3.tight_layout()
fig3.savefig('/home/icgguest/Project/final figures/mass hists')

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
ax.set_title('Primary')
figtest.savefig('/home/icgguest/Project/final figures/mass hist comparison with my weights all')



#Set up so it can remove the same rows from both stars_dict_frac_2 and 
#gas_dict_frac_2
remove_cant_tell_from_stars = np.where(stars_dict_frac_2['not enough data present cant tell (stars)'] > 0.5)
remove_cant_tell_from_gas = np.where(gas_dict_frac_2['not enough data present cant tell (gas)'] > 0.5)
remove_cant_tell_all = np.unique(np.append(remove_cant_tell_from_stars,remove_cant_tell_from_gas),return_index = True)
fig4, ax = plt.subplots()
variable_1 = variables_stars_all[0]
variable_2 = variables_gas_all[0]
#Removing same rows from both dictionaries
for b in range(len(stars_dict_frac_2)):
    stars_dict_frac_2[variables_stars_all[b]] = np.delete(stars_dict_frac_2[variables_stars_all[b]],remove_cant_tell_all[1],axis=None)
    gas_dict_frac_2[variables_gas_all[b]] = np.delete(gas_dict_frac_2[variables_gas_all[b]],remove_cant_tell_all[1],axis=None)
    
pair_arrays, pairs = pair_matching(stars_dict_frac_2[variable_1],gas_dict_frac_2[variable_2])
#Sum of the volume limited weighted points at each unique position
weighted_number_of_points = np.zeros(len(pairs[1]))
for cake in range(len(pairs[1])):
    weighted_number_of_points[cake] = np.sum(stars_dict_frac_2['volume weights'][np.where(pair_arrays[pairs[1][cake]]==pair_arrays)])

ax.scatter(stars_dict_frac_2[variable_1][pairs[1]],gas_dict_frac_2[variable_2][pairs[1]],s=weighted_number_of_points*30,c='r')
if variable_1 == 'mass':
    ax.set_xscale('log')
if variable_2 == 'mass':
    ax.set_yscale('log')
ax.set_xlabel('Vote fraction for '+variable_1,fontsize=14)
ax.set_ylabel('Vote Fraction for '+variable_2,fontsize=14)
fig4.tight_layout()
fig4.savefig('/home/icgguest/Project/final figures/scaling_scatter_plot_regular_rotation')
    


#lambda r vs ellipticity plot
fig5, ax = plt.subplots()
gr_colour = stars_dict_frac['petromag Mg'] - stars_dict_frac['petromag Mr']
cb1 = ax.scatter(stars_dict_frac['ellipticity'],stars_dict_frac['lambda r'],c = stars_dict_frac['regular rotation (stars)'],cmap='rainbow')
ellipticity_values = np.linspace(0,0.9,1000)
ax.plot(ellipticity_values,0.31*np.sqrt(ellipticity_values),'k--',linewidth=2)
ax.set_xlabel('$\epsilon$',fontsize=20)
ax.set_ylabel('$\lambda _R$',fontsize=16)
cb = fig5.colorbar(cb1, ax=ax)
ax.set_xlim([0,0.9])
ax.set_ylim([0,0.9])
cb.set_label('Vote fraction for regular rotation (stars)')
fig5.tight_layout()
fig5.savefig('/home/icgguest/Project/final figures/ellipticity_lambda_r_regular_rotation_stars')



#Plot of kinematic maps on the lambda r vs ellipticity plot
fig6, ax = plt.subplots()
ellipticity_values = np.linspace(0,0.9,1000)
ax.plot(ellipticity_values,0.31*np.sqrt(ellipticity_values),'k--',linewidth=2)
ax.set_xlabel('$\epsilon$',fontsize=20)
ax.set_ylabel('$\lambda _R$',fontsize=16)
varx = variables_stars_all[10]
vary = variables_stars_all[15]
#Selects indices where the arguments are true
stars_index = np.zeros(len(stars_dict_frac[varx]))
for scone in range(len(stars_dict_frac[varx])):
    if stars_dict_frac[varx][scone] > 0.8 and stars_dict_frac[vary][scone]==0:
        stars_index[scone] = scone +1
stars_index = np.delete(stars_index,np.where(stars_index==0)[0],axis=None)-1
stars_index = stars_index.astype(int)
kmaps_on_lambdar_ellipticity_plot(stars_dict_frac,stars_index,ax)
ax.set_xlim([0,np.nanmax(stars_dict_frac['ellipticity'][stars_index]+0.05)])
ax.set_ylim([0,np.nanmax(stars_dict_frac['lambda r'][stars_index]+0.05)])
fig6.tight_layout()
fig6.savefig('/home/icgguest/Project/final figures/slow_features')

fig7, ax = plt.subplots()
#kinematic_twist_mask_bar = gas_dict_frac['vote fraction for features or disk'] > 0.5 and gas_dict_frac['no edge on fraction'] > 0.5 and gas_dict_frac['bar fraction'] >0.5
#kinematic_twist_mask_not_bar = gas_dict_frac['vote fraction for features or disk'] > 0.5 and gas_dict_frac['no edge on fraction'] > 0.5 and gas_dict_frac['bar fraction'] <=0.5
bar_twist_or_not = 0
not_bar_twist_or_not = 0
bar_count = 0
not_bar_count = 0
for pie in range(len(gas_dict_frac['kinematic twist (gas)'])):
    if gas_dict_frac['vote fraction for features or disk'][pie] > 0.5:
        if gas_dict_frac['no edge on fraction'][pie] > 0.5:
            if gas_dict_frac['bar fraction'][pie] > 0.5:
                bar_count +=1
                if (gas_dict_frac['kinematic twist (gas)'])[pie] > 0.5:
                    bar_twist_or_not += 1
            elif gas_dict_frac['bar fraction'][pie] < 0.2:
                not_bar_count += 1
                if (gas_dict_frac['kinematic twist (gas)'])[pie] > 0.5:
                    not_bar_twist_or_not +=1
bar_twist_perc = np.array([bar_twist_or_not/bar_count,np.sqrt(bar_twist_or_not)/bar_count])*100
ar_twist_perc = np.array([not_bar_twist_or_not/not_bar_count,np.sqrt(not_bar_twist_or_not)/not_bar_count])*100
print('Bars with twists: '+str(bar_twist_perc[0])+'%'+'+-'+str(bar_twist_perc[1])+'%')
print('Non-bars with twists: '+str(ar_twist_perc[0])+'%'+'+-'+str(ar_twist_perc[1])+'%')

primary_only_weights = ancillary_data.field('SWEIGHT')[sec_index]
ESW = ancillary_data.field('ESWEIGHT')[sec_index]
my_prim_calc_weights = 1e6/(cosmo.comoving_volume(ancillary_data.field('SZMAX')[sec_index])-cosmo.comoving_volume(ancillary_data.field('SZMIN')[sec_index])*0.769)
area_norm = np.sum(primary_only_weights)/np.sum(my_prim_calc_weights)
area_norm1 = np.sum(primary_only_weights)/np.sum(ESW)
mass_sec = (ancillary_data.field('NSA_ELPETRO_MASS'))[sec_index]
figh, ax = plt.subplots()
ax.hist(mass_sec,weights=primary_only_weights,histtype='step',color='b',label='Weights')
ax.hist(mass_sec,weights=my_prim_calc_weights*area_norm,histtype='step',color='k',label='My Weights')
#ax.hist(mass_sec,weights=ESW*area_norm1,histtype='step',color='r',label='My weights from pplus')
ax.set_xlabel('$\log _{10}(\mathrm{mass})$', fontsize=16)
ax.set_ylabel('Frequency', fontsize = 14)
ax.legend()
figh.tight_layout()
#ax.set_title('Primary only sample')
#figh.savefig('/home/icgguest/Project/final figures/primary comp')

figg, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(ancillary_data.field('ZMIN')[primary_index],ancillary_data.field('EZMIN')[primary_index],'r.')
ax[1].plot(ancillary_data.field('ZMAX')[primary_index],ancillary_data.field('EZMAX')[primary_index],'r.')
ax[0].set_xlabel('ZMIN')
ax[0].set_ylabel('EZMIN')
ax[1].set_xlabel('ZMAX')
ax[1].set_ylabel('EZMAX')



#savepath = '/home/icgguest/Project/data/Galaxy images and maps/slow features/'
#Galaxy_image_star_and_gas_maps(stars_dict_frac,stars_index,savepath)

