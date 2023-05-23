# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:05:33 2021

@author: jeanh
"""


import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
#os.environ["pRT_input_data_path"] = "/scratch/software/petitRADTRANS/petitRADTRANS/input_data"
os.environ["pRT_input_data_path"] = "/home/ipa/quanz/shared/petitRADTRANS/input_data"
#sys.path.append("/scratch/software/petitRADTRANS/")
#sys.path.append("/home/ipa/quanz/shared/petitRADTRANS/")

from doubleRetrieval.forward_model import ForwardModel
from doubleRetrieval.plotting import plot_data,plot_profiles
from doubleRetrieval.util import convert_units,save_photometry,save_spectrum,trim_spectrum,calc_cov_matrix,save_spectra,save_lines
from doubleRetrieval.rebin import rebin_to_CC,rebin,doppler_shift,rebin_to_RES,rebin_to_PHOT,only_gaussian_filter
from doubleRetrieval.data2 import Data
from doubleRetrieval.rotbroad_utils import add_rot_broad,cut_spectrum
from sim_config2 import *

#from petitRADTRANS import radtrans as rt
#from petitRADTRANS import nat_cst as nc

import csv
import pickle
import os

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

import shutil

cwd = os.getcwd()
source = cwd+'/sim_config2.py'
destination = OUTPUT_DIR+'/sim_config2_copy.py'
shutil.copyfile(source,destination)
print('Config file copied')


# redesign how to call data object for simulation
# idea: import real data, and copy format

data_obj = Data(
                data_dir = None,
                use_sim_files = ['CC','RES','PHOT'],
                PHOT_flux_format = 2,
                PHOT_filter_dir = FILTER_DIR,
                PHOT_flux_dir = PHOT_DATA_FILE,
                CC_data_dir=CC_WVL_DIR, # if sinfoni, then WVL_FILE, if CRIRES then CC_WVL_DIR
                RES_data_dir=RES_WVL_DIR,
                RES_err_dir=RES_COV_DIR
                )

data_obj.distribute_FMs()

lbl_itvls = data_obj.disjoint_lbl_intervals
CC_to_lbl_itvls = data_obj.CC_to_lbl_intervals
RES_to_lbl_itvls = data_obj.RES_to_lbl_intervals
lbl_intervals_max_step = data_obj.disjoint_lbl_intervals_max_stepsize

wlen_borders_ck = data_obj.ck_FM_interval

do_scat_emis = False
cloud_species = []
clouds = None
if len(CLOUDS_PARAMS.keys()) > 0:
    if CLOUDS_PARAMS['model'] == 'ackermann':
        print('USING CLOUDS')
        do_scat_emis = False
        cloud_species = CLOUDS_OPACITIES
        clouds = 'ackermann'
print(cloud_species)
forwardmodel_ck = ForwardModel(
     wlen_borders = wlen_borders_ck,
     max_wlen_stepsize = wlen_borders_ck[1]/1000,
     mode = 'c-k',
     line_opacities = ABUNDANCES,
     clouds = clouds,
     cloud_species = cloud_species.copy(),
     do_scat_emis = do_scat_emis,
     model = MODEL,
     max_RV = 0,
     max_winlen = 0
     )
forwardmodel_ck.calc_rt_obj(lbl_sampling = None)

wlen_borders_lbl,max_wlen_stepsize = data_obj.wlen_details_lbl()
# wlen stepsize actually only relevant for CC spectrum
max_wlen_stepsize = data_obj.max_stepsize(data_obj.getCCSpectrum()[0])
print(cloud_species,clouds)
forwardmodel_lbl = ForwardModel(
     wlen_borders = wlen_borders_lbl,
     max_wlen_stepsize = max_wlen_stepsize,
     mode = 'lbl',
     line_opacities = ABUNDANCES,
     clouds = clouds,
     cloud_species = cloud_species,
     do_scat_emis = do_scat_emis,
     model = MODEL,
     max_RV = 0,
     max_winlen = WINDOW_LENGTH_LBL
     )
forwardmodel_lbl.calc_rt_obj(lbl_sampling = LBL_SAMPLING)

wlen_ck,flux_ck = forwardmodel_ck.calc_spectrum(
        ab_metals = AB_METALS,
        temp_params = TEMP_PARAMS,
        clouds_params = CLOUDS_PARAMS,
        external_pt_profile = None)

wlen_lbl,flux_lbl,pressures,temperatures,abundances = forwardmodel_lbl.calc_spectrum(
        ab_metals = AB_METALS,
        temp_params = TEMP_PARAMS,
        clouds_params = CLOUDS_PARAMS,
        external_pt_profile = None,
        return_profiles = True)

plot_profiles(
            pressures,
            temperatures = temperatures,
            abundances = abundances,
            output_dir=OUTPUT_DIR,
            fontsize=20)


CC_wlen_data,CC_flux_data = data_obj.getCCSpectrum()

if INSTRUMENT == 'ERIS':
    if len(CC_wlen_data.keys()) == 1:
        print('CONVERTING TO ERIS RESOLUTION')
        sinf_key = list(CC_wlen_data.keys())[0]
        print('SINFONI BINS: ',len(CC_wlen_data[sinf_key]))
        sinfoni_resolution = CC_wlen_data[sinf_key][0]/(CC_wlen_data[sinf_key][1]-CC_wlen_data[sinf_key][0])
        eris_resolution = 2*sinfoni_resolution
        dlambda = CC_wlen_data[sinf_key][0]/eris_resolution
        number_eris_bins = int((CC_wlen_data[sinf_key][-1]-CC_wlen_data[sinf_key][0])/dlambda)
        CC_wlen_data[sinf_key] = np.linspace(CC_wlen_data[sinf_key][0],CC_wlen_data[sinf_key][-1],number_eris_bins)
        print('ERIS BINS: ',len(CC_wlen_data[sinf_key]))

if NOISE != 0 or SNR != 0:
    max_stepsize = data_obj.max_stepsize(CC_wlen_data)
    wlen_bords = data_obj.min_max_range(CC_wlen_data)
    print(wlen_bords,max_stepsize)
    wvl_template = np.linspace(wlen_bords[0],wlen_bords[1],int((wlen_bords[1]-wlen_bords[0])/max_stepsize))
    CC_wlen_removed,CC_flux_removed,sgfilter,CC_wlen_rebin,CC_flux_rebin = rebin_to_CC(wlen_lbl,flux_lbl,wvl_template,win_len = WINDOW_LENGTH_LBL,method='datalike',filter_method = 'only_gaussian',convert = CONVERT_SINFONI_UNITS,log_R=TEMP_PARAMS['log_R'],distance=DISTANCE)
    
    std_flux = np.std(CC_flux_removed)
    noise_std = NOISE*std_flux
    
    if SNR != 0:
        noise_std = np.mean(CC_flux_rebin)/SNR
    print('noise',noise_std)
    print('Relative noise',noise_std/std_flux)

CC_wlen_shifted,CC_flux_shifted = doppler_shift(wlen_lbl,flux_lbl,RV)
CC_wlen_removed,CC_flux_removed,CC_wlen_rebin,CC_flux_rebin,sgfilter = {},{},{},{},{}

if CONVERT_SINFONI_UNITS:
    CC_wlen_shifted,CC_flux_shifted = convert_units(CC_wlen_shifted,CC_flux_shifted,TEMP_PARAMS['log_R'],DISTANCE)
else:
    CC_wlen_shifted = 1e4*CC_wlen_shifted

for key in CC_wlen_data.keys():
    
    # cut what I don't need
    """
    keeping = 10000
    if OUTPUT_FORMAT == 'datalike':
        keeping = 2000
    
    CC_wlen_cut,CC_flux_cut = trim_spectrum(CC_wlen_shifted,CC_flux_shifted,CC_wlen_data[key],threshold=10000,keep=keeping)
    """
    # rebinning
    CC_wlen_rebin[key],CC_flux_rebin[key] = rebin(CC_wlen_shifted,CC_flux_shifted,CC_wlen_data[key])
    
    if PHYSICAL_PARAMS['rot_vel'] != 0:
        CC_wlen_rebin[key],CC_flux_rebin[key] = add_rot_broad(CC_wlen_rebin[key],CC_flux_rebin[key],PHYSICAL_PARAMS['rot_vel'],method='fast',edgeHandling = 'cut')
    
    if NOISE != 0 or SNR != 0:
        # noise
        CC_flux_rebin[key] = CC_flux_rebin[key] + np.random.normal(loc=0,scale=noise_std,size=len(CC_flux_rebin[key]))
        print('NOISE ADDED')
        
    # remove continuum with filter
    wlen_after = None
    if OUTPUT_FORMAT == 'datalike':
        wlen_after = CC_wlen_data[key]
    CC_wlen_removed[key],CC_flux_removed[key],sgfilter[key] = only_gaussian_filter(CC_wlen_rebin[key],CC_flux_rebin[key],sigma=WINDOW_LENGTH_LBL,wlen_after=wlen_after)

save_spectra(CC_wlen_removed,CC_flux_removed,save_dir= OUTPUT_DIR + '/CC_spectrum',save_name='')

RES_wlen_data,RES_flux_data,RES_cov_err,RES_inverse_cov,RES_flux_err_data = data_obj.getRESSpectrum()
RES_wlen_rebin,RES_flux_rebin,RES_cov={},{},{}

CC_wlen_shifted,CC_flux_shifted = doppler_shift(wlen_lbl,flux_lbl,RV)
print('mean flux before resampling RES data',np.mean(CC_flux_shifted))
for key in RES_wlen_data.keys():
    RES_flux_data[key] = np.array(RES_flux_data[key])
    
    max_resolution = max(RES_wlen_data[key][1:]/(RES_wlen_data[key][1:]-RES_wlen_data[key][:-1]))
    if max_resolution < 900:
        RES_wlen_rebin[key],RES_flux_rebin[key] = rebin_to_RES(wlen_ck,flux_ck,RES_wlen_data[key],TEMP_PARAMS['log_R'],DISTANCE)
    else:
        RES_wlen_rebin[key],RES_flux_rebin[key] = rebin_to_RES(CC_wlen_shifted,CC_flux_shifted,RES_wlen_data[key],TEMP_PARAMS['log_R'],DISTANCE)
    print('mean flux after resampling RES data',np.mean(RES_flux_rebin[key]))
    RES_cov[key] = calc_cov_matrix(cov_data = RES_cov_err[key],flux_sim = RES_flux_rebin[key],flux_data = RES_flux_data[key])
    if key == 'SPHERE_IFS':
        print('USING OTHER UNCERTAINTY FOR SPHERE DATA')
        RES_cov[key] = np.diag(RES_flux_rebin[key]*RES_flux_rebin[key]*(10/100)**2)
    RES_flux_err = [np.sqrt(RES_cov[key][i][i]) for i in range(len(RES_cov[key]))]
    
    
    if NOISE_RES != 0 and key != 'SPHERE_IFS':
        RES_flux_rebin[key] = RES_flux_rebin[key] + np.random.normal(loc=0,scale=RES_flux_err)
        print('RES noise added for',key)
     
    save_lines(RES_cov[key],save_dir = OUTPUT_DIR +'/RES_error',save_name='/'+str(key))
save_spectra(RES_wlen_rebin,RES_flux_rebin,save_dir= OUTPUT_DIR+'/RES_spectrum',save_name='')


PHOT_flux,PHOT_flux_err,filt,filt_func,filt_mid,filt_width = data_obj.getPhot() #dictionaries
model_photometry,model_photometry_err,phot_midpoint,phot_width,PHOT_wlen,PHOT_flux = rebin_to_PHOT(wlen_ck,flux_ck,filt_func,log_R=TEMP_PARAMS['log_R'],distance=DISTANCE,phot_flux_data = PHOT_flux,phot_flux_err_data = PHOT_flux_err)
save_photometry(model_photometry,model_photometry_err,phot_midpoint,phot_width,save_dir=OUTPUT_DIR)
save_spectrum(PHOT_wlen,PHOT_flux,save_dir=OUTPUT_DIR,save_name='/spectrum_cksim')

plot_data(CONFIG_DICT,
            CC_wlen = CC_wlen_removed,
            CC_flux = CC_flux_removed,
            CC_wlen_w_cont = CC_wlen_rebin,
            CC_flux_w_cont = CC_flux_rebin,
            model_CC_wlen = None,
            model_CC_flux = None,
            sgfilter = sgfilter,
            RES_wlen = RES_wlen_rebin,
            RES_flux = RES_flux_rebin,
            RES_flux_err = RES_cov,
            model_RES_wlen = None,
            model_RES_flux = None,
            PHOT_midpoint = phot_midpoint,
            PHOT_width = phot_width,
            PHOT_flux = model_photometry,
            PHOT_flux_err = model_photometry_err,
            PHOT_filter = filt,
            PHOT_sim_wlen = PHOT_wlen,
            PHOT_sim_flux = PHOT_flux,
            model_PHOT_flux = None,
            output_file = OUTPUT_DIR,
            plot_name='plot')
"""

plot_data(CONFIG_DICT,
            PHOT_sim_wlen = PHOT_wlen,
            PHOT_sim_flux = PHOT_flux,
            output_file = OUTPUT_DIR,
            plot_name='plot_cksim')

plot_data(CONFIG_DICT,
            CC_wlen = CC_wlen_removed,
            CC_flux = CC_flux_removed,
            CC_wlen_w_cont = CC_wlen_rebin,
            CC_flux_w_cont = CC_flux_rebin,
            model_CC_wlen = None,
            model_CC_flux = None,
            sgfilter = sgfilter,
            output_file = OUTPUT_DIR,
            plot_name='plot_CC')

plot_data(CONFIG_DICT,
            RES_wlen = RES_wlen_rebin,
            RES_flux = RES_flux_rebin,
            RES_flux_err = RES_cov,
            output_file = OUTPUT_DIR,
            plot_name='plot_RES')

plot_data(CONFIG_DICT,
            PHOT_midpoint = phot_midpoint,
            PHOT_width = phot_width,
            PHOT_flux = model_photometry,
            PHOT_flux_err = model_photometry_err,
            PHOT_filter = filt,
            model_PHOT_flux = None,
            output_file = OUTPUT_DIR,
            plot_name='plot_PHOT')
"""
