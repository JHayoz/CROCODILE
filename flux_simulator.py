# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:05:33 2021

@author: Jean Hayoz
"""
print('IMPORTING LIBRARIES')
import os
os.environ['OMP_NUM_THREADS'] = '1'
from sim_config import *
print('    CONFIG READ')

import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
os.environ["pRT_input_data_path"] = "/home/ipa/quanz/shared/petitRADTRANS/input_data"

from core.forward_model import ForwardModel
from core.plotting import plot_data,plot_profiles
from core.util import convert_units,save_photometry,save_spectrum,trim_spectrum,calc_cov_matrix,save_spectra,save_lines
from core.rebin import rebin_to_CC,rebin,doppler_shift,rebin_to_RES,rebin_to_PHOT,only_gaussian_filter
from core.data import Data
from core.rotbroad_utils import add_rot_broad

import csv
import pickle

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

import shutil

cwd = os.getcwd()
source = cwd+'/sim_config.py'
destination = OUTPUT_DIR+'/sim_config_copy.py'
shutil.copyfile(source,destination)
print('Config file copied')

# importing data to copy format
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
        wlen_borders=wlen_borders_ck,
        max_wlen_stepsize=0.01,
        mode='c-k',
        line_opacities=ABUNDANCES,
        cont_opacities= ['H2-H2','H2-He'],
        rayleigh_scat = ['H2','He'],
        cloud_model = clouds,
        cloud_species = cloud_species.copy(),
        do_scat_emis = do_scat_emis,
        chem_model = CHEM_MODEL,
        temp_model = TEMP_MODEL,
        max_RV = RV,
        max_winlen = WINDOW_LENGTH_LBL,
        include_H2 = True,
        only_include = 'all')
forwardmodel_ck.calc_rt_obj(lbl_sampling = None)

wlen_borders_lbl,max_wlen_stepsize = data_obj.wlen_details_lbl()
max_wlen_stepsize = data_obj.max_stepsize(data_obj.getCCSpectrum()[0])
print(cloud_species,clouds)
forwardmodel_lbl = ForwardModel(wlen_borders=wlen_borders_lbl,
        max_wlen_stepsize=max_wlen_stepsize,
        mode='lbl',
        line_opacities=ABUNDANCES,
        cont_opacities= ['H2-H2','H2-He'],
        rayleigh_scat = ['H2','He'],
        cloud_model = clouds,
        cloud_species = cloud_species.copy(),
        do_scat_emis = do_scat_emis,
        chem_model = CHEM_MODEL,
        temp_model = TEMP_MODEL,
        max_RV = RV,
        max_winlen = WINDOW_LENGTH_LBL,
        include_H2 = True,
        only_include = 'all')
forwardmodel_lbl.calc_rt_obj(lbl_sampling = LBL_SAMPLING)

wlen_ck,flux_ck = forwardmodel_ck.calc_spectrum(
        CHEM_MODEL_PARAMS,
        TEMP_MODEL_PARAMS,
        CLOUDS_PARAMS,
        PHYSICAL_PARAMS,
        external_pt_profile = None,
        return_profiles = False
        )

wlen_lbl,flux_lbl,pressures,temperatures,abundances = forwardmodel_lbl.calc_spectrum(
        CHEM_MODEL_PARAMS,
        TEMP_MODEL_PARAMS,
        CLOUDS_PARAMS,
        PHYSICAL_PARAMS,
        external_pt_profile = None,
        return_profiles = True
        )

plot_profiles(
            pressures,
            temperatures = temperatures,
            abundances = abundances,
            output_dir=OUTPUT_DIR,
            fontsize=20)


CC_wlen_data,CC_flux_data = data_obj.getCCSpectrum()

if NOISE != 0 or SNR != 0:
    max_stepsize = data_obj.max_stepsize(CC_wlen_data)
    wlen_bords = data_obj.min_max_range(CC_wlen_data)
    print(wlen_bords,max_stepsize)
    wvl_template = np.linspace(wlen_bords[0],wlen_bords[1],int((wlen_bords[1]-wlen_bords[0])/max_stepsize))
    CC_wlen_removed,CC_flux_removed,sgfilter,CC_wlen_rebin,CC_flux_rebin = rebin_to_CC(wlen_lbl,flux_lbl,wvl_template,win_len = WINDOW_LENGTH_LBL,method='datalike',filter_method = 'only_gaussian',convert = True,log_R=TEMP_MODEL_PARAMS['log_R'],distance=DISTANCE)
    
    std_flux = np.std(CC_flux_removed)
    noise_std = NOISE*std_flux
    
    if SNR != 0:
        noise_std = np.mean(CC_flux_rebin)/SNR
    print('noise',noise_std)
    print('Relative noise',noise_std/std_flux)

CC_wlen_shifted,CC_flux_shifted = doppler_shift(wlen_lbl,flux_lbl,RV)
CC_wlen_removed,CC_flux_removed,CC_wlen_rebin,CC_flux_rebin,sgfilter = {},{},{},{},{}

CC_wlen_shifted,CC_flux_shifted = convert_units(CC_wlen_shifted,CC_flux_shifted,TEMP_MODEL_PARAMS['log_R'],DISTANCE)

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
        RES_wlen_rebin[key],RES_flux_rebin[key] = rebin_to_RES(wlen_ck,flux_ck,RES_wlen_data[key],TEMP_MODEL_PARAMS['log_R'],DISTANCE)
    else:
        RES_wlen_rebin[key],RES_flux_rebin[key] = rebin_to_RES(CC_wlen_shifted,CC_flux_shifted,RES_wlen_data[key],TEMP_MODEL_PARAMS['log_R'],DISTANCE)
    print('mean flux after resampling RES data',np.mean(RES_flux_rebin[key]))
    RES_cov[key] = calc_cov_matrix(cov_data = RES_cov_err[key],flux_sim = RES_flux_rebin[key],flux_data = RES_flux_data[key])
    if key == 'SPHERE_IFS':
        print('USING OTHER UNCERTAINTY FOR SPHERE DATA')
        RES_cov[key] = np.diag(RES_flux_rebin[key]*RES_flux_rebin[key]*(NOISE_RES)**2)
    RES_flux_err = [np.sqrt(RES_cov[key][i][i]) for i in range(len(RES_cov[key]))]
    if ADD_RANDOM_SCATTER:
        if key == 'SPHERE_IFS':
            RES_flux_rebin[key] = RES_flux_rebin[key] + np.random.normal(loc=0,scale=RES_flux_err)
            print('RES noise added for',key,' using normal iid noise')

        if NOISE_RES != 0 and key != 'SPHERE_IFS':
            if not USE_COV:
                RES_flux_rebin[key] = RES_flux_rebin[key] + np.random.normal(loc=0,scale=RES_flux_err)
                print('RES noise added for',key,' using normal iid noise')
            else:
                RES_flux_rebin[key] = RES_flux_rebin[key] + np.random.multivariate_normal(mean=np.zeros_like(RES_flux_rebin[key]),cov=RES_cov[key],size=1).reshape((-1))
                print('RES noise added for',key,' using covarying multivariate normal noise')
    
    
    save_lines(RES_cov[key],save_dir = OUTPUT_DIR +'/RES_error',save_name='/'+str(key))
save_spectra(RES_wlen_rebin,RES_flux_rebin,save_dir= OUTPUT_DIR+'/RES_spectrum',save_name='')


PHOT_flux_data,PHOT_flux_err,filt,filt_func,filt_mid,filt_width = data_obj.getPhot() #dictionaries
model_photometry,model_photometry_err,phot_midpoint,phot_width,PHOT_wlen_ck,PHOT_flux_ck = rebin_to_PHOT(wlen_ck,flux_ck,filt_func,log_R=TEMP_MODEL_PARAMS['log_R'],distance=DISTANCE,phot_flux_data = PHOT_flux_data,phot_flux_err_data = PHOT_flux_err)
if ADD_RANDOM_SCATTER:
    for key in PHOT_flux_data.keys():
        model_photometry[key] = model_photometry[key] + np.random.normal(loc=0,scale=model_photometry_err[key],size=1)[0]
        print('PHOT noise added for',key,' using normal iid noise')
save_photometry(model_photometry,model_photometry_err,phot_midpoint,phot_width,save_dir=OUTPUT_DIR)
save_spectrum(PHOT_wlen_ck,PHOT_flux_ck,save_dir=OUTPUT_DIR,save_name='/spectrum_cksim')

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
            PHOT_sim_wlen = PHOT_wlen_ck,
            PHOT_sim_flux = PHOT_flux_ck,
            model_PHOT_flux = None,
            output_file = OUTPUT_DIR,
            plot_name='plot')
