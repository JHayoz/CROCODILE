# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 13:00:49 2021

@author: jeanh
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import scipy.stats, scipy
import pymultinest
import pickle

from core.plotting import plot_corner,plot_retrieved_temperature_profile,plot_CO_ratio,plot_retrieved_abunds,plot_retrieved_spectra_FM_dico
from core.data import Data
from core.retrievalClass import Retrieval
from core.forward_model import ForwardModel
#from core.util import *
#from core.rebin import *
#from core.plot_errorbars_abundances import Posterior_Classification_Errorbars,compute_model_line,compute_SSG_errorbars

#matplotlib.rcParams['agg.path.chunksize'] = 10000

IPAGATE_ROUTE = '/home/ipa/quanz/user_accounts/jhayoz/Projects/'
retrieval = 'My_spectrum_v01_01_CROC'
sim_data_input = 'My_spectrum_v01'
input_dir = IPAGATE_ROUTE + 'Quick_spectra/'+sim_data_input+'/'+retrieval
input_dir = '/scratch/jhayoz/RunningJobs/My_spectrum_v01_chem_equ_CROC'

BAYESIAN_METHOD = 'pymultinest'

output_dir = input_dir + '/' 

with open(input_dir + '/SAMPLESpos.pickle','rb') as f:
    samples = pickle.load(f)

import sys
sys.path.append(input_dir)
from config import *

fontsize=12
lw=0.8
figsize=(8,5)

PLOT_ALL = True

if PLOT_ALL:
    
    plot_corner(
        CONFIG_DICT,
        samples,
        param_range = None,
        percent_considered = 0.90,
        output_file = output_dir,
        fontsize=fontsize,
        include_abunds = True,
        title = None,
        plot_format = 'pdf')
    
    plot_corner(
        CONFIG_DICT,
        samples,
        param_range = None,
        percent_considered = 0.90,
        output_file = output_dir,
        fontsize=fontsize,
        include_abunds = False,
        title = None,
        plot_format = 'pdf')
    
    plot_retrieved_temperature_profile(
        CONFIG_DICT,
        samples,
        output_file = output_dir,
        nb_stds = 3,
        fontsize = fontsize,
        lw = lw,
        figsize=figsize,
        color = 'g',
        title='Thermal profile',
        ax = None)
    if 'C/O' not in CONFIG_DICT['PARAMS_NAMES']:
        plot_CO_ratio(
            CONFIG_DICT,
            samples,
            percent_considered = 1.,
            abundances_considered = 'all',
            output_file = output_dir,
            fontsize = fontsize,
            lw = lw,
            figsize=figsize,
            color = 'g',
            title='C/O ratio',
            ax = None)
        
        plot_retrieved_abunds(
            CONFIG_DICT,
            samples,
            pressure_distr = None,
            output_dir = output_dir,
            nb_stds = 0,
            fontsize = fontsize,
            lw =lw,
            figsize=(8,4),
            xlim = [-10,0],
            title='Molecular abundance profiles',
            add_xlabel = True,
            add_ylabel = True,
            add_legend = True,
            errorbar_color = None,
            plot_marker = False,
            ax = None)

data_obj = Data(
    data_dir = SIM_DATA_DIR,
    use_sim_files = USE_SIM_DATA,
    PHOT_flux_format = 4,
    PHOT_filter_dir = PHOT_DATA_FILTER_FILE,
    PHOT_flux_dir = PHOT_DATA_FLUX_FILE,
    CC_data_dir=CC_DATA_FILE,
    RES_data_dir=RES_DATA_FILE,
    RES_err_dir=RES_ERR_FILE)

prior_obj = None
print('CONFIGURATION')
for key in CONFIG_DICT.keys():
    print(key,': ',CONFIG_DICT[key])

retrieval = Retrieval(
    data_obj,
    prior_obj,
    config=CONFIG_DICT,
    chem_model=CHEM_MODEL, # or chem_equ
    temp_model=TEMP_MODEL,
    cloud_model=CLOUD_MODEL,
    retrieval_name = RETRIEVAL_NAME,
    output_path = OUTPUT_DIR,
    plotting=PLOTTING,
    printing=PRINTING,
    timing=TIMING,
    for_analysis=True)

wlen_CC,flux_CC,wlen_RES,flux_RES,photometry = plot_retrieved_spectra_FM_dico(
        retrieval,
        samples,
        output_file = output_dir,
        title = 'Retrieved spectrum for '+RETRIEVAL_NAME+' '+VERSION,
        show_random = None,
        saving=True,
        output_results = True)