# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:27:11 2021

@author: jeanh
"""

import numpy as np
import scipy.constants as cst
from doubleRetrieval.util import save_1d_data

#organise after sections in paper
instrument = 'JWST'
NAME = 'TEST_OTHER_PARAMS'
VERSION = '04'
NAME += '_v'+VERSION

MODEL = 'chem_equ' # 'free' or 'chem_equ'
OUTPUT_FORMAT = 'datalike' # determines whether output has same bins as data ('datalike') or if it still contains its extensions
INSTRUMENT = 'SINFONI'

USE_FORECASTER = False

USE_PRIOR = None

IPAGATE_ROUTE = '/home/ipa/quanz/user_accounts/jhayoz/Projects/'

#OUTPUT_DIR = IPAGATE_ROUTE + 'VHS1256b_JWST/'
OUTPUT_DIR = IPAGATE_ROUTE + 'MT_paper/PAPER_RETRIEVALS/'
#OUTPUT_DIR = IPAGATE_ROUTE + 'Proposals/CRIRES/'
OUTPUT_DIR += NAME
INPUT_FILE = IPAGATE_ROUTE + 'retrieval_input/'


#WVL_FILE = INPUT_FILE + 'CC_data/CC_SINFONI'
#CC_WVL_DIR = IPAGATE_ROUTE + '/Proposals/CRIRES/flux_at_all_bands_v2/H1567'
CC_WVL_DIR = INPUT_FILE + 'CC_data/CC_SINFONI'

#CC_WVL_DIR = INPUT_FILE + 'CC_data/ERIS_WVL_RANGE_'+instrument
#WVL_FILE = INPUT_FILE+'wavelength_range_2.088-2.451_right_format.txt'

#RES_WVL_DIR = INPUT_FILE + 'RES_data/Flux'
#RES_COV_DIR = INPUT_FILE + 'RES_data/Error'

#RES_WVL_FILE = INPUT_FILE + 'RES_data/Flux/data.txt'
#RES_COV_FILE = INPUT_FILE + 'RES_data/Error/data.txt'

RES_WVL_DIR = INPUT_FILE + 'RES_data/Flux'
RES_COV_DIR = INPUT_FILE + 'RES_data/Error'

FILTER_DIR = INPUT_FILE + 'PHOT_data/filter/'
PHOT_DATA_FILE = INPUT_FILE + 'PHOT_data/flux/'


#JWST data
#RES_WVL_DIR = IPAGATE_ROUTE + 'VHS1256b_JWST/spectrum_NIR_MIRI_noNaN'
#RES_COV_DIR = IPAGATE_ROUTE + 'VHS1256b_JWST/spectrum_NIR_MIRI_noNaN_err'

EXTERNAL_PROFILES = False

MODE = 'lbl'
LBL_SAMPLING = None
CONVERT_SINFONI_UNITS = True

RESOLUTION = 8000


WLEN_BORDERS = [2.0880364682002703,2.4506398060442036]
WLEN_BORDERS = [0.8,20]
EXTENSIONS = None

ABUNDANCES = ['H2O_main_iso', 'CO_main_iso','CH4_main_iso', 'CO2_main_iso','H2S_main_iso','VO','TiO_all_iso','K','FeH_main_iso']
ABUNDANCES = ['H2O_main_iso']
#ABUNDANCES = ['H2O_main_iso']
ABUNDANCES = ['C/O','FeHs']

pc_to_m = 3.086*1e16
distance_pc = 19.7538
DISTANCE = distance_pc*pc_to_m

RADIUS_J = 69911*1000
MASS_J = 1.898*1e27

TEMP_PARAMS = {}

TEMP_PARAMS['log_gamma']      = np.log10(0.4)
TEMP_PARAMS['t_int']          = 200.
TEMP_PARAMS['t_equ']          = 1742
#TEMP_PARAMS['t_equ']          = 2200
TEMP_PARAMS['log_R']          = np.log10(1.36)
TEMP_PARAMS['log_gravity']    = 4.35
#TEMP_PARAMS['log_gravity']    = 5.1
TEMP_PARAMS['log_kappa_IR']   = np.log10(0.01)
TEMP_PARAMS['P0']             = 2

if USE_FORECASTER:
    import scipy.constants as cst
    from doubleRetrieval.util import predict_RM_distr,sample_to_pdf,predict_g_distr
    import forecaster
    from forecaster import mr_forecast as mr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    #TEMP_PARAMS['log_M'] = TEMP_PARAMS['log_gravity'] + 2*TEMP_PARAMS['log_R'] + 2*np.log10(RADIUS_J) - np.log10(cst.gravitational_constant) - np.log10(MASS_J)
    
    TEMP_PARAMS['log_M'] = np.log10(15.84)
    M_mean,M_std = 10**TEMP_PARAMS['log_M'],6
    print('calculating M-R relation',M_mean,M_std)
    radius_sample,mass_sample = predict_RM_distr(M_mean,M_std,predict='radius',classify = True,N_picks=50000,return_input=True)
    
    radius_pdf = sample_to_pdf(radius_sample)
    radius_pos = np.linspace(min(radius_sample),max(radius_sample),1000)
    mass_pdf = sample_to_pdf(mass_sample)
    mass_pos = np.linspace(min(mass_sample),max(mass_sample),1000)
    
    TEMP_PARAMS['log_R'] = np.log10(np.median(radius_sample))
    print('calculating log(g)')
    log_g_sample = predict_g_distr(mass_sample,radius_sample,N_picks = 10000)
    log_g_pdf = sample_to_pdf(log_g_sample)
    log_g_pos = np.linspace(min(log_g_sample),max(log_g_sample),1000)
    
    TEMP_PARAMS['log_gravity'] = np.median(log_g_sample)
    
    plt.figure(figsize=(10,6))
    plt.subplot(1,3,1)
    plt.hist(radius_sample,bins=20,density=True,alpha=0.3)
    plt.plot(radius_pos,radius_pdf(radius_pos))
    plt.axvline(10**TEMP_PARAMS['log_R'],color='r',label='Median radius: {numb:.2f}'.format(numb=10**TEMP_PARAMS['log_R']))
    plt.title('Radius')
    plt.subplot(1,3,2)
    plt.hist(mass_sample,bins=20,density=True,alpha=0.3)
    plt.plot(mass_pos,mass_pdf(mass_pos))
    plt.axvline(10**TEMP_PARAMS['log_M'],color='r',label='Median mass: {numb:.2f}'.format(numb = 10**TEMP_PARAMS['log_M']))
    plt.legend()
    plt.title('Mass')
    plt.subplot(1,3,3)
    plt.hist(log_g_sample,bins=20,density=True,alpha=0.3)
    plt.plot(log_g_pos,log_g_pdf(log_g_pos))
    plt.axvline(TEMP_PARAMS['log_gravity'],color='r',label='Median log g: {numb:.2f}'.format(numb=TEMP_PARAMS['log_gravity']))
    plt.axvline(np.mean(log_g_sample),color='b',label='Mean log g: {numb:.2f}'.format(numb=np.mean(log_g_sample)))
    plt.legend()
    plt.title('Surface gravity')
    plt.savefig(IPAGATE_ROUTE + 'diverse_images/M_R_prior_sim.png')
    
    save_1d_data(radius_sample,save_dir=OUTPUT_DIR,save_name='/radius_sample_prior')
    save_1d_data(mass_sample,save_dir=OUTPUT_DIR,save_name='/mass_sample_prior')
    save_1d_data(log_g_pos,save_dir=OUTPUT_DIR,save_name='/log_g_sample_prior')
elif USE_PRIOR is not None:
    if USE_PRIOR == 'M':
        prior_quantity = TEMP_PARAMS['log_gravity'] - 2 + 2*TEMP_PARAMS['log_R'] + 2*np.log10(RADIUS_J) - np.log10(cst.gravitational_constant) - np.log10(MASS_J)
        TEMP_PARAMS['log_M'] = prior_quantity
    elif USE_PRIOR == 'R':
        prior_quantity = TEMP_PARAMS['log_R']
    
    print('PLANET MASS FOR GIVEN RADIUS AND SURFACE GRAVITY: {value:.2f}'.format(value=10**prior_quantity))
    quantity_std = 0.2*(10**prior_quantity)
    quantity_samples = np.random.normal(loc=10**prior_quantity,scale=quantity_std,size=int(1e4))
    
    save_1d_data(quantity_samples,save_dir=OUTPUT_DIR,save_name='/' + USE_PRIOR + '_sample_prior')

print(TEMP_PARAMS)
CLOUDS_PARAMS = {}

#CLOUDS_PARAMS['log_Pcloud']     = np.log10(0.5)
#CLOUDS_OPACITIES = ['Fe(c)_cm']
#CLOUDS_PARAMS['model'] = None # ackermann
#CLOUDS_PARAMS['log_kzz'] = np.log10(4)
#CLOUDS_PARAMS['fsed'] = 4
#CLOUDS_PARAMS['sigma_lnorm'] = 4
#CLOUDS_PARAMS['cloud_abunds'] = {}
#CLOUDS_PARAMS['cloud_abunds']['Fe(c)'] = -0.75


AB_METALS = {}
#AB_METALS['H2O_main_iso']   = -2.4
#AB_METALS['CO_main_iso']    = -3.3
#AB_METALS['CH4_main_iso']   = -4.5
#AB_METALS['CO2_main_iso']   = -4.2
#AB_METALS['H2O_main_iso']   = -1.8
#AB_METALS['CO_main_iso']    = -1.6
"""
AB_METALS['CH4_main_iso']   = -3.8
AB_METALS['CO2_main_iso']   = -4.6
AB_METALS['H2S_main_iso']   = -2.9
AB_METALS['VO']   = -6.6
AB_METALS['TiO_all_iso']    = -6.7
AB_METALS['K']   = -5.1
AB_METALS['FeH_main_iso']   = -6.2
"""
AB_METALS['C/O']   = 0.44
AB_METALS['FeHs']   = 0.

PHYSICAL_PARAMS = {}
#PHYSICAL_PARAMS['rot_vel'] = 25
PHYSICAL_PARAMS['rot_vel'] = 0

DATA_PARAMS = {}
DATA_PARAMS.update(TEMP_PARAMS)
DATA_PARAMS.update(AB_METALS)
DATA_PARAMS.update(CLOUDS_PARAMS)
DATA_PARAMS.update(PHYSICAL_PARAMS)


WINDOW_LENGTH_LBL = 151
WINDOW_LENGTH_LBL = 101
WINDOW_LENGTH_LBL = 64 # now referring to the length of the window for the median filter
WINDOW_LENGTH_LBL = 600 # now referring to the length of the window for the median filter, too slow
WINDOW_LENGTH_LBL = 300 # now referring to the length of the window for the only gaussian filter
WINDOW_LENGTH_LBL = 60 # now referring to the length of the window for the only gaussian filter, for sinfoni data
#WINDOW_LENGTH_LBL = 20 # now referring to the length of the window for the only gaussian filter, for sinfoni data
WINDOW_LENGTH_CK = 41

RV = 31
NOISE = 3
NOISE_RES = 1
SNR= 0

CONFIG_DICT = {}
CONFIG_DICT['ALL_PARAMS'] = ABUNDANCES
CONFIG_DICT['ABUNDANCES'] = ABUNDANCES
CONFIG_DICT['NEEDED_LINE_SPECIES'] = ABUNDANCES
CONFIG_DICT['TEMPS'] = ['log_gamma','t_int','t_equ','log_gravity','log_kappa_IR','log_R','P0']
CONFIG_DICT['CLOUDS'] = CLOUDS_PARAMS.keys()
#CONFIG_DICT['CLOUDS_OPACITIES'] = CLOUDS_OPACITIES
CONFIG_DICT['UNSEARCHED_ABUNDANCES'] = []
CONFIG_DICT['UNSEARCHED_TEMPS'] = []
CONFIG_DICT['UNSEARCHED_CLOUDS'] = []
CONFIG_DICT['PARAMS_NAMES'] = []
CONFIG_DICT['UNSEARCHED_PARAMS'] = []
CONFIG_DICT['RVMAX'] = 0
CONFIG_DICT['RVMIN'] = 0
CONFIG_DICT['DRV'] = 0
CONFIG_DICT['DISTANCE'] = DISTANCE
CONFIG_DICT['WIN_LEN'] = WINDOW_LENGTH_LBL
CONFIG_DICT['LBL_SAMPLING'] = LBL_SAMPLING
CONFIG_DICT['CONVERT_SINFONI_UNITS'] = CONVERT_SINFONI_UNITS
CONFIG_DICT['WRITE_THRESHOLD'] = 0
CONFIG_DICT['DATA_PARAMS'] = DATA_PARAMS