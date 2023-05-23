# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:27:11 2021

@author: jeanh
"""

import numpy as np
import scipy.constants as cst
from core.util import save_1d_data

# Define where to save spectrum

NAME = 'My_spectrum'
VERSION = '01'
NAME += '_v'+VERSION

IPAGATE_ROUTE = '/home/ipa/quanz/user_accounts/jhayoz/Projects/'

OUTPUT_DIR = IPAGATE_ROUTE + 'Quick_spectra/'
OUTPUT_DIR += NAME

# Define where to read input data to copy format from

INPUT_FILE = IPAGATE_ROUTE + 'retrieval_input/'
CC_WVL_DIR = INPUT_FILE + 'CC_data/CC_SINFONI'
RES_WVL_DIR = INPUT_FILE + 'RES_data/Flux'
RES_COV_DIR = INPUT_FILE + 'RES_data/Error'
FILTER_DIR = INPUT_FILE + 'PHOT_data/filter/'
PHOT_DATA_FILE = INPUT_FILE + 'PHOT_data/flux/'

# Define bounds of spectrum
OUTPUT_FORMAT = 'datalike' # determines whether output has same bins as data ('datalike') or if it still contains its extensions
MODE = 'lbl'
LBL_SAMPLING = None
WLEN_BORDERS = [0.8,20]
EXTENSIONS = None
RESOLUTION = 8000
WINDOW_LENGTH_LBL = 20 # now referring to the length of the window for the only gaussian filter, for sinfoni data
WINDOW_LENGTH_CK = 41

# Define model
# chemistry
CHEM_MODEL = 'chem_equ' # 'free' or 'chem_equ'
#ABUNDANCES = ['H2O_main_iso', 'CO_main_iso','CH4_main_iso', 'CO2_main_iso','H2S_main_iso','VO','TiO_all_iso','K','FeH_main_iso']
ABUNDANCES = ['C/O','FeHs']

CHEM_MODEL_PARAMS = {}
CHEM_MODEL_PARAMS['C/O']   = 0.44
CHEM_MODEL_PARAMS['FeHs']   = 0.
CHEM_MODEL_PARAMS['Pquench_carbon'] = False

# P-T profile
TEMP_MODEL = 'guillot'
TEMP_MODEL_PARAMS = {}
TEMP_MODEL_PARAMS['log_gamma']      = np.log10(0.4)
TEMP_MODEL_PARAMS['t_int']          = 200.
TEMP_MODEL_PARAMS['t_equ']          = 1742
TEMP_MODEL_PARAMS['log_R']          = np.log10(1.36)
TEMP_MODEL_PARAMS['log_gravity']    = 4.35
TEMP_MODEL_PARAMS['log_kappa_IR']   = np.log10(0.01)
TEMP_MODEL_PARAMS['P0']             = 2

# Cloud model
CLOUDS_PARAMS = {}

#CLOUDS_PARAMS['log_Pcloud']     = np.log10(0.5)
#CLOUDS_OPACITIES = ['Fe(c)_cm']
#CLOUDS_PARAMS['model'] = None # ackermann
#CLOUDS_PARAMS['log_kzz'] = np.log10(4)
#CLOUDS_PARAMS['fsed'] = 4
#CLOUDS_PARAMS['sigma_lnorm'] = 4
#CLOUDS_PARAMS['cloud_abunds'] = {}
#CLOUDS_PARAMS['cloud_abunds']['Fe(c)'] = -0.75

# Physical model

PHYSICAL_PARAMS = {}
#PHYSICAL_PARAMS['rot_vel'] = 25
PHYSICAL_PARAMS['rot_vel'] = 0
PHYSICAL_PARAMS['log_gravity']=TEMP_MODEL_PARAMS['log_gravity']

# Define a few physical constraints
pc_to_m = 3.086*1e16
distance_pc = 19.7538
DISTANCE = distance_pc*pc_to_m

RADIUS_J = 69911*1000
MASS_J = 1.898*1e27

RV = 31

# Add random scatter in the data
ADD_RANDOM_SCATTER = False
USE_SEED = True
USE_COV = True
NOISE = 3
NOISE_RES = 1
SNR= 0

# create configuration
DATA_PARAMS = {}
DATA_PARAMS.update(TEMP_MODEL_PARAMS)
DATA_PARAMS.update(CHEM_MODEL_PARAMS)
DATA_PARAMS.update(CLOUDS_PARAMS)
DATA_PARAMS.update(PHYSICAL_PARAMS)

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
CONFIG_DICT['WRITE_THRESHOLD'] = 0
CONFIG_DICT['DATA_PARAMS'] = DATA_PARAMS