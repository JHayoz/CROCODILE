# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:08:49 2021

@author: jeanh
"""

from numpy import log10
from doubleRetrieval.prior_functions import a_b_range,uniform_prior,gaussian_prior,log_gauss

# Define where to store results of retrieval

#MACHINE = 'rainbow'
MACHINE = 'sunray'
#MACHINE = 'bluesky'
#MACHINE = 'guenther38'

# name of retrieval run
NUMBER = ''
RETRIEVAL_NAME_INPUT = 'My_spectrum_v01'
VERSION = '01'
#DATA_TYPE = 'LRSP'
#DATA_TYPE = 'CROC'
DATA_TYPE = 'CC'
if MODEL == 'free':
    VERSION += '_free'
RETRIEVAL_NAME = RETRIEVAL_NAME_INPUT +'_'+ VERSION + '_' + DATA_TYPE

# configure the paths of the input and output files
OUTPUT_DIR = '/scratch/'
if MACHINE == 'rainbow':
    OUTPUT_DIR += 'user/'
OUTPUT_DIR += 'jhayoz/RunningJobs/' + RETRIEVAL_NAME + '/'


IPAGATE_ROUTE = '/home/ipa/quanz/user_accounts/jhayoz/Projects/'
INPUT_DIR = IPAGATE_ROUTE + 'MT_paper/PAPER_RETRIEVALS/REFEREE_NEW_ANALYSIS/' + RETRIEVAL_NAME_INPUT

SIM_DATA_DIR = INPUT_DIR
CC_DATA_FILE = INPUT_DIR+'/CC_spectrum'
RES_DATA_FILE = INPUT_DIR+'/RES_spectrum' #/SPHERE_IFS.txt'
RES_ERR_FILE = INPUT_DIR+'/RES_error' #/SPHERE_IFS.txt'

if DATA_TYPE == 'CROC':
    USE_SIM_DATA = ['CC','RES','PHOT'] #['CC','RES','PHOT']
elif DATA_TYPE == 'LRSP':
    USE_SIM_DATA = ['RES','PHOT'] #['CC','RES','PHOT']
else: # DATA_TYPE == 'CC':
    USE_SIM_DATA = ['CC'] #['CC','RES','PHOT']
print(USE_SIM_DATA)

# Name of files for respective data. If don't want to use one, write None
PHOT_DATA_FILE = IPAGATE_ROUTE+ 'retrieval_input/PHOT_data'
if 'PHOT' not in USE_SIM_DATA:
    PHOT_DATA_FILE = None
PHOT_DATA_FILTER_FILE,PHOT_DATA_FLUX_FILE = None,None
if 'PHOT' in USE_SIM_DATA:
    PHOT_DATA_FILTER_FILE = PHOT_DATA_FILE + '/filter'
    PHOT_DATA_FLUX_FILE = PHOT_DATA_FILE + '/flux'
    PHOT_DATA_FLUX_FILE = INPUT_DIR

RETRIEVAL_NOTES = [
    'Testing CROCODILE'
                   ]





# Define configuration
MODE = 'lbl'
LBL_SAMPLING = 10

USE_PRIOR = None
USE_COV = False
WINDOW_LENGTH_lbl = 20 # ERIS SPIFFIER
WINDOW_LENGTH_ck = 41


# Hyperparameters of retrieval
RVMIN = -400. # 400
RVMAX = 400.
DRV = 0.5
BAYESIAN_METHOD = 'pymultinest' # pymultinest, ultranest or mcmc
#pymultinest
N_LIVE_POINTS = 800

# diagnostic parameters
WRITE_THRESHOLD = 50
PRINTING = True
PLOTTING = False
TIMING = True
SHOW_REF_VALUES = False
PLOTTING_THRESHOLD = 10

# Define forward model

# Chemical model
CHEM_MODEL = 'chem_equ' # 'free' or 'chem_equ'
#ABUNDANCES = ['H2O_main_iso','CO_main_iso','CH4_main_iso']
if CHEM_MODEL == 'free':
    #ABUNDANCES = ['H2O_main_iso','CH4_main_iso', 'CO_main_iso', 'CO2_main_iso','H2S_main_iso','FeH_main_iso','TiO_all_iso','K','VO']
    ABUNDANCES = ['H2O_main_iso', 'CO_main_iso']
    #ABUNDANCES = [ 'H2O_main_iso','CO_main_iso','CH4_main_iso','CO2_main_iso','HCN_main_iso','FeH_main_iso','H2S_main_iso','NH3_main_iso','TiO_all_iso','VO']
if CHEM_MODEL == 'chem_equ':
    ABUNDANCES = ['C/O','FeHs']
UNSEARCHED_ABUNDS = []

# P-T model
TEMP_MODEL = 'guillot'
ALL_TEMPS = ['log_gamma','t_int','t_equ','log_gravity','log_kappa_IR','R','P0']
TEMP_PARAMS = ['t_equ','log_gravity'#,'R'
               ]#Pick from ALL_TEMPS, and order is relevant: must be like in ALL_TEMPS
UNSEARCHED_TEMPS = [item for item in ALL_TEMPS if not(item in TEMP_PARAMS)]

# Cloud model
USE_CHEM_DISEQU = False
CLOUD_MODEL = None # None or 'grey_cloud' or 'ackermann'
DO_SCAT_CLOUDS = False
CLOUDS_OPACITIES = []
ALL_CLOUDS = []
CLOUDS = []
UNSEARCHED_CLOUDS = []
# available species ['Fe(c)_cm','Mg2SiO4(c)_cm','MgSiO3(c)_cm']
if CLOUD_MODEL is not None:
    DO_SCAT_CLOUDS = False
    CLOUDS_OPACITIES = ['Mg2SiO4(c)_cm']
    ALL_CLOUDS = ['log_kzz','fsed','sigma_lnorm','Mg2SiO4(c)']
    CLOUDS = ['log_kzz','fsed','Mg2SiO4(c)']
    UNSEARCHED_CLOUDS = [param for param in ALL_CLOUDS if not(param in CLOUDS)]


PARAMS = [TEMP_PARAMS, ABUNDANCES, CLOUDS]
PARAMS_NAMES = TEMP_PARAMS + ABUNDANCES + CLOUDS
UNSEARCHED_PARAMS = [UNSEARCHED_TEMPS, UNSEARCHED_ABUNDS, UNSEARCHED_CLOUDS]
ALL_PARAMS = TEMP_PARAMS + ABUNDANCES + CLOUDS + UNSEARCHED_TEMPS + UNSEARCHED_ABUNDS + UNSEARCHED_CLOUDS
NEEDED_LINE_SPECIES = ABUNDANCES + UNSEARCHED_ABUNDS

if CHEM_MODEL == 'chem_equ':
    NEEDED_LINE_SPECIES += MOL_ABUNDS_KEYS_LBL
    if 'C/O' in NEEDED_LINE_SPECIES:
        NEEDED_LINE_SPECIES.remove('C/O')
    if 'FeHs' in NEEDED_LINE_SPECIES:
        NEEDED_LINE_SPECIES.remove('FeHs')


# enter reference values here (if data was simulated, their true values)

pc_to_m = 3.086*1e16
distance_pc = 19.7538
DISTANCE = distance_pc*pc_to_m

RADIUS_J = 69911*1000
MASS_J = 1.898*1e27

DATA_PARAMS = {}

# P-T model
DATA_PARAMS['TEMP_MODEL'] = 'guillot'

DATA_PARAMS['log_gamma']      = log10(0.4)
DATA_PARAMS['t_int']          = 200.
DATA_PARAMS['t_equ']          = 1742
DATA_PARAMS['log_kappa_IR']   = log10(0.01)
DATA_PARAMS['R']          = 1.36
DATA_PARAMS['log_gravity']    = 4.35
DATA_PARAMS['P0']             = 2

# Chemical model
DATA_PARAMS['CHEM_MODEL'] = 'free'
#DATA_PARAMS['C/O'] = 0.3
#DATA_PARAMS['FeHs'] = 0.66
DATA_PARAMS['H2O_main_iso']   = -1.8
DATA_PARAMS['CO_main_iso']    = -1.6

# Cloud model
#DATA_PARAMS['log_kzz'] = 9.8
#DATA_PARAMS['fsed'] = 1.88
#DATA_PARAMS['sigma_lnorm'] = 1.9 # as HR8799e

DATA_PARAMS['cloud_abunds'] = {}
#DATA_PARAMS['cloud_abunds']['Mg2SiO4(c)'] = -1
#DATA_PARAMS['log_Pcloud']     = log10(0.5)


# configure prior distributions

# define parameters of prior distributions

RANGE = {}
RANGE['log_gamma']      = [-4,0]
RANGE['t_int']          = [0,1000]
RANGE['t_equ']          = [0,5000]
RANGE['log_gravity']    = [1,8] #[-2,10]
RANGE['log_kappa_IR']   = [-2,2]
#RANGE['log_R']          = [DATA_PARAMS['log_R']-0.5,DATA_PARAMS['log_R']+0.5]
RANGE['R']          = [0.01,20]
RANGE['P0']             = [-2,2]
RANGE['log_Pcloud']     = [-3,1.49]
RANGE['abundances']     = [-10,0]
RANGE['C/O']            = [0.1,1.6]
RANGE['FeHs']           = [-2,3]

RANGE['log_kzz']        = [0.1,20]
RANGE['fsed']           = [0,10]
RANGE['sigma_lnorm']    = [1.001,6]
for name in DATA_PARAMS['cloud_abunds']:
    RANGE[name]     = [-10,0]

# define prior distributions
LOG_PRIORS = {}
LOG_PRIORS['log_gamma']      = lambda x: a_b_range(x,RANGE['log_gamma'])
LOG_PRIORS['t_int']          = lambda x: a_b_range(x,RANGE['t_int'])
LOG_PRIORS['t_equ']          = lambda x: a_b_range(x,RANGE['t_equ'])
#LOG_PRIORS['t_equ']          = lambda x: log_gauss(x,DATA_PARAMS['t_equ'],60) # Gaussian prior using prior analyses
LOG_PRIORS['log_gravity']    = lambda x: a_b_range(x,RANGE['log_gravity'])
#LOG_PRIORS['log_gravity']          = lambda x: log_gauss(x,DATA_PARAMS['log_gravity'],0.5) # Gaussian prior using prior analyses
LOG_PRIORS['log_kappa_IR']   = lambda x: a_b_range(x,RANGE['log_kappa_IR'])
LOG_PRIORS['R']          = lambda x: a_b_range(x,RANGE['R'])
#LOG_PRIORS['R']          = lambda x: log_gauss(x,DATA_PARAMS['R'],0.5)
LOG_PRIORS['P0']             = lambda x: a_b_range(x,RANGE['P0'])
LOG_PRIORS['log_Pcloud']     = lambda x: a_b_range(x,RANGE['log_Pcloud'])
for name in ABUNDANCES:
    LOG_PRIORS[name]     = lambda x: a_b_range(x,RANGE['abundances'])
LOG_PRIORS['C/O']  = lambda x: a_b_range(x,RANGE['C/O'])
LOG_PRIORS['FeHs'] = lambda x: a_b_range(x,RANGE['FeHs'])

LOG_PRIORS['log_kzz']  = lambda x: a_b_range(x,RANGE['log_kzz'])
LOG_PRIORS['fsed'] = lambda x: a_b_range(x,RANGE['fsed'])
LOG_PRIORS['sigma_lnorm'] = lambda x: a_b_range(x,RANGE['sigma_lnorm'])
for name in DATA_PARAMS['cloud_abunds']:
    LOG_PRIORS[name]     = lambda x: a_b_range(x,RANGE['abundances'])


# pymultinest uses unit cube, so define transformation of the unit cube corresponding to the prior distributions
CUBE_PRIORS = {}
CUBE_PRIORS['log_gamma']      = lambda x: uniform_prior(x,RANGE['log_gamma'])
CUBE_PRIORS['t_int']          = lambda x: uniform_prior(x,RANGE['t_int'])
CUBE_PRIORS['t_equ']          = lambda x: uniform_prior(x,RANGE['t_equ'])
#CUBE_PRIORS['t_equ']          = lambda x: gaussian_prior(x,DATA_PARAMS['t_equ'],60)
CUBE_PRIORS['log_gravity']    = lambda x: uniform_prior(x,RANGE['log_gravity'])
#CUBE_PRIORS['log_gravity']          = lambda x: gaussian_prior(x,DATA_PARAMS['log_gravity'],0.5)
CUBE_PRIORS['log_kappa_IR']   = lambda x: uniform_prior(x,RANGE['log_kappa_IR'])
CUBE_PRIORS['R']          = lambda x: uniform_prior(x,RANGE['R'])
#CUBE_PRIORS['R']          = lambda x: gaussian_prior(x,DATA_PARAMS['R'],0.5)
CUBE_PRIORS['P0']             = lambda x: uniform_prior(x,RANGE['P0'])
CUBE_PRIORS['log_Pcloud']     = lambda x: uniform_prior(x,RANGE['log_Pcloud'])
for name in ABUNDANCES:
    CUBE_PRIORS[name]     = lambda x: uniform_prior(x,RANGE['abundances'])
CUBE_PRIORS['C/O']        = lambda x: uniform_prior(x,RANGE['C/O'])
CUBE_PRIORS['FeHs']       = lambda x: uniform_prior(x,RANGE['FeHs'])

CUBE_PRIORS['log_kzz']  = lambda x: uniform_prior(x,RANGE['log_kzz'])
CUBE_PRIORS['fsed'] = lambda x: uniform_prior(x,RANGE['fsed'])
CUBE_PRIORS['sigma_lnorm'] = lambda x: uniform_prior(x,RANGE['sigma_lnorm'])
for name in DATA_PARAMS['cloud_abunds'].keys():
    CUBE_PRIORS[name]     = lambda x: uniform_prior(x,RANGE['abundances'])

if USE_PRIOR is not None:
    print('Creating priors on Mass')
    if USE_PRIOR == 'M':
        RANGE['M'] = [1,30]
        LOG_PRIORS['M'] = lambda x: gauss_pdf(x,*popt_param)
        CUBE_PRIORS['M']= lambda x: gauss_ppf(x,*popt_param)
    elif USE_PRIOR == 'R':
        RANGE['log_R'] = [0.1,3]
        LOG_PRIORS['log_R'] = lambda x: gauss_pdf(x,*popt_param)
        CUBE_PRIORS['log_R']= lambda x: gauss_ppf(x,*popt_param)

CONFIG_DICT = {}
CONFIG_DICT['CHEM_MODEL'] = CHEM_MODEL
CONFIG_DICT['MODE'] = MODE
CONFIG_DICT['USE_PRIOR'] = USE_PRIOR
CONFIG_DICT['USE_COV'] = USE_COV
CONFIG_DICT['ALL_PARAMS'] = ALL_PARAMS
CONFIG_DICT['ABUNDANCES'] = ABUNDANCES
CONFIG_DICT['NEEDED_LINE_SPECIES'] = NEEDED_LINE_SPECIES
CONFIG_DICT['TEMPS'] = TEMP_PARAMS
CONFIG_DICT['CLOUD_MODEL'] = CLOUD_MODEL
CONFIG_DICT['CLOUDS'] = CLOUDS
CONFIG_DICT['CLOUDS_OPACITIES'] = CLOUDS_OPACITIES
CONFIG_DICT['CLOUDS_ABUNDS'] = list(DATA_PARAMS['cloud_abunds'].keys())
CONFIG_DICT['DO_SCAT_CLOUDS'] = DO_SCAT_CLOUDS
CONFIG_DICT['UNSEARCHED_ABUNDANCES'] = UNSEARCHED_ABUNDS
CONFIG_DICT['UNSEARCHED_TEMPS'] = UNSEARCHED_TEMPS
CONFIG_DICT['UNSEARCHED_CLOUDS'] = UNSEARCHED_CLOUDS
CONFIG_DICT['PARAMS_NAMES'] = PARAMS_NAMES
CONFIG_DICT['UNSEARCHED_PARAMS'] = UNSEARCHED_PARAMS
CONFIG_DICT['RVMAX'] = RVMAX
CONFIG_DICT['RVMIN'] = RVMIN
CONFIG_DICT['DRV'] = DRV
CONFIG_DICT['DISTANCE'] = DISTANCE
CONFIG_DICT['WIN_LEN'] = WINDOW_LENGTH_lbl
CONFIG_DICT['LBL_SAMPLING'] = LBL_SAMPLING
CONFIG_DICT['CONVERT_SINFONI_UNITS'] = CONVERT_SINFONI_UNITS
CONFIG_DICT['WRITE_THRESHOLD'] = WRITE_THRESHOLD
CONFIG_DICT['PLOTTING_THRESHOLD'] = PLOTTING_THRESHOLD
CONFIG_DICT['DATA_PARAMS'] = DATA_PARAMS