# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:32:01 2021

@author: jeanh
"""
# Main code to run an atmospheric retrieval

# import all relevant Python libraries
print('IMPORTING LIBRARIES')
import os
os.environ['OMP_NUM_THREADS'] = '1'
from config3 import *
print('    CONFIG READ')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import sys
import numpy as np
#os.environ["pRT_input_data_path"] = "/scratch/software/petitRADTRANS/petitRADTRANS/input_data"
os.environ["pRT_input_data_path"] = "/home/ipa/quanz/shared/petitRADTRANS/input_data"

from os import path
#sys.path.append("/scratch/software/petitRADTRANS/")
#sys.path.append("/home/ipa/quanz/shared/petitRADTRANS/")
from petitRADTRANS import Radtrans
#from petitRADTRANS import nat_cst as nc
import pickle
import json
#import scipy.stats, scipy

# import all modules
#from doubleRetrieval.util import *
from core.priors import Prior
from core.data2 import Data
from core.retrievalClass3 import Retrieval
from core.plotting import plot_corner,plot_SNR,plot_walkers,plot_retrieved_spectra_FM_dico

print('    DONE')

if BAYESIAN_METHOD == 'pymultinest':
    import pymultinest
elif BAYESIAN_METHOD == 'ultranest':
    import ultranest
    from joblib import delayed, Parallel
    from ultranest.plot import cornerplot,PredictionBand
else: #mcmc
    from emcee import EnsembleSampler
    from schwimmbad import MPIPool
    from tqdm import tqdm

if not os.path.exists(OUTPUT_DIR):
    try:
        os.mkdir(OUTPUT_DIR)
    except FileExistsError:
        print('Avoided error')

import shutil

cwd = os.getcwd()
source = cwd+'/config3.py'
destination = OUTPUT_DIR+'config3_copy.py'
shutil.copyfile(source,destination)
print('Config file copied')


with open(OUTPUT_DIR+'NOTES.txt','w') as f:
    for line in RETRIEVAL_NOTES:
        f.write(line +  '\n')

# create data, prior, and retrieval class

data_obj = Data(data_dir = None,
                use_sim_files = USE_SIM_DATA,
                PHOT_flux_format = 4,
                PHOT_filter_dir = PHOT_DATA_FILTER_FILE,
                PHOT_flux_dir = PHOT_DATA_FLUX_FILE,
                CC_data_dir=CC_DATA_FILE,
                RES_data_dir=RES_DATA_FILE,
                RES_err_dir=RES_ERR_FILE,
                verbose=True)

# data_obj.plot(CONFIG_DICT,OUTPUT_DIR+'data',plot_errorbars=False,inset_plot=False)

# Check that the retrieval does what I want it to do
if 'CC' in USE_SIM_DATA:
    assert(data_obj.CCinDATA())
    wlen_data_temp,flux_data_temp = data_obj.getCCSpectrum()
    assert(len(wlen_data_temp.keys()) == 1 or len(wlen_data_temp.keys()) > 5)
if 'PHOT' in USE_SIM_DATA:
    assert(data_obj.PHOTinDATA())
if 'RES' in USE_SIM_DATA:
    assert(data_obj.RESinDATA())

print('Check passed')

prior_obj = Prior(RANGE,LOG_PRIORS,CUBE_PRIORS)
prior_obj.plot(CONFIG_DICT,OUTPUT_DIR)
print('FORWARD MODEL')
print('Chemistry: %s, p-T: %s, clouds: %s' %(CHEM_MODEL,TEMP_MODEL,CLOUD_MODEL))
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
    for_analysis=False)

print('Starting Bayesian inference')
if BAYESIAN_METHOD == 'pymultinest':
    n_params = len(PARAMS_NAMES)
    
    pymultinest.run(retrieval.lnprob_pymultinest,
                    retrieval.Prior,
                    n_params,
                    outputfiles_basename=OUTPUT_DIR,
                    resume = False,
                    verbose = True,
                    n_live_points = N_LIVE_POINTS)
    print('############### FINISHING THE SAMPLER #####################')
    # save positions
    json.dump(PARAMS_NAMES, open(OUTPUT_DIR+'params.json', 'w'))
    
    # create analyzer object
    a = pymultinest.Analyzer(n_params, outputfiles_basename = OUTPUT_DIR)
    
    stats = a.get_stats()
    bestfit_params = a.get_best_fit()
    samples = np.array(a.get_equal_weighted_posterior())[:,:-1]
    
    f = open(OUTPUT_DIR+'SAMPLESpos.pickle','wb')
    pickle.dump(samples,f)
    f.close()
elif BAYESIAN_METHOD == 'ultranest':
    
    with Parallel(n_jobs=ULTRANEST_JOBS) as parallel:
        def new_likelihood(points):
            likelihoods = parallel(delayed(retrieval.calc_log_likelihood)(p) for p in points)
            return np.asarray(list(likelihoods))
        
        def new_prior(points):
            prior_likelihoods = parallel(delayed(retrieval.ultranest_prior)(p) for p in points)
            return np.asarray(list(prior_likelihoods))
        
        print('starting retrieval')
        sampler = ultranest.ReactiveNestedSampler(PARAMS_NAMES, new_likelihood, new_prior,
                                                  log_dir=OUTPUT_DIR, resume='overwrite',vectorized=True)
        result = sampler.run()
        
        samples = result['samples']
        
        f = open(OUTPUT_DIR+'SAMPLESpos.pickle','wb')
        pickle.dump(samples,f)
        f.close()
        
else: #MCMC
    def MCMC_log_likelihood_workaround(x):
        return retrieval.lnprob_mcmc(x)
    """ pre-burn"""
    n_dim = len(PARAMS_NAMES)
    """Set up initial walker position"""
    p0 = [np.array([
            np.random.uniform(low=prior_obj.RANGE['abundances'][0],
                              high=prior_obj.RANGE['abundances'][1])
            if param_key in ABUNDANCES else
            np.random.uniform(low=prior_obj.RANGE[param_key][0],
                              high=prior_obj.RANGE[param_key][1])
            for param_key in PARAMS_NAMES
        ]) for i in range(N_WALKERS)]
    
    """run pre-burn"""
    if CLUSTER:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = EnsembleSampler(N_WALKERS, n_dim, MCMC_log_likelihood_workaround, \
                                            a = STEPSIZE, pool = pool)
    else:
        if N_THREADS > 1:
            sampler = EnsembleSampler(N_WALKERS, n_dim, MCMC_log_likelihood_workaround, \
                                                a = STEPSIZE, threads = N_THREADS)
        else:
            sampler = EnsembleSampler(N_WALKERS, n_dim, MCMC_log_likelihood_workaround, \
                                                a = STEPSIZE)
    
    
    pos, prob, state = sampler.run_mcmc(p0, PRE_BURN_ITER,progress=True)
    
    samples_preburn = sampler.chain[:, :, :].reshape((-1, n_dim))
    
    """save sampled parameter positions"""
    f = open(OUTPUT_DIR+'preburn_pos.pickle','wb')
    pickle.dump(pos,f)
    pickle.dump(prob,f)
    pickle.dump(state,f)
    pickle.dump(samples_preburn,f)
    f.close()
    with open(OUTPUT_DIR+'lnprob.pickle', 'wb') as f:
        pickle.dump([sampler.lnprobability], f, protocol=pickle.HIGHEST_PROTOCOL)
    
    """Run main MCMC chain"""
    highest_prob_index = np.unravel_index(sampler.lnprobability.argmax(), \
                                              sampler.lnprobability.shape)
    best_position = sampler.chain[highest_prob_index]
    median_position = np.median(samples_preburn, axis=0)
    std_position = np.std(samples_preburn, axis=0)
    
    f = open(OUTPUT_DIR+'best_position_pre_burn_in_.txt', 'w')
    f.write(str(best_position))
    f.close()
    
    """Run actual chain"""
    
    p1_params = {}
    p1 = [np.array([
            np.random.normal(loc = median_position[k],scale = std_position[k])
            for k,param_key in enumerate(PARAMS_NAMES)
        ]) for i in range(N_WALKERS)]
    
    """if we want to start using cluster"""
    if CLUSTER:
        sampler = EnsembleSampler(N_WALKERS, n_dim, MCMC_log_likelihood_workaround, \
                                            a = STEPSIZE, pool = pool)
    else:
        if N_THREADS > 1:
            sampler = EnsembleSampler(N_WALKERS, n_dim, MCMC_log_likelihood_workaround, \
                                                a = STEPSIZE, threads = N_THREADS)
        else:
            sampler = EnsembleSampler(N_WALKERS, n_dim, MCMC_log_likelihood_workaround, \
                                                a = STEPSIZE)
    
    pos, prob, state = sampler.run_mcmc(p1, N_ITER,progress=True)
    
    if CLUSTER:
        pool.close()
    samples = sampler.chain[:, :, :].reshape((-1, n_dim))
    """save sampled parameter positions"""
    f = open(OUTPUT_DIR+'pos.pickle','wb')
    pickle.dump(pos,f)
    pickle.dump(prob,f)
    pickle.dump(state,f)
    pickle.dump(samples,f)
    f.close()
    with open(OUTPUT_DIR+'lnprob.pickle', 'wb') as f:
        pickle.dump([sampler.lnprobability], f, protocol=pickle.HIGHEST_PROTOCOL)
    if not path.exists(OUTPUT_DIR + 'walkers'):
        plot_walkers(CONFIG_DICT,
                     samples,
                     samples_preburn,
                     prob,
                     quantiles = (0.16,0.5,0.84),
                     percent_considered = 0.5,
                     output_files = OUTPUT_DIR,
                     title = 'Walkers of '+RETRIEVAL_NAME
                     )
"""
nb_positions = len(samples)
percent_considered = 1.
if BAYESIAN_METHOD == 'mcmc':
    percent_considered = 0.5
if not path.exists(OUTPUT_DIR + 'cornerplot'):
    plot_corner(CONFIG_DICT,
                samples,
                param_range = None,
                percent_considered = percent_considered,
                output_file = OUTPUT_DIR,
                title = 'Corner plot of '+RETRIEVAL_NAME+' '+VERSION
                )

if not path.exists(OUTPUT_DIR + 'plot_retrieved_spectrum'):
    try:
        wlen_CC,flux_CC,wlen_RES,flux_RES,photometry = plot_retrieved_spectra_FM_dico(
            retrieval,
            samples,
            output_file = OUTPUT_DIR,
            title = 'Retrieved spectrum for '+RETRIEVAL_NAME+' '+VERSION,
            show_random = None,
            output_results = True)
    except:
        pass

if not path.exists(OUTPUT_DIR + 'CC_function'):
    if data_obj.CCinDATA():
        CC_wlen_data,CC_flux_data = data_obj.getCCSpectrum()
        try:
            plot_SNR(CONFIG_DICT,wlen_CC,flux_CC,CC_wlen_data,CC_flux_data,output_file = OUTPUT_DIR,title='C-C function for '+RETRIEVAL_NAME+' '+VERSION,printing=True)
        except:
            pass
"""
