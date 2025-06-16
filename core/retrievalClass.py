# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 12:40 2025

@author: Jean Hayoz
"""
import sys
import numpy as np
from pathlib import Path
from time import time
from PyAstronomy.pyasl import crosscorrRV
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os
from os import path

# from petitRADTRANS import radtrans as rt
# from petitRADTRANS import nat_cst as nc
# from petitRADTRANS import physics

from config_petitRADTRANS import *
from core.util import name_lbl_to_ck,convert_units_to_SI,scale_flux
from core.read import read_forward_model_from_config
from core.forward_model import ForwardModel
from core.rebin import rebin_to_RES,rebin_to_CC,rebin_to_PHOT,doppler_shift
from core.plotting import plot_data

# class to define an atmospheric retrieval

class Retrieval:
    def __init__(
        self,
        data_obj,
        prior_obj,
        config_file,
        for_analysis,
        continue_retrieval):
        
        self.data_obj = data_obj
        self.prior_obj = prior_obj
        self.config = config_file.copy() # dictionary, with keys from the config file
        
        # Define forward model used
        self.FM = self.config['retrieval']['FM']['model']
        self.chem_model = self.config['retrieval']['FM']['chemistry']['model']
        self.temp_model = self.config['retrieval']['FM']['p-T']['model']
        self.cloud_model = self.config['retrieval']['FM']['clouds']['model']
        if self.chem_model == 'free':
            self.abundances_lbl = list(self.config['retrieval']['FM']['chemistry']['parameters'].keys())
        else:
            self.abundances_lbl = POOR_MANS_ABUND_LBL
        self.abundances_ck = [name_lbl_to_ck(abund) for abund in self.abundances_lbl]

        self.params_names = self.prior_obj.params_names
        print('PARAMETERS',self.params_names)
        
        # meta data
        self.retrieval_name = self.config['metadata']['retrieval_id']
        self.output_path = Path(self.config['metadata']['output_dir'])
        
        # diagnostics
        self.plotting = str(self.config['hyperparameters']['diagnostics']['plotting']) == 'True'
        self.plotting_threshold = self.config['hyperparameters']['diagnostics']['plotting_threshold']
        self.printing = str(self.config['hyperparameters']['diagnostics']['printing']) == 'True'
        self.timing = str(self.config['hyperparameters']['diagnostics']['timing']) == 'True'
        self.writing_threshold = self.config['hyperparameters']['diagnostics']['writing_threshold']
        self.diag_file = self.output_path / 'diag.txt'
        
        # Declare diagnostics
        self.start_time = time()
        self.function_calls = 0
        self.computed_spectra = 0
        self.outside_priors = 0
        self.NaN_spectra = 0
        self.NaN_spectRES = 0
        self.NaN_savgolay = 0
        self.NaN_crosscorrRV = 0
        self.NaN_photometry = 0
        self.nb_failed_DS = 0
        
        # diagnostic for C-C log
        self.diag_sf2 = 0.
        self.diag_sg2 = 0.
        self.diag_CC = 0.
        self.diag_log_CC = 0.
        self.diag_params = []
        self.diag_CC_file = self.output_path / 'diag_CC.txt'
        
        if not for_analysis:
            if not continue_retrieval:
                open(self.diag_file,'w').close()
                open(self.diag_CC_file,'w').close()
            
            if self.plotting:
                if not os.path.exists(self.output_path / 'model'):
                    os.mkdir(self.output_path / 'model')
        
        
        self.data_obj.distribute_FMs()
        
        if self.data_obj.RESinDATA() or self.data_obj.CCinDATA():
            self.lbl_itvls = self.data_obj.disjoint_lbl_intervals
            self.CC_to_lbl_itvls = self.data_obj.CC_to_lbl_intervals
            self.RES_to_lbl_itvls = self.data_obj.RES_to_lbl_intervals
            self.lbl_intervals_max_step = self.data_obj.disjoint_lbl_intervals_max_stepsize
            self.lbl_intervals_min_step = self.data_obj.disjoint_lbl_intervals_min_stepsize
        
        
        self.forwardmodel_lbl = None
        self.forwardmodel_ck = None
        
        if self.data_obj.ck_FM_interval is not None:
            print('CALLING FORWARD MODEL WITH C-K MODE')
            wlen_borders_ck = self.data_obj.ck_FM_interval
            if self.config['retrieval']['FM']['clouds']['opacities'] is None:
                cloud_species = None
            else:
                cloud_species = self.config['retrieval']['FM']['clouds']['opacities'].copy()
            self.forwardmodel_ck = ForwardModel(
                 wlen_borders = wlen_borders_ck,
                 max_wlen_stepsize = wlen_borders_ck[1]/1000,
                 mode = 'c-k',
                 line_opacities = self.abundances_ck,
                 cont_opacities= ['H2-H2','H2-He'],
                 rayleigh_scat = ['H2','He'],
                 cloud_model = self.cloud_model,
                 cloud_species = cloud_species,
                 do_scat_emis = self.config['retrieval']['FM']['clouds']['scattering_emission'],
                 chem_model = self.chem_model,
                 temp_model = self.temp_model,
                 max_RV = 0,
                 max_winlen = 0,
                 include_H2 = True,
                 only_include = 'all'
                 )
            self.forwardmodel_ck.calc_rt_obj(lbl_sampling = None)
            
        
        if self.data_obj.CCinDATA() or (self.data_obj.RESinDATA() and 'lbl' in [self.data_obj.RES_data_info[key][0] for key in self.data_obj.RES_data_info.keys()]):
            print('CALLING FORWARD MODEL WITH LBL MODE')
            self.forwardmodel_lbl = {}
            
            for interval_key in self.lbl_itvls.keys():
                wlen_borders_lbl = self.lbl_itvls[interval_key]
                max_wlen_stepsize = self.lbl_intervals_max_step[interval_key]
                
                win_len = self.config['hyperparameters']['CC']['filter_size']/1e3/max_wlen_stepsize # in wvl channel
                
                if self.config['retrieval']['FM']['clouds']['opacities'] is None:
                    cloud_species = None
                else:
                    cloud_species = self.config['retrieval']['FM']['clouds']['opacities'].copy()
                
                self.forwardmodel_lbl[interval_key] = ForwardModel(
                     wlen_borders = wlen_borders_lbl,
                     max_wlen_stepsize = max_wlen_stepsize,
                     mode = 'lbl',
                     line_opacities = self.abundances_lbl,
                     cont_opacities= ['H2-H2','H2-He'],
                     rayleigh_scat = ['H2','He'],
                     cloud_model = self.cloud_model,
                     cloud_species = cloud_species,
                     do_scat_emis = self.config['retrieval']['FM']['clouds']['scattering_emission'],
                     chem_model = self.chem_model,
                     temp_model = self.temp_model,
                     max_RV = 1.01*max(abs(self.config['hyperparameters']['rv']['rv_min']),self.config['hyperparameters']['rv']['rv_max']),
                     max_winlen = int(1.01*win_len),
                     include_H2 = True,
                     only_include = 'all'
                     )
                self.forwardmodel_lbl[interval_key].calc_rt_obj(lbl_sampling = self.config['hyperparameters']['radtrans']['lbl_sampling'])
            
            print('THERE ARE %i FMs USING LBL MODE' % len(list(self.forwardmodel_lbl.keys())))
            print('Their wlen range are:',[self.forwardmodel_lbl[key].wlen_borders for key in self.forwardmodel_lbl.keys()])
            
        return
    
    def Prior(self,
              cube,
              ndim,
              nparam):
        for i,name in enumerate(self.params_names):
            cube[i] = self.prior_obj.log_cube_priors[name](cube[i])
    
    def ultranest_prior(self,
                        cube):
        params = cube.copy()
        for i in range(len(self.params_names)):
            params[i] = self.prior_obj.log_cube_priors[self.params_names[i]](cube[i])
        return params
    
    def lnprob_pymultinest(self,cube,ndim,nparams):
        params = []
        for i in range(ndim):
            params.append(cube[i])
        params = np.array(params)
        return self.calc_log_likelihood(params)
    
    def lnprob_mcmc(self,x):
        return self.calc_log_likelihood(x)
    
    def calc_log_L_CC(
        self,
        wlen,
        flux,
        data_wlen,
        data_flux,
        data_N,
        data_sf2,
        data_params,
        data_key):
        
        log_L_CC = 0.

        flux_scaling_key = '%s___%s' % (data_key,'flux_scaling')
        flux_data_scaled = data_params[flux_scaling_key]*data_flux

        data_sf2_scaled = 1./data_N*np.sum(flux_data_scaled**2)
        
        # cut what I don't need to improve speed
        #wlen_cut,flux_cut = trim_spectrum(wlen,flux,wlen_data,threshold=5000,keep=1000)
        
        # filter_size given in nm, not in wvl channels
        dwvl = np.mean(data_wlen[1:]-data_wlen[:-1])*1e3 # size of a wvl channel in nm
        win_len = self.config['hyperparameters']['CC']['filter_size']/dwvl # in wvl channel
        
        wlen_removed,flux_removed,sgfilter,wlen_rebin,flux_rebin = rebin_to_CC(
            wlen.copy(),
            flux.copy(),
            data_wlen,
            win_len = win_len,
            method='linear',
            filter_method = 'only_gaussian')
        
        if sum(np.isnan(flux_rebin))>0:
            self.NaN_spectRES += 1
            log_L_CC += -1e299
            return log_L_CC,wlen_removed,flux_removed,sgfilter
        
        assert(len(wlen_removed) == len(flux_removed))
        
        if sum(np.isnan(flux_removed))>0:
            self.NaN_savgolay += 1
            log_L_CC += -1e299
            return log_L_CC,wlen_removed,flux_removed,sgfilter
        
        # flux_removed_rescaled = flux_removed/np.std(flux_removed)
        
        # cross-correlation
        dRV,CCF=crosscorrRV(
            data_wlen,
            flux_data_scaled,
            wlen_removed,
            flux_removed, # flux_removed_rescaled
            rvmin=self.config['hyperparameters']['rv']['rv_min'],
            rvmax=self.config['hyperparameters']['rv']['rv_max'],
            drv=self.config['hyperparameters']['rv']['drv'])
        
        if sum(np.isnan(CCF)>0):
            self.NaN_crosscorrRV += 1
            log_L_CC += -1e299
            return log_L_CC,wlen_removed,flux_removed,sgfilter
        
        CC=CCF/np.abs(data_N)
        RV_max_i=np.argmax(CC)
        CC_max = CC[RV_max_i]
        
        # need to doppler shift the model to the argmax of the CC-function. For that, we need to go back to high-resolution spectrum out of petitRADTRANS
        
        wlen_removed_DS,flux_removed_DS = wlen.copy(),flux.copy()
        if abs(dRV[RV_max_i])<max(abs(self.config['hyperparameters']['rv']['rv_min']),abs(self.config['hyperparameters']['rv']['rv_max']))*0.75:
            wlen_removed_DS,flux_removed_DS = doppler_shift(wlen.copy(),flux.copy(),dRV[RV_max_i])
        else: 
            print('Cant Dopplershift too much')
            self.nb_failed_DS += 1
        
        wlen_removed_DS,flux_removed_DS,sgfilter,wlen_rebin,flux_rebin = rebin_to_CC(
            wlen_removed_DS,
            flux_removed_DS,
            data_wlen,
            win_len = win_len,
            method='datalike',
            filter_method = 'only_gaussian')
        
        assert(len(wlen_removed_DS) == len(data_wlen))
        
        if sum(np.isnan(flux_removed_DS))>0:
            self.NaN_spectRES += 1
            log_L_CC += -1e299
            return log_L_CC,wlen_removed,flux_removed,sgfilter
        
        # flux_removed_rescaled = flux_removed_DS/np.std(flux_removed_DS)
        
        # s_g2 = 1./data_N*np.sum(flux_removed_rescaled**2)
        s_g2 = 1./data_N*np.sum(flux_removed_DS**2)
        
        # CROCODILE
        # log_term = data_sf2_scaled-2*CC_max+s_g2
        # Zucker 2003
        CC_2_norm = (CC_max**2)/s_g2/data_sf2_scaled
        log_term = 1.-CC_2_norm
        
        if log_term <= 0:
            self.NaN_crosscorrRV += 1
            log_L_CC += -1e299
            print('Negative values inside logarithm')
            return log_L_CC,wlen_removed,flux_removed,sgfilter
        
        print('C-C: ',CC_max)
        print('sf2 (D): ',data_sf2_scaled)
        print('sg2 (M): ',s_g2)
        log_L_CC += -data_N*np.log(log_term)/2
        
        # diagnostic for C-C log
        self.diag_sg2 = s_g2
        self.diag_sf2 = data_sf2_scaled
        self.diag_CC = CC_max
        self.diag_log_CC = log_L_CC
        
        return log_L_CC,wlen_removed,flux_removed,sgfilter
    
    def calc_log_L_PHOT(
        self,
        wlen,
        flux):
        
        log_L_PHOT = 0.
        
        # get data
        PHOT_flux,PHOT_flux_err,filt,filt_func,filt_mid,filt_width = self.data_obj.getPhot() #dictionaries
        
        model_photometry = rebin_to_PHOT(
            wlen.copy(),
            flux.copy(),
            filt_func = filt_func)
        
        for instr in PHOT_flux.keys():
            
            if np.isnan(model_photometry[instr]):
                self.NaN_photometry += 1
                return -1e299,model_photometry,wlen_rebin,flux_rebin
            
            log_L_PHOT += -0.5*((model_photometry[instr]-PHOT_flux[instr])/PHOT_flux_err[instr])**2
        
        return log_L_PHOT,model_photometry
    
    def calc_log_L_RES(
        self,
        wlen,
        flux,
        data_params,
        data_key,
        wlen_data,
        flux_data,
        flux_err,
        inverse_cov,
        flux_data_std,
        mode='lbl'):
        
        log_L_RES = 0
        if mode == 'lbl' or max(wlen_data[1:]/(wlen_data[1:]-wlen_data[:-1])) < 950:
            wlen_rebin,flux_rebin = rebin_to_RES(wlen.copy(),flux.copy(),wlen_data)
        else:
            # very edge case if I need to rebin a 1000 res spectrum into 980 res spectrum
            flux_interped = interp1d(wlen,flux)
            wlen_rebin,flux_rebin = wlen_data.copy(),flux_interped(wlen_data)
        
        flux_scaling_key = '%s___%s' % (data_key,'flux_scaling')
        flux_data_scaled = data_params[flux_scaling_key]*flux_data
        error_scaling_key = '%s___%s' % (data_key,'error_scaling')
        inverse_cov_scaled = inverse_cov/data_params[error_scaling_key]**2
        
        log_L_RES += -0.5*np.dot((flux_data_scaled-flux_rebin),np.dot(inverse_cov_scaled,(flux_data_scaled-flux_rebin)))
        
        if sum(np.isnan(flux_rebin))>0 or np.isnan(log_L_RES):
            self.NaN_spectRES += 1
            log_L_RES += -1e299
            return log_L_RES,wlen_rebin,flux_rebin
        
        return log_L_RES,wlen_rebin,flux_rebin
    
    def calc_log_likelihood(self,params):
        
        # construct parameters of FM out of fixed and sampled parameters
        
        chem_params = {}
        temp_params = {}
        clouds_params = {}
        physical_params = {}
        data_params = {}
        
        # go through all parameters
        # if they're a retrieved parameter, extract its value from params
        # else take the fixed parameter from the config file
        chem_params,temp_params,clouds_params,physical_params,data_params = read_forward_model_from_config(
            config_file=self.config,
            params=params,
            params_names=self.params_names,
            extract_param=True)
        
        self.function_calls += 1
        
        if self.timing:
            t0 = time()
        
        """Prior calculation of all input parameters"""
        log_prior = 0.
        
        """Metal abundances: check that their summed mass fraction is below 1."""
        # only applies for the free chem model
        if self.chem_model == 'free':
            metal_sum = 0.
            for param_name in chem_params.keys():
                if param_name in self.params_names:
                    log_prior += self.prior_obj.log_priors[param_name](chem_params[param_name])
                if not param_name in ['C/O','FeHs','log_Pquench_carbon']:
                    abund,model,param_nb = param_name.split('___')
                    if model == 'constant':
                        metal_sum += 1e1**chem_params[param_name]
                    else:
                        print('Only constant abundance profiles are currently taken into account')
            
            if metal_sum > 1.:
                log_prior += -1e299
        
        """prior of other parameters"""
        for param_name in self.params_names:
            if param_name in temp_params.keys():
                # temperature parameters
                log_prior += self.prior_obj.log_priors[param_name](temp_params[param_name])
            elif param_name in clouds_params.keys():
                # clouds parameters
                log_prior += self.prior_obj.log_priors[param_name](clouds_params[param_name])
            elif param_name in data_params.keys():
                # data parameters
                log_prior += self.prior_obj.log_priors[param_name](data_params[param_name])
        
        """return -inf if parameters fall outside prior distribution"""
        
        if (log_prior < -1e297):
            self.outside_priors += 1
            return -1e299
        
        """Calculate the log-likelihood"""
        log_likelihood = 0.
        log_L_PHOT = 0.
        log_L_CC = 0.
        log_L_RES = 0.
        
        wlen_ck,flux_ck,wlen_lbl,flux_lbl = None,None,None,None
        
        if self.data_obj.RESinDATA():
            RES_data_wlen,RES_data_flux,flux_err,inverse_cov,flux_data_std = self.data_obj.getRESSpectrum()
        
        # evaluate log-likelihood for FM using c-k mode
        wlen_RES,flux_RES = {},{}
        wlen_ck,flux_ck=None,None
        wlen_ck_SI,flux_ck_SI_scaled=None,None
        model_photometry = {}
        if self.forwardmodel_ck is not None:
            wlen_ck,flux_ck = self.forwardmodel_ck.calc_spectrum(
                      chem_model_params = chem_params,
                      temp_model_params = temp_params,
                      cloud_model_params = clouds_params,
                      physical_params = physical_params,
                      external_pt_profile = None,
                      return_profiles = False)
            
            wlen_ck_SI,flux_ck_SI = convert_units_to_SI(wlen_ck,flux_ck)
            flux_ck_SI_scaled = scale_flux(flux_ck_SI,radius=physical_params['R'],distance=physical_params['distance'])
            
            if sum(np.isnan(flux_ck_SI))>0 or sum(np.isnan(wlen_ck_SI))>0:
                self.NaN_spectra += 1
                return -1e299
            
            if self.data_obj.PHOTinDATA():
                # print('c-k photometry Log-L')
                log_L_PHOT,model_photometry = self.calc_log_L_PHOT(
                    wlen=wlen_ck_SI,
                    flux=flux_ck_SI_scaled)
            
            if self.data_obj.RES_data_with_ck:
                for instr in RES_data_wlen.keys():
                    if self.data_obj.RES_data_info[instr][0] == 'c-k':
                        # print('c-k RES-Log-L for',instr)
                        log_L_RES_temp,wlen_RES[instr],flux_RES[instr] = self.calc_log_L_RES(
                            wlen=wlen_ck_SI,
                            flux=flux_ck_SI_scaled,
                            data_params=data_params,
                            data_key=instr,
                            wlen_data=RES_data_wlen[instr],
                            flux_data=RES_data_flux[instr],
                            flux_err=flux_err[instr],
                            inverse_cov=inverse_cov[instr],
                            flux_data_std=flux_data_std[instr],
                            mode='c-k')
                        log_L_RES += log_L_RES_temp
        
        if self.data_obj.CCinDATA():
            CC_data_wlen,CC_data_flux = self.data_obj.getCCSpectrum()
            data_N,data_sf2 = self.data_obj.CC_data_N,self.data_obj.CC_data_sf2
        
        # evaluate log-likelihood for FM using lbl mode
        wlen_CC,flux_CC,sgfilter={},{},{}
        if self.forwardmodel_lbl is not None:
            
            for interval_key in self.lbl_itvls.keys():
                wlen_lbl,flux_lbl = self.forwardmodel_lbl[interval_key].calc_spectrum(
                      chem_model_params = chem_params,
                      temp_model_params = temp_params,
                      cloud_model_params = clouds_params,
                      physical_params = physical_params,
                      external_pt_profile = None,
                      return_profiles = False)

                wlen_lbl_SI,flux_lbl_SI = convert_units_to_SI(wlen_lbl,flux_lbl)
                flux_lbl_SI_scaled = scale_flux(flux_lbl_SI,radius=physical_params['R'],distance=physical_params['distance'])
                
                if sum(np.isnan(flux_lbl_SI_scaled))>0 or sum(np.isnan(wlen_lbl_SI))>0:
                    self.NaN_spectra += 1
                    return -1e299
                
                if self.data_obj.CCinDATA():
                    for instr in CC_data_wlen.keys():
                        if self.CC_to_lbl_itvls[instr] == interval_key:
                            # print('lbl CC-Log-L for',instr)
                            log_L_CC_temp,wlen_CC[instr],flux_CC[instr],sgfilter[instr] = self.calc_log_L_CC(
                                wlen=wlen_lbl_SI,
                                flux=flux_lbl_SI, # flux_lbl_SI_scaled
                                data_wlen=CC_data_wlen[instr],
                                data_flux=CC_data_flux[instr],
                                data_N=data_N[instr],
                                data_sf2=data_sf2[instr],
                                data_params=data_params,
                                data_key=instr
                                )
                            log_L_CC += log_L_CC_temp
                
                if self.data_obj.RESinDATA():
                    for instr in RES_data_wlen.keys():
                        if self.data_obj.RES_data_info[instr][0] == 'lbl':
                            if self.RES_to_lbl_itvls[instr] == interval_key:
                                # print('lbl RES-Log-L for',instr)
                                log_L_RES_temp,wlen_RES[instr],flux_RES[instr] = self.calc_log_L_RES(
                                    wlen=wlen_lbl_SI,
                                    flux=flux_lbl_SI_scaled,
                                    data_params=data_params,
                                    data_key=instr,
                                    wlen_data=RES_data_wlen[instr],
                                    flux_data=RES_data_flux[instr],
                                    flux_err=flux_err[instr],
                                    inverse_cov=inverse_cov[instr],
                                    flux_data_std=flux_data_std[instr],
                                    mode='lbl')
                                log_L_RES += log_L_RES_temp
        
        if self.timing:
            t1 = time()
            print('Forward Models and likelihood functions: ',t1-t0)
        
        self.computed_spectra += 1

        # diagnostic for C-C log
        self.diag_params = params
        
        log_likelihood += log_L_CC + log_L_RES + log_L_PHOT
        print(log_prior + log_likelihood)
        print("--> ", self.function_calls, " --> ", self.computed_spectra)
        if self.printing:
            if (self.function_calls%self.writing_threshold == 0):
                hours = (time() - self.start_time)/3600.0
                info_list = [self.function_calls, self.computed_spectra,
                             log_L_CC,log_L_RES,log_L_PHOT,log_likelihood, hours, 
                             self.nb_failed_DS ,self.NaN_spectra, self.NaN_spectRES, 
                             self.NaN_savgolay, self.NaN_crosscorrRV, self.NaN_photometry, self.outside_priors]
                with open(self.diag_file,'a') as f:
                    for i in np.arange(len(info_list)):
                        if (i == len(info_list) - 1):
                            f.write(str(info_list[i]).ljust(15) + "\n")
                        else:
                            f.write(str(info_list[i]).ljust(15) + " ")
                
                # diagnostic for C-C log
                info_list_CC = [self.function_calls,hours,self.diag_log_CC,self.diag_CC,self.diag_sg2,self.diag_sf2] + list(self.diag_params)
                with open(self.diag_CC_file,'a') as f:
                    for i in np.arange(len(info_list_CC)):
                        if (i == len(info_list_CC) - 1):
                            f.write(str(info_list_CC[i]).ljust(15) + "\n")
                        else:
                            f.write(str(info_list_CC[i]).ljust(15) + " ")
        
        if self.plotting:
            if (self.function_calls%self.plotting_threshold == 0):
                plot_path = self.output_path / 'model' / ('plot_%i.png' % int(self.function_calls/self.plotting_threshold))
                # only to avoid overwriting a plot
                if not path.exists(plot_path):
                    CC_data_wlen={}
                    CC_data_flux={}
                    data_RES_wlen={}
                    data_RES_flux={}
                    data_RES_err={}
                    data_RES_inv_cov={}
                    data_RES_flux_std={}
                    data_sim_wlen={}
                    data_sim_flux={}
                    data_PHOT_flux={}
                    data_PHOT_err={}
                    data_PHOT_filter={}
                    data_PHOT_filter_function={}
                    data_PHOT_filter_midpoint={}
                    data_PHOT_filter_width={}
                    if self.data_obj.CCinDATA():
                        CC_data_wlen,CC_data_flux = self.data_obj.getCCSpectrum()
                    if self.data_obj.RESinDATA():
                        data_RES_wlen,data_RES_flux,data_RES_err,data_RES_inv_cov,data_RES_flux_std = self.data_obj.getRESSpectrum()
                    if self.data_obj.PHOTinDATA():
                        data_sim_wlen,data_sim_flux = self.data_obj.getSimSpectrum()
                        data_PHOT_flux,data_PHOT_err,data_PHOT_filter,data_PHOT_filter_function,data_PHOT_filter_midpoint,data_PHOT_filter_width = self.data_obj.getPhot()
                    print('PLOTTING MODEL')
                    # plt.figure(figsize=(20,8))
                    # for key in wlen_RES.keys():
                    #     plt.plot(wlen_RES[key],flux_RES[key])
                    # plt.savefig(self.output_path / 'model' / ('plot%i.png' % int(self.function_calls/self.plotting_threshold)))
                    
                    plot_data(
                            self.config,
                            CC_wlen = CC_data_wlen,
                            CC_flux = CC_data_flux,
                            model_CC_wlen = wlen_CC,
                            model_CC_flux = flux_CC,
                            sgfilter = None,
                            RES_wlen = data_RES_wlen,
                            RES_flux = data_RES_flux,
                            RES_flux_err = data_RES_flux_std,
                            model_RES_wlen = wlen_RES,
                            model_RES_flux = flux_RES,
                            PHOT_midpoint = data_PHOT_filter_midpoint,
                            PHOT_width = data_PHOT_filter_width,
                            PHOT_flux = data_PHOT_flux,
                            PHOT_flux_err = data_PHOT_err,
                            PHOT_filter = data_PHOT_filter,
                            PHOT_sim_wlen = wlen_ck_SI,
                            PHOT_sim_flux = flux_ck_SI_scaled,
                            model_PHOT_flux = model_photometry,
                            output_file = str(self.output_path / 'model'),
                            plot_name = 'plot_%s' % int(self.function_calls/self.plotting_threshold),
                            save_plot=True)
                
                # plot CCF function
                plot_path = self.output_path / 'model' / ('CC_fct_%s.png' % int(self.function_calls/self.plotting_threshold))
                if self.data_obj.CCinDATA() and (not path.exists(plot_path)):
                    CC_data_wlen,CC_data_flux = self.data_obj.getCCSpectrum()
                    print('PLOTTING CCF')
                    ncols = len(CC_data_wlen.keys())
                    fig,axes = plt.subplots(ncols=ncols,figsize=(4,4*ncols))
                    for instr_i,instr in enumerate(CC_data_wlen.keys()):
                        if ncols==1:
                            ax=axes
                        else:
                            ax = axes[instr_i]
                        dRV,CC=crosscorrRV(
                            CC_data_wlen[instr],
                            CC_data_flux[instr],
                            wlen_CC[instr],
                            flux_CC[instr],
                            rvmin=self.config['hyperparameters']['rv']['rv_min'],
                            rvmax=self.config['hyperparameters']['rv']['rv_max'],
                            drv=self.config['hyperparameters']['rv']['drv'])
                        RV_max_i=np.argmax(CC)
                        ax.plot(dRV,CC,label=instr)
                        ax.axvline(dRV[RV_max_i],label='Max CC at RV={rv}'.format(rv=dRV[RV_max_i]))
                    plt.legend()
                    plt.title('log-L C-C ' + str(int(self.function_calls/self.plotting_threshold)))
                    plt.savefig(self.output_path / 'model' / ('CC_fct_%s.png' % int(self.function_calls/self.plotting_threshold)),dpi=100)
        
        if self.timing and self.plotting:
            t2 = time()
            print('Printing and plotting: ',t2-t1)
        if np.isnan(log_prior + log_likelihood):
            return -1e299
        else:
            return log_prior + log_likelihood