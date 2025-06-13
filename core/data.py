# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:07 2025

@author: Jean Hayoz
"""
from scipy.interpolate import interp1d
import numpy as np
from numpy.linalg import inv
import pandas as pd
from glob import glob
import json
from pathlib import Path
import os
from itertools import combinations
from .plotting import plot_data
from .util import get_extinction

class Data:
    def __init__(
        self,
        photometry_file = 'photometry.csv',
        spectroscopy_files = {},
        contrem_spectroscopy_files = {},
        photometry_filter_dir = ''):
        
        # properties
        self.extinction = None
        
        # spectroscopy
        self.spectral_resolution = {}
        # continuum-removed spectrum, aka used with cross-correlation spectroscopy
        self.CC_data_wlen = {}
        self.CC_data_flux = {}
        self.CC_data_N    = {}
        self.CC_data_sf2  = {}
        # continuum-included spectrum, aka used with method of residuals
        self.RES_data_wlen = {}
        self.RES_data_flux = {}
        self.RES_cov_err = {}
        self.RES_inv_cov = {}
        self.RES_data_flux_err = {}
        
        # photometric data
        self.PHOT_data_flux = {}
        self.PHOT_data_err = {}
        self.PHOT_data_filter = {}
        self.PHOT_filter_function = {}
        self.PHOT_filter_midpoint = {}
        self.PHOT_filter_width = {}
        # simulated continuum-included spectrum for photometry
        self.PHOT_sim_spectrum_wlen = {}
        self.PHOT_sim_spectrum_flux = {}
        
        photometry_included = not photometry_file is None
        spectroscopy_included = len(spectroscopy_files) > 0
        contrem_spectroscopy_included = len(contrem_spectroscopy_files) > 0
        
        if photometry_included:
            if photometry_file[-4:] != '.csv':
                print('WARNING: only csv files accepted for the moment.')
            else:
                self.read_photometry(photometry_file=photometry_file,photometry_filter_dir=photometry_filter_dir)
        
        if spectroscopy_included:
            self.read_spectroscopy_calib(spectroscopy_files=spectroscopy_files)
        
        if contrem_spectroscopy_included:
            self.read_spectroscopy_contrem(spectroscopy_files=contrem_spectroscopy_files)
        
        # calculate spectral resolution for spectroscopy
        for file_key,wlen in {**self.CC_data_wlen,**self.RES_data_wlen}.items():
            self.spectral_resolution[file_key] = np.mean(wlen[1:]/(wlen[1:]-wlen[:-1]))
        return 

    def read_spectroscopy_contrem(self,spectroscopy_files):
        for file_key in spectroscopy_files.keys():
            spectrum_pd = pd.read_csv(spectroscopy_files[file_key])
            columns = spectrum_pd.columns
            wlen,flux = spectrum_pd[['wlen','flux']].values.transpose()
            
            self.CC_data_wlen[file_key] = wlen
            self.CC_data_flux[file_key] = flux
            self.CC_data_N[file_key]    = len(wlen)
            self.CC_data_sf2[file_key]  = 1./len(wlen)*np.sum(flux**2)
        return

    def read_spectroscopy_calib(self,spectroscopy_files):
        for file_key in spectroscopy_files.keys():
            spectrum_pd = pd.read_csv(spectroscopy_files[file_key])
            columns = spectrum_pd.columns
            wlen,flux = spectrum_pd[['wlen','flux']].values.transpose()
            self.RES_data_wlen[file_key] = wlen
            self.RES_data_flux[file_key] = flux
            if 'error' in columns:
                error = spectrum_pd['error'].values
                cov = np.diag(error**2)
                self.RES_inv_cov[file_key] = np.diag(1./error**2) # np.where(np.diag(error)==0,0,np.diag(1./error**2)) if 0 in error
            elif 'cov_0' in columns:
                cov = spectrum_pd[['cov_%i' % i for i in range(len(wlen))]].values
                error = np.sqrt(np.diag(cov))
                self.RES_inv_cov[file_key] = inv(cov)
            
            self.RES_cov_err[file_key] = cov
            self.RES_data_flux_err[file_key] = error
        return
            
        
    def read_photometry(self,photometry_file,photometry_filter_dir):
        photometry_pd = pd.read_csv(photometry_file)
        columns = np.array(['Author','Filter','effective_wavelength','effective_width','Flux','Error'])
        mask_found_cols = np.isin(columns,photometry_pd.columns)
        if np.sum(mask_found_cols) != len(columns):
            print('WARNING: columns ',columns[mask_found_cols],' not found!')
        for index,row in photometry_pd.iterrows():
            author,filt,wlen,eff_width,flux,flux_err = row[columns]
            
            db_name = '%s_%s' % (author,filt)
            self.PHOT_data_flux[db_name] = flux
            self.PHOT_data_err[db_name] = flux_err
            self.PHOT_filter_midpoint[db_name] = wlen
            self.PHOT_filter_width[db_name] = eff_width

            all_filters = sorted(glob(photometry_filter_dir + '*.json'))
            all_filters_names = list(map(lambda x: Path(x).stem,all_filters))
            filt_name_db = filt.replace('/','_').replace('.','-')
            filt_index = all_filters_names.index(filt_name_db)
            filter_path = all_filters[filt_index]
            with open(filter_path,'r') as f:
                filter_info = json.loads(f.read())
            self.PHOT_data_filter[db_name] = [filter_info['filter_profile']['wlen'],filter_info['filter_profile']['trsm']]
            self.PHOT_filter_function[db_name] = interp1d(self.PHOT_data_filter[db_name][0],self.PHOT_data_filter[db_name][1],bounds_error=False,fill_value=0.)
        return

    def deredden_all_data(self,Av):
        print('Dereddening all data by Av=%.2f' % Av)
        self.extinction = Av
        # CC data
        for instr in self.CC_data_wlen.keys():
            print('Dereddening %s' % instr)
            extinction = get_extinction(self.CC_data_wlen[instr],Av)
            self.CC_data_flux[instr] = self.CC_data_flux[instr]/extinction
            self.CC_data_sf2[instr]  = 1./len(self.CC_data_wlen[instr])*np.sum(self.CC_data_flux[instr]**2)
        # RES data
        for instr in self.RES_data_wlen.keys():
            print('Dereddening %s' % instr)
            extinction = get_extinction(self.RES_data_wlen[instr],Av)
            self.RES_data_flux[instr] = self.RES_data_flux[instr]/extinction
            self.RES_data_flux_err[instr] = self.RES_data_flux_err[instr]*(1./extinction)
            self.RES_cov_err[instr] = self.RES_cov_err[instr]*(1./extinction)**2
            self.RES_inv_cov[instr] = self.RES_inv_cov[instr]/(1./extinction)**2
        # PHOT data
        for instr in self.PHOT_data_flux.keys():
            print('Dereddening %s' % instr)
            extinction = get_extinction(self.PHOT_filter_midpoint[instr],Av)
            self.PHOT_data_flux[instr] = self.PHOT_data_flux[instr]/extinction
            self.PHOT_data_err[instr] = self.PHOT_data_err[instr]/extinction
        print('Done!')
    
    def rescale_CC_data(self,scale=1e-15):
        print('Rescaling all data by ', scale)
        # CC data
        for instr in self.CC_data_wlen.keys():
            std = np.std(self.CC_data_flux[instr])
            self.CC_data_flux[instr] = self.CC_data_flux[instr]*scale/std
            self.CC_data_sf2[instr]  = 1./len(self.CC_data_wlen[instr])*np.sum(self.CC_data_flux[instr]**2)
    
    def getCCSpectrum(self):
        return self.CC_data_wlen,self.CC_data_flux
    def getRESSpectrum(self):
        return self.RES_data_wlen,self.RES_data_flux,self.RES_cov_err,self.RES_inv_cov,self.RES_data_flux_err
    def getSimSpectrum(self):
        return self.PHOT_sim_spectrum_wlen,self.PHOT_sim_spectrum_flux
    def getPhot(self):
        return self.PHOT_data_flux,self.PHOT_data_err,self.PHOT_data_filter,self.PHOT_filter_function,self.PHOT_filter_midpoint,self.PHOT_filter_width
    
    def CCinDATA(self):
        if isinstance(self.CC_data_wlen,dict):
            return len(self.CC_data_wlen.keys())>0
        else:
            return self.CC_data_wlen is not None
    def RESinDATA(self):
        if isinstance(self.RES_data_wlen,dict):
            return len(self.RES_data_wlen.keys())>0
        else:
            return self.RES_data_wlen is not None
    def PHOTinDATA(self):
        if isinstance(self.PHOT_data_filter,dict):
            return len(self.PHOT_data_filter.keys())>0
        else:
            return self.PHOT_data_filter is not None
    
    # calculates necessary range for retrieval using lbl mode
    
    def wlen_details(self,wlen):
        wlen_range = [wlen[0],
                         wlen[-1]]
        wlen_stepsize = max([wlen[i+1]-wlen[i] for i in range(len(wlen)-1)])
        return wlen_range,wlen_stepsize
    
    def min_max_range(self,wlen_dict):
        return [
                min([wlen_dict[key][0] for key in wlen_dict.keys()]),
               max([wlen_dict[key][-1] for key in wlen_dict.keys()])]
    def max_stepsize(self,wlen_dict):
        return max([max([wlen_dict[key][i+1]-wlen_dict[key][i] for i in range(len(wlen_dict[key])-1)]) for key in wlen_dict.keys()])
    
    def wlen_details_lbl(self):
        wlen_range_CC,wlen_stepsize_CC = {},{}
        wlen_range_RES,wlen_stepsize_RES = {},{}
        if self.CCinDATA():
            for key in self.CC_data_wlen.keys():
                wlen_range_CC[key],wlen_stepsize_CC[key] = self.wlen_details(self.CC_data_wlen[key])
        if self.RESinDATA():
            for key in self.RES_data_wlen.keys():
                wlen_range_RES[key],wlen_stepsize_RES[key] = self.wlen_details(self.RES_data_wlen[key])
        if self.CCinDATA() and self.RESinDATA():
            outer_wlen_range_CC=self.min_max_range(wlen_range_CC)
            outer_wlen_range_RES=self.min_max_range(wlen_range_RES)
            outer_wlen_range = [min(outer_wlen_range_CC[0],outer_wlen_range_RES[0]),
                                max(outer_wlen_range_CC[1],outer_wlen_range_RES[1])]
            # actually only care about CC stepsize
            #larger_stepsize = max(self.max_stepsize(self.CC_data_wlen),self.max_stepsize(self.RES_data_wlen))
            larger_stepsize = self.max_stepsize(self.CC_data_wlen)
            return outer_wlen_range,larger_stepsize
        elif self.CCinDATA():
            return self.min_max_range(wlen_range_CC),self.max_stepsize(self.CC_data_wlen)
        elif self.RESinDATA():
            return self.min_max_range(wlen_range_RES),self.max_stepsize(self.RES_data_wlen)
        else:
            return None,None
    
    
    # calculates necessary range for retrieval using c-k mode
    
    def wlen_range_ck(self):
        wlen_range_PHOT = None
        if self.PHOTinDATA():
            MinWlen = min([self.PHOT_data_filter[instr][0][0] for instr in self.PHOT_data_filter.keys()])
            MaxWlen = max([self.PHOT_data_filter[instr][0][-1] for instr in self.PHOT_data_filter.keys()])
            wlen_range_PHOT = [MinWlen,MaxWlen]
        return wlen_range_PHOT
    
    def distribute_FMs(self):
        self.ck_FM_interval = None
        self.RES_data_with_ck = False
        self.RES_data_info = {}
        self.CC_data_info = {}
        
        self.disjoint_lbl_intervals = {}
        self.RES_to_lbl_intervals = {}
        self.CC_to_lbl_intervals = {}
        
        all_lbl_intervals = {}
        interval_naming = 0
        
        if self.PHOTinDATA():
            self.ck_FM_interval = self.wlen_range_ck()
        
        if not (self.CCinDATA() or self.RESinDATA()):
            return 
        
        # collect all intervals of data that need a forward model with lbl mode, and those that need c-k mode
        if self.CCinDATA():
            for instr in self.CC_data_wlen.keys():
                all_lbl_intervals[interval_naming] = ['CC',instr,[self.CC_data_wlen[instr][0],self.CC_data_wlen[instr][-1]]]
                self.CC_data_info[instr] = ['lbl',[self.CC_data_wlen[instr][0],self.CC_data_wlen[instr][-1]],max(self.CC_data_wlen[instr][1:] - self.CC_data_wlen[instr][:-1])]
                interval_naming += 1
        if self.RESinDATA():
            for instr in self.RES_data_wlen.keys():
                max_resolution = max(self.RES_data_wlen[instr][1:]/(self.RES_data_wlen[instr][1:] - self.RES_data_wlen[instr][:-1]))
                if max_resolution < 1001:
                    self.RES_data_info[instr] = ['c-k',[self.RES_data_wlen[instr][0],self.RES_data_wlen[instr][-1]],max(self.RES_data_wlen[instr][1:] - self.RES_data_wlen[instr][:-1])]
                    self.RES_data_with_ck = True
                else:
                    self.RES_data_info[instr] = ['lbl',[self.RES_data_wlen[instr][0],self.RES_data_wlen[instr][-1]],max(self.RES_data_wlen[instr][1:] - self.RES_data_wlen[instr][:-1])]
                    all_lbl_intervals[interval_naming] = ['RES',instr,[self.RES_data_wlen[instr][0],self.RES_data_wlen[instr][-1]]]
                    interval_naming += 1
        
        # increase range of c-k FM
        if self.RESinDATA():
            if not self.PHOTinDATA():
                self.ck_FM_interval = [min([self.RES_data_wlen[key][0] for key in self.RES_data_wlen.keys()]),max([self.RES_data_wlen[key][-1] for key in self.RES_data_wlen.keys()])]
            else:
                print(self.ck_FM_interval)
                for key in self.RES_data_wlen.keys():
                    if self.RES_data_info[key][0] == 'c-k':
                        if self.RES_data_wlen[key][0] < self.ck_FM_interval[0]:
                            self.ck_FM_interval[0] = self.RES_data_wlen[key][0]
                        if self.RES_data_wlen[key][-1] > self.ck_FM_interval[1]:
                            self.ck_FM_interval[1] = self.RES_data_wlen[key][-1]
        
        
        final_intervals = {}
        for key in all_lbl_intervals.keys():
            final_intervals[key] = all_lbl_intervals[key][2]
        
        # merge intervals of data requiring lbl mode that overlap
        if len(final_intervals.keys()) > 1:
            working = True
            while working:
                for key_i,key_j in combinations(final_intervals.keys(),2):
                    if key_i == key_j:
                        continue
                    if do_arr_intersect(final_intervals[key_i],final_intervals[key_j]):
                        final_intervals[key_i] = [min(final_intervals[key_i][0],final_intervals[key_j][0]),max(final_intervals[key_i][-1],final_intervals[key_j][-1])]
                        final_intervals.pop(key_j)
                        break
                else:
                    working=False
        
        self.disjoint_lbl_intervals = final_intervals
        self.disjoint_lbl_intervals_max_stepsize = {key:0 for key in self.disjoint_lbl_intervals.keys()}
        self.disjoint_lbl_intervals_min_stepsize = {key:100 for key in self.disjoint_lbl_intervals.keys()}
        
        if self.CCinDATA():
            for key in self.CC_data_wlen.keys():
                for interval_i in self.disjoint_lbl_intervals.keys():
                    if do_arr_intersect(self.CC_data_wlen[key],self.disjoint_lbl_intervals[interval_i]):
                        self.CC_to_lbl_intervals[key] = interval_i
                        self.disjoint_lbl_intervals_max_stepsize[interval_i] = max(self.disjoint_lbl_intervals_max_stepsize[interval_i],self.CC_data_info[key][2])
                        self.disjoint_lbl_intervals_min_stepsize[interval_i] = min(self.disjoint_lbl_intervals_min_stepsize[interval_i],self.CC_data_info[key][2])
                        break
        
        if self.RESinDATA():
            for key in self.RES_data_wlen.keys():
                if self.RES_data_info[key][0] == 'lbl':
                    for interval_i in self.disjoint_lbl_intervals.keys():
                        if do_arr_intersect(self.RES_data_wlen[key],self.disjoint_lbl_intervals[interval_i]):
                            self.RES_to_lbl_intervals[key] = interval_i
                            self.disjoint_lbl_intervals_max_stepsize[interval_i] = max(self.disjoint_lbl_intervals_max_stepsize[interval_i],self.RES_data_info[key][2])
                            self.disjoint_lbl_intervals_min_stepsize[interval_i] = min(self.disjoint_lbl_intervals_min_stepsize[interval_i],self.RES_data_info[key][2])
                            break
    
    
    def plot(self,config,
             output_dir='',
             plot_name = 'plot',
             title = 'Spectrum',
             inset_plot=True,
             plot_errorbars=True,
             save_plot=True):
        
        fig = plot_data(
            config,
            CC_wlen       = self.CC_data_wlen,
            CC_flux       = self.CC_data_flux,
            RES_wlen      = self.RES_data_wlen,
            RES_flux      = self.RES_data_flux,
            RES_flux_err  = self.RES_cov_err,
            PHOT_midpoint = self.PHOT_filter_midpoint,
            PHOT_width    = self.PHOT_filter_width,
            PHOT_flux     = self.PHOT_data_flux,
            PHOT_flux_err = self.PHOT_data_err,
            PHOT_filter   = self.PHOT_data_filter,
            #PHOT_sim_wlen = self.PHOT_sim_spectrum_wlen,
            #PHOT_sim_flux = self.PHOT_sim_spectrum_flux,
            inset_plot    = inset_plot,
            output_file   = output_dir,
            title         = title,
            plot_name     = plot_name,
            plot_errorbars= plot_errorbars,
            save_plot=save_plot,
            extinction = self.extinction
            )
        return fig


def index_last_slash(string):
    if '/' not in string:
        return 0
    else:
        return len(string)-string[::-1].index('/')

def do_arr_intersect(arr1,arr2):
    # whether arrays arr1 and arr2 almost intersect (by 0.1 micron)
    a=arr1[0]
    b=arr1[-1]
    c=arr2[0]
    d=arr2[-1]
    return not (c > b + 0.1 or a > d + 0.1)