# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:40:40 2021

@author: jeanh
"""
import matplotlib.pyplot as plt
import numpy as np
from core.read import read_forward_model_from_config
from pathlib import Path

# prior likelihood functions
class Prior:
    def __init__(self,
                 config_file
                 ):
        self.RANGE = {}
        self.log_priors = {}
        self.log_cube_priors = {}

        self.read_prior_config(config_file)

        self.params_names = self.getParams()
        return
    
    
    def getParams(self):
        return list(self.RANGE.keys())
    
    def getRANGE(self):
        return self.RANGE
    def getLogPriors(self):
        return self.log_priors
    def getLogCubePriors(self):
        return self.log_cube_priors
    
    def plot(self,output_dir='',save_plot=True):
        nb_params=len(self.RANGE.keys())
        fig,ax = plt.subplots(nrows = 1,ncols=nb_params,figsize=(nb_params*3,3))
        for col_i,name in enumerate(self.RANGE.keys()):
            x_arr = np.linspace(self.RANGE[name][0],self.RANGE[name][1],100)
            ax[col_i].plot(x_arr,[self.log_priors[name](x) for x in x_arr])
            ax[col_i].set_title(name)
        plt.tight_layout()
        if save_plot:
            fig.savefig(Path(output_dir) / 'priors.png',dpi=300)
    
    def read_prior_config(self,config_file):
        
        RANGE,LOG_PRIORS,CUBE_PRIORS = read_forward_model_from_config(config_file,params=[],params_names=[],extract_param=False)
        
        self.RANGE = RANGE
        self.log_priors = LOG_PRIORS
        self.log_cube_priors = CUBE_PRIORS
        
        return 