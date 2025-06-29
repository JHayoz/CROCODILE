# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:39:54 2021

@author: jeanh
"""
import csv
import numpy as np
import os
import re
from mendeleev import element
import sys
from PyAstronomy.pyasl import crosscorrRV,fastRotBroad,rotBroad
import scipy.constants as cst
from scipy.interpolate import interp1d
from scipy.special import erfcinv
from scipy.stats import truncnorm,skewnorm,gaussian_kde,norm
from scipy.optimize import curve_fit
from itertools import product
from dust_extinction.parameter_averages import G23
from astropy import units as u

from config_petitRADTRANS import *

RADIUS_J = 69911*1000
MASS_J = 1.898*1e27
mlower = 3e-4
mupper = 3e5
SQRT2 = np.sqrt(2.)
SQRT2PI = np.sqrt(2*np.pi)

def get_std_percentiles(nb_stds):
    quantiles = [
        (1-0.9973)/2,
        (1-0.9545)/2,
        (1-0.6827)/2,
        0.5,
        1-(1-0.6827)/2,
        1-(1-0.9545)/2,
        1-(1-0.9973)/2]
    return quantiles[3-nb_stds:3+nb_stds+1]

def open_spectrum(file_dir):
    try:
        with open(file_dir,'r') as f:
            datareader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
            data = np.array([row for row in datareader])
            if np.shape(data)[0] == 1 and len(np.shape(data)) > 1:
                data = data[0]
    except ValueError:
        data = np.loadtxt(file_dir)
    return data

def open_spectra(dir_path):
    data = {}
    for file in os.listdir(dir_path):
        name = file[:-4]
        data[name] = open_spectrum(dir_path + '/' + file)
    return data

def open_photometry(file_dir):
    with open(file_dir,'r') as f:
        datareader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        data = np.array([row for row in datareader][0]).astype(np.float64)
    return data

def open_filter(file_dir):
    with open(file_dir,'r') as f:
        datareader = csv.reader(f)
        data = np.array([row for row in datareader]).astype(np.float64)
    return data

def open_filter_dir(file_dir,true_dir = False):
    cwd = os.getcwd()
    filter_data = {}
    if true_dir:
        file_dir_to_search = file_dir
    else:
        file_dir_to_search = cwd+'/'+file_dir
    for file in os.listdir(file_dir_to_search):
        instr = file[:-4]
        filter_data[instr] = open_filter(file_dir+'/'+file)
    return filter_data

def save_spectrum(wlen,flux,save_dir= '',save_name='/spectrum'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_dir+save_name+'.txt','w') as f:
        writer = csv.writer(f)
        writer.writerow(wlen)
        writer.writerow(flux)

def save_spectra(wlen_dict,flux_dict,save_dir = 'spectra',save_name = ''):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for key in wlen_dict.keys():
        save_spectrum(wlen_dict[key],flux_dict[key],save_dir= save_dir,save_name='/'+save_name+str(key))
    print('Spectra saved')

def save_photometry(photometry,photometry_err,phot_midpoint,phot_width,save_dir):
    for instr in photometry.keys():
        with open(save_dir+'/'+instr+'.txt','w') as f:
            writer = csv.writer(f)
            writer.writerow([photometry[instr],photometry_err[instr],phot_midpoint[instr],phot_width[instr]])

def save_1d_data(data,save_dir='data',save_name=''):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_dir+save_name+'.txt','w') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def save_lines(data,save_dir = 'data',save_name=''):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_dir+save_name+'.txt','w') as f:
        writer = csv.writer(f)
        for line in data:
            writer.writerow(line)

def calc_FWHM(array):
    # assuming that it's a gauss-like curve, not necessarily symmetrical
    
    max_index = np.argmax(array)
    max_y = array[max_index]
    
    values_higher_half = np.array(array) > max_y/2
    index_lower,index_higher = 0,len(array)
    for index in range(len(array)):
        if values_higher_half[index]:
            index_higher = index
        if values_higher_half[len(array)-index-1]:
            index_lower = len(array)-index - 1
    return index_lower,max_index,index_higher

def calc_median(array):
    return calc_quantile(array,q=0.5)

def calc_quantile(array,q=0.5):
    cum_distr = 0
    index = 0
    while cum_distr < sum(array)*q and index < len(array):
        cum_distr += array[index]
        index += 1
    if index == len(array)-1:
        print('percentile {quant}-th not found'.format(quant=q*100))
        return None
    else:
        return (array[index] + array[index-1])*0.2
    

def calc_median_filter(f,N_points):
    """
    f is a filter transmission function
    output: median of the filter
    """
    wvl = np.linspace(0.2,8,N_points)
    transmission = [f(xx) for xx in wvl]
    integral = np.trapz(transmission,wvl)
    wvl_i = 4
    cum_distr = 0.
    while cum_distr < integral/2 and wvl_i < len(wvl):
        cum_distr = np.trapz([f(xx) for xx in wvl[:wvl_i]],wvl[:wvl_i])
        wvl_i += 1
    if wvl_i == len(wvl):
        print('median wvl not found')
        return None
    return wvl[wvl_i]

def effective_width_filter(f,N_points):
    """
    f is filter transmission function
    output: width of transmission function if it were a rectangle of equivalent surface area
    """
    wvl = np.linspace(0.2,8,N_points)
    transmission = [f(xx) for xx in wvl]
    area = np.trapz(transmission,wvl)
    max_transm = max(transmission)
    return area/max_transm

def synthetic_photometry(wlen,flux,f):
    """
    f is filter transmission function
    output: synthetic photometry of flux through f
    """
    integrand1 = np.trapz([f(x)*flux[i] for i,x in enumerate(wlen)],wlen)
    integrand2 = np.trapz([f(x) for i,x in enumerate(wlen)],wlen)
    return integrand1/integrand2

def set_const_abundance_param(chem_model_params,abund,val):
    abund,model,param_i = abund,'constant','param_0'
    param = '___'.join([abund,model,param_i])
    chem_model_params[param]=val
    return chem_model_params

def get_abundance_params(config_file):
    abundances_names = []
    if config_file['retrieval']['FM']['chemistry']['model'] == 'free':
        for mol in config_file['retrieval']['FM']['chemistry']['parameters'].keys():
            for param_nb in config_file['retrieval']['FM']['chemistry']['parameters'][mol].keys():
                if param_nb == 'model':
                    continue
                param_name = '%s___%s___%s' % (mol,config_file['retrieval']['FM']['chemistry']['parameters'][mol]['model'],param_nb)
                abundances_names += [param_name]
    else:
        print('Chemistry model is not free')
    return abundances_names

def nice_param_name(param,config_file):
    abundances_names = get_abundance_params(config_file)
    if param == 'FeHs':
        return '[Fe/H]'
    if param in abundances_names:
        return abundance_param_to_nice(param)
    else:
        if param == 't_equ':
            return '$T_{\mathrm{equ}}$'
        elif param == 't_int':
            return '$T_{\mathrm{int}}$'
        elif param == 'log_gravity':
            return '$\log g$'
        elif param == 'log_R':
            return '$\log R$'
        elif param == 'log_M':
            return '$\log M$'
        elif param == 'log_gamma':
            return '$\log \gamma$'
        elif param == 'log_kappa_IR':
            return '$\log \kappa_{\mathrm{IR}}$'
        elif param == 'P0':
            return 'P0'
        else:
            return param

def nice_name(molecule):
    final_name = ''
    if '_' in molecule:
        final_name += molecule[:molecule.index('_')]
    else:
        final_name += molecule
    final_string = ''
    for char in final_name:
        if char.isnumeric():
            final_string += '$_{'+char+'}$'
        else:
            final_string += char
    if molecule in ['CO_36','13CO']:
        final_string = '$^{13}$C$^{16}$O'
    return final_string

def abundance_param_to_nice(param):
    mol,model,param_nb = param.split('___')
    mol_name = nice_name(mol)
    if model == 'constant':
        mol_name_str = mol_name
    else:
        mol_name_str = mol_name + ':' + param_nb[-1]
    return mol_name_str

def name_lbl_to_ck(abundance):
    name_ck_dict = {
        'CO_36':'13CO',
        'H2S_main_iso':'H2S',
        'H2O_main_iso':'H2O',
        'CO2_main_iso':'CO2',
        'CO_main_iso':'CO',
        'HCN_main_iso':'HCN',
        'H2_main_iso':'H2',
        'CH4_main_iso':'CH4',
        'CH4_hargreaves_main_iso':'CH4',
        'NH3_main_iso':'NH3_HITRAN',
        'TiO_all_iso':'TiO',
        'O3_main_iso':'O3',
        'SiO_main_iso':'SiO_Chubb',
        'VO':'VO',
        'PH3_main_iso':'PH3',
        'FeH_main_iso':'FeH_Chubb'}
    if abundance in name_ck_dict.values() or '_' not in abundance:
        return abundance
    else:
        return name_ck_dict[abundance]
def name_ck(molecule):
    if molecule == 'NH3_main_iso':
        return 'NH3_HITRAN'
    if molecule == 'FeH_main_iso':
        return 'FeH'
    if molecule == 'SiO_main_iso':
        return 'SiO_Chubb'
    if not('_' in molecule):
        return molecule
    else:
        return molecule[:molecule.index('_')]
    
def convert_to_ck_names(abundances):
    return [name_lbl_to_ck(mol) for mol in abundances]

def convert_units_to_SI(wlen, flux):
    # converts a flux in CSG units to SI units, and from F_nu to F_lambda
    wlen_SI = wlen*1e4
    #flux_temp = flux*1e-3/1e-6*R_planet**2/distance**2
    #flux_temp = flux*1e-3*nc.c*1e-2/1e-6/wlen_temp**2*(R_planet**2/distance**2)
    # flux_temp = flux*1e-3*cst.c/1e-6/wlen_temp**2*(R_planet**2/distance**2)
    flux_SI = flux*1e-3*cst.c/1e-6/wlen_SI**2
    
    return wlen_SI,flux_SI

def scale_flux(flux,radius,distance):
    # assuming wlen and flux in SI units, flux in W/m2/microns
    # distance in parsec
    # radius in Jupiter radius
    RADIUS_J = 69911*1000
    R_planet_m=radius*RADIUS_J
    pc_to_meter = 30856775812799588
    distance_m = distance*pc_to_meter
    
    flux_scaled = flux*((R_planet_m**2)/(distance_m**2))
    return flux_scaled
    

def make_phot(phot_filters,wlen,flux,phot_data_flux = None,phot_data_flux_err = None,N_points=1000):
    filter_function={}
    for instr in phot_filters.keys():
        filter_function[instr] = interp1d(phot_filters[instr][0],phot_filters[instr][1],bounds_error=False,fill_value=0.)
    
    photometry = {}
    photometry_err={}
    phot_midpoint = {}
    phot_width = {}
    for instr in phot_filters.keys():
        photometry[instr] = synthetic_photometry(wlen,flux,filter_function[instr])
        if phot_data_flux is None and phot_data_flux_err is None:
            photometry_err[instr] = 0.1*photometry[instr]
        else:
            photometry_err[instr] = phot_data_flux_err[instr]/phot_data_flux[instr]*photometry[instr]
        phot_midpoint[instr] = calc_median_filter(filter_function[instr],N_points)
        phot_width[instr] = effective_width_filter(filter_function[instr],N_points)
    return photometry,photometry_err,phot_midpoint,phot_width

def calc_cov_matrix(cov_data,flux_sim,flux_data):
    cov_RES = np.array([[cov_data[i][j]*flux_sim[i]/flux_data[i]
                             *flux_sim[j]/flux_data[j] 
                             for i in range(len(flux_data))]
                            for j in range(len(flux_data))])
    return cov_RES

def print_infos_sim_data(output_dir,
                         ab_metals,
                         temp_params,
                         clouds_params,
                         model,
                         mode,
                         resolution,
                         rv,
                         mol_abund_profile,
                         pt_profile,
                         CC_wvl_file,
                         RES_file,
                         filter_dir
                         ):
        with open(output_dir+'/details.txt','w') as f:
            f.write('This file gives details about the simulated data \n \n')
            f.write('Line opacities: '+ ', '.join(list(map(str,ab_metals.keys()))) + '\n')
            if mol_abund_profile is not None:
                f.write('The molecular abundances were simulated using an external profile \n')
            else:
                f.write('Vertically constant molecular abundances used (in log10): ' + ', '.join([name+':'+str(ab_metals[name]) for name in ab_metals.keys()]) +'\n')
            if pt_profile is not None:
                f.write('Temperature parameters: an external profile was used \n')
            else:
                f.write('Temperature parameters: '+ ', '.join(list(map(str,temp_params.keys()))) + '\n')
            f.write('Forward model used: '+model + '\n')
            f.write('Clouds parameters: '+ ', '.join(list(map(str,clouds_params.keys()))) + '\n')
            f.write('Mode: '+mode +'Resolution: '+str(resolution) +'\n')
            f.write('RV: '+str(rv) + '\n')
            f.write('Wvl file used for CC: ' +str(CC_wvl_file) +'\n')
            f.write('File used for RES: ' + RES_file + '\n')
            f.write('Filter trsm fct used: '+str(filter_dir) +'\n')

def quantiles_to_string(quantiles,decimals = 2):
    string1 = '{q:.'+str(decimals)+'f}'
    string2 = '{q2:.'+str(decimals)+'f}'
    string3 = '{q1:.'+str(decimals)+'f}'
    return string1.format(q = quantiles[1]) +'$^{+'+string2.format(q2=quantiles[2]-quantiles[1]) +'}_{-'+string3.format(q1=quantiles[1]-quantiles[0])+'}$'

def nb_Os(name):
    Os = 0
    for i,char in enumerate(name):
        if char == 'O':
            if i < len(name)-1 and name[i+1].isdigit():
                Os += int(name[i+1])
            else:
                Os += 1
    return Os

def nb_Cs(name):
    Cs = 0
    for i,char in enumerate(name):
        if char == 'C':
            if i < len(name)-1 and name[i+1].isdigit():
                Cs += int(name[i+1])
            else:
                Cs += 1
    return Cs

def get_MMWs(mol):
    
    MMWs = {}
    MMWs['H2'] = 2.
    MMWs['H2_main_iso'] = 2.
    MMWs['He'] = 4.
    MMWs['H2O'] = 18.
    MMWs['H2O_main_iso'] = 18.
    MMWs['CH4'] = 16.
    MMWs['CH4_main_iso'] = 16.
    MMWs['CH4_hargreaves_main_iso'] = 16.
    MMWs['CO2'] = 44.
    MMWs['CO2_main_iso'] = 44.
    MMWs['CO2_Chubb'] = 44.
    MMWs['CO'] = 28.
    MMWs['CO_all_iso'] = 28.
    MMWs['CO_main_iso'] = 28.
    MMWs['CO_36'] = 29.
    MMWs['13CO'] = 29.
    MMWs['Na'] = 23.
    MMWs['K'] = 39.
    MMWs['NH3'] = 17.
    MMWs['NH3_main_iso'] = 17.
    MMWs['HCN'] = 27.
    MMWs['HCN_main_iso'] = 27.
    MMWs['C2H2,acetylene'] = 26.
    MMWs['PH3'] = 34.
    MMWs['H2S'] = 34.
    MMWs['H2S_main_iso'] = 34.
    MMWs['VO'] = 67.
    MMWs['TiO'] = 64.
    MMWs['TiO_all_iso'] = 64.
    MMWs['SiO_main_iso'] = 44.
    MMWs['SiO'] = 44.
    MMWs['SiO_Chubb'] = 44.
    MMWs['O3_main_iso'] = 48.
    MMWs['PH3_main_iso'] = 34.
    MMWs['NH3_HITRAN'] = 17.
    MMWs['FeH_main_iso'] = 56.8
    MMWs['FeH_Chubb'] = 56.8
    MMWs['FeH'] = 56.8
    MMWs['N2'] = 14
    return MMWs[mol]

def CO_ratio_VMRs(abundances):
    VMR_Os = 0.
    VMR_Cs = 0.
    for j,name in enumerate(abundances.keys()):
        if nb_Os(name) > 0:
            VMR_Os += nb_Os(name)*10**abundances[name]/get_MMWs(name)
        if nb_Cs(name) > 0:
            VMR_Cs += nb_Cs(name)*10**abundances[name]/get_MMWs(name)
    if not(VMR_Os == 0):
        RATIO = VMR_Cs/VMR_Os
    else:
        print('there is no oxygen')
        RATIO = 0
    return RATIO
    
def CO_ratio_new(abundances):
    mass_Os = 0.
    mass_Cs = 0.
    for j,name in enumerate(abundances.keys()):
        if nb_Os(name) > 0:
            mass_Os += 16*nb_Os(name)*10**abundances[name]/get_MMWs(name)
        if nb_Cs(name) > 0:
            mass_Cs += 12*nb_Cs(name)*10**abundances[name]/get_MMWs(name)
    if not(mass_Os == 0):
        RATIO = mass_Cs/mass_Os
    else:
        print('there is no oxygen')
        RATIO = 0
    return RATIO

def CO_ratio_standard(abundances):
    return 16/12*CO_ratio_mass(abundances)

def CO_ratio_mass(abundances):
    mass_Os = 0.
    mass_Cs = 0.
    for j,name in enumerate(abundances.keys()):
        mass_Os += nb_Os(name)*10**abundances[name]
        mass_Cs += nb_Cs(name)*10**abundances[name]
    if not(mass_Os == 0):
        RATIO = mass_Cs/mass_Os
    else:
        print('there is no oxygen')
        RATIO = 1e20
    return RATIO

def CO_ratio_correct(abundances):
    n_Cs = 0
    n_Os = 0
    for key in abundances.keys():
        if nb_Cs(name) > 0:
            n_Cs += nb_Cs(key)*10**abundances[key]/get_MMWs(key)
        if nb_Os(name) > 0:
            n_Os += nb_Os(key)*10**abundances[key]/get_MMWs(key)
    return n_Cs/n_Os
    
    
def calc_CO_ratio(samples,params_names,abundances,percent_considered = 1.,abundances_considered = 'all',method = 'mine'):
    samples_used = samples
    if percent_considered < 1:
        nb_iter = len(samples)
        index_consider = int(nb_iter*(1.-percent_considered))
        samples_used = samples[index_consider:,:]
    CO_ratio_sampled = []
    for params in samples_used:
        ab_metals = {}
        for name_i,name in enumerate(params_names):
            if name in abundances:
                mol,model,param_nb = name.split('___')
                if abundances_considered == 'all' or name in abundances_considered:
                    ab_metals[mol] = params[name_i]
        CO_ratio = 0
        if method == 'mine':
            CO_ratio = CO_ratio_mass(ab_metals)
        elif method == 'VMRs':
            CO_ratio = CO_ratio_VMRs(ab_metals)
        elif method == 'standard':
            CO_ratio = CO_ratio_standard(ab_metals)
        elif method == 'new':
            CO_ratio = CO_ratio_new(ab_metals)
        elif method == 'correct':
            CO_ratio = CO_ratio_correct(ab_metals)
        if abs(CO_ratio - 1e20) < 100:
            continue
        else:
            CO_ratio_sampled.append(CO_ratio)
    return CO_ratio_sampled

def get_molecule_name(molecule):
    if not('_' in molecule):
        return molecule
    else:
        return molecule[:molecule.index('_')]

def split_mol_in_atoms(molecule):
    mol_pure = get_molecule_name(molecule)
    atom_list = re.sub( r"([A-Z])", r" \1", mol_pure).split()
    return atom_list

def split_atom_number(element):
    element_list = re.sub( r"([0-9])", r" \1", element).split()
    if len(element_list)==1:
        return [element_list[0],1]
    else:
        return [element_list[0],int(''.join(element_list[1:]))]

def calc_atoms_mass(abundances):
    atom_mass = {}
    for mol in abundances.keys():
        mol_pure = get_molecule_name(mol)
        atom_list = split_mol_in_atoms(mol)
        for el in atom_list:
            atom,count = split_atom_number(el)
            atomic_mass = element(atom).atomic_weight
            if atom in atom_mass.keys():
                atom_mass[atom] += (10**abundances[mol])*count*atomic_mass/get_MMWs(mol_pure)
            else:
                atom_mass[atom] = (10**abundances[mol])*count*atomic_mass/get_MMWs(mol_pure)
    return atom_mass

def calc_metallicity(abundances):
    new_abunds = fill_h2_he(abundances)
    atom_mass = calc_atoms_mass(new_abunds)
    metals_mass = 0
    for key in atom_mass.keys():
        if key not in ['H','He']:
            metals_mass += atom_mass[key]
    return metals_mass/(atom_mass['H']+atom_mass['He'])

def fill_h2_he(abundances):
    new_abunds_temp = {}
    metal_sum = 0
    for key in abundances.keys():
        if key in ['He','H2','H2_main_iso']:
            continue
        metal_sum += 10**abundances[key]
        new_abunds_temp[key] = abundances[key]
    h2he = 1.-metal_sum
    new_abunds_temp['H2'] = np.log10(0.75*h2he)
    new_abunds_temp['He'] = np.log10(0.25*h2he)
    return new_abunds_temp

def calc_FeH_ratio(abundances,return_abunds=False):
    """
    FeH_number = 0
    if 'FeH' in abundances.keys() or 'FeH_main_iso' in abundances.keys():
        if 'FeH' in abundances.keys():
            FeH_number = 10**abundances['FeH']/56.845
        else:
            FeH_number = 10**abundances['FeH_main_iso']/56.845
    if FeH_number == 0:
        return -100
    """
    new_abunds = fill_h2_he(abundances)
    atom_count = count_atoms(new_abunds)
    if return_abunds:
        return (np.log10((1.-atom_count['H']-atom_count['He'])/atom_count['H']) - np.log10(0.0134)),new_abunds
    else:
        return (np.log10((1.-atom_count['H']-atom_count['He'])/atom_count['H']) - np.log10(0.0134))
    #return np.log10(FeH_number/atom_count['H'])# - np.log10(27/(909964))



list_atoms = ['H','He','C','O','V','Ti','Fe','S','K','Na','N','P']
atomic_weight = {}
for atom in list_atoms:
    atomic_weight[atom] = element(atom).atomic_weight


def count_atoms(abundances):
    atom_count = {}
    for mol in abundances.keys():
        mol_pure = get_molecule_name(mol)
        atom_list = split_mol_in_atoms(mol)
        mol_weight = sum([split_atom_number(el)[1]*atomic_weight[split_atom_number(el)[0]] for el in atom_list])
        for el in atom_list:
            atom,count = split_atom_number(el)
            if atom in atom_count.keys():
                atom_count[atom] += count*(10**abundances[mol])/mol_weight*atomic_weight[atom]
            else:
                atom_count[atom] = count*(10**abundances[mol])/mol_weight*atomic_weight[atom]
    
    return atom_count

def calc_FeH_ratio_from_samples(samples,params_names,abundances,percent_considered = 1.,abundances_considered = 'all'):
    samples_used = samples.copy()
    if percent_considered < 1:
        nb_iter = len(samples)
        index_consider = int(nb_iter*(1.-percent_considered))
        samples_used = samples[index_consider:,:]
    FeH_ratio_sampled = []
    for params in samples_used:
        ab_metals = {}
        for name_i,name in enumerate(params_names):
            if name in abundances:
                mol,model,param_nb = name.split('___')
                if abundances_considered == 'all' or name in abundances_considered:
                    ab_metals[mol] = params[name_i]
        FeH_ratio_sampled.append(calc_FeH_ratio(ab_metals))
    return FeH_ratio_sampled

def poor_mans_abunds_lbl():
    return POOR_MANS_ABUND_LBL
def poor_mans_abunds_ck():
    return POOR_MANS_ABUND_CK

def poor_mans_ck_and_lbl(molecule,mode):
    abunds_ck = poor_mans_abunds_ck()
    abunds_lbl = poor_mans_abunds_lbl()
    if mode == 'lbl':
        return abunds_lbl[abunds_ck.index(molecule)]
    else:
        return abunds_ck[abunds_lbl.index(molecule)]

def filter_relevant_mass_fractions(mass_fractions,mode):
    #print('SiO IS STILL INCLUDED, CONTINUE??')
    #print('SiO not INCLUDED, CONTINUE??')
    poor_mans_species = list(mass_fractions.keys())
    
    mol_abund_keys_ck = poor_mans_abunds_ck()
    mol_abund_keys_lbl = poor_mans_abunds_lbl()
    abundances = {}
    for species in poor_mans_species:
        if species in mol_abund_keys_ck:
            abundances[species] = mass_fractions[species]
        elif species=='NH3' and 'NH3_HITRAN' in mol_abund_keys_ck:
            abundances['NH3_HITRAN'] = mass_fractions['NH3']
        elif species == 'FeH' and 'FeH_Chubb' in mol_abund_keys_ck:
            abundances['FeH_Chubb'] = mass_fractions['FeH']
        #elif species == 'SiO' and 'SiO_Chubb' in mol_abund_keys_ck:
        #    abundances['SiO_Chubb'] = mass_fractions['SiO']
    
    if mode == 'c-k':
        return abundances
    else:
        abundances_final = {}
        for species in abundances.keys():
            abund_spec = abundances[species]
            lbl_species_name = poor_mans_ck_and_lbl(species,'lbl')
            abundances_final[lbl_species_name] = abund_spec
        return abundances_final

def trim_spectrum(wlen,flux,wlen_data,threshold=5000,keep=1000):
    mask_wlen = np.logical_and(np.array(wlen) >= wlen_data[0],np.array(wlen) <= wlen_data[-1])
    
    high_ind = np.min([len(wlen)-1,np.where(mask_wlen)[0][-1] + keep])
    low_ind = np.max([0,np.where(mask_wlen)[0][0] - keep])
    return wlen[low_ind:high_ind],flux[low_ind:high_ind]

# def trim_spectrum(wlen,flux,wlen_data,threshold=5000,keep=1000):
#     wvl_stepsize = np.mean([wlen_data[i+1]-wlen_data[i] for i in range(len(wlen_data)-1)])
#     nb_bins_left = int(abs(wlen_data[0]-wlen[0])/wvl_stepsize)
#     nb_bins_right = int(abs(wlen_data[-1]-wlen[-1])/wvl_stepsize)
#     cut_right=False
#     cut_left=False
#     nb_bins_to_cut_left,nb_bins_to_cut_right=0,0
#     if nb_bins_left >= threshold:
#         nb_bins_to_cut_left = nb_bins_left - keep
#         cut_left=True
#     if nb_bins_right >= threshold:
#         nb_bins_to_cut_right = nb_bins_right - keep
#         cut_right=True
#     if cut_right or cut_left:
#         CC_wlen_cut,CC_flux_cut = cut_spectrum(wlen,flux,nb_bins_to_cut_left,nb_bins_to_cut_right)
#         return CC_wlen_cut,CC_flux_cut
#     else:
#         return wlen,flux
#     
#     
# 

def cut_spectrum(wlen,flux,nb_bins_left,nb_bins_right):
    if nb_bins_left + nb_bins_right > len(wlen) or nb_bins_left + nb_bins_right == 0:
        return wlen,flux
    wlen_cut,flux_cut = np.transpose([[wlen[i],flux[i]] for i in range(len(wlen)) if i >= nb_bins_left and i <= len(wlen) - 1 - nb_bins_right])
    return wlen_cut,flux_cut

def add_rot_broad(wlen,flux,rot_vel,method='fast',edgeHandling = 'cut'):
    
    if method == 'fast':
        flux_broad = fastRotBroad(wlen,flux,0,rot_vel)
    else:
        # method == 'slow'
        flux_broad = rotBroad(wlen,flux,0,rot_vel)
    
    if edgeHandling=='cut':
        skipping = abs(wlen[-1]*rot_vel*1000/cst.c)
        wvl_stepsize=np.mean([wlen[i+1]-wlen[i] for i in range(len(wlen)-1)])
        skipped_bins = int(skipping/wvl_stepsize)
        #print('Bins skipped for rotational broadening {b}'.format(b=skipped_bins))
        wlen_cut,flux_cut = cut_spectrum(wlen,flux_broad,nb_bins_left = skipped_bins,nb_bins_right=skipped_bins)
        return wlen_cut,flux_cut
    else:
        # edgeHandling == 'keep'
        return wlen,flux_broad

def calc_SNR(dRV,CC):
    RV_max_i = np.argmax(CC)
    CC_max = CC[RV_max_i]
    
    left_bord,right_bord=RV_max_i-1,RV_max_i+1
    
    while left_bord > 0 and right_bord < len(dRV)-1:
        if CC[left_bord-1] > CC[left_bord] and CC[right_bord+1] > CC[right_bord]:
            break
        else:
            if CC[left_bord-1] < CC[left_bord]:
                left_bord -= 1
            if CC[right_bord+1] < CC[right_bord]:
                right_bord += 1
    nb_CC_bins = len(dRV)
    #left_bord = RV_max_i - int(nb_CC_bins/10)
    #right_bord = RV_max_i + int(nb_CC_bins/10)
    
    noisy_CC_function = [CC[i] for i in range(len(CC)) if i not in range(left_bord,right_bord)]
    std_CC = np.std(noisy_CC_function)
    SNR = CC_max/std_CC
    return SNR,std_CC,RV_max_i,left_bord,right_bord,noisy_CC_function

def predict_g_distr(M_sample,R_sample,N_picks = 1000):
    if N_picks == 'all':
        values = product(np.log10(M_sample),np.log10(R_sample))
    else:
        log_M_picks = np.log10(np.random.choice(M_sample,N_picks))
        log_R_picks = np.log10(np.random.choice(R_sample,N_picks))
        values = zip(log_M_picks,log_R_picks)
    gravity_cgs_si = 100
    log_g_sample = np.array([np.log10(cst.gravitational_constant) + log_M + np.log10(MASS_J) - 2*log_R - 2*np.log10(RADIUS_J) + np.log10(gravity_cgs_si) for log_M,log_R in values])
    
    return log_g_sample
"""
def prior_transform(sample):

def prior_distr(sample):
"""
def filter_position(PHOT_midpoint):
    filter_pos = {}
    filter_names_list = list(PHOT_midpoint.keys())
    filter_position_i = 0
    while len(filter_names_list)>0:
        filt_min = filter_names_list[0]
        for instr in filter_names_list:
            if PHOT_midpoint[instr] < PHOT_midpoint[filt_min]:
                filt_min = instr
        filter_names_list.remove(filt_min)
        filter_pos[filt_min] = filter_position_i
        filter_position_i += 1
    return filter_pos

def get_extinction(wlen,Av):
    extmod = G23(Rv=3.1)
    extinction = extmod.extinguish(wlen*u.micron,Av=Av)
    return extinction

def calc_retrieved_params(config,samples):
    params_names = config['PARAMS_NAMES']
    data_params = config['DATA_PARAMS']
    
    ab_metals_params = config['ABUNDANCES']
    unsearched_ab_metals = config['UNSEARCHED_ABUNDANCES']
    
    temp_params_names = config['TEMPS']
    unsearched_temp_params = config['UNSEARCHED_TEMPS']
    
    ab_metals = {}
    for param in unsearched_ab_metals:
        ab_metals[param] = data_params[param]
    
    temp_params = {}
    for param in unsearched_temp_params:
        temp_params[param] = data_params[param]
    
    median_params = np.median(samples,axis=0)
    for param_i,param in enumerate(params_names):
        if param in temp_params_names:
            temp_params[param] = median_params[param_i]
        if param in ab_metals_params:
            ab_metals[param] = median_params[param_i]
    return ab_metals,temp_params
    