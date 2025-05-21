import numpy as np
import scipy.constants as cst
from scipy.interpolate import interp1d
from scipy.special import erfcinv
from scipy.stats import truncnorm,skewnorm,gaussian_kde,norm
from scipy.optimize import curve_fit
SQRT2 = sqrt(2.)
SQRT2PI = sqrt(2*pi)
# log(0) outside [a,b], else log(1)
def a_b_range(x,arr):
    a,b = arr
    if x > b or x < a:
        return -np.inf
    else:
        return 0.

def gauss(x,mu,sigma):
    return 1./SQRT2PI/sigma*np.exp(-1./2*((x-mu)/sigma)**2)

def log_gauss(x,mu,sigma):
    return -1./2*((x-mu)/sigma)**2

# x in [0,1]
# output in [x1,x2]
def uniform_prior(x,arr):
    x1,x2 = arr
    return x1 + x*(x2-x1)

# Priors stolen from https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90
def log_prior(cube,lx1,lx2):
    return 10**(lx1+cube*(lx2-lx1))

def gaussian_prior(cube,mu,sigma):
    return mu + sigma*SQRT2*erfcinv(2.0*(1.0 - cube))
    #return -(((cube-mu)/sigma)**2.)/2.

def log_gaussian_prior(cube,mu,sigma):
    bracket = sigma*sigma + sigma*SQRT2*erfcinv(2.0*cube)
    return bracket

def delta_prior(cube,x1,x2):
    return x1

def sample_to_pdf(sample):
    kernel = gaussian_kde(sample)
    return kernel

def gauss_pdf(x,A,mu,std):
    return A*np.exp(-0.5*(x-mu)**2/std**2)

def gauss_ppf(q,A,mu,std):
    return norm.ppf(q,loc=mu,scale=std)

def skew_gauss_pdf(x,A,mu,std,a):
    return A*skewnorm.pdf(x, a, loc=mu, scale=std)

def skew_gauss_ppf(q,A,mu,std,a):
    return skewnorm.ppf(q, a, loc=mu, scale=std)

def fit_gauss(pos,pdf):
    popt,pcov = curve_fit(gauss_pdf,pos,pdf)
    return popt,pcov

def fit_skewed_gauss(pos,pdf):
    popt,pcov = curve_fit(skew_gauss_pdf,pos,pdf)
    return popt,pcov

def create_prior_from_parameter(param,parameter_data,RANGE,LOG_PRIORS,CUBE_PRIORS):
    if parameter_data[0] == 'uniform':
        create_uniform_prior(param=param,parameter_list=parameter_data[1],RANGE=RANGE,LOG_PRIORS=LOG_PRIORS,CUBE_PRIORS=CUBE_PRIORS)
    elif parameter_data[0] == 'gaussian':
        create_gaussian_prior(param=param,parameter_list=parameter_data[1],RANGE=RANGE,LOG_PRIORS=LOG_PRIORS,CUBE_PRIORS=CUBE_PRIORS)
    else:
        print('Prior %s not taken into account!' % parameter_data[0])
    
def create_uniform_prior(param,parameter_list,RANGE,LOG_PRIORS,CUBE_PRIORS):
    RANGE[param] = parameter_list
    LOG_PRIORS[param]      = lambda x: a_b_range(x,parameter_list)
    CUBE_PRIORS[param]      = lambda x: uniform_prior(x,parameter_list)
    return
def create_gaussian_prior(param,parameter_list,RANGE,LOG_PRIORS,CUBE_PRIORS):
    RANGE[param] = [parameter_list[0]-parameter_list[1],parameter_list[0]+parameter_list[1]]
    LOG_PRIORS[param]      = lambda x: log_gauss(x,parameter_list[0],parameter_list[1])
    CUBE_PRIORS[param]      = lambda x: gaussian_prior(x,parameter_list[0],parameter_list[1])
    return