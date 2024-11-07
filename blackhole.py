# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:54:52 2024

@author: ISAFA
"""
#sistemare esercizio 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm 
from astropy.visualization.hist import hist as fancyhist
from sklearn.neighbors import KernelDensity
import scipy.stats
from scipy import integrate

def Mirr(M, chi):    #function of the irreducible mass
    return M*np.sqrt((1+np.sqrt(1-chi**2))/2)

N = 1000
mu = 1              #all masses in the problem are measured in units of mu
sigma = 0.02

chi = np.random.uniform(0, 1, N)  #the spin is uniformly distributed between 0 and 1.
M = np.random.normal(mu, sigma, N) #the mass is normally distributed with mean mu and standard deviation sigma

M_irr = Mirr(M, chi)  #array of float, passa ogni M e ogni chi e restituisce per ognuno un M_irr

fig = plt.figure(figsize=(8,6))
ax = fig.gca()

#I plot the histogram using the freedman and the scott rule for the bins
M_irr = np.sort(M_irr) #va messo in ordine
scott = fancyhist(M_irr, bins="scott", density=True, alpha = 0.8, label = 'Scott')
freedman = fancyhist(M_irr, bins="freedman", density=True, alpha = 0.8, label ='Freedman')

Mgrid = np.linspace(M_irr.min(), M_irr.max(),1000)  # Use this instead of 'M_irr' for plotting

#I define the function for genereting the distribution of M_irr using a KDE
def KDE(M_irr, bandwidth, kernel):    
   KDE = KernelDensity(bandwidth=bandwidth, kernel= kernel) 
   KDE.fit(M_irr[:, np.newaxis]) # sklearn returns log(density)
   return np.exp(KDE.score_samples(Mgrid[:, np.newaxis]))

#I plot it using different kernel
kde_gauss = KDE(M_irr, 0.01, kernel = 'gaussian')
ax.plot(Mgrid, kde_gauss, color = 'crimson', label='gaussian')

kde_lin = KDE(M_irr, 0.01, 'linear')
ax.plot(Mgrid, kde_lin, color = 'springgreen', label='linear')

kde_top = KDE(M_irr, 0.01, 'tophat')
ax.plot(Mgrid, kde_top, color = 'purple', label='tophat')

ax.legend()

def f (M, M_irr):   #function required by the exercise
    f=M_irr/M
    return f

#Now I need to evaluate the KS distance with different sigmas
#first we need different distribution of M and M_irr with different sigmas
sigmas = np.linspace(0, 5, 50)
masses = []

for i in sigmas:
    masses.append(np.random.normal(mu, i, N)) #list of 50 elements, each an array of float

masses_irr = []
for i in masses:
    masses_irr.append(Mirr(i, chi)) #list of 50 elements, each an array of float
    
fig1 = plt.figure(figsize=(8,6))
ax1 = fig1.gca()

function = []
for i in range(len(masses)):
    function.append(f(masses[i], masses_irr[i]))
    
#ax1.plot(sigmas, function)

ks_test_irr = []
for i in range(len(masses)) :
     ks_test_irr.append(scipy.stats.kstest(masses_irr[i], function[i]).pvalue)
     
ax1.plot(sigmas, ks_test_irr)
ax1.set_yscale('log')

ks_test = []
for i in range(len(masses)) :
     ks_test.append(scipy.stats.kstest(masses_irr[i], masses[i]).pvalue)

fig2 = plt.figure(figsize=(8,6))
ax2 = fig2.gca()    
ax2.plot(sigmas, ks_test)
ax2.set_yscale('log')

# ks_test_M = scipy.stats.kstest(M_irr, M)
# print("the ks distance bt M and M_irr is: ", ks_test_M)

#pdf of Mirr
func= f(M, M_irr)
integrals=[]
for i in range(len(func)):
    if func[i]>1/np.sqrt(2) and func[i]<1:
        integrals.append((np.exp(-((M_irr[i]/func[i] - mu)**2)/2*sigma**2))*((2*func[i]**2-1)/(func[i]**np.sqrt(1-func[i]**2)))) 

integral =(np.sqrt(2/np.pi)/sigma)*integrate.simpson(integrals)


