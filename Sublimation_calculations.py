#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:48:23 2022

@author: laferrierek


Sublimation and such
"""

# ** Import libaries **
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import constants as const
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
import json

# ** Import custom Libraries **
import Orbital_Parameters as op
import Make_Layers as ml

# ** Define things needed to run **    
loc = '/Users/laferrierek/Box Sync/Desktop/Mars_Troughs/Project_MCMC/Thermal_model_KLL/'

#%%
def sublimation():
    
#%% Equation 4 from Bramson 2019
def retreat(D, iceVapRho, atmRho, z, d, rhoIce):
    # z is , d is dsut fraction
    return (D*(iceVapRho-atmRho))/(z*(1-d)*rhoIce)
Dreg = 3*10**(-4) # m2/s
rhoIce = 920 # kg/m3
waterVapDensity = 0.013 #kg/m3 (google)
dust = 3/100 #(Grima et al.)
'''
atmo_rhov = 

R = retreat(Dreg, iceTable_rhov, atmo_rhov, lag, dust, rhoIce)
#%% This idoes not work
def convective_loss(mol_mass, T_bl, A, u_wind, esat, e, del_eta, rho_Ave, D_at, del_rho, rho, v, g, rhoIce):
    m_forced = mol_mass * (1/(kb*T_bl)) * A * u_wind * (esat - e) * rhoIce
    m_free =  0.14 * del_eta * rho_Ave * D_at * ((del_rho/rho) * (g/v**2) * (v/D_at))**(1/3)
    msub = m_forced+m_free
    return msub

water_mol = m_gas_h2o/NA

b = 0.2
Tatm = minimumDiurnalSurfTemps**b * T_reg_Diurnal**(1-b)
atmwater = np.linspace(0, 1, 670)
#%%
#tm_density = pressure/R_gas* T # 700 Pa (cO2) and water from e. 
#dh = (iceTable_rhov - 0.013)/atm__density
# rho_a
def diffusion_coeff(Tbl, Patm):
    return (1.387*10**(-5))*(Tbl/273.15)**(3/2)*(10**5/Patm)
Dat = diffusion_coeff(Tbl, Patm) # diffusion coefficient of H2O in CO2
def kinematic_viscosity(R, Tbl, mc, Patm):
    # R, mc = universal gas const, molar mass co2 0.044 kg
    return (1.48*10**(-5))*(R*Tbl/(mc*Patm))*(240+293.15/(240+Tbl))*(Tbl/293.15)**(3/2)
v = kinematic_viscosity(R, Tbl, mc, Patm) # kinematice viscosity ?

def rho_ratio(mc, mw, esat, Patm):
    # mc, mw, 0.044 kg, 0.018 kg.
    return ((mc-mw)*esat)/(mc*Patm - (mc-mw)*esat)
del_rho = rho_ratio(mc, mw, iceTable_Pv, Patm) # atmospheric and surface gas density difference

convective_loss(water_mol, np.average(averageDiurnalSurfTemps, Tatm), 0.002, 2.5, iceTable_Pv, atmwater, dh, rho_a, Dat,  )


'''
