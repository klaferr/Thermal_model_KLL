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

# Constants - which of these are needed here?
G = 6.67259*10**(-11)       # Gravitational Constant; m3/kg/s2
NA = 6.022 * 10**(23)       # Avogadros number, per mol
h = const.h                 # Planck's constant;  Js
c = const.c                 # Speed of light; meters/second
k = const.k                 # Boltzmann constant;  J/K
b = 2.89777199*10**(-3)     # ?; meters
sigma = 5.670*10**(-8)      # Stefan-Boltzmann; W/m^2K^4
au = 1.4959787061e11        # AU; meters
sm  = 1.9891e30             # Solar Mass; kg

# Molecules
R_bar = 8.314       # units?
hydrogen = 1.004                    # g/mol
oxygen = 15.999                     # g/mol
carbon = 12.01                      # g/mol
# H2O/ ice 1/ water vapor
m_gas_h2o = (hydrogen*2+oxygen)/1000    # kg/mol
mass_molecule_h2o = m_gas_h2o / NA
triple_P_h2o = 611.657              # Pa
triple_Temp_h2o = 273.16            # K
Lc_H2O =  -51058                    # Latent heat of CO2 frost; J/kg
#H2O_FrostPoints = 150 

# CO2
m_gas_co2 = (carbon+oxygen*2)/1000  # kg/mol
triple_P_co2 = 516757               # Pa
triple_Temp_co2 = 216.55            # K
Lc_CO2 =  589.9*10**3               # Latent heat of CO2 frost; J/kg
CO2_FrostPoints = 150


#other 
A_drag_coeff = 0.002
u_wind = 2.5 # Dundas 2010

# conversion
mbar_to_Pascal = 100
gpercm3_tokgperm3 = 1000

# Earth Specific constants
EarthYearLength = 2*np.pi*np.sqrt(au**3/(G*sm))             # Length of one Earth year in seconds
solarflux_at_1AU = 1367                                     # Current; W/m2

# Mars Specific constants
Mars_semimajor = 1.52366231                                # Distance; AU
MarsyearLength = 2*np.pi/np.sqrt(G*sm/(Mars_semimajor*au)**3)   # Length of Mars year in seconds using Kepler's 3rd law
MarsyearLength_days = 668.6
MarsdayLength = 88775 
solarflux_at_Mars = solarflux_at_1AU/Mars_semimajor**2
Mars_g = 3.71                                       # m/s2
P_atm = 700         # Pa

# Surface conditions
#"""
albedo = 0.25
emissivity = 1.0
Q = 0.03 #30*10**(-3) # W/m2
Tref = 250

# Frost
emisFrost = 0.95
albedoFrost = 0.6
Tfrost = CO2_FrostPoints
windupTime = 8
convergeT = 0.01

runTime = 15
f = 0.5
dt = 500
#%% functions
def clapyeron(triple_pressure, triple_T, R_bar, Lc, T):
    return triple_pressure*np.exp( (Lc/R_bar) * ((1/triple_T) - (1/T)))
def rho_molecules(molarMassa, MolarMassb, atmP, esat, T_sub):
    return (((molarMassa/NA) * atmP) + ((((molarMassb-molarMassa)/NA)) * esat) /(k*T_sub))
def diffusion_coeff(T, triple_T, P_atm):
    return (1.387*10**(-5)) * ((T/triple_T)**(3/2)) * (10**5/P_atm)
def gas_density_diff(molarMassa, molarMassb, atmP, T_surf, T_atm, esat, ewatvap):
    # for mars, a is co2 and b is h2o
    num = (molarMassa * atmP * ((T_surf/T_atm) - 1)) + ((molarMassa - molarMassb) * (esat - ((T_surf/T_atm) * ewatvap)))
    denom = 0.5*((molarMassCa * atmP * ((T_surf/T_atm) + 1)) - ((molarMassa - molarMassb) * ((T_surf/T_atm) * ewatvap + esat)))
    value = num/denom
    if value < 0:
        value = 0
    return value
def kin_visc(molarMass, T):
    # Sutherland formula
    refT = 293.15
    sut_con = 240
    return (1.48*10**5)*(R_gas * T/(molarMass * atmP)) * ((sut_con+refT)/(sut_con+T)) * (T/refT)**(3/2)
#%% free and forced convection

# partial pressure H2O scheme

# Partial pressure of water at present day

# Free and forced
numDays_sublimation = np.size(diurnal curves)

summedForcedDayTotals = np.zeros((numDays_sublimation)) 
summedFreeDayTotals = np.zeros((numDays_sublimation))
summedDayTotals = np.zeros((numDays_sublimation))

# Loop over every day
for day in range(1, numDays_sublimation):
    # Calculate surface sublimation  
        
    T_surf_subl = SLOPEDaverageDiurnalSurfTemps[day]
    
    reg_Tmin = REGminDiurnalSurfTemps[day]
    reg_all_Tsurfs = REGdiurnalTsurfCurves[day]
    
    numTimestep_subl = np.size(reg_all_Tsurf)
    
    diurnal_m_free_thickness = np.zeros((numTimesteps_subl))
    diurnal_m_forced_thickness = np.zeros((numTimesteps_subl))
    diurnal_total = np.zeros((numTimesteps_subl))
    
    for timestep in range(1, numTimestep_subl):
        reg_Tsurf = reg_all_Tsurf(timestep)
        
        # Compute Temp
        T_min = reg_Tmin# most recent diurnal minimum surface temp
        T_reg = reg_Tsurf # current surface temp
        b_exp = 0.2     # viking
        T_atm_subl = T_min**b_exp * T_reg**(1-b_exp) # near sirface atmospheric temperature
        
        T_bl = (T_surf_subl + T_atm_sibl)/2 # boundary layer tempearture
        
        # Sensible heat for forced convection, (Eq 5 Dundas et al. 2010)
        T_subl = T_bl
        e_sat = clapyeron(triple_P_h2o, triple_Temp_h2o, R_bar, Lc_H2O, T_surf_subl)    # Saturation vapor pressure given temo
        
        e_watvap = PPHO(counter)
        
        m_forced = (mas_molecule_H2O/(k*T_subl))*A_drag_coeff * u_wind * (e_sat - e_watvap)
        
        diurnal_m_forced_thickness(timestep) = (m_forced*dt/densityIce)
        
        # Sensible heat for free convection (Eq 6 from Dundas et al. 2010)
        rho_surf =  rho_molecules(m_gas_co2, m_gas_h2o, P_atm, e_sat, T_surf_subl)
        rho_atm = rho_molecules(m_gas_co2, m_gas_h2o, P_atm, e_watvap, T_atm_subl)
        rho_ave = (rho_atm + rho_surf )/2
        
        rho_bl_H2O = rho_molecules(0, m_gas_h2o, 0, e_sat, T_bl)
        rho_atm_H2O = rho_molecules(0, m_gas_h2o, 0, e_watvap, T_atm_subl)
        rho_surf_H2O = rho_molecules(0, m_gas_h2o, 0, e_sat, T_surf_subl)
        # difference between atmospheric and surface gas water mass
        deltaEta = (rho_surf_H2O - rho_atm_H2O) /rho_atm
        
        # diffusion coeefient for H2O in Co2 at T_bl
        Diff_H2OinCO2 = diffusion_coeff(T_bl, triple_Temp_H2O, P_atm
        
        # Atmospheric and surface gas density difference divided by reference density
        delta_rho_over_rho = gas_density_diff(m_gas_co2, m_gas_h2o, P_atm, T_surf_subl, T_atm_subl, e_sat, e_watvap)
        
        # Kinematic viscosity
        num = kin_visc(m_gas_co2, T_bl)    

        # free
        x = (delta_rho_over_rho*(g/nu**2)*(nu/Diff_H2OinCO2))**(1/3) # check sign
        m_free = 0.14*deltaEta * rho_Ave * Diff_H2OinCO2 * x # kg/m2?
        
        diurnal_m_free_thickness(timestep) = m_free * dt / densityIce
        
    m_forced_thickness[day] = diurnal_m_forced_thickness
    m_free_thickness[day] = diurnal_m_free_thickness
    total[day] = diurnal_m_free_thickness+ diurnal_m_forced_thickness
    
    summedForcedDayTotals[day] = sum(diurnal_m_forced_thickness)
    summedFreeDayTotals[day] = sum(diurnal_m_free_thickness)
    summedDayTotals[day]= sum(total[day])
                                    

TotalFree = sum(summedFreeDayTotals)
TotalForced = sum(summedForcedDayTotals)
TotalSublimation = sum(summedDayTotals)

print('Final forced convection sublimation is %10.9f mm' %TotalForced*1000);
print('Final free convection sublimation is %10.9f mm' %TotalFree*1000);
print('Final total sublimation is %10.9f mm'%TotalSublimation*1000);
#%%

# potential write functions for free and forced. 

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

#%%
import numpy as np

R_gas = 8.3145                      # J/mol K

def mean_gas_const(f1, m1, f2, m2, f3, m3):
    mbar = f1*m1+f2*m2+f3*m3
    return R_gas/mbar

def clapyeron(triple_pressure, triple_T, R_bar, Lc, T):
    # Pascal
    return triple_pressure*np.exp( (Lc/R_bar) * ((1/triple_T) - (1/T)))

# Molecules
hydrogen = 1.004                    # g/mol
oxygen = 15.999                     # g/mol
m_gas_h2o = (hydrogen*2+oxygen)/1000    # kg/mol
m_gas_h2o_molar = m_gas_h2o*1000        # g/mol
R_bar = mean_gas_const(1, m_gas_h2o, 0, 0, 0, 0)    #J/kgK

# Molecule - Water
triple_T = 273.1575                 # K
triple_P = 611.657                  # Pa 
triple_P_bar = triple_P * 10**(-5)  # bar (still doesn't wokr. )
Latent_heat = 2.834*10**6 #3340720               # J/kg - is this wrong

# define sublimation
def sublimation(alpha, ps, T, mu):
    # kg/m2 s
    E = alpha*ps*np.sqrt(mu/(2*np.pi*R_gas*T))
    return E

alpha = 1
ps = clapyeron(triple_P, triple_T, R_bar, Latent_heat, Tsurf)
mu = m_gas_h2o
equ_1 = sublimation(alpha, ps, Tsurf, mu)

