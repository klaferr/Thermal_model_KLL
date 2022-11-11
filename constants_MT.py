#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 09:51:27 2022

@author: laferrierek
"""
# constants

import numpy as np
from scipy import constants as const


# Constants
G = 6.67259*10**(-11)       # Gravitational Constant; m3/kg/s2
NA = 6.022 * 10**(23)       # Avogadros number, per mol
h = const.h                 # Planck's constant;  Js
c = const.c                 # Speed of light; meters/second
k = const.k                 # Boltzmann constant;  J/K
b = 2.89777199*10**(-3)     # ?; meters
sigma = 5.670*10**(-8)      # Stefan-Boltzmann; W/m^2K^4
au = 1.4959787061e11        # AU; meters
sm  = 1.9891e30             # Solar Mass; kg


# Conversion
mbar_to_Pascal = 100
gpercm3_tokgperm3 = 1000


# Molecules
R_bar = 8.314       # units?
hydrogen = 1.004                    # g/mol
oxygen = 15.999                     # g/mol
carbon = 12.01                      # g/mol

# H2O
m_gas_h2o = (hydrogen*2+oxygen)/1000    # kg/mol
mass_molecule_h2o = m_gas_h2o / NA
triple_P_h2o = 611.657                  # Pa
triple_Temp_h2o = 273.1575              # K
Lc_H2O = 2257*10**3                     # J/kg

# CO2
m_gas_co2 = (carbon+oxygen*2)/1000  # kg/mol
triple_P_co2 = 516757               # Pa
triple_Temp_co2 = 216.55            # K
Lc_CO2 =  589.9*10**3               # Latent heat of CO2 frost; J/kg
#CO2_FrostPoints = 150


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
atmofactor = 0 # 0 == no water vapor in atmo


# Thermal (Bramosn et al. 2017, JGR Planets)
# compositional values - this may need to be read in
# Orbital solutions
eccentricity = 0.09341233
obl = np.deg2rad(25.19)
Lsp = np.deg2rad(250.87)
dt_orb = 500


# Atmospheric: Parameterization from Schorghofer and Forget (2012)
SF_a1 = -1.27  
SF_b1 = 0.139 
SF_c1 = -0.00389 

# possible surface conditions
# Rock properties
k_rock = 2
density_rock = 3300
c_rock = 837
TI_rock = 1200 # J/m2/K/s^1/2

# Ice properties
k_ice = 3.2
density_ice = 920
c_ice = 1540


def smooth(yval, box_size):
    # Used to smooth the data 
    box = np.ones(box_size) / box_size
    y_smooth = np.convolve(yval, box, mode='same')
    return y_smooth