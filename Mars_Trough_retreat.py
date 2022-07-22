#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:48:39 2022

@author: laferrierek

Main code: runs CN, everything else
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
from pathlib import Path

# ** Import custom Libraries **
import Orbital_Parameters as op
import Make_Layers as ml
import CN_1D as cn


# ** Define things needed to run **    
loc = '/Users/laferrierek/Box Sync/Desktop/Mars_Troughs/Project_MCMC/Thermal_model_KLL/'

# Aesthetic
fs = (10, 8)
res = 350
plt.rc("font", size=18, family="serif")
plt.style.use('ggplot')
#%%

def flat_save(nStepsInYear, frostMasses, emisFrost, emis, sigma, Tsurf, Afrost, A, sf, trough_num):
    flatIR = np.zeros((nStepsInYear))*np.nan
    flatVis = np.zeros((nStepsInYear))*np.nan
    for n in range(1, nStepsInYear):
        if frostMasses[n] > 0:
           flatIR[n] = emisFrost * sigma * Tsurf[n-1]**4
           flatVis[n] = Afrost * sf[n]
        else:
           flatIR[n] = emis * sigma * Tsurf[n-1]**4
           flatVis[n] = A * sf[n]

    if frostMasses[0] > 0:
        flatIR[0] = emisFrost * sigma * Tsurf[-1]**4
        flatVis[0] = Afrost * sf[0]
    else:
        flatIR[0] = emis * sigma * Tsurf[-1]**4
        flatVis[0] = A * sf[0]
    np.savetxt(loc+'data/flatVis_saved_Trough_%1.0f.txt'%trough_num, np.vstack((flatIR, flatVis)).T, delimiter=',')
    return

#%% Constants - From files and basic.

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

# conversion
mbar_to_Pascal = 100
gpercm3_tokgperm3 = 1000

# Molecules
hydrogen = 1.004                    # g/mol
oxygen = 15.999                     # g/mol
carbon = 12.01                      # g/mol
# H2O
m_gas_h2o = (hydrogen*2+oxygen)/1000    # kg/mol
triple_P_h2o = 611.657                  # Pa
triple_Temp_h2o = 273.1575              # K
Lc_H2O = 2257*10**3                     # J/kg

# CO2
m_gas_co2 = (carbon+oxygen*2)/1000  # kg/mol
triple_P_co2 = 516757               # Pa
triple_Temp_co2 = 216.55            # K
Lc_CO2 =  589.9*10**3               # Latent heat of CO2 frost; J/kg
CO2_FrostPoints = 150

# Earth Specific constants
EarthYearLength = 2*np.pi*np.sqrt(au**3/(G*sm))             # Length of one Earth year in seconds
solarflux_at_1AU = 1367                                     # Current; W/m2

# Mars Specific constants
Mars_semimajor = 1.52366231                                # Distance; AU
MarsyearLength = 2*np.pi/np.sqrt(G*sm/(Mars_semimajor*au)**3)   # Length of Mars year in seconds using Kepler's 3rd law
MarsyearLength_days = 668.6
MarsdayLength = 88775 
solarflux_at_Mars = solarflux_at_1AU/Mars_semimajor**2

# Thermal (Bramosn et al. 2017, JGR Planets)
# compositional values - this may need to be read in
# Orbital solutions
eccentricity = 0.09341233
obl = np.deg2rad(25.19)
Lsp = np.deg2rad(250.87)
dt_orb = 500

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
#%% Datas
class Profile:
  def reader(self,input_dict,*kwargs):
    for key in input_dict:
      try:
        setattr(self, key, input_dict[key])
      except:
        print("no such attribute, please consider add it at init")
        continue
    
    
with open(loc+"data/Trough_slope_02_9.json",'r') as file:
    a=file.readlines()
Mars_Trough=Profile()
Mars_Trough.reader(json.loads(a[0]))


with open(loc+'data/Trough_flat_00.json','r') as file:
    a = file.readlines()
flat_Mars_Trough=Profile()
flat_Mars_Trough.reader(json.loads(a[0]))


with open(loc+"data/Layers_Bramson2019.json",'r') as file:
    a=file.readlines()
bramson=Profile()
bramson.reader(json.loads(a[0]))

#%% 2. Use the pre-made from solar_flux_trough.py to run for xMyrs
# read in min_mean,max, use max. 
#time_frame = 100
#time_op, sfmin, sfmean, sf = np.loadtxt(loc+'solar_flux_minmeanmax_%3.0f_kyr_Trough1.txt'%time_frame, skiprows=1, delimiter=',', unpack=True)



#%% Actual run
if __name__ == "__main__":
    # given specific trough:
    print("Step 1: find orbital parameters")
    
    # grabbing the real values. 
    ecc = Mars_Trough.eccentricity
    obl = np.deg2rad(Mars_Trough.obl)
    Lsp = np.deg2rad(Mars_Trough.Lsp)
    dt_orb = Mars_Trough.dt
    
    print("Step 2: make layers")
    # need layer infos
    nLayers, ktherm, dz, rho, cp, kappa, depthsAtMiddleOfLayers = ml.Make(bramson, MarsyearLength, Mars_Trough.Rotation_rate) 
    
    print('Step 3: Run for flat, save')    
    #if flatvis_saved_trough1.txt doesn't exist:
    trough_num = 2

    path_to_file = loc+'data/flatVis_Saved_Trough_%1.0f.txt'%(trough_num)
    path = Path(path_to_file)
    if path.is_file():
        flatIR, flatVis = np.loadtxt(path_to_file, unpack='True', delimiter=',')
    else:
        soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, flat_Mars_Trough, trough_num)
        np.savetxt(loc+'data/raw_flat_Saved_Trough_%1.0f.txt'%(trough_num), np.vstack((IRdown, visScattered)).T, delimiter=',')
        fTemps, fwindupTemps, ffinaltemps, fTsurf, ffrostMasses = cn.Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, kappa, emissivity, Tfrost, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR) 
        flat_save(nStepsInYear, ffrostMasses, emisFrost, emissivity, sigma, fTsurf, albedoFrost, albedo, sf, trough_num)

    # write a save file
    #np.savetxt('CN_flat_Trough1.txt', T_regs, delimiter=',', header='Temp (K) for flat')
    #%%
    print("Step 3: Run for real slope")
    soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, Mars_Trough, trough_num)
    Temps, windupTemps, finaltemps, Tsurf, frostMasses = cn.Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, kappa, emissivity, Tfrost, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR)
    
    print("Step 4: return last year values")
    cn.final_year(Tsurf, Temps, frostMasses)
   
    print("Step 5: find the diurnal averages throughout the year")
    avg_all = cn.diurnal_average(hr, Temps, fTemps, nLayers)

    print("Step 6: Calculate ice table")
    # use a fumction. 
    iceTableIndex = 2
    numPropLayers = np.size(ktherm)
    if numPropLayers > 1:
    
        iceTableTemps = (ktherm[iceTableIndex]*dz[iceTableIndex-1]*Temps[iceTableIndex,:] + ktherm[iceTableIndex-1]*dz[iceTableIndex]*Temps[iceTableIndex-1,:])/(ktherm[iceTableIndex]*dz[iceTableIndex-1] + kappa[iceTableIndex-1]*dz[iceTableIndex])
        iceTableTemps = iceTableTemps
        
        iceTable_Pv = 611 * np.exp( (-51058/8.31)*(1/iceTableTemps - 1/273.16) ) # compute vapor pressure at ice table
        iceTable_rhov = iceTable_Pv * (0.01801528 / 6.022140857e23) / (1.38064852e-23 * iceTableTemps); # compute vapor densities at ice table
        
        meanIceTable_Pv = np.nanmean(iceTable_Pv) # get mean for that year of ice table vapor pressures
        meanIceTable_rhov = np.nanmean(iceTable_rhov) # get mean for that year of ice table vapor densities
        meanIceTableT = np.nanmean(iceTableTemps) # mean temperature at the ice table over the year
        meansurfT = np.nanmean(Tsurf) # mean surface temperature
    
    print("Step 7: Calculate sublimation")
    #= sublimation()
    
    print("Step 8: make retreat table")
    
    print("Step 9; Output retreat info")
    
    
    
#%% minimum to run
# Datas
class Profile:
  def reader(self,input_dict,*kwargs):
    for key in input_dict:
      try:
        setattr(self, key, input_dict[key])
      except:
        print("no such attribute, please consider add it at init")
        continue
    
    
with open(loc+"data/Trough_slope_02_9.json",'r') as file:
    a=file.readlines()
Mars_Trough=Profile()
Mars_Trough.reader(json.loads(a[0]))


with open(loc+'data/Trough_flat_00.json','r') as file:
    a = file.readlines()
flat_Mars_Trough=Profile()
flat_Mars_Trough.reader(json.loads(a[0]))


with open(loc+"data/Layers_Bramson2019.json",'r') as file:
    a=file.readlines()
bramson=Profile()
bramson.reader(json.loads(a[0]))


# grabbing the real values. 
ecc = Mars_Trough.eccentricity
obl = np.deg2rad(Mars_Trough.obl)
Lsp = np.deg2rad(Mars_Trough.Lsp)
dt_orb = Mars_Trough.dt

# run
soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, flat_Mars_Trough)
nLayers, ktherm, dz, rho, cp, kappa, depthsAtMiddleOfLayers = ml.Make(bramson, MarsyearLength, Mars_Trough.Rotation_rate) 
t, wt, ltimeT, Tsurf, frost = Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, kappa, emissivity, Tfrost, Tref, depthsAtMiddleOfLayers)    
    
