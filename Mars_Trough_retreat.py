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

# ** Import custom Libraries **
import Orbital_Parameters as op
import Make_Layers as ml

# ** Define things needed to run **    
loc = '/Users/laferrierek/Box Sync/Desktop/Mars_Troughs/Project_MCMC/Thermal_model_KLL/'

# Aesthetic
fs = (10, 8)
res = 350
plt.rc("font", size=18, family="serif")
plt.style.use('ggplot')


#%% 2. Use the pre-made from solar_flux_trough.py to run for xMyrs
# read in min_mean,max, use max. 
time_frame = 100
time_op, sfmin, sfmean, sf = np.loadtxt(loc+'solar_flux_minmeanmax_%3.0f_kyr_Trough1.txt'%time_frame, skiprows=1, delimiter=',', unpack=True)



#%% Actual run
if __name__ == "__main__":
    # given specific trough:
    print("Step 1: find orbital parameters")
    
    # grabbing the real values. 
    ecc = Mars_Trough.eccentricity
    obl = np.deg2rad(Mars_Trough.obl)
    Lsp = np.deg2rad(Mars_Trough.Lsp)
    dt_orb = Mars_Trough.dt
    
    # if flatvis_saved_trough1.txt doesn't exist:
    soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, flat_Mars_Trough)
    #np.savetxt('flatVis_Saved_Trough1.txt', sf)

    print("Step 2: make layers")
    # need layer infos
    nLayers, ktherm, dz, rho, cp, kappa, depthsAtMiddleOfLayers = ml.Make(bramson, MarsyearLength, Mars_Trough.Rotation_rate) 
    
    T_regs, fwindupTemps, ffinaltemps, fTsurf, ffrostMasses = Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref, depthsAtMiddleOfLayers)
    # write a save file
    #np.savetxt('CN_flat_Trough1.txt', T_regs, delimiter=',', header='Temp (K) for flat')
    
    print("Step 3: Run for real slope")
    soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, Mars_Trough)
    Temps, windupTemps, finaltemps, Tsurf, frostMasses = Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref, depthsAtMiddleOfLayers)
    
    print("Step 4: return last year values")
    final_year(Tsurf, Temps, frostMasses)
   
    print("Step 5: find the diurnal averages throughout the year")
    avg_all = diurnal_average(hr, Temps, T_regs, nLayers)

    print("Step 6: Calculate ice table")
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
    = sublimation()
    
    print("Step 8: make retreat table")
    
    print("Step 9; Output retreat info")
    
    
    
    
    
    