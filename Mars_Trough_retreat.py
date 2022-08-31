#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:48:39 2022

@author: laferrierek

Main code: runs CN, everything else
"""

# ** Import libaries **
import numpy as np
import json
from pathlib import Path
import csv


# ** Import custom Libraries **
import Orbital_Parameters as op
import Make_Layers as ml
import CN_1D as cn
import constants_MT as cT
import Sublimation_calculations as sub

# ** Define things needed to run **    
loc = '/Users/laferrierek/Box Sync/Desktop/Mars_Troughs/Project_MCMC/Thermal_model_KLL/'

# ask user: is test trun?
test_run = True
trough_num = 2 # ask for trough number
# for a test run, do:
time_stop = 110
time_start = 10
time_step = 1

#%% Functions
# save flat data
def flat_save(nStepsInYear, frostMasses, emisFrost, emis, sigma, Tsurf, Temps, Afrost, A, sf, trough_num, timef, locf):
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
    #filename = 'data/flatVis_saved_Trough_Lt{:0>3}'.format(time) + '_Tr{:0>2}'.format(trough_num) + '.txt'
    flat_save = np.vstack((flatIR, flatVis, Tsurf)).T
    np.savetxt(locf, flat_save, delimiter=',')
    return


def pick_input_depth(time_start, L04_lsp, L04_ecc, L04_obl, annualSkinDepths, s):
    z_ei = annualSkinDepths[0]*s.annualLayers # look within 6 (or whatever annualLayers is) skin depths of dry lag and have no pore filling ice
    z_pf = annualSkinDepths[0]*s.annualLayers # no pore-filling ice- set pf value to be same as z_ei
    nLayers, ktherm, dz, rhoc, kappa, z_ei_index, z_pf_index, depthsAtMiddleOfLayers, depthsAtLayerBoundaries = ml.Make(bramson, terrain, cT.MarsyearLength, Mars_Trough.Rotation_rate, z_ei, z_pf)

    obl = L04_obl[time_start]
    
    atmoPress = sub.calc_atmPress(obl)
    
    PFICE, z_eq, z_pf = ml.EquilibriumDepth(s, bramson, atmoPress)
    
    if PFICE == 0:
        # if no equilibrium depth found within several annual skin depths
        print('No equilibrium depth found. Ice not stable at all. Setting excess ice interface at 6 meters depth.')
        z_ei = 6
        z_pf = 6
    else:
        z_ei = z_eq # Will set initial thickness of dry lag to be equilibrium depth
        z_pf = z_eq # No pore-filling ice to start, excess ice starts at equilibrium depth, z_pf >= z_ei is that condition
    return z_ei, z_pf

class Profile:
  def reader(self,input_dict,*kwargs):
    for key in input_dict:
      try:
        setattr(self, key, input_dict[key])
      except:
        print("no such attribute, please consider add it at init")
        continue

#%% Datas
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


with open(loc+"data/paramFile.json",'r') as file:
    a=file.readlines()
    # 1 is a yes, 0 is a no
params=Profile()
params.reader(json.loads(a[0]))

#%% Constants - From files and basic.
# Surface conditions
albedo = 0.25
emissivity = 1.0
Q = 0.03 #30*10**(-3) # W/m2
Tref = 165

# Frost
emisFrost = 0.95
albedoFrost = 0.6
Tfrost = cT.CO2_FrostPoints
windupTime = 8
convergeT = 0.05
porosity = params.iceDust *(1/(1-params.icePorosity - params.iceDust))/(1-params.regolithPorosity)

runTime = 15
f = 0.5
dt = 500


#%% Actual run
if __name__ == "__main__":
    # ask for trough number
    print('Running for trough: %2.0f'%(trough_num))
    # steps from Bramson et al. (2019) v12.m - 
    
    # Step 1; Find orbital parameteres from Laskar 2004 
    print("Step 1: find orbital parameters")
    
    # Format
    L04_timestep, L04_ecc, L04_obl, L04_lsa = np.loadtxt(loc+'/data/Laskar2004_ecc_obl_LSP.txt', skiprows=1, delimiter='\t', unpack=True)
    L04_lsp = L04_lsa - np.pi
    numLaskarRuns = np.round((time_stop - time_start)/time_step)
    dt_orb = Mars_Trough.dt
    
    # Step 2: Set layer properites 
    print("Step 2: Make Layers - pore-filling ice, excess ice, regolith")
    terrain = ml.layer_types(params.regolithPorosity, params.icePorosity, params.iceDust)
    
    diurnalSkinDepths = np.sqrt(terrain.Kappa*cT.MarsdayLength/np.pi) # meters
    annualSkinDepths = np.sqrt(terrain.Kappa*cT.MarsdayLength/np.pi) # meters
    
    # Step 3: Initial Lag thickness
    print("Step 3: Inital Lag thickness - check")
    if params.InputDepth == 1:
        z_ei = params.initialLagThickness
        z_pf = params.initialLagThickness
        print('Initial lag thickness is %2.2f'%z_ei)
    elif params.InputDepth == 0:
        z_ei, z_pf = pick_input_depth(time_start, L04_lsp, L04_ecc, L04_obl, annualSkinDepths, bramson)
    else:
        print('Oh no! what do we do with lag?')

    
    # Step 4: create layers with initial lag thickness
    print("Step 4: set grids - check")
    nLayers, ktherm, dz, rhoc, kappa, z_ei_index, z_pf_index, depthsAtMiddleOfLayers, depthsAtLayerBoundaries = ml.Make(bramson, terrain, cT.MarsyearLength, Mars_Trough.Rotation_rate, z_ei, z_pf) 
    
    # Step 5: run model through orbital
    print('Step 5: Run for Laskar timesteps (%3.0f); from %3.0f to %3.0f'%(time_step, time_start,time_stop))

    #outputretreat = np.ones((np.int64((time_stop-time_start)/time_step), 9))*np.nan
    outputretreat = np.ones((np.int64((time_stop-time_start)/time_step), 9))*np.nan
    header_line = ['timebeforepresent, ecc, oblDeg, Lsp_rad, meanTsurf, atmosPressSF12, TotalForced, TotalFree, TotalSublimation']
    filename = loc+'data/'+ 'Retreat_saved_Trough_Tr{:0>2}'.format(trough_num) + '.txt'
        
    f = open(filename, 'w')
    fwriter=csv.writer(f)
    fwriter.writerow(header_line)

    for timef in range(time_start, time_stop, time_step):

        # Calculate at Laskar times tep
        ecc = L04_ecc[timef]
        Lsp = L04_lsp[timef]
        obl = L04_obl[timef]
        timebeforepresent = timef*1000

        # - why is this here?        
        atmoP = sub.calc_atmPress(obl)
        atmoPressure = atmoP * cT.atmofactor                         
        
        # the flat case:
        print('Step 5a: Run for flat, open or save')  
        
        ## check if exists
        path_to_flat = loc+'data/Trough%1.0f/'%trough_num +'flatVis_saved_Trough_Lt{:0>3}'.format(timef) + '_Tr{:0>2}'.format(trough_num) + '.txt'
        path_to_flat2 = loc+'data/Trough%1.0f/'%trough_num +'flatREGmin_saved_Trough_Lt{:0>3}'.format(timef) + '_Tr{:0>2}'.format(trough_num) + '.txt'
        path_to_flat3 = loc+'data/Trough%1.0f/'%trough_num +'flatREGdi_saved_Trough_Lt{:0>3}'.format(timef) + '_Tr{:0>2}'.format(trough_num) + '.txt'

        path = Path(path_to_flat)
        if path.is_file():
            print('Files found.')
            flatIR, flatVis, fTsurf = np.loadtxt(path_to_flat, unpack='True', delimiter=',')
            REGmin = np.loadtxt(path_to_flat2, unpack='True', delimiter=',')
            with open(path_to_flat3, "r") as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                REGdiurnal = []
                for line in reader:
                    REGdiurnal.append(np.array((line)))

        else:
            print('Did not exist:'+ path_to_flat)
            soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, flat_Mars_Trough, path_to_flat)

            fTemps, fwindupTemps, ffinaltemps, fTsurf, ffrostMasses = cn.Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rhoc, kappa, emissivity, Tfrost, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR) 
            flat_save(nStepsInYear, ffrostMasses, emisFrost, emissivity, cT.sigma, fTsurf, fTemps, albedoFrost, albedo, sf, trough_num, timef, path_to_flat)
                       
            REGmin, REGdiurnal = cn.diurnal_average(hr,fTsurf, 0)
            REGmin = REGmin[0]
            
            np.savetxt(path_to_flat2, REGmin.T, delimiter=',')

            with open(path_to_flat3, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                for line in REGdiurnal:
                    writer.writerow(line)      
            print('flat files ran and saved')

        print("Step 6: Run for real slope")
        if test_run == True:
            print("opening")
            path_to_slope = loc+'data/Trough%1.0f/'%trough_num+'slopedTest_Temps_saved_Lt{:0>3}'.format(timef)+'_Tr{:0>3}'.format(trough_num)+'.txt'
            path_to_slope2 = loc+'data/Trough%1.0f/'%trough_num+'slopedTest_Tsurf_saved_Lt{:0>3}'.format(timef)+'_Tr{:0>3}'.format(trough_num)+'.txt'

            path = Path(path_to_slope)
            if path.is_file():
                Tsurf, frostMasses, hr, lsWrapped = np.loadtxt(path_to_slope2, unpack=True, delimiter=',', skiprows=1)
                Temps = np.loadtxt(path_to_slope, delimiter=',', skiprows=1)
                nStepsInYear = np.size(hr)
            else:
                print('File does not exist, running and saving')
                soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, Mars_Trough, path_to_flat)
                Temps, windupTemps, finaltemps, Tsurf, frostMasses = cn.Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rhoc, kappa, emissivity, Tfrost, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR)
                
                save_arr = np.vstack((Tsurf, frostMasses,hr, lsWrapped)).T
                header_arr = 'Tsurf, frostmasses, hr, lsWrapped'
                np.savetxt(path_to_slope, Temps, delimiter=',', header='Temps per layer (K)')
                np.savetxt(path_to_slope2, save_arr, delimiter=',', header=header_arr)
        else:
            soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, Mars_Trough, path_to_flat)
            Temps, windupTemps, finaltemps, Tsurf, frostMasses = cn.Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rhoc, kappa, emissivity, Tfrost, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR)
            

        print("Step 6a: return last year values")
        cn.final_year(Tsurf, Temps, frostMasses)

        print("Step 6b: find the diurnal averages throughout the year")
        SLOPED = cn.diurnal_average(hr, Tsurf, Mars_Trough.Slope)
        
        # diurnal wrap
        print('Step 6c: Reorganize to diurnal')
        Ls0, TsurfLs0, hrLs0 = cn.diurnal_wrap(nStepsInYear, nLayers, Temps, hr, lsWrapped) #, nStepsInYear = 118780)

        print('Step 7: calculate sublimation')
        Free, Forced, Sublimation = sub.sublimation_free_forced(Ls0, L04_obl, cT.atmofactor, atmoPressure, SLOPED, REGmin, REGdiurnal)

        # ice loss in Earth years
        print('Step 8: ice loss in Earth years')
        TotalFree_Eyr = Free*cT.EarthYearLength/cT.MarsyearLength
        TotalForced_Eyr = Forced*cT.EarthYearLength/cT.MarsyearLength
        TotalSublimation_Eyr = Sublimation*cT.EarthYearLength/cT.MarsyearLength
        
        meanTsurf = np.mean(TsurfLs0)

        print("Step 9; Output retreat info")
        outputretreat[np.int64(timef/time_step), :] = np.array(([timebeforepresent, ecc, np.rad2deg(obl), Lsp, meanTsurf, atmoPressure, TotalForced_Eyr, TotalFree_Eyr, TotalSublimation_Eyr])).T
        
        fwriter.writerow(outputretreat[np.int64(timef/time_step), :])
        toc = time.time()

    

    print("Step 10: make retreat table")
    f.close()


    
