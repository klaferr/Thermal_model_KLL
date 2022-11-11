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

import matplotlib.pyplot as plt

# ** Import custom Libraries **
import Orbital_Parameters as op
import Make_Layers as ml
#import testing_matrixmath as cn
import CN_1D as cn
import constants_MT as cT
import Sublimation_calculations as sub

# ** Define things needed to run **    
loc = '/Users/laferrierek/Box Sync/Desktop/Mars_Troughs/Project_MCMC/Thermal_model_KLL/'

# ask user: is test trun?
test_run = True
trough_num = 1 # ask for trough number
# for a test run, do:
time_stop = 5
time_start = 0
time_step = 1

#%% Functions
# save flat data
def flat_save(nStepsInYear, frostMasses, s, Tsurf, sf, trough_num, timef, locf):
    flatIR = np.zeros((nStepsInYear))*np.nan
    flatVis = np.zeros((nStepsInYear))*np.nan
    for n in range(0, nStepsInYear):
        if frostMasses[n] > 0:
           flatIR[n] = s.emisFrost * cT.sigma * Tsurf[n-1]**4
           flatVis[n] = s.albedoFrost * sf[n]
        else:
           flatIR[n] = s.emis * cT.sigma * Tsurf[n-1]**4
           flatVis[n] = s.albedo * sf[n]

    #if frostMasses[0] > 0:
    #    flatIR[0] = emisFrost * sigma * Tsurf[-1]**4
    #    flatVis[0] = Afrost * sf[0]
    #else:
    #    flatIR[0] = emis * sigma * Tsurf[-1]**4
    #    flatVis[0] = A * sf[0]
    #filename = 'data/flatVis_saved_Trough_Lt{:0>3}'.format(time) + '_Tr{:0>2}'.format(trough_num) + '.txt'
    flat_save = np.vstack((flatIR, flatVis, Tsurf)).T
    np.savetxt(locf, flat_save, delimiter=',')
    return flatIR, flatVis


class Profile:
  def reader(self,input_dict,*kwargs):
    for key in input_dict:
      try:
        setattr(self, key, input_dict[key])
      except:
        print("no such attribute, please consider add it at init")
        continue

#%% Datas
with open(loc+"data/Trough_slope_{:0>2}".format(trough_num)+".json",'r') as file:
    a=file.readlines()
Mars_Trough=Profile()
Mars_Trough.reader(json.loads(a[0]))


with open(loc+'data/Trough_flat_{:0>2}'.format(trough_num)+'.json','r') as file:
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
#albedo = 0.25
Q = 0.03 #30*10**(-3) # W/m2ma
Tref = 165

#emissivity = params.emis
#emisFrost = params.emisFros
#albedoFrost = params.albedoFrost

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
    #terrain = ml.layer_types(params.regolithPorosity, params.icePorosity, params.iceDust)
    terrain = ml.preset_layertypes(bramson)

    diurnalSkinDepths = np.sqrt(terrain.Kappa*cT.MarsdayLength/np.pi) # meters
    annualSkinDepths = np.sqrt(terrain.Kappa*cT.MarsyearLength/np.pi) # meters
    
    # Step 3: Initial Lag thickness
    print("Step 3: Inital Lag thickness - check")
    if params.InputDepth == 1:
        z_ei = params.initialLagThickness
        z_pf = params.initialLagThickness
        print('Initial lag thickness is %2.2f'%z_ei)
    elif params.InputDepth == 0:
        # this is not done, there is more to be done here. but were not using this right now 
        z_ei, z_pf = ml.pick_input_depth(time_start, L04_lsp, L04_ecc, L04_obl, annualSkinDepths, bramson)
    else:
        print('Oh no! what do we do with lag?')

    
    # Step 4: create layers with initial lag thickness
    print("Step 4: set grids - check")
    nLayers, ktherm, dz, rhoc, kappa, z_ei_index, z_pf_index, depthsAtMiddleOfLayers, depthsAtLayerBoundaries = ml.Make(bramson, terrain, cT.MarsyearLength, Mars_Trough.Rotation_rate, z_ei, z_pf) 
 
    # Step 5: run model through orbital
    print('Step 5: Run for Laskar timesteps (%3.0f); from %3.0f to %3.0f'%(time_step, time_start,time_stop))
    #%%
    #outputretreat = np.ones((np.int64((time_stop-time_start)/time_step), 9))*np.nan
    outputretreat = np.ones((np.int64((time_stop-time_start)/time_step), 9))*np.nan
    header_line = ['timebeforepresent, ecc, oblDeg, Lsp_rad, meanTsurf, atmosPressSF12, TotalForced, TotalFree, TotalSublimation']
    filename = loc+'data/'+ 'Trough%1.0f/'%trough_num+'Retreat_saved_Trough_Tr{:0>2}'.format(trough_num) + '_Laskar{:0>5}'.format(time_start)+ '_{:0>5}'.format(time_stop) +'.txt' # these needs a new name, stop overwriting
    
    f = open(filename, 'w')
    fwriter=csv.writer(f)
    fwriter.writerow(header_line)

    # the following loop is simplied as the flat (surrounding ice of NPLD) and sloped case
    # are treated as having the same input values for k, rhoc, kappa, skin depths
    for timef in range(time_start, time_stop, time_step):
        #%%
        path_to_slope = loc+'data/Trough%1.0f/'%trough_num+'slopedTest_Temps_saved_Lt{:0>3}'.format(timef)+'_Tr{:0>3}'.format(trough_num)+'.txt'
        path_to_slope2 = loc+'data/Trough%1.0f/'%trough_num+'slopedTest_Tsurf_saved_Lt{:0>3}'.format(timef)+'_Tr{:0>3}'.format(trough_num)+'.txt'
    
        path_to_flat = loc+'data/Trough%1.0f/'%trough_num +'flatVis_saved_Trough_Lt{:0>3}'.format(timef) + '_Tr{:0>2}'.format(trough_num) + '.txt'
        path_to_flat2 = loc+'data/Trough%1.0f/'%trough_num +'flatREGmin_saved_Trough_Lt{:0>3}'.format(timef) + '_Tr{:0>2}'.format(trough_num) + '.txt'
        path_to_flat3 = loc+'data/Trough%1.0f/'%trough_num +'flatREGdi_saved_Trough_Lt{:0>3}'.format(timef) + '_Tr{:0>2}'.format(trough_num) + '.npy'

        # Calculate at Laskar timestep
        ecc = L04_ecc[timef]
        Lsp = L04_lsp[timef]
        obl = L04_obl[timef]
        timebeforepresent = timef*1000

        # - why is this here?        
        atmoP = sub.calc_atmPress(obl)
        atmoPressure = atmoP * cT.atmofactor   

        if (Mars_Trough.Slope != 0) & (Mars_Trough.Slope_rerad == 1):
            # flat case first. purley ice around
            z_ei = 0; z_pf = 0
            
            # things we don't need to do:
            # layers using flat
            
            # Need to do:
            # Temps, orbital params, reorg diurnal
            
            # the flat case:
            print('Step 5a: Run for flat, open or save')  
            
            ## check if exists

            path = Path(path_to_flat3)
            if path.is_file():
                print('Files found.')
                flatIR, flatVis, fTsurf = np.loadtxt(path_to_flat, unpack='True', delimiter=',')
                
                soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, flat_Mars_Trough, path_to_flat)

                Ls0, fTsurfLs0, hrLs0 = cn.diurnal_wrap(nStepsInYear, nLayers, fTsurf, hr, lsWrapped) #, nStepsInYear = 118780)

                REGmin, REGdiurnal_rs = cn.diurnal_average(hrLs0, fTsurfLs0, 0, Ls0, nStepsInYear)
                REGmin = REGmin[0]
                REGdiurnal = REGdiurnal_rs.reshape(669)
                
                #REGmin = np.loadtxt(path_to_flat2, unpack='True', delimiter=',')
                #with open(path_to_flat3, "rb") as file:
                #    REGdiurnal_rs = np.load(file, allow_pickle=True)
                #REGdiurnal = REGdiurnal_rs.reshape(669)

            else:
                print('Did not exist:'+ path_to_flat)
                # not built to handle different k, TI for flat case   
                fnLayers, fktherm, fdz, frhoc, fkappa, fz_ei_index, fz_pf_index, fdepthsAtMiddleOfLayers, fdepthsAtLayerBoundaries = ml.Make(bramson, terrain, cT.MarsyearLength, Mars_Trough.Rotation_rate, 0, 0) 

                soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, flat_Mars_Trough, path_to_flat)
                CO2_FrostPoints = sub.CO2_frost_T(lsrad, Mars_Trough)

                fTemps, fwindupTemps, ffinaltemps, fTsurf, ffrostMasses = cn.Crank_Nicholson(fnLayers, nStepsInYear, windupTime, runTime, fktherm, fdz, dt, frhoc, fkappa, params.flatAlbedo, Tref, fdepthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR, CO2_FrostPoints, flat_Mars_Trough, params)
                plt.plot(fTsurf)
                plt.show()
                
                #flatIR, flatVis = cn.reradFlat(nStepsInYear, fTsurf, params, sf, ffrostMasses)
                flatIR, flatVis = flat_save(nStepsInYear, ffrostMasses, params, fTsurf, sf, trough_num, timef, path_to_flat)
                #flatIR, flatVis = flat_save(nStepsInYear, ffrostMasses, emisFrost, emissivity, cT.sigma, fTsurf, fTemps, albedoFrost, albedo, sf, trough_num, timef, path_to_flat)
                         
                Ls0, fTsurfLs0, hrLs0 = cn.diurnal_wrap(nStepsInYear, nLayers, fTsurf, hr, lsWrapped) #, nStepsInYear = 118780)

                REGmin, REGdiurnal_rs = cn.diurnal_average(hrLs0, fTsurfLs0, 0, Ls0, nStepsInYear)
                REGmin = REGmin[0]
                REGdiurnal = REGdiurnal_rs.reshape(669)
                
                np.savetxt(path_to_flat2, REGmin.T, delimiter=',')
                with open(path_to_flat3, "wb") as file:
                    np.save(file, REGdiurnal)
                print('flat files ran and saved')
                
            
        if z_ei > 0:
            # lag deposit
            
            # things we don't need to do:
            # layers using input
            
            # things I do need to do:            
            # find equil, reorg
            soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, Mars_Trough, path_to_flat)
            Temps, windupTemps, finaltemps, Tsurf, frostMasses = cn.Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rhoc, kappa, params.albedo, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR, CO2_FrostPoints, Mars_Trough, params)

            z_eq, z_pf = ml.EquilibriumDepth(Temps, Tsurf, atmoPressure, ktherm, dz, z_ei_index, z_pf_index)
            
            Ls0, TsurfLs0, hrLs0 = cn.diurnal_wrap(nStepsInYear, nLayers, Tsurf, hr, lsWrapped) #, nStepsInYear = 118780)

            #REGmin, REGdiurnal_rs = cn.diurnal_average(hrLs0, TsurfLs0, 0, Ls0, nStepsInYear)
            #REGmin = REGmin[0]
            #REGdiurnal = REGdiurnal_rs.reshape(669)
            
        else:
            # no lag, excess ice all the way down
            z_ei = 0; z_pf = 0
            
            # things we don't need to do:
            # layers using input 
            
            # things we do need to do:
            # calculate orbital params, temps, reorg diurnal
            
            
            print("Step 6: Run for real slope")
            if test_run == True:

                path = Path(path_to_slope)
                if path.is_file():
                    Tsurf, frostMasses, hr, lsWrapped = np.loadtxt(path_to_slope2, unpack=True, delimiter=',', skiprows=1)
                    nStepsInYear = np.size(hr)
                    Temps = np.loadtxt(path_to_slope, delimiter=',', skiprows=1)
                    
                    Ls0, TsurfLs0, hrLs0 = cn.diurnal_wrap(nStepsInYear, nLayers, Tsurf, hr, lsWrapped) #, nStepsInYear = 118780)

                    #REGmin, REGdiurnal_rs = cn.diurnal_average(hrLs0, TsurfLs0, 0, Ls0, nStepsInYear)
                    #REGmin = REGmin[0]
                    #REGdiurnal = REGdiurnal_rs.reshape(669)

                else:
                    print('File does not exist, running and saving')
                    soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, Mars_Trough, path_to_flat)
                    CO2_FrostPoints = sub.CO2_frost_T(lsrad, Mars_Trough)
                    Temps, windupTemps, finaltemps, Tsurf, frostMasses = cn.Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rhoc, kappa, params.albedo, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR, CO2_FrostPoints, Mars_Trough, params)
                    
                    Ls0, TsurfLs0, hrLs0 = cn.diurnal_wrap(nStepsInYear, nLayers, Tsurf, hr, lsWrapped) #, nStepsInYear = 118780)

                    #REGmin, REGdiurnal_rs = cn.diurnal_average(hrLs0, TsurfLs0, 0, Ls0, nStepsInYear)
                    #REGmin = REGmin[0]
                    #REGdiurnal = REGdiurnal_rs.reshape(669)
                    
                    save_arr = np.vstack((Tsurf, frostMasses, hr, lsWrapped)).T
                    header_arr = 'Tsurf, frostmasses, hr, lsWrapped'
                    np.savetxt(path_to_slope, Temps, delimiter=',', header='Temps per layer (K)')
                    np.savetxt(path_to_slope2, save_arr, delimiter=',', header=header_arr)
                        
            else:
                soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, Mars_Trough, path_to_flat)
                CO2_FrostPoints = sub.CO2_frost_T(lsrad, Mars_Trough)
    
                Temps, windupTemps, finaltemps, Tsurf, frostMasses = cn.Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rhoc, kappa, params.albedo, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR, CO2_FrostPoints, Mars_Trough, params)
                
                Ls0, TsurfLs0, hrLs0 = cn.diurnal_wrap(nStepsInYear, nLayers, Tsurf, hr, lsWrapped) #, nStepsInYear = 118780)

                #REGmin, REGdiurnal_rs = cn.diurnal_average(hrLs0, TsurfLs0, 0, Ls0, nStepsInYear)
                #REGmin = REGmin[0]
                #REGdiurnal = REGdiurnal_rs.reshape(669)
        #%%   
        # issue is in reg diurnal
        print("Step 6a: return last year values")
        cn.final_year(Tsurf, Temps, frostMasses)

        # diurnal wrap
        print('Step 6c: Reorganize to diurnal')
        #Ls0, TsurfLs0, hrLs0 = cn.diurnal_wrap(nStepsInYear, nLayers, Tsurf, hr, lsWrapped) #, nStepsInYear = 118780)

        print("Step 6b: find the diurnal averages throughout the year")
        SLOPED = cn.diurnal_average(hrLs0, TsurfLs0, Mars_Trough.Slope, Ls0, nStepsInYear)
        

        print('Step 7: calculate sublimation')
        print('differs in free!! due to issue in regmin ')
        Free, Forced, Sublimation = sub.sublimation_free_forced(Ls0, L04_obl, cT.atmofactor, atmoPressure, SLOPED, REGmin, REGdiurnal)

        # ice loss in Earth years
        print('Step 8: ice loss in Earth years')
        TotalFree_Eyr = Free*cT.EarthYearLength/cT.MarsyearLength
        TotalForced_Eyr = Forced*cT.EarthYearLength/cT.MarsyearLength
        TotalSublimation_Eyr = Sublimation*cT.EarthYearLength/cT.MarsyearLength
        
        meanTsurf = np.mean(TsurfLs0)
        
        print("Step 9; Output retreat info")
        time_p = time_start - timef
        outputretreat[np.int64(time_p/time_step), :] = np.array(([timebeforepresent, ecc, np.rad2deg(obl), np.rad2deg(Lsp), meanTsurf, atmoPressure, TotalForced_Eyr, TotalFree_Eyr, TotalSublimation_Eyr])).T
        
        fwriter.writerow(outputretreat[np.int64(time_p/time_step), :])
    

    print("Step 10: make retreat table")
    f.close()


    
