#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:07:44 2021

@author: laferrierek
"""

# Layer maker
import numpy as np
import math
import json
from scipy.interpolate import interp1d
import constants_MT as cT
import Sublimation_calculations as sub # - don't want to do this!


dt = 500

loc = '/Users/laferrierek/Desktop/Mars_Troughs/Codes/thermal_model/'

class Profile:
  def reader(self,input_dict,*kwargs):
    for key in input_dict:
      try:
        setattr(self, key, input_dict[key])
      except:
        print("no such attribute, please consider add it at init")
        continue
    
with open(loc+"trough_parameters.json",'r') as file:
  a=file.readlines()
Mars_Trough=Profile()
Mars_Trough.reader(json.loads(a[0]))

#%% Layers - changes at depth

def Make(filename, layers, lengthOfYear, lengthOfDay, z_ei, z_pf):

    k = np.array(layers.k)
    rhoc = np.array(layers.rhoc)
    allKappas = np.array(layers.Kappa)
    allThermal = np.sqrt(rhoc*k)  # units: J/m^2/K/s^(1/2)
    
    layerGrowth = filename.Growth
    dailyLayers = filename.daily
    annualLayers = filename.annual


    print('Thermal inertia is: %2.2f - %2.2f'%(allThermal[0], allThermal[-1]))
    
    diurnalSkinDepths = np.sqrt(allKappas*lengthOfDay/np.pi) # meters
    print('Diurnal thermal skin depth of top layer = %.8f m'%diurnalSkinDepths[0])
    annualSkinDepths = np.sqrt(allKappas*lengthOfYear/np.pi) # meters
    print ('Annual thermal skin depth of bottom layer = %.8f m'%annualSkinDepths[-1])
    
    # z's
    #z_ei = annualSkinDepths[0]*annualLayers
    #z_pf = annualSkinDepths[0]*annualLayers
    
    if z_ei <= 0:
        # no lag - check this works, excess ice
        firstLayerThickness = diurnalSkinDepths[2]/((1-layerGrowth**dailyLayers)/(1-layerGrowth))
        numberLayers = math.ceil(np.log(1-(1-layerGrowth)*(annualLayers*annualSkinDepths[2]/firstLayerThickness))/np.log(layerGrowth) ) # Number of subsurface layers based on annual skin depth of deepest layer
        dz = (firstLayerThickness * layerGrowth**np.arange(0, numberLayers, 1)) # transpose to make column vector

        #
        k_vector = np.zeros(numberLayers)       # Thermal conductivities (W m^-1 K^-1)
        rhoc_vector = np.zeros(numberLayers) # Densities of subsurface (kg/m^3)
        
        # set to excess
        k_vector[:] = k[2]
        rhoc_vector[:] = rhoc[2] 
        z_ei_index = 0
        z_pf_index = -1

    else: # lag exists
        # dry lag
        firstLayerThickness = diurnalSkinDepths[0]/((1-layerGrowth**dailyLayers)/(1-layerGrowth))

        if z_pf <= 0: # this could be it's own function
            # porefilling ice is stable up to surface - check
            firstLayerThickness = diurnalSkinDepths[0]/((1-layerGrowth**dailyLayers)/(1-layerGrowth))
            numberLayers = math.ceil(np.log(1-(1-layerGrowth)*(annualLayers*annualSkinDepths[2]/firstLayerThickness))/np.log(layerGrowth) ) # Number of subsurface layers based on annual skin depth of deepest layer
            dz = (firstLayerThickness * layerGrowth**np.arange(0, numberLayers, 1)) # transpose to make column vector

            # setup
            k_vector = np.zeros(numberLayers)       # Thermal conductivities (W m^-1 K^-1)
            rhoc_vector = np.zeros(numberLayers) # Densities of subsurface (kg/m^3)
            
            # excess ice table depth
            depthsAtLayerBoundaries = np.cumsum(dz)
            z_ei_index =np.argwhere(np.abs(depthsAtLayerBoundaries - z_ei) == np.min(np.abs(depthsAtLayerBoundaries - z_ei)))
            
            if z_ei_index ==1:
                z_ei_index+=1
            z_pf_index = 0 # pore ice to surface
            depthsAtLayerBoundaries[z_ei_index] = z_ei
            
            # remesh
            dz[z_ei_index] = np.abs(depthsAtLayerBoundaries[z_ei_index] - depthsAtLayerBoundaries[z_ei_index-1])
            dz[z_ei_index+1] = np.abs(depthsAtLayerBoundaries[z_ei_index+1] - depthsAtLayerBoundaries[z_ei_index])
            
            # set reset of layers
            k_vector[0:z_ei_index] = k[1]
            rhoc_vector[0:z_ei_index] = rhoc[1]
            
            k_vector[z_ei_index:] = k[2]
            rhoc_vector[z_ei_index:] = rhoc[2]
            
        elif (z_pf < firstLayerThickness) | (z_ei < firstLayerThickness):
            # one layer dry, thickness is thinner than first lyer
            firstLayerThickness = np.min([z_pf, z_ei])
            numberLayers = math.ceil(np.log(1-(1-layerGrowth)*(annualLayers*annualSkinDepths[2]/firstLayerThickness))/np.log(layerGrowth) ) # Number of subsurface layers based on annual skin depth of deepest layer
            dz = (firstLayerThickness * layerGrowth**np.arange(0, numberLayers, 1)) # transpose to make column vector

            # se tup
            k_vector = np.zeros(numberLayers)       # Thermal conductivities (W m^-1 K^-1)
            rhoc_vector = np.zeros(numberLayers) # Densities of subsurface (kg/m^3)
            
            if firstLayerThickness == z_ei:
                # one think later o dry with no pf ice
                z_ei_index = 1
                z_pf_index = -1
                
                # explore chnging the indexes.
                k_vector[0] = k[0]
                rhoc_vector[0] = rhoc[0] 
                
                k_vector[1:] = k[2]
                rhoc_vector[1:] = rhoc[2] 
                
            elif firstLayerThickness == z_pf:
                # oen thin layer of dry lag with pf ice between excess and eqil. depth
                
                k_vector[0] = k[0]
                rhoc_vector[0] = rhoc[0] 
        
                z_pf_index = 1
                
                # exces sice table depthdepthsAtLayerBoundaries = np.cumsum(dz)
                z_ei_index = np.argwhere(np.abs(depthsAtLayerBoundaries - z_ei) == np.min(np.abs(depthsAtLayerBoundaries - z_ei)))
                
                if z_ei_index ==1:
                    z_ei_index+=1
                    
                depthsAtLayerBoundaries[z_ei_index] = z_ei
                
                # remesh
                dz[z_ei_index] = np.abs(depthsAtLayerBoundaries[z_ei_index] - depthsAtLayerBoundaries[z_ei_index-1])
                dz[z_ei_index+1] = np.abs(depthsAtLayerBoundaries[z_ei_index+1] - depthsAtLayerBoundaries[z_ei_index])
                
                # set
                k_vector[z_pf_index:z_ei_index] = k[1]
                rhoc_vector[z_pf_index:z_ei_index] = rhoc[1]
                k_vector[z_ei_index:] = k[2]
                rhoc_vector[z_ei_index:] = rhoc[2]
            else:
                print('Error in first layer thickness')
                
        else:
            # more than one layer of lag.  - concerned
            firstLayerThickness = diurnalSkinDepths[0]/((1-layerGrowth**dailyLayers)/(1-layerGrowth))
            numberLayers = math.ceil(np.log(1-(1-layerGrowth)*(annualLayers*annualSkinDepths[2]/firstLayerThickness))/np.log(layerGrowth) ) # Number of subsurface layers based on annual skin depth of deepest layer
            dz = (firstLayerThickness * layerGrowth**np.arange(0, numberLayers, 1)) # transpose to make column vector
            
            # se tup
            k_vector = np.zeros(numberLayers)       # Thermal conductivities (W m^-1 K^-1)
            rhoc_vector = np.zeros(numberLayers) # Densities of subsurface (kg/m^3)
            #c_vector = np.zeros(numberLayers)       # Specific heats J/(kg K)
            
            # excess ice table depth
            depthsAtLayerBoundaries = np.cumsum(dz)

            z_ei_index = np.argwhere(np.abs(depthsAtLayerBoundaries - z_ei) == np.min(np.abs(depthsAtLayerBoundaries - z_ei)))[0][0]
            z_pf_index = np.argwhere(np.abs(depthsAtLayerBoundaries - z_pf) == np.min(np.abs(depthsAtLayerBoundaries - z_pf)))[0][0]
            
            if z_pf < z_ei:
                # case : if porefilling ice is barely stable, within the same layer as exess ice. move it. 
                if z_pf_index == z_ei_index:
                    z_ei_index+=1
                    
                depthsAtLayerBoundaries[z_pf_index] = z_pf
                
                # remesh
                dz[z_pf_index] = np.abs(depthsAtLayerBoundaries[z_pf_index] - depthsAtLayerBoundaries[z_pf_index-1])
                dz[z_pf_index+1] = np.abs(depthsAtLayerBoundaries[z_pf_index+1] - depthsAtLayerBoundaries[z_pf_index])
                depthsAtLayerBoundaries = np.cumsum(dz)
                
                # excess ice depth
                depthsAtLayerBoundaries[z_ei_index] = z_ei
                
                # remesh
                dz[z_ei_index] = np.abs(depthsAtLayerBoundaries[z_ei_index] - depthsAtLayerBoundaries[z_ei_index-1])
                dz[z_ei_index+1] = np.abs(depthsAtLayerBoundaries[z_ei_index+1] - depthsAtLayerBoundaries[z_ei_index])
                
                # set
                k_vector[0:z_pf_index] = k[0]
                rhoc_vector[:z_pf_index] = rhoc[0]
                
                k_vector[z_pf_index:z_ei_index] = k[1]
                rhoc_vector[z_pf_index:z_ei_index] = rhoc[1]
                
                k_vector[z_ei_index:] = k[2]
                rhoc_vector[z_ei_index:] = rhoc[2]
            else:
                # no pore ice, set index to -1
                z_pf_index = -1
                depthsAtLayerBoundaries[z_ei_index] = z_ei

                if z_ei_index > 1:
                    dz[z_ei_index] = np.abs(depthsAtLayerBoundaries[z_ei_index] - depthsAtLayerBoundaries[z_ei_index-1])
                    dz[z_ei_index+1] = np.abs(depthsAtLayerBoundaries[z_ei_index+1] - depthsAtLayerBoundaries[z_ei_index])
                else:
                    dz[z_ei_index] = z_ei
                    dz[z_ei_index+1] = np.abs(depthsAtLayerBoundaries[z_ei_index+1] - depthsAtLayerBoundaries[z_ei_index])
                                
                # set
                k_vector[0:z_ei_index] = k[0]
                rhoc_vector[:z_ei_index] = rhoc[0]
                
                k_vector[z_ei_index:] = k[2]
                rhoc_vector[z_ei_index:] = rhoc[2]

    depthsAtLayerBoundaries = np.cumsum(dz)
    depthsAtMiddleOfLayers = np.cumsum(dz) - dz/2
    Kappa_vector = k_vector/(rhoc_vector)     
    return numberLayers, k_vector, dz, rhoc_vector, Kappa_vector, z_ei_index, z_pf_index, depthsAtMiddleOfLayers, depthsAtLayerBoundaries
       
def EquilibriumDepth(Temps, Tsurf, atmoPressure, ktherm, dz, z_ei_index, z_pf_index):
    # bramson makes mares torughs only, update one day
    #sTrough = s # typically, flat trough
    #sTrough.slope = 0
    #sTrough.slope_aspect = 0
    
    FOUNDZeq = 0
    groundicecounter= 1
    PFICE = 1
    triedTwice = 0
    
    while FOUNDZeq == 0:
        dz_test = dz
        
        depthsAtLayerBoundary = np.cumsum(dz_test)
        
        # is ground ice in lag deposit? - from matlab code
        onesMatrix = np.ones((1, np.size(Temps[1:z_ei_index+1, :], 1)))
        boundary_temps = (ktherm[1:z_ei_index+1] * dz[0:z_ei_index] * onesMatrix * Temps[1:z_ei_index+1, :] + ktherm[0:z_ei_index] * dz[1:z_ei_index+1] * onesMatrix * Temps[0:z_ei_index,:]) / ((ktherm[1:z_ei_index+1] * dz[0:z_ei_index] + ktherm[0:z_ei_index] * dz[1:z_ei_index+1]) * onesMatrix)
        boundary_Pv = sub.clapyeron(cT.triple_P_h2o, cT.triple_Temp_h2o, cT.R_bar, 51058, boundary_temps)
        boundary_rhov = sub.rho_molecules(cT.m_gas_h2o, 0, boundary_Pv, 0, boundary_temps)
        boundary_rhov_mean = np.mean(boundary_rhov)
        
        Tsurf_mean = np.mean(Tsurf)
        
        atmoDensity = (atmoPressure * (cT.m_gas_h20/cT.NA/cT.k))/Tsurf_mean
        
        
        if np.all(boundary_rhov_mean < atmoDensity):
            # ground ice is not stable
            z_pf = depthsAtLayerBoundary[z_ei_index]
            z_eqConverge = -1
            
            print("Ground ice counter = %2.0f"%groundicecounter)
            print("Ground ice not stable, z_pf = %3.0f"%z_pf)
            
            if triedTwice > 0:
                FOUNDZeq = 1
                PFICE = 0
            else:
                triedTwice += 1
                FOUNDZeq = 0
         
        else:
            # ground ice is stable
            # middle of layers
            middle_Pv = sub.clapyeron(cT.triple_P_h2o, cT.triple_Temp_h20, cT.R_bar, 51058, Temps[0:z_ei_index])
            middle_rhov = sub.rho_molecules(cT.m_gas_h2o, 0, middle_Pv, 0, Temps[0:z_ei_index])
            middle_rhov_mean = np.mean(middle_rhov)
        
            # Combines into one array to interpolate from
            alldepths = np.concatenate((depthsAtMiddleOfLayers[0:z_ei_index], depthsAtLayerBoundary[0:z_ei_index]))
            print(np.shape(alldepths))
            all_rhov = np.hstack((middle_rhov_mean, boundary_rhov_mean))
            print(np.shape(all_rhov))
            
            if groundicecounter > 1:
                # is atmospheric value larger than layer? rhov
                if atmoDensity > all_rhov[0]:
                    z_eq = 0
                    z_pf = 0
                    FOUNDZeq = 1
                    
                    print("Ground ice counter = %2.0f"%groundicecounter)
                    print("Ground ice not stable, z_pf = %3.0f"%z_pf)
                    
                else:
                    z_eq = interp1d(all_rhov, alldepths)(atmoDensity) # itnerpolae
                    z_eqConverge = np.abs(z_pf - z_eq)
                    
                    tmp_z_pf_index  = np.argwhere(np.abs(depthsAtLayerBoundary - z_eq) == np.min(np.abs(depthsAtLayerBoundary - z_eq)))
                    
                    print("Ground ice counter = %2.0f"%groundicecounter)
                    print("Ground ice not stable, z_pf = %3.0f"%z_pf)
                    
                    if (z_eqConverge < np.min(np.array([dz_test[tmp_z_pf_index+1]/2, 0.003])))  and (groundicecounter > 1):
                        z_pf = z_eq
                        FOUNDZeq = 1
                    else:
                        if groundicecounter > 14:
                             # 14
                             FOUNDZeq = 1
                             print('Ground ice counter = %2.0f: z_pf = %3.9f, z_eq based on Temps = %3.9f, difference is = %3.9f and convergence criteria = %3.9f. Set to mean of the values: %3.9f \n'%(groundicecounter, z_pf, z_eq, z_eqConverge, min([dz_test[tmp_z_pf_index+1]/2, 0.003]),np.mean([z_pf, z_eq])))
                             z_pf = np.mean(np.array([z_pf, z_eq]))
                        else:
                             FOUNDZeq = 0
                             z_pf = z_eq
            else:
                if atmoDensity > all_rhov[0]:
                    z_eq = 0
                    z_pf = 0
                    print('Ground ice counter %2.0f, ice is stable upt to surface'%groundicecounter)
                else:
                    z_eq = z_pf
                    print('Ground ice counter %2.0f, ice is not stable upt to surface'%groundicecounter)
        groundicecounter+=1
    return z_eq, z_pf  
  


def layer_types(regoPoro, icePoro, iceDust):
    # this is used when the bramosn layer is not sassumed to be true
    rhoc_regolith = cT.density_rock* cT.c_rock*(1-regoPoro)
    rhoc_porefillingice = rhoc_regolith + cT.density_ice*cT.c_ice*(regoPoro)
    rhoc_excessice = cT.density_rock* cT.c_rock*(iceDust) + cT.density_ice*cT.c_ice*(1-iceDust-icePoro)
    
    k_regolith = cT.TI_rock**2/rhoc_regolith
    k_porefillingice = (1-regoPoro)*cT.k_rock + regoPoro*cT.k_ice
    k_excessice = (iceDust/(1-icePoro))*cT.k_rock + ((1-iceDust-icePoro)**2 /(1-icePoro))*cT.k_ice
    
    k = np.hstack([k_regolith, k_porefillingice, k_excessice])
    rhoc = np.hstack([rhoc_regolith, rhoc_porefillingice, rhoc_excessice])
    Kappa = k/rhoc
    
    class Folder:
        def __init__(self, rhoc, k, Kappa):
            self.rhoc = rhoc
            self.k = k
            self.Kappa = Kappa
    
    folder = Folder(rhoc, k, Kappa)

    return folder    

def preset_layertypes(tr):
    # tr == bramson
    # the values in the bramson file come from bramson + 2019
    class Folder:
        def __init__(self, rhoc, k, Kappa):
            self.rhoc = rhoc
            self.k = k
            self.Kappa = Kappa  
    
    rhoc = tr.rhoc
    k = np.array((tr.TI))**2/rhoc
    Kappa = k/rhoc
        
    folder = Folder(rhoc, k, Kappa)
    return folder   

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


#%%
def old_Make(filename, lengthOfYear, lengthOfDay):
    k = np.array(filename.k)
    rho = np.array(filename.rho)
    cp = np.array(filename.cp)
    depth = np.array(filename.depth)
    layerGrowth = filename.Growth
    dailyLayers = filename.daily
    annualLayers = filename.annual
    # out need modelLayers 0-4, len
    nPropLayers = np.size(filename.k)
    
    allKappas = k / (rho * cp)
    allThermal = np.sqrt(cp*rho*k)  # units: J/m^2/K/s^(1/2)
    print('Thermal inertia is: %2.2f - %2.2f'%(allThermal[0], allThermal[-1]))
    
    diurnalSkinDepths = np.sqrt(allKappas*lengthOfDay/np.pi) # meters
    # these are not what i am doing, kappas for rock, ice excessice
    print('Diurnal thermal skin depth of top layer = %.8f m'%diurnalSkinDepths[0])
    annualSkinDepths = np.sqrt(allKappas*lengthOfYear/np.pi) # meters
    print ('Annual thermal skin depth of bottom layer = %.8f m'%annualSkinDepths[-1])
    
    #dz = (firstLayerThickness * layerGrowth**np.arange(0, numberLayers, 1)) # transpose to make column vector
    
    # needs to be adjust to handle lag.
    firstLayerThickness = diurnalSkinDepths[0]/((1-layerGrowth**(dailyLayers)/(1-layerGrowth)))
    numberLayers = math.ceil(np.log(1-(1-layerGrowth)*(annualLayers*annualSkinDepths[-1]/firstLayerThickness))/np.log(layerGrowth) ) # Number of subsurface layers based on annual skin depth of deepest layer
    print('Number of layers = %2.2f'%numberLayers)
    print('First layer thickness = %.4f'%firstLayerThickness)
    
    dz = (firstLayerThickness * layerGrowth**np.arange(0, numberLayers, 1)) # transpose to make column vector
    
    depthsAtMiddleOfLayers = np.cumsum(dz) - dz/2
    depthsAtLayerBoundaries = sum(dz)
    print('Layer depth = %2.4f'%depthsAtLayerBoundaries)
 
    k_vector = np.zeros(numberLayers)       # Thermal conductivities (W m^-1 K^-1)
    density_vector = np.zeros(numberLayers) # Densities of subsurface (kg/m^3)
    c_vector = np.zeros(numberLayers)       # Specific heats J/(kg K)
    
    for ii in range(0, nPropLayers):
        if filename.depth[ii] > depthsAtLayerBoundaries:
            print('Warning: Model domain isn''t deep enough to have a layer at %f m.' %(depth[ii]))
        else:
            nPropLayersToUse = ii

    layerIndices = np.zeros(nPropLayersToUse+1) #% numerical layer index for top of each layer
    for ii in range(0, nPropLayersToUse+1):
                
        if ii==0:
            
            indexStart = 0
            
            if ii+1 <= nPropLayersToUse:
                indexesBelowDepth1 = np.argwhere(depthsAtMiddleOfLayers<depth[ii+1])
                indexEnd = indexesBelowDepth1[-1][0]
            else:
                indexEnd = numberLayers #[0]
            
        else:
            indexesBelowDepth2 = np.argwhere(depthsAtMiddleOfLayers<depth[ii])
            indexStarta = indexesBelowDepth2[-1] 
            indexStart = indexStarta[0]
            if ii+1 <= nPropLayersToUse:
                indexesBelowDepth3 = np.argwhere(depthsAtMiddleOfLayers<depth[ii+1])
                indexEnd = indexesBelowDepth3[-1][0]

            else:
                indexEnd = numberLayers
                        
        k_vector[indexStart:indexEnd] =k[ii]
        density_vector[indexStart:indexEnd] = rho[ii]
        c_vector[indexStart:indexEnd] = cp[ii]
        layerIndices[ii] = indexStart
     
    #% Calculate diffusivity
    Kappa_vector = k_vector / (density_vector * c_vector)
    
    #% Calculate timestep needed to fulfill Courant Criterion for
    #% numerical stability
    courantCriteria = dz * dz / (5 * Kappa_vector)
    courantdt = min(courantCriteria)
    print('For numerical stability, delta t will be %2.5f s.'%courantdt)
    
    numPropLayers = np.size(layerIndices)
    if numPropLayers > 1:
        iceTableIndex = int(layerIndices[1])
    else:
        iceTableIndex = 1

    return numberLayers, k_vector, dz, density_vector, c_vector, Kappa_vector, depthsAtMiddleOfLayers
    
    