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

MarsyearLength = 59350658.844609514
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

def Make(filename, lengthOfYear, lengthOfDay):
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
    allThermal = np.sqrt(cp*rho*k)
    print('Thermal inertia is: %2.2f - %2.2f'%(allThermal[0], allThermal[-1]))
    
    diurnalSkinDepths = np.sqrt(allKappas*lengthOfDay/np.pi) #% meters
    print('Diurnal thermal skin depth of top layer = %.8f m'%diurnalSkinDepths[0])
    annualSkinDepths = np.sqrt(allKappas*lengthOfYear/np.pi) #% meters
    print ('Annual thermal skin depth of bottom layer = %.8f m'%annualSkinDepths[-1])
      
    firstLayerThickness = 0.2*diurnalSkinDepths[0] 
    numberLayers = math.ceil(np.log(1-(1-layerGrowth)*(annualLayers*annualSkinDepths[-1]/firstLayerThickness))/np.log(layerGrowth) ) #% Number of subsurface layers based on annual skin depth of deepest layer
    print('Number of layers = %2.2f'%numberLayers)
    print('First layer thickness = %.4f'%firstLayerThickness)
    
    dz = (firstLayerThickness * layerGrowth**np.arange(0, numberLayers, 1)) #% transpose to make column vector
    depthsAtMiddleOfLayers = np.cumsum(dz) - dz/2
    depthBottom = sum(dz)
    
    print('Layer depth = %2.4f'%depthBottom)
 
    k_vector = np.zeros(numberLayers)       #% Thermal conductivities (W m^-1 K^-1)
    density_vector = np.zeros(numberLayers) #% Densities of subsurface (kg/m^3)
    c_vector = np.zeros(numberLayers)       #% Specific heats J/(kg K)
    
    for ii in range(0, nPropLayers):
        if filename.depth[ii] > depthBottom:
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
    
    