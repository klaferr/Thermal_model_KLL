#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:46:55 2022

@author: laferrierek
"""

# plot test
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np


loc = '/Users/laferrierek/Box Sync/Desktop/Mars_Troughs/Project_MCMC/Thermal_model_KLL/'

trough_num = 1

path_to_file = loc+'data/flatVis_Saved_Trough_%1.0f.txt'%(trough_num)
flatIR, flatVis = np.loadtxt(path_to_file, unpack='True', delimiter=',')

path_to_file = loc+'data/raw_flat_Saved_Trough_%1.0f.txt'%(trough_num+1)
IR, Vis = np.loadtxt(path_to_file, unpack='True', delimiter=',')

plt.plot(flatIR, c='b')
plt.plot(IR, c='lightblue')

plt.plot(flatVis, c='orange')
plt.plot(Vis, c='r')
plt.show()


#%%%
eccentricity = 0.09341233
obl = np.deg2rad(25.19)
Lsp = np.deg2rad(250.87)
dt_orb = 500

# Surface conditions
#"""
A = 0.25
emis = 1.0
Q = 0.03 #30*10**(-3) # W/m2
Tref = 250

# Frost
emisFrost = 0.95
Afrost = 0.6
sigma = 5.670*10**(-8)      # Stefan-Boltzmann; W/m^2K^4
#frostMasses = np.sin(np.arange((0, 2*np.pi, np.size(flatIR))))
#Tsurf = np.cos(np.arange(100, 250, np.size(flatIR)))
#%%
nStepsInYear = np.size(flatIR)
flatIR = np.zeros((nStepsInYear))*np.nan
flatVis = np.zeros((nStepsInYear))*np.nan
for n in range(0, nStepsInYear):
    if frostMasses[n] > 0:
       flatIR[n] = emisFrost * sigma * Tsurf[n]**4
       flatVis[n] = Afrost * sf[n]
    else:
       flatIR[n] = emis * sigma * Tsurf[n]**4
       flatVis[n] = A * sf[n]

#if frostMasses[0] > 0:
#    flatIR[0] = emisFrost * sigma * Tsurf[-1]**4
#    flatVis[0] = Afrost * sf[0]
#else:
#    flatIR[0] = emis * sigma * Tsurf[-1]**4
#    flatVis[0] = A * sf[0]


plt.plot(flatIR, c='b')


plt.plot(flatVis, c='orange')

plt.show()


# frm CN
#%% Plot checks - i think temperature issues are rleated to the layer formation - not the thermal conductivity. 
def plot_checks_CN(Temps, windupTemps, finaltemps,Tsurf, frostMasses, depthsAtMiddleOfLayers, daily_depth, annual_depth, hr, nLayers):
    # Create diurnal averages throughout the last year
    beginDayIndex = np.zeros((670))*np.nan
    beginDayIndex[0] = 0
    dayIndex = 1
    for n in range(1, np.size(hr)):
        if (hr[n] > 0) & (hr[n-1] < 0):
            beginDayIndex[dayIndex] = n
            dayIndex = dayIndex + 1
    
    beginDayIndex = beginDayIndex.astype('int')
    numDays = np.max(np.size(beginDayIndex))
    averageDiurnalTemps = np.zeros((nLayers, numDays))
    averageDiurnalSurfTemps = np.zeros((numDays))
    
    for n in np.arange(0, numDays):
        if n == numDays-1:
            averageDiurnalTemps[:,n] = np.mean(Temps[:, beginDayIndex[n]:np.size(Temps, 1)], 1)
            averageDiurnalSurfTemps[n] = np.nanmean(Temps[beginDayIndex[n]:np.size(Temps,1)]) #- this seems wrong
        else:
            averageDiurnalTemps[:,n] = np.mean(Temps[:, beginDayIndex[n]:beginDayIndex[n+1]-1], 1)
            averageDiurnalSurfTemps[n] = np.mean(Temps[beginDayIndex[n]:beginDayIndex[n+1]-1])
    
    averageDiurnalAllTemps = np.concatenate((averageDiurnalSurfTemps.reshape(1,numDays), averageDiurnalTemps))
    days = 25
    
    color = cm.plasma(np.linspace(0, 1, np.int(670/days)))

    plt.figure(figsize=(5,5), dpi=300)
    # show depth every 50 days
    for i in range(0, 669, days):
        plt.plot(averageDiurnalAllTemps[1:, i], depthsAtMiddleOfLayers)
    plt.ylim((-0.5, np.nanmax(depthsAtMiddleOfLayers)))
    plt.xlim((140, 210))
    plt.gca().invert_yaxis()
    plt.ylabel('Depth (m)')
    plt.xlabel('Temperature (K)')
    plt.title('%2.0f day steps'%days)
    plt.show()

    # show avearge Diurnal results.     
    plt.figure(figsize=(5, 5), dpi=300)
    for i, c in zip(range(0, 669, days), color):
        plt.plot(averageDiurnalAllTemps[1:, i], depthsAtMiddleOfLayers, c=c) 
    plt.xlim((0, 250))
    plt.ylim((0, 3))
    plt.gca().invert_yaxis()
    plt.ylabel('Depth (m)')
    plt.xlabel('Temperature (K)')
    plt.title('Days per line: %2.0f '%(days))
    sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=0, vmax=i))
    plt.colorbar(sm)
    plt.show()
    
    # i want to plot the aveargae tempearure of a layer across time of day
    # why is the surface temp lower than subsurface?     - becuase we start in winter. 
    plt.figure(figsize=(5, 5), dpi=300)
    st_i = 400
    st_e = 402
    ltst =  hr * (180/(np.pi*15))
    daily = np.argwhere(depthsAtMiddleOfLayers >= daily_depth)[0]
    print(daily)
    plt.scatter(ltst[beginDayIndex[st_i]:beginDayIndex[st_e]], Temps[0, beginDayIndex[st_i]:beginDayIndex[st_e]], s=2, c='r', label='Surface')
    plt.scatter(ltst[beginDayIndex[st_i]:beginDayIndex[st_e]], Temps[daily, beginDayIndex[st_i]:beginDayIndex[st_e]], s=2, c='g', label='Skin depth diurnal')
    plt.scatter(ltst[beginDayIndex[st_i]:beginDayIndex[st_e]], Temps[-1, beginDayIndex[st_i]:beginDayIndex[st_e]], s=2,  c='b', alpha=0.5, label='Bottom')
    mean= np.nanmean(Temps[0, beginDayIndex[st_i]:beginDayIndex[st_e]])
    std = np.std(Temps[0, beginDayIndex[st_i]:beginDayIndex[st_e]]) 
    plt.hlines(mean, -12, 12, colors='k')
    plt.hlines(std/np.e+mean, -12, 12, colors='k', linestyle='dashed')
    plt.hlines(mean - std/np.e, -12, 12, colors='k', linestyle='dashed')
    plt.ylabel('temp (k)')
    plt.xlabel('Local time (hr)')
    plt.legend()
    plt.xlim((-12, 12))
    plt.ylim((120, 280))
    plt.show()
    
    # yearly:
    plt.figure(figsize=(5, 5), dpi=300)

    ltst =  hr * (180/(np.pi*15))
    day_ar = np.arange(0, numDays, 1)
    annual = np.argwhere(depthsAtMiddleOfLayers >= annual_depth)[0]
    print(annual)
    plt.scatter(day_ar, averageDiurnalAllTemps[1, :], s=2, c='r', label='Surface')
    plt.scatter(day_ar, averageDiurnalAllTemps[annual, :], s=2, c='g', label='Annual skin')
    plt.scatter(day_ar, averageDiurnalAllTemps[-1, :], s=2, c='b', label='bottom')
    mean= np.nanmean(averageDiurnalAllTemps[1, :])
    std = np.std(averageDiurnalAllTemps[1, :]) 
    #plt.hlines(mean, 0, 670, colors='k')
    plt.hlines(std/np.e+mean, 0, 670, colors='k', linestyle='dashed')
    plt.hlines(mean - std/np.e, 0, 670, colors='k', linestyle='dashed')
    plt.ylabel('temp (k)')
    plt.xlabel('days of year')
    plt.legend()
    plt.xlim((0, 670))
    #plt.ylim((120, 280))
    plt.show()
    
    return averageDiurnalAllTemps

# - Plots
def plot_layers(layer_depth, layer_number, layer_thickness):
    plt.rc("font", size=18, family="serif")
    plt.figure(figsize=(10,10), dpi=360)
    subsurface = np.linspace(0, np.nanmax(layer_depth)+5, (np.int(np.nanmax(layer_depth)+5)))
    image = np.ones((np.int(np.nanmax(layer_depth)+5), 20))*np.reshape(subsurface, ((np.int(np.nanmax(layer_depth)+5)), 1))
    layer_number = np.arange(1, 16, 1)
    plt.imshow(image, vmin=0, vmax=(np.int(np.nanmax(layer_depth)+5)), cmap='summer')
    plt.colorbar(label='Depth (cm)')
    plt.scatter(np.ones((15))*10, layer_depth, c='k', marker='o')
    for i in range(0, 15):
        plt.annotate('%2.0f'%(layer_number[i]), (10.5, layer_depth[i]+0.5))
    plt.hlines(layer_thickness/2+layer_depth, 0, 20, colors='k', linestyle='dashed')
    plt.hlines(layer_depth[0]-layer_thickness[0]/2, 0, 20, colors='r', linestyle='dashed')
    plt.ylim((-0.1, (np.int(np.nanmax(layer_depth)+3))))
    plt.xlim((0, 19))
    plt.gca().invert_yaxis()
    plt.show()


#avgT = plot_checks_CN(Temps, windupTemps, finaltemps,Tsurf, frostMasses, depthsAtMiddleOfLayers, diurnal_depth, annual_depth)
# write a save file