#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:00:37 2021

@author: laferrierek

Runs a crank - nichoson 1d thermal model

Builds off notes+papers provided by A. Bramson

!!! Everything after CN only handles 1 year. need to run wihtin CN loop potentially. 

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
#import layer_maker   # what is this

import Orbital_Parameters as op

# ** Define things needed to run **    
loc = '/Users/laferrierek/Box Sync/Desktop/Mars_Troughs/Codes/Thermal_model/'

# Aesthetic
fs = (10, 8)
res = 350
plt.rc("font", size=18, family="serif")
plt.style.use('ggplot')

#%% Constants - pull the specific mars ones from a paramfille. 

class Profile:
  def reader(self,input_dict,*kwargs):
    for key in input_dict:
      try:
        setattr(self, key, input_dict[key])
      except:
        print("no such attribute, please consider add it at init")
        continue
    
    
with open(loc+"trough_parameters_2021.json",'r') as file:
    a=file.readlines()
Mars_Trough=Profile()
Mars_Trough.reader(json.loads(a[0]))


with open(loc+'trough_flat_parameters_1.json','r') as file:
    a = file.readlines()
flat_Mars_Trough=Profile()
flat_Mars_Trough.reader(json.loads(a[0]))


with open(loc+"bramson_layers.json",'r') as file:
    a=file.readlines()
bramson=Profile()
bramson.reader(json.loads(a[0]))
    

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

#%% get orbital solutions (solar flux, etc) for the trough provided. 
#%% 1. Read in values for trough, get orbitial solutions
ecc = Mars_Trough.eccentricity
obl = np.deg2rad(Mars_Trough.obl)
Lsp = np.deg2rad(Mars_Trough.Lsp)
dt_orb = Mars_Trough.dt

soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op(ecc, obl, Lsp, dt_orb, flat_Mars_Trough)
np.savetxt('flatVis_Saved_Trough1.txt', sf)

# Run - flat
T_regs, fwindupTemps, ffinaltemps, fTsurf, ffrostMasses = Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref)

# write a save file
np.savetxt('CN_flat_Trough1.txt', T_regs, delimiter=',', header='Temp (K) for flat')

#%% actual with slope
soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op(ecc, obl, Lsp, dt_orb, Mars_Trough)
#%% 2. Use the pre-made from solar_flux_trough.py to run for xMyrs
# read in min_mean,max, use max. 
time_frame = 100
time_op, sfmin, sfmean, sf = np.loadtxt(loc+'solar_flux_minmeanmax_%3.0f_kyr_Trough1.txt'%time_frame, skiprows=1, delimiter=',', unpack=True)


#%% Make layers from Bramson et al. 2019
k_input = bramson.k 
density_input = bramson.rho
c_input = bramson.cp
depths_input = bramson.depth
layerGrowth = bramson.Growth     
dailyLayers  = bramson.daily         
annualLayers = bramson.annual
dt = bramson.dt

layerPropertyVectors = np.vstack((k_input, density_input, c_input, depths_input)).T

modelLayers, cdt, layerIndices, diurnal_depth, annual_depth = layer_maker.layerProperties(layerPropertyVectors, MarsyearLength, Mars_Trough.Rotation_rate, layerGrowth,dailyLayers, annualLayers)

ktherm = modelLayers[:, 0]
rho = modelLayers[:, 1]
cp = modelLayers[:, 2]
kappa = modelLayers[:, 3]
dz = modelLayers[:, 4]
depthsAtMiddleOfLayers = modelLayers[:, 5]
timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
nLayers = np.size(dz)

numPropLayers = np.size(layerIndices)
if numPropLayers > 1:
    iceTableIndex = int(layerIndices[1])
else:
    iceTableIndex = 1
    

'''
#Bramson 2017 model
k_input = np.array([0.0459, 2.952])  
density_input = np.array([1626.18, 1615])  
c_input = np.array([837, 925]) 
depths_input = np.array([0, 0.5])
layerGrowth = 1.03                
dailyLayers  = 10                  
annualLayers = 6   
dt = 500

layerPropertyVectors = np.vstack((k_input, density_input, c_input, depths_input)).T

modelLayers, cdt, layerIndicies = layer_maker.layerProperties(layerPropertyVectors, MarsyearLength, Mars_Trough.Rotation_rate, layerGrowth,dailyLayers, annualLayers)

ktherm = modelLayers[:, 0]
rho = modelLayers[:, 1]
cp = modelLayers[:, 2]
kappa = modelLayers[:, 3]
dz = modelLayers[:, 4]
depthsAtMiddleOfLayers = modelLayers[:, 5]
timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
nLayers = np.size(dz)
'''
#%% Functions 

# - Phase diagrams
def clapyeron(triple_pressure, triple_T, R_bar, Lc, T):
    return triple_pressure*np.exp( (Lc/R_bar) * ((1/triple_T) - (1/T)))

def mean_gas_const(f1, m1, f2, m2, f3, m3):
    '''
    f's are the % volume of the atmosphere as a fraction
    m's are the molar mass of the molecule (sometimes a compound)
    '''
    mbar = f1*m1+f2*m2+f3*m3
    return 8.314/mbar

# Thermal - Bramson et al. 2017
def thermal_diffusivity_calc(k, rho, cp):
    return k/(rho*cp)

def thermal_skin_depth(k, rho, cp, P):
    thermal_diff = thermal_diffusivity_calc(k, rho, cp)
    skin = np.sqrt(4*np.pi*thermal_diff*P)
    return skin
    
def surface_enegry_balance(Solar, incidence_angle, albedo, dmco2_dt, dTemp_dz, IR_downwelling):
    T_surface = ((Solar*np.cos(incidence_angle)*(1-albedo)+Lc_CO2 * dmco2_dt +k*dTemp_dz*IR_downwelling)/(emissivity*sigma))**(1/4)
    return T_surface

def stability_depth(triple_pressure, triple_T, Lc, T, f1, m1, rho_atmo, layer_depth):
    R_bar = mean_gas_const(f1, m1, 0, 0, 0, 0)
    pressure_sublimation = clapyeron(triple_pressure, triple_T, R_bar, Lc, T)
    rho_vapor = pressure_sublimation/(R_bar*T)
    match = np.argwhere(rho_vapor >= rho_atmo)[0]
    excess_ice_depth = layer_depth[match[0]]
    return excess_ice_depth

# - Plots
def plot_layers(layer_depth, layer_number, layer_thickness):
    plt.rc("font", size=18, family="serif")
    plt.figure(figsize=(10,10), dpi=res)
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


# - Define Crank Nicholson
# missing inputs: 
def Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref):
    Temps = np.zeros((nLayers, nStepsInYear))
    Tsurf = np.zeros((nStepsInYear))
    lastTimestepTemps = np.zeros((nLayers, runTime))
    oldTemps = np.zeros((nLayers))
        
    frostMass = 0
    frostMasses = np.zeros((nStepsInYear))
    
    oldTemps[:] = Tref
    print('Initial Surface Temperature = %2.2f K' %Tref)
    timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
    
    # Define alpha values for matrix
    alpha_u = (2*ktherm*np.roll(ktherm,1)/(ktherm*np.roll(dz,1) + np.roll(ktherm, 1)*dz))*(dt/(rho*cp*dz))
    alpha_u[0] = 0
    alpha_d = (2*ktherm*np.roll(ktherm,-1)/(ktherm*np.roll(dz,-1) +np.roll(ktherm,-1)*dz))*(dt/(rho*cp*dz))
    alpha_d[-1] = 0
    
    #define diagnols, e for explicit, i for implicit
    dia_e = np.zeros((nLayers))
    dia_e = 1 - (1-f)*alpha_u - (1-f)*alpha_d
    dia_e[nLayers-1] = 1 - (1-f)*alpha_u[-1]
       
    # Boundary conditions
    boundary = np.zeros((nLayers))
    boundary[-1] = dt*Q/(rho[-1]*cp[-1]*dz[-1])
    
    dia_i = np.zeros((nLayers));
    dia_i = 1 + f*alpha_u + f*alpha_d;
    dia_i[nLayers-1] = 1+f*alpha_u[-1]
    
    B_implicit = np.array([(-f*np.roll(alpha_u,-1)), (dia_i), (-f*np.roll(alpha_d,1))])
    
    Amatrix_i = sparse.spdiags(B_implicit, [-1, 0, 1], nLayers, nLayers)
    A = sparse.csc_matrix(Amatrix_i) 
    A.eliminate_zeros()

    beta = ktherm[0]*dt/(rho[0]*cp[0]*dz[0]*dz[0])
            
    # Total fluxes
    Fin = (sf + visScattered*sky + flatVis*(1-sky))*(1-albedo) + (IRdown*sky + flatIR*(1-sky))*emissivity;
    Fin_frost = (sf + visScattered*sky +flatVis*(1-sky))*(1-albedoFrost)+(IRdown*sky + flatIR*(1-sky))*emisFrost
    Fin_i = (np.roll(sf, -1) + np.roll(visScattered, -1)*sky + np.roll(flatVis, -1)*(1-sky))*(1-albedo) + (np.roll(IRdown, -1)*sky + np.roll(flatIR, -1)*(1-sky))*emissivity
    
    # Calculate a and b's for surface temperature calculation
    aa = (dz[0]/(2*ktherm[0])*(Fin[0] + 3*emissivity*sigma*Tref**4)/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0]))))
    b = 1/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0])))
    Tsurf[0] = aa+b*Tref
    
    # Frost mass
    gamma_frost = (-1/Lc_CO2)*(2*ktherm[0]*(dt/dz[0]))
    theta_frost = (dt/Lc_CO2)*(2*ktherm[0]*CO2_FrostPoints/dz[0] - Fin_frost +emisFrost*sigma*CO2_FrostPoints**4)
    theta_frost_i = np.roll(theta_frost, -1)
    
    defrosting_decrease = np.exp(-depthsAtMiddleOfLayers/timestepSkinDepth)

    st = time.time()
    for yr in range(0, runTime):  # this is the run up time before actually starting.   
        for n in range(0, nStepsInYear):
            if frostMass == 0:
                # Have to recacluate each time  
                # Mssing: if ktherm is temperature dependent.
                b = 1/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0])))
                a_e = (dz[0]/(2*ktherm[0]))*(Fin[n] + 3*emissivity*sigma*Tref**4/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0]))))
                a_i = (dz[0]/(2*ktherm[0])*(Fin_i[n] + 3*emissivity*sigma*Tref**4)/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0]))))
                
                boundary[0] = 2*beta*((1-f)*a_e + f*a_i)
          
                # Explicit Part
                dia_e[0] = 1 - (1-f)*(alpha_d[0]+(2-2*b)*beta)
                T_e = (1-f)*alpha_u*np.roll(oldTemps,1) + (1-f)*alpha_d*np.roll(oldTemps,-1) + dia_e*oldTemps + boundary
        
                # Implicit Part - can't this be doen outside?
                A[0, 0] = 1 + f*(alpha_d[0]+(2-2*b)*beta)
                B = T_e
                
                # New temps
                Temps[:, n] = spsolve(A, B) 

                Tsurf[n] = a_i + b*Temps[0,n]  # Uses implicit a with new T calculation- instantanous balance
                frostMass = 0
                
                # uf surface Temp is below Frost Temp, make a frost layer. 
                if Tsurf[n] < Tfrost:
                    deltaTsurf = Tfrost - Tsurf[n]
                    frostMass = deltaTsurf*rho[0]*cp[0]*timestepSkinDepth/Lc_CO2
                    Temps[:, n] = Temps[:, n] +deltaTsurf*defrosting_decrease
                    Tsurf[n] = Tfrost
            
            elif frostMass > 0:
                # Frost affects surface temp
                boundary[0] = 2*beta*Tfrost
                
                # Explicit Part
                dia_e[0] = 1 - (1-f)*(alpha_d[0]+2*beta)
                T_e = (1-f)*alpha_u*np.roll(oldTemps,1) + (1-f)*alpha_d*np.roll(oldTemps,-1) + dia_e*oldTemps + boundary
        
                # Implicit Part - can't this be doen outside?
                A[0, 0] = 1 + f*(alpha_d[0]+2*beta)
                B = T_e

                Temps[:, n] = spsolve(A, B) #
                Tsurf[n] = Tfrost
                
                frostMass = frostMass + (1-f)*(gamma_frost*oldTemps[0] +theta_frost[n]) + f*(gamma_frost*Temps[0, n] + theta_frost_i[n]) 
                if frostMass < 0:
                    shiftedFrostMasses = np.roll(frostMasses, 1)
                    timeDefrosted = np.sqrt((0-frostMass)/shiftedFrostMasses[n] -frostMass)
                    deltaTsurf2 = -frostMass*Lc_CO2/(rho[0]*cp[0]*timestepSkinDepth*timeDefrosted)
                    Tsurf[n] = Tfrost+deltaTsurf2
                    Temps[:, n] = Temps[:, n]+deltaTsurf2*defrosting_decrease
                    frostMass = 0
            
            else:
                print('Frost mass is negative, issue', n)
            
            # record Temps, Tsurf, frost Mass
            oldTemps[:] = Temps[:, n]
            Tref = Tsurf[n]
            frostMasses[n] = frostMass
            
        lastTimestepTemps[:,yr] = Temps[:,n]  # To compare for convergence 
        print('Youre %2.0f / %2.0f'%(yr, runTime))
        if yr == windupTime:
            windupTemps = np.nanmean(Tsurf)
            oldTemps[:] = windupTemps
            print('Windup done, Setting all temps to %4.2f'%windupTemps)
        if yr == runTime-1:
            tempDiffs = lastTimestepTemps[:, runTime-1] -lastTimestepTemps[:, runTime-2]
            whichConverge = np.abs(tempDiffs) < convergeT
            if np.sum(whichConverge) == np.size(whichConverge):
                print('Converge to %3.7f'%(np.max(np.abs(tempDiffs))))
            else:
                print('Did not converge, increase run Time')
        if yr > 1:
            tempDiffs = lastTimestepTemps[:,yr] - lastTimestepTemps[:,yr-1]
            print('Still at least %3.7f K off' %np.max(np.abs(tempDiffs)))
    print('Took %2.2f'%(time.time()-st))
      
    return Temps, windupTemps, lastTimestepTemps, Tsurf, frostMasses

#%% Run - the old way
nStepsInYear = 118780
Temps, windupTemps, finaltemps, Tsurf, frostMasses = Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref)


#%% Run - as a loop

st = time.time()

for i in range(np.size(ecc)):
    soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op(ecc[i], obl[i], Lsp[i], dt_orb, Mars_Trough)
    Temps, windupTemps, finaltemps, Tsurf, frostMasses = Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref)
    if i %100 == 0:
        print('Step: %2.0f'%i)
        print(np.nanmax(Temps))
print('end', time.time()-st)

#%% Plot checks - i think temperature issues are rleated to the layer formation - not the thermal conductivity. 
def plot_checks_CN(Temps, windupTemps, finaltemps,Tsurf, frostMasses, depthAtMiddleOfLayers, daily_depth, annual_depth):
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
    fig = plt.figure(figsize=(5, 5), dpi=300)
    for i, c in zip(range(0, 669, days), color):
        im = plt.plot(averageDiurnalAllTemps[1:, i], depthsAtMiddleOfLayers, c=c) 
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
    daily = np.argwhere(depthsAtMiddleOfLayers >= daily_depth[0])[0]
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
    annual = np.argwhere(depthsAtMiddleOfLayers >= annual_depth[-1])[0]
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

avgT = plot_checks_CN(Temps, windupTemps, finaltemps,Tsurf, frostMasses, depthsAtMiddleOfLayers, diurnal_depth, annual_depth)
# write a save file
#%% this is not done.
#Calculate and print some output
#Temps, windupTemps, Tsurf, frostMasses = Crank_Nicholson(nLayers, nStepsInYear, windupTime, 15, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref)

# Find min, max and average temperatures at each depth over the last year
print('Minimum Surface Temp: %8.4f K'%np.min(Tsurf))
print('Maximum Surface Temp: %8.4f K'%np.max(Tsurf))
print('Mean Surface Temp: %8.4f K'%np.nanmean(Tsurf))

minT = np.min(Temps)
maxT =  np.max(Temps)
averageTemps = np.mean(Temps)

rho_CO2ice = 1600
equivalentCO2Thicknesses = frostMasses/rho_CO2ice;

print('Max frost thickness during the year: %5.4f m'%max(equivalentCO2Thicknesses))

print(r'Minimum Frost Mass: %8.4f kg/m^2' %min(frostMasses))
print(r'Maximum Frost Mass: %8.4f kg/m^2'%max(frostMasses))
print(r'Mean Frost Mass: %8.4f kg/m^2'%np.nanmean(frostMasses))

#%%
# diunral min tep
# create diurnal averages throughout the last year:# Create diurnal averages throughout the last year
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
minimumDiurnalSurfTemps = np.zeros((numDays))
T_reg_Diurnal = np.zeros((nLayers, numDays))

for n in np.arange(0, numDays):
    if n == numDays-1:
        averageDiurnalTemps[:,n] = np.mean(Temps[:, beginDayIndex[n]:np.size(Temps, 1)], 1)
        averageDiurnalSurfTemps[n] = np.nanmean(Temps[0, beginDayIndex[n]:np.size(Temps,1)])
        minimumDiurnalSurfTemps[n] = np.nanmin(Temps[0, beginDayIndex[n]:np.size(Temps,1)])
        T_reg_Diurnal[:, n] = np.mean(T_regs[:, beginDayIndex[n]:np.size(Temps,1)], 1)
    else:
        averageDiurnalTemps[:,n] = np.mean(Temps[:, beginDayIndex[n]:beginDayIndex[n+1]-1], 1)
        averageDiurnalSurfTemps[n] = np.nanmean(Temps[0, beginDayIndex[n]:beginDayIndex[n+1]-1])
        minimumDiurnalSurfTemps[n] = np.nanmin(Temps[0, beginDayIndex[n]:beginDayIndex[n+1]-1])
        T_reg_Diurnal[:, n] = np.mean(T_regs[:, beginDayIndex[n]:beginDayIndex[n+1]-1], 1)


averageDiurnalAllTemps = np.concatenate((averageDiurnalSurfTemps.reshape(1,numDays), averageDiurnalTemps))


#%% Calculate ice table/top of subsurface layer temperatures
numPropLayers = np.size(ktherm)
#iceTableIndex = 2
if numPropLayers > 1:
    
    iceTableTemps = (ktherm[iceTableIndex]*dz[iceTableIndex-1]*Temps[iceTableIndex,:] + ktherm[iceTableIndex-1]*dz[iceTableIndex]*Temps[iceTableIndex-1,:])/(ktherm[iceTableIndex]*dz[iceTableIndex-1] + kappa[iceTableIndex-1]*dz[iceTableIndex])
    iceTableTemps = iceTableTemps
    
    iceTable_Pv = 611 * np.exp( (-51058/8.31)*(1/iceTableTemps - 1/273.16) ) # compute vapor pressure at ice table
    iceTable_rhov = iceTable_Pv * (0.01801528 / 6.022140857e23) / (1.38064852e-23 * iceTableTemps); # compute vapor densities at ice table
    
    meanIceTable_Pv = np.nanmean(iceTable_Pv) # get mean for that year of ice table vapor pressures
    meanIceTable_rhov = np.nanmean(iceTable_rhov) # get mean for that year of ice table vapor densities
    meanIceTableT = np.nanmean(iceTableTemps) # mean temperature at the ice table over the year
    meansurfT = np.nanmean(Tsurf) # mean surface temperature

#%% Equation 4 from Bramson 2019
def retreat(D, iceVapRho, atmRho, z, d, rhoIce):
    # z is , d is dsut fraction
    return (D*(iceVapRho-atmRho))/(z*(1-d)*rhoIce)
Dreg = 3*10**(-4) # m2/s
rhoIce = 920 # kg/m3
waterVapDensity = 0.013 #kg/m3 (google)
dust = 3/100 #(Grima et al.)
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

















