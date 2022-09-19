#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:07:33 2022

@author: laferrierek
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

sigma = 5.670*10**(-8)      # Stefan-Boltzmann; W/m^2K^4
Lc_CO2 =  589.9*10**3               # Latent heat of CO2 frost; J/kg
CO2_FrostPoints = 150

nLayers = 300
nStepsInYear = 118640
windupTime = 2
runTime = 5

ktherm = np.linspace(1, 3, nLayers)
rhoc = np.linspace(100, 500, nLayers)
kappa = ktherm/rhoc
dz = np.linspace(0, 100, nLayers) +0.1
depth = dz +0.

sf = np.sin(np.linspace(np.pi/8, np.pi/2, nStepsInYear))
vis = np.ones((nStepsInYear))*0.1
IRdown = np.ones((nStepsInYear))*0.1
flatVis =  np.ones((nStepsInYear))*0.1
flatIR =  np.ones((nStepsInYear))*0.1

Tfrost = 150
Tref = 250
dt = 5
f=0.5
Q = 30
emissivity = 1
sky = 0.999
albedo = 0.7
albedoFrost = 0.9
emisFrost = 0.1
convergeT = 0.01

#%%
def compute_a_e(Fin: np.ndarray, Tsurf: np.ndarray, const_term, n: int):
    """ Compute the value of a_e per step of loop n 
    this is quicker if we only give it the row, nstead of using n to call the row"""    
    return const_term*(Fin[n] + 3*emissivity*sigma*Tsurf[n-1]**4/(1+(4*emissivity*sigma*Tsurf[n-1]**3*const_term)))

def compute_a_i(Fin_i: np.ndarray, Tsurf: np.ndarray, const_term, n: int):
    return const_term*((Fin_i[n] + 3*emissivity*sigma*Tsurf[n-1]**4)/(1+(4*emissivity*sigma*Tsurf[n-1]**3*const_term)))

def compute_b(Tsurf: np.ndarray, const_term, n: int):
    return 1/(1+(4*emissivity*sigma*Tsurf[n-1]**3*const_term))

def initialize_arrs(nLayers: int, nStepsInYear: int, runTime: int, Tref: int):
    Temps = np.zeros((nLayers, nStepsInYear))
    Tsurf = np.zeros((nStepsInYear))
    lastTimestepTemps = np.zeros((nLayers, runTime))
    oldTemps = np.zeros((nLayers))
    
    frostMasses = np.zeros((nStepsInYear))
    
    Temps[:, 0] = Tref 
    oldTemps[:] = Tref
    return Temps, Tsurf, lastTimestepTemps, oldTemps, frostMasses

def initialize_matrix_arrs(nLayers):
    # Define alpha values for matrix
    ktherm_r = np.concatenate((ktherm[-1:], ktherm[:-1]))
    ktherm_nr = np.concatenate((ktherm[1:], ktherm[:1]))
    dz_r = np.concatenate((dz[-1:], dz[:-1]))
    dz_nr = np.concatenate((dz[1:], dz[:1]))
    
    alpha_u = (2*ktherm*ktherm_r/(ktherm*dz_r + ktherm_r*dz))*(dt/(rhoc*dz))
    alpha_u[0] = 0
    alpha_d = (2*ktherm*ktherm_nr/(ktherm*dz_nr +ktherm_nr*dz))*(dt/(rhoc*dz))
    alpha_d[-1] = 0
    
    #define diagnols, e for explicit, i for implicit
    dia_e = np.zeros((nLayers))
    dia_e = 1 - (1-f)*alpha_u - (1-f)*alpha_d
    dia_e[nLayers-1] = 1 - (1-f)*alpha_u[-1]
       
    # Boundary conditions
    boundary = np.zeros((nLayers))
    boundary[-1] = dt*Q/(rhoc[-1]*dz[-1])
    
    dia_i = np.zeros((nLayers));
    dia_i = 1 + f*alpha_u + f*alpha_d;
    dia_i[nLayers-1] = 1+f*alpha_u[-1]
    return alpha_u, alpha_d, dia_e, dia_i, boundary

def initialize_matrix(nLayers, alpha_u, alpha_d, dia_i):    
    alpha_u_nr = np.concatenate((alpha_u[1:], alpha_u[:1]))
    alpha_d_r = np.concatenate((alpha_d[-1:], alpha_d[:-1]))
    B_implicit = np.array([(-f*alpha_u_nr), (dia_i), (-f*alpha_d_r)])
    
    Amatrix_i = sparse.spdiags(B_implicit, [-1, 0, 1], nLayers, nLayers)
    A = sparse.csc_matrix(Amatrix_i) 
    
    return B_implicit, A

def compute_Fins(visScattered):
    Fin = (sf + visScattered*sky + flatVis*(1-sky))*(1-albedo) + (IRdown*sky + flatIR*(1-sky))*emissivity;
    Fin_frost = (sf + visScattered*sky +flatVis*(1-sky))*(1-albedoFrost)+(IRdown*sky + flatIR*(1-sky))*emisFrost
    #Fin_i = (np.roll(sf, -1) + np.roll(visScattered, -1)*sky + np.roll(flatVis, -1)*(1-sky))*(1-albedo) + (np.roll(IRdown, -1)*sky + np.roll(flatIR, -1)*(1-sky))*emissivity
    
    sf_r =  np.concatenate((sf[1:], sf[:1]))
    visScattered_r = np.concatenate((visScattered[1:], visScattered[:1]))
    flatVis_r = np.concatenate((flatVis[1:], flatVis[:1]))
    IRdown_r = np.concatenate((IRdown[1:], IRdown[:1]))
    flatIR_r = np.concatenate((flatIR[1:], flatIR[:1]))
    
    Fin_i = (sf_r + visScattered_r*sky + flatVis_r*(1-sky))*(1-albedo) + (IRdown_r*sky + flatIR_r*(1-sky))*emissivity

    return Fin, Fin_frost, Fin_i

def initial_Tsurf(Fin: np.ndarray, Tref: int):
    # Calculate a and b's for surface temperature calculation
    aa = (dz[0]/(2*ktherm[0])*(Fin[0] + 3*emissivity*sigma*Tref**4)/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0]))))
    b = 1/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0])))
    Tsurf_0 = aa+b*Tref
    return Tsurf_0

def initial_frostMass(Fin_frost):
    gamma_frost = (-1/Lc_CO2)*(2*ktherm[0]*(dt/dz[0]))
    theta_frost = (dt/Lc_CO2)*(2*ktherm[0]*CO2_FrostPoints/dz[0] - Fin_frost +emisFrost*sigma*CO2_FrostPoints**4)
    theta_frost_i = np.concatenate((theta_frost[1:], theta_frost[:1])) #! # np.roll(theta_frost, -1)
    return gamma_frost, theta_frost, theta_frost_i
    
##
def A_00_set(alpha_d_0, b, beta, frostMass):
    if frostMass == 0:
        A00 = 1 + f*(alpha_d_0+(2+2*b)*beta)
    elif frostMass > 0:
        A00 = 1+f*(alpha_d_0+2*beta)
    return A00

def matrix_setup(alpha_u, alpha_d, dia_e, b, beta, boundary, Temps, n):
    oldTemps_r = np.concatenate((Temps[-1:, n-1], Temps[:-1, n-1])) 
    oldTemps_nr =  np.concatenate((Temps[1:, n-1], Temps[:1, n-1])) 
    B = (1-f)*alpha_u*oldTemps_r + (1-f)*alpha_d*oldTemps_nr + dia_e*Temps[:, n-1] + boundary
    return B

def boundary_0(Fin, Fin_i, Tsurf, Tfrost, const_term, n, beta, frostMass):
    a_i = compute_a_i(Fin_i, Tsurf, const_term, n)
    if frostMass == 0:
        a_e = compute_a_e(Fin, Tsurf, const_term, n)
        boundary0 = 2*beta*((1-f)*a_e + f*a_i)
    elif frostMass > 0:
        boundary0 = 2*beta*Tfrost  
    return boundary0, a_i

def matrix_inloop(alpha_u, alpha_d, dia_e, b, beta, boundary, Temps, frostMass, n):
    B = matrix_setup(alpha_u, alpha_d, dia_e, b, beta, boundary, Temps, n)
    A_00 = A_00_set(alpha_d[0], b, beta, frostMass)
    return A_00, B

def surf_inloop(frostMass, Temps_surf, Tfrost, b, a_i):
    if frostMass == 0:
        Tsurf_n = a_i + b*Temps_surf
    elif frostMass > 0:
        Tsurf_n = Tfrost
    return Tsurf_n

def handle_Frost(frostMass, Tfrost, Tsurf, Temps, n, frostMasses, Fin_frost):
    if frostMass == 0:
        if Tsurf[n] < Tfrost:
            Tsurf[n], Temps[:, n], frostMass = make_frost_layer(Tfrost, Tsurf, Temps, n)

    elif frostMass > 0: 
        gamma_frost, theta_frost, theta_frost_i = initial_frostMass(Fin_frost) # these match

        frostMass = frostMass + (1-f)*(gamma_frost*Temps[0, n-1] +theta_frost[n]) + f*(gamma_frost*Temps[0, n] + theta_frost_i[n]) 
        if frostMass < 0:
            Tsurf[n], Temps[:, n], frostMass = lose_frost_layer(Temps, frostMasses, frostMass, n)

    return frostMass, Tsurf, Temps
##
def make_frost_layer(Tfrost, Tsurf, Temps, n):
    timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
    defrosting_decrease = np.exp(-depth/timestepSkinDepth)
    
    deltaTsurf = Tfrost - Tsurf[n]
    frostMass = deltaTsurf*rhoc[0]*timestepSkinDepth/Lc_CO2
    Temps_n = Temps[:, n] +deltaTsurf*defrosting_decrease
    Tsurf_n = Tfrost
    return Tsurf_n, Temps_n, frostMass

def lose_frost_layer(Temps, frostMasses, frostMass, n):
    timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
    defrosting_decrease = np.exp(-depth/timestepSkinDepth)
    
    shiftedFrostMasses = np.concatenate((frostMasses[-1:], frostMasses[:-1]))
    #np.roll(frostMasses, 1)
    timeDefrosted = np.sqrt((0-frostMass)/shiftedFrostMasses[n] -frostMass)
    deltaTsurf2 = -frostMass*Lc_CO2/(rhoc[0]*timestepSkinDepth*timeDefrosted)
    Tsurf_n = Tfrost+deltaTsurf2
    Temps_n = Temps[:, n]+deltaTsurf2*defrosting_decrease
    frostMass = 0
    return Tsurf_n, Temps_n, frostMass    

# - Main focus
def Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rhoc, kappa, emissivity, Tfrost, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR):
    Temps, Tsurf, lastTimestepTemps, oldTemps, frostMasses = initialize_arrs(nLayers = nLayers, nStepsInYear = nStepsInYear, runTime = runTime, Tref = Tref)
    frostMass = 0
    
    print('Initial Surface Temperature = %2.2f K' %Tref)
    const_term = (dz[0]/(2*ktherm[0])) # where does this belong?
    
    alpha_u, alpha_d, dia_e, dia_i, boundary = initialize_matrix_arrs(nLayers)
    B_implicit, A = initialize_matrix(nLayers, alpha_u, alpha_d, dia_i)

    beta = ktherm[0]*dt/(rhoc[0]*dz[0]*dz[0])
            
    # Total fluxes
    # requires: soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, flat_Mars_Trough)
    Fin, Fin_frost, Fin_i = compute_Fins(visScattered)

    Tsurf[0] = initial_Tsurf(Fin, Tref)
    
    # Frost mass
    
    #timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
    #defrosting_decrease = np.exp(-depth/timestepSkinDepth)
    
    for yr in range(0, runTime):  # this is the run up time before actually starting.   
        for n in range(0, nStepsInYear):
            b = compute_b(Tsurf, const_term, n)

            boundary[0], a_i = boundary_0(Fin, Fin_i, Tsurf, Tfrost, const_term, n, beta, frostMass)

            A_00, B = matrix_inloop(alpha_u, alpha_d, dia_e, b, beta, boundary, Temps, frostMass, n)

            A[0, 0] = A_00
            
            Temps[:, n] = spsolve(A, B, permc_spec='NATURAL', use_umfpack=True) 
            Tsurf[n] = surf_inloop(frostMass, Temps[0, n], Tfrost, b, a_i)
            
            frostMass, Tsurf, Temps = handle_Frost(frostMass, Tfrost, Tsurf, Temps, n, frostMasses, Fin_frost)
                
            # record Temps, Tsurf, frost Mass
            frostMasses[n] = frostMass
            
            # record Temps, Tsurf, frost Mass
            Tref = Tsurf[n]
            
        lastTimestepTemps[:,yr] = Temps[:,n]  # To compare for convergence 
        print('Youre %2.0f / %2.0f'%(yr+1, runTime))
        
        if yr == windupTime:
            windupTemps = np.nanmean(Tsurf)
            oldTemps[:] = windupTemps
            print('Windup done, Setting all temps to %4.2f'%windupTemps)
        elif yr == runTime-1:
            tempDiffs = lastTimestepTemps[:, runTime-1] -lastTimestepTemps[:, runTime-2]
            whichConverge = np.abs(tempDiffs) < convergeT
            if np.sum(whichConverge) == np.size(whichConverge):
                print('Converge to %3.7f'%(np.max(np.abs(tempDiffs))))
            else:
                print('Did not converge, increase run Time')
        elif yr > 1:
            tempDiffs = lastTimestepTemps[:,yr] - lastTimestepTemps[:,yr-1]
            #print('Still at least %3.7f K off' %np.max(np.abs(tempDiffs)))
      
    return Temps, windupTemps, lastTimestepTemps, Tsurf, frostMasses
#%%
import constants_MT as cT
import time
albedo = 0.25
emissivity = 1.0
Q = 0.03 #30*10**(-3) # W/m2
Tref = 250

# Frost
emisFrost = 0.95
albedoFrost = 0.6
Tfrost = cT.CO2_FrostPoints
windupTime = 8
convergeT = 0.01

runTime = 15
f = 0.5
dt = 500

tic = time.time()
Crank_Nicholson(nLayers, nStepsInYear, 8, 15, ktherm, dz, dt, rhoc, kappa, emissivity, Tfrost, Tref, depth, sf, vis, sky, IRdown, flatVis, flatIR)
toc = time.time()
print(toc-tic)




#%%



    
    