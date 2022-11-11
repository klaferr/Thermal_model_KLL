#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:07:33 2022

@author: laferrierek
"""
# ** Import libaries **
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import constants_MT as cT

# ** Define things needed to run **    
loc = '/Users/laferrierek/Box Sync/Desktop/Mars_Troughs/Project_MCMC/Thermal_model_KLL/'

## Constants - From files and basic.
# Surface conditions
#albedo = 0.25
#emissivity = 1.0
Q = 0.03 #30*10**(-3) # W/m2
Tref = 250

# Frost
#emisFrost = 0.95
#albedoFrost = 0.6
#Tfrost = cT.CO2_FrostPoints
windupTime = 8
convergeT = 0.05

runTime = 15
f = 0.5
dt = 500


#%%
def diurnal_wrap(nStepsInYear, nLayers, Temps, hr, lsWrapped):
    whereCrossOver360to0 = np.argwhere((lsWrapped[1:]-lsWrapped[0:-1]) <= 0)[0][0] + 1
    currenthr = hr[whereCrossOver360to0]
    beforehr = hr[whereCrossOver360to0-1]
    
    while ~((currenthr < 0) & (beforehr > 0)):
        whereCrossOver360to0 += -1
        currenthr = hr[whereCrossOver360to0]
        beforehr = hr[whereCrossOver360to0-1]
    
    Ls0 = []
    T0 = []
    hr0= []
    
    Ls0[:nStepsInYear-whereCrossOver360to0] = lsWrapped[whereCrossOver360to0:]
    Ls0[nStepsInYear - whereCrossOver360to0:nStepsInYear] = lsWrapped[:whereCrossOver360to0]
    T0[:nStepsInYear-whereCrossOver360to0] = Temps[whereCrossOver360to0:]
    T0[nStepsInYear - whereCrossOver360to0:nStepsInYear] = Temps[:whereCrossOver360to0]
    hr0[:nStepsInYear-whereCrossOver360to0] = hr[whereCrossOver360to0:]
    hr0[nStepsInYear - whereCrossOver360to0:nStepsInYear]= hr[:whereCrossOver360to0]
    
    return Ls0, T0, hr0

def Ls0_wrap(nStepsInYear, lsWrapped, v):
    whereCrossOver360to0 = np.argwhere((lsWrapped[1:]-lsWrapped[0:-1]) <= 0)[0][0] +1

    v0 = []
    
    v0[:nStepsInYear-whereCrossOver360to0] = v[whereCrossOver360to0:]
    v0[nStepsInYear - whereCrossOver360to0:nStepsInYear] = v[:whereCrossOver360to0]
    return v0

def diurnal_average(hr0, Tsurf, slope, lsWrapped, nStepsInYear):
    #print('broken, fix me!')
    # adjust for Ls0 affect too
    #hr0 = Ls0_wrap(nStepsInYear, lsWrapped, hr)
    
    # Temps, T_regs, nlayers
    days = round(cT.MarsyearLength_days) #np.round(cT.MarsyearLength_days).astype("int")
    beginDayIndex = np.zeros((days))*np.nan
    beginDayIndex[0] = 0
    dayIndex = 0
    for n in range(0, np.size(hr0)):
        if (hr0[n] < 0) & (hr0[n-1] > 0):
            beginDayIndex[dayIndex] = n
            dayIndex = dayIndex + 1
    beginDayIndex = beginDayIndex.astype('int')
    numDays = np.max(np.size(beginDayIndex)) # this feels useless

    averageDiurnalSurfTemps = np.zeros((numDays))
    minimumDiurnalSurfTemps = np.zeros((numDays))
    diurnalTsurfCurves = [] 
    
    for n in np.arange(0, numDays):
        if n == numDays-1:
            averageDiurnalSurfTemps[n] = np.mean(Tsurf[beginDayIndex[n]:np.size(Tsurf)])
            minimumDiurnalSurfTemps[n] = np.min(Tsurf[beginDayIndex[n]:np.size(Tsurf)])
            dTsurfstep = np.array((Tsurf[beginDayIndex[n]:np.size(Tsurf)]))
            diurnalTsurfCurves.append(dTsurfstep)
            

        else:
            averageDiurnalSurfTemps[n] = np.mean(Tsurf[beginDayIndex[n]:beginDayIndex[n+1]])
            minimumDiurnalSurfTemps[n] = np.min(Tsurf[beginDayIndex[n]:beginDayIndex[n+1]])
            
            dTsurfstep = np.array((Tsurf[beginDayIndex[n]:beginDayIndex[n+1]]))
            diurnalTsurfCurves.append(dTsurfstep)

    
    if slope == 0:
        REGminDiurnalSurfTemps = minimumDiurnalSurfTemps
        REGdiurnalTsurfCurves = diurnalTsurfCurves
        out = np.array([[REGminDiurnalSurfTemps], [REGdiurnalTsurfCurves]], dtype='object')
        
    else:
        SLOPEDaverageDiurnalSurfTemps = averageDiurnalSurfTemps
        out = SLOPEDaverageDiurnalSurfTemps
        
    return out 

def compute_a_e(Fin: np.ndarray, Tref: float, const_term, n: int, s):
    """ Compute the value of a_e per step of loop n 
    this is quicker if we only give it the row, nstead of using n to call the row"""    
    return const_term*(Fin[n] + 3*s.emis*cT.sigma*Tref**4)/(1+(4*s.emis*cT.sigma*const_term*Tref**3))

def compute_a_i(Fin_i: np.ndarray, Tref: float, const_term, n: int, s):
    return const_term*((Fin_i[n] + 3*s.emis*cT.sigma*Tref**4)/(1+(4*s.emis*cT.sigma*Tref**3*const_term)))

def compute_b(Tref: float, const_term, n: int, s):
    return 1/(1+(4*s.emis*cT.sigma*Tref**3*const_term))

def initialize_arrs(nLayers: int, nStepsInYear: int, runTime: int, Tref: int):
    Temps = np.zeros((nLayers, nStepsInYear))
    Tsurf = np.zeros((nStepsInYear))
    lastTimestepTemps = np.zeros((nLayers, runTime))
    oldTemps = np.zeros((nLayers))
    
    frostMasses = np.zeros((nStepsInYear))
    
    Temps[:, 0] = Tref 
    oldTemps[:] = Tref
    return Temps, Tsurf, lastTimestepTemps, oldTemps, frostMasses

def initialize_matrix_arrs(nLayers, ktherm, dz, rhoc):
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
    B_implicit = np.array([(-f*alpha_u_nr), (dia_i), (-f*alpha_d_r)]) #, dtype=np.float32)
    
    Amatrix_i = sparse.spdiags(B_implicit, [-1, 0, 1], nLayers, nLayers)
    A = sparse.csc_matrix(Amatrix_i) 
    
    return B_implicit, A

def compute_Fins(visScattered, maxIRdown, IRdown, sf, sky, s, albedo, flatIR, flatVis):
    Fin = (sf + visScattered*sky + flatVis*(1-sky))*(1-albedo) + (IRdown*sky + flatIR*(1-sky))*s.emis;
    Fin_frost = (sf + visScattered*sky +flatVis*(1-sky))*(1-s.albedoFrost)+(maxIRdown*sky + flatIR*(1-sky))*s.emisFrost
    
    #Fin = (sf + visScattered*sky + flatVis*(1-sky))*(1-albedo) + (IRdown*sky + flatIR*(1-sky))*emissivity;
    #Fin_frost = (sf + visScattered*sky +flatVis*(1-sky))*(1-albedoFrost)+(maxIRdown*sky + flatIR*(1-sky))*emisFrost
    #Fin_frost = (sf + visScattered.*sky + flatVis.*(1-sky)).*(1-s.albedoFrost) + (maxIRDown + flatIR.*(1-sky)).*s.emisFrost;

    sf_r =  np.concatenate((sf[1:], sf[:1]))
    visScattered_r = np.concatenate((visScattered[1:], visScattered[:1]))
    flatVis_r = np.concatenate((flatVis[1:], flatVis[:1]))
    IRdown_r = np.concatenate((IRdown[1:], IRdown[:1]))
    flatIR_r = np.concatenate((flatIR[1:], flatIR[:1]))
    
    Fin_i = (sf_r + visScattered_r*sky + flatVis_r*(1-sky))*(1-albedo) + (IRdown_r*sky + flatIR_r*(1-sky))*s.emis

    return Fin, Fin_frost, Fin_i

def initial_Tsurf(Fin: np.ndarray, Tref: int, emissivity, dz, ktherm, con):
    # Calculate a and b's for surface temperature calculation
    aa = ((con)*(Fin[0] + 3*emissivity*cT.sigma*Tref**4)/(1+(4*emissivity*cT.sigma*con*Tref**3)))
    b = 1/(1+(4*emissivity*cT.sigma*con*Tref**3))
    Tsurf_0 = aa+b*Tref
    return Tsurf_0

def initial_frostMass(Fin_frost, CO2_FrostPoints, ktherm, dz, s):
    gamma_frost = (-1/cT.Lc_CO2)*(2*ktherm[0]*(dt/dz[0]))
    theta_frost = (dt/cT.Lc_CO2)*(2*ktherm[0]*CO2_FrostPoints/dz[0] - Fin_frost + s.emisFrost*cT.sigma*CO2_FrostPoints**4)
    theta_frost_i = np.concatenate((theta_frost[1:], theta_frost[:1])) #! # np.roll(theta_frost, -1)
    return gamma_frost, theta_frost, theta_frost_i
    
##
def A_00_set(alpha_d_0, b, beta, frostMass):
    if frostMass == 0:
        A00 = 1 + f*(alpha_d_0+(2-2*b)*beta)
        # 1+ f*(alpha_d(1)+(2-2*b)*beta)
    elif frostMass > 0:
        A00 = 1+f*(alpha_d_0+2*beta)
    return A00

def matrix_setup(alpha_u, alpha_d, dia_e, b, beta, boundary, Temps, n):
    oldTemps_r = np.concatenate((Temps[-1:], Temps[:-1])) 
    oldTemps_nr =  np.concatenate((Temps[1:], Temps[:1])) 
    B = (1-f)*alpha_u*oldTemps_r + (1-f)*alpha_d*oldTemps_nr + dia_e*Temps + boundary
    return B

def boundary_0(Fin, Fin_i, Tsurf, Tfrost, const_term, n, beta, frostMass, s):
    a_i = compute_a_i(Fin_i, Tsurf, const_term, n, s)
    if frostMass == 0:
        a_e = compute_a_e(Fin, Tsurf, const_term, n, s)
        boundary0 = 2*beta*((1-f)*a_e + f*a_i)
    elif frostMass > 0:
        boundary0 = 2*beta*Tfrost  
    return boundary0, a_i

def matrix_inloop(alpha_u, alpha_d, dia_e, b, beta, boundary, Temps, frostMass, n):
    if frostMass == 0:
        dia_e[0] = 1-(1-f)*(alpha_d[0]+(2-2*b)*beta)
        # 1 - (1-f)*alpha_b2beta;
    elif frostMass >0:
        dia_e[0] = 1-(1-f)*(alpha_d[0]+2*beta)
    B = matrix_setup(alpha_u, alpha_d, dia_e, b, beta, boundary, Temps, n)
    A_00 = A_00_set(alpha_d[0], b, beta, frostMass)
    return A_00, B

def surf_inloop(frostMass, Temps_surf, Tfrost, b, a_i):
    if frostMass == 0:
        Tsurf_n = a_i + b*Temps_surf
    elif frostMass > 0:
        Tsurf_n = Tfrost
    return Tsurf_n

def handle_Frost(frostMass, Tfrost, Tsurf, Temps, oldTemps, n, frostMasses, Fin_frost, kappa, depth, dz, rhoc, CO2_FrostPoints, ktherm, s, timestepSkinDepth, defrosting_decrease):
    if frostMass == 0:
        #if round(Tsurf, 5) < round(Tfrost, 5):
        if Tsurf < Tfrost:
            Tsurf_n, Temps_n, frostMass = make_frost_layer(Tfrost, Tsurf, Temps, n, kappa, rhoc, timestepSkinDepth, defrosting_decrease)
        else:
            Tsurf_n = Tsurf
            Temps_n = Temps
            frostMass = 0
            
    elif frostMass > 0: 
        gamma_frost, theta_frost, theta_frost_i = initial_frostMass(Fin_frost, CO2_FrostPoints, ktherm, dz, s) # these match

        frostMass = frostMass + (1-f)*(gamma_frost*oldTemps[0] + theta_frost[n]) + f*(gamma_frost*Temps[0] + theta_frost_i[n]) 
        
        if frostMass < 0:
            Tsurf_n, Temps_n, frostMass = lose_frost_layer(Temps, frostMasses, frostMass, n, kappa, depth, rhoc, Tfrost, timestepSkinDepth, defrosting_decrease)
        else:
            Tsurf_n = Tsurf
            Temps_n = Temps
    else:
        print('error!!')
    return frostMass, Tsurf_n, Temps_n

##
def make_frost_layer(Tfrost, Tsurf_n, Temps_n, n, kappa, rhoc, timestepSkinDepth, defrosting_decrease):
    deltaTsurf = Tfrost - Tsurf_n
    frostMass = deltaTsurf*rhoc[0]*timestepSkinDepth/cT.Lc_CO2
    Temps_n = Temps_n +deltaTsurf*defrosting_decrease
    Tsurf_n = Tfrost
    return Tsurf_n, Temps_n, frostMass

def lose_frost_layer(Temps, frostMasses, frostMass, n, kappa, depth, rhoc, Tfrost, timestepSkinDepth, defrosting_decrease):

    defrosting_decrease = np.exp(-depth/timestepSkinDepth)
    
    shiftedFrostMasses = np.concatenate((frostMasses[-1:], frostMasses[:-1]))
    timeDefrosted = np.sqrt((0-frostMass)/(shiftedFrostMasses[n] - frostMass))
    deltaTsurf2 = -frostMass*cT.Lc_CO2/(rhoc[0]*timestepSkinDepth*timeDefrosted)
    Tsurf_n = Tfrost+deltaTsurf2
    Temps_n = Temps+deltaTsurf2*defrosting_decrease
    frostMass = 0
    return Tsurf_n, Temps_n, frostMass   

# - Main focus
def Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rhoc, kappa, albedo, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR, CO2_FrostPoints, s, p):
    Temps, Tsurf, lastTimestepTemps, oldTemps, frostMasses = initialize_arrs(nLayers = nLayers, nStepsInYear = nStepsInYear, runTime = runTime, Tref = Tref)
    frostMass = 0
    
    print('Initial Surface Temperature = %2.2f K' %Tref)
    const_term = (dz[0]/(2*ktherm[0])) # where does this belong?
    
    alpha_u, alpha_d, dia_e, dia_i, boundary = initialize_matrix_arrs(nLayers, ktherm, dz, rhoc)
    B_implicit, A = initialize_matrix(nLayers, alpha_u, alpha_d, dia_i)

    beta = ktherm[0]*dt/(rhoc[0]*dz[0]*dz[0])
    timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
    defrosting_decrease = np.exp(-depthsAtMiddleOfLayers/timestepSkinDepth)

            
    # Total fluxes
    # requires: soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, flat_Mars_Trough)
    # Keiffer - downwelling
    DownIRPolarNight = s.downwellingPerc *cT.sigma * CO2_FrostPoints**4
    maxIRdown = np.maximum(IRdown*sky, DownIRPolarNight)
    
    Fin, Fin_frost, Fin_i = compute_Fins(visScattered, maxIRdown, IRdown, sf, sky, p, albedo, flatIR, flatVis)

    Tsurf[0] = initial_Tsurf(Fin, Tref, p.emis, dz, ktherm, const_term)
    #Tsurf[-1] = Tref #initial_Tsurf(Fin, Tref, p.emis, dz, ktherm, const_term)

    # Frost mass
    
    #timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
    #defrosting_decrease = np.exp(-depth/timestepSkinDepth)

    for yr in range(0, runTime):  # this is the run up time before actually starting.   
        for n in range(0, nStepsInYear):

            Tfrost = CO2_FrostPoints[n]

            b = compute_b(Tref, const_term, n, p)

            boundary[0], a_i = boundary_0(Fin, Fin_i, Tref, Tfrost, const_term, n, beta, frostMass, p)

            A_00, B = matrix_inloop(alpha_u, alpha_d, dia_e, b, beta, boundary, oldTemps, frostMass, n)

            A[0, 0] = A_00
            
            Temps[:, n] = spsolve(A, B, permc_spec='NATURAL', use_umfpack=True) 
            Tsurf[n] = surf_inloop(frostMass, Temps[0, n], Tfrost, b, a_i)

            frostMass, Tsurf[n], Temps[:, n] = handle_Frost(frostMass, Tfrost, Tsurf[n], Temps[:, n], oldTemps, n, frostMasses, Fin_frost, kappa, depthsAtMiddleOfLayers, dz, rhoc, CO2_FrostPoints, ktherm, p, timestepSkinDepth, defrosting_decrease)
                
            # record Temps, Tsurf, frost Mass
            frostMasses[n] = frostMass
            
            # record Temps, Tsurf, frost Mass
            Tref = Tsurf[n]
            
            oldTemps = Temps[:, n]

        #print(np.mean(Tsurf), Tfrost)    
  
        lastTimestepTemps[:,yr] = Temps[:,n]  # To compare for convergence 
        print('Youre %2.0f / %2.0f'%(yr+1, runTime))
        
        if yr == windupTime:
            windupTemps = np.nanmean(Tsurf)
            #oldTemps[:] = windupTemps
            Temps[:, 0] = windupTemps
            Temps[:, -1] = windupTemps

            print('Windup done, Setting all temps to %4.2f'%windupTemps)
        elif yr == runTime:
            tempDiffs = lastTimestepTemps[:, runTime] -lastTimestepTemps[:, runTime-1]
            whichConverge = np.abs(tempDiffs) < convergeT
            if np.sum(whichConverge) == np.size(whichConverge):
                print('Converge to %3.7f'%(np.max(np.abs(tempDiffs))))
            else:
                print('Did not converge, increase run Time')
                print('Converge to %3.7f'%(np.max(np.abs(tempDiffs))))

        elif yr > 1:
            tempDiffs = lastTimestepTemps[:,yr] - lastTimestepTemps[:,yr-1]
            print('Still at least %3.7f K off' %np.max(np.abs(tempDiffs)))
      
    return Temps, windupTemps, lastTimestepTemps, Tsurf, frostMasses

def final_year(Tsurf, Temps, frostMasses):
    # Find min, max and average temperatures at each depth over the last year
    print('Minimum Surface Temp: %8.4f K'%np.min(Tsurf))
    print('Maximum Surface Temp: %8.4f K'%np.max(Tsurf))
    print('Mean Surface Temp: %8.4f K'%np.nanmean(Tsurf))
    
    minT = np.min(Temps)
    maxT = np.max(Temps)
    averageTemps = np.mean(Temps)
    
    print('Minimum subSurface Temp: %8.4f K'%minT)
    print('Maximum subSurface Temp: %8.4f K'%maxT)
    print('Mean subSurface Temp: %8.4f K'%averageTemps)
    
    rho_CO2ice = 1600
    equivalentCO2Thicknesses = frostMasses/rho_CO2ice;
    
    print('Max frost thickness during the year: %5.4f m'%max(equivalentCO2Thicknesses))
    
    print(r'Minimum Frost Mass: %8.4f kg/m^2' %min(frostMasses))
    print(r'Maximum Frost Mass: %8.4f kg/m^2'%max(frostMasses))
    print(r'Mean Frost Mass: %8.4f kg/m^2'%np.nanmean(frostMasses))
    return 

    
