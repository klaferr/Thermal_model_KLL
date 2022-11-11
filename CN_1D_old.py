#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:00:37 2021

@author: laferrierek

Crank-Nicholson 1D thermal

Builds off notes+papers provided by A. M. Bramson
Day/Year loop has been checked

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
albedo = 0.25
emissivity = 1.0
Q = 0.03 #30*10**(-3) # W/m2
Tref = 250

# Frost
emisFrost = 0.95
albedoFrost = 0.6
#Tfrost = cT.CO2_FrostPoints
windupTime = 8
convergeT = 0.05

runTime = 15
f = 0.5
dt = 500

#%% Define functions

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
def thermal_conductivity_T(k, T, R350):
    # for metals, k proportional to T*e_conductivity
    # from Hayne code 
    # k * (1+R350* T**3)
    # used grain sized parameter R350 ==  ~2.7/(350**3)
    
    # Flekkoy and Brodin (2022) - fractal
    # 16/3 * r_0 * sigma * T**3 * (a/r_o)**((3-D))
    # where r_0 = 1 micron
    # T = 250 K
    # D = 2.35
    # a = ? (semi major? 113 m?)
    # silicate rocks ~1 W/Km
    # of a solid:
        #phi = (r_0/a)**(3-D)~10^-5
        
    # Wiedemann-Franz law: k = L*sigma*T
    # L is hte Lorenz number. 
    
    #  Michell and de Pater (1994)
    # K  = Kc* (1+chi * (T/T_350)**3)
    # Kc is the phonon conductivity, chi is the rradiative conductivity to phonon conductivity at 350 K (Cuzzzi 1974)
    # varies with depth
    
    # vasvada et al. (2012)
    # k = kd - (kd-ks) * np.exp(-z/0.06) + chi*ks*(T/350)**3
    # where ks = 0.0006 W/m/K
    # kd is 0.007 W/m/K W/m/K
    # chi is 2.7
    
    # Lawson and Srivastava 2009
    # saturation value is = 1/3 * Cv*v*a, where Cv is specific heat
    # v is the velocity of sound
    # a is interatomic distance
    return  k*(1+R350*T**3)

def thermal_diffusivity_calc(k, rho, cp):
    return k/(rho*cp)

def thermal_skin_depth(k, rho, cp, P):
    #not used here
    thermal_diff = thermal_diffusivity_calc(k, rho, cp)
    skin = np.sqrt(4*np.pi*thermal_diff*P)
    return skin
    
def surface_enegry_balance(Solar, incidence_angle, albedo, dmco2_dt, dTemp_dz, IR_downwelling):
    # not used in this
    T_surface = ((Solar*np.cos(incidence_angle)*(1-albedo)+cT.Lc_CO2 * dmco2_dt +cT.k*dTemp_dz*IR_downwelling)/(emissivity*cT.sigma))**(1/4)
    return T_surface
 
def stability_depth(triple_pressure, triple_T, Lc, T, f1, m1, rho_atmo, layer_depth):
    # not used
    R_bar = mean_gas_const(f1, m1, 0, 0, 0, 0)
    pressure_sublimation = clapyeron(triple_pressure, triple_T, R_bar, Lc, T)
    rho_vapor = pressure_sublimation/(R_bar*T)
    match = np.argwhere(rho_vapor >= rho_atmo)[0]
    excess_ice_depth = layer_depth[match[0]]
    return excess_ice_depth

# - Dealing with CN results
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

def diurnal_average(hr, Tsurf, slope, lsWrapped, nStepsInYear):
    # adjust for Ls0 affect too
    hr0 = Ls0_wrap(nStepsInYear, lsWrapped, hr)
    
    # Temps, T_regs, nlayers
    days = round(cT.MarsyearLength_days) #np.round(cT.MarsyearLength_days).astype("int")
    beginDayIndex = np.zeros((days))*np.nan
    beginDayIndex[0] = 0
    dayIndex = 0
    for n in range(0, np.size(hr)):
        if (hr0[n] < 0) & (hr0[n-1] > 0):
            beginDayIndex[dayIndex] = n
            dayIndex = dayIndex + 1
    beginDayIndex = beginDayIndex.astype('int')
    numDays = np.max(np.size(beginDayIndex)) # this feels useless
    #averageDiurnalTemps = np.zeros((nLayers, numDays))

    averageDiurnalSurfTemps = np.zeros((numDays))
    minimumDiurnalSurfTemps = np.zeros((numDays))
    diurnalTsurfCurves = [] #np.zeros((numDays, 200))#[]
    #T_reg_Diurnal = np.zeros((nLayers, numDays))
    
    for n in np.arange(0, numDays):
        if n == numDays-1:
            averageDiurnalSurfTemps[n] = np.mean(Tsurf[beginDayIndex[n]:np.size(Tsurf)])
            minimumDiurnalSurfTemps[n] = np.min(Tsurf[beginDayIndex[n]:np.size(Tsurf)])
            dTsurfstep = np.array((Tsurf[beginDayIndex[n]:np.size(Tsurf)]))
            diurnalTsurfCurves.append(dTsurfstep)
            
            #diurnalTsurfCurves[n, :] = Tsurf[beginDayIndex[n]:np.size(Tsurf)]
    
            # if temps is shape
            #averageDiurnalTemps[:,n] = np.mean(Temps[:, beginDayIndex[n]:np.size(Temps, 1)], 1)
            #averageDiurnalSurfTemps[n] = np.nanmean(Temps[0, beginDayIndex[n]:np.size(Temps,1)])
            #minimumDiurnalSurfTemps[n] = np.nanmin(Temps[0, beginDayIndex[n]:np.size(Temps,1)])
            #T_reg_Diurnal[:, n] = np.mean(T_regs[:, beginDayIndex[n]:np.size(Temps,1)], 1)
        else:
            averageDiurnalSurfTemps[n] = np.mean(Tsurf[beginDayIndex[n]:beginDayIndex[n+1]-1])
            minimumDiurnalSurfTemps[n] = np.min(Tsurf[beginDayIndex[n]:beginDayIndex[n+1]-1])
            
            dTsurfstep = np.array((Tsurf[beginDayIndex[n]:beginDayIndex[n+1]-1]))
            diurnalTsurfCurves.append(dTsurfstep)
            
            #diurnalTsurfCurves[n, :] = Tsurf[beginDayIndex[n]:beginDayIndex[n+1]-1]
            
            #averageDiurnalTemps[:,n] = np.mean(Temps[:, beginDayIndex[n]:beginDayIndex[n+1]-1], 1)
            #averageDiurnalSurfTemps[n] = np.nanmean(Temps[0, beginDayIndex[n]:beginDayIndex[n+1]-1])
            #minimumDiurnalSurfTemps[n] = np.nanmin(Temps[0, beginDayIndex[n]:beginDayIndex[n+1]-1])
            #T_reg_Diurnal[:, n] = np.mean(T_regs[:, beginDayIndex[n]:beginDayIndex[n+1]-1], 1)
    
    #averageDiurnalAllTemps = np.concatenate((averageDiurnalSurfTemps.reshape(1,numDays), averageDiurnalTemps))
    
    if slope == 0:
        REGminDiurnalSurfTemps = minimumDiurnalSurfTemps
        REGdiurnalTsurfCurves = diurnalTsurfCurves
        out = np.array([[REGminDiurnalSurfTemps], [REGdiurnalTsurfCurves]], dtype='object')
        
    else:
        SLOPEDaverageDiurnalSurfTemps = averageDiurnalSurfTemps
        out = SLOPEDaverageDiurnalSurfTemps
        
    return out    
    #return REGminDiurnalSurfTemps, REGdiurnalTsurfCurves#, SLOPEDaverageDiurnalSurfTemps #averageDiurnalAllTemps,

def diurnal_wrap(nStepsInYear, nLayers, Temps, hr, lsWrapped):
    whereCrossOver360to0 = np.argwhere((lsWrapped[1:]-lsWrapped[0:-1]) <= 0)[0][0] +1
    
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
    

# - Main focus
def Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rhoc, kappa, emissivity, Tref, depthsAtMiddleOfLayers, sf, visScattered, sky, IRdown, flatVis, flatIR, CO2_FrostPoints, Mars_Trough):
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
    alpha_u = (2*ktherm*np.roll(ktherm,1)/(ktherm*np.roll(dz,1) + np.roll(ktherm, 1)*dz))*(dt/(rhoc*dz))
    alpha_u[0] = 0
    alpha_d = (2*ktherm*np.roll(ktherm,-1)/(ktherm*np.roll(dz,-1) +np.roll(ktherm,-1)*dz))*(dt/(rhoc*dz))
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
    
    B_implicit = np.array([(-f*np.roll(alpha_u,-1)), (dia_i), (-f*np.roll(alpha_d,1))])
    
    Amatrix_i = sparse.spdiags(B_implicit, [-1, 0, 1], nLayers, nLayers)
    A = sparse.csc_matrix(Amatrix_i) 
    A.eliminate_zeros()

    beta = ktherm[0]*dt/(rhoc[0]*dz[0]*dz[0])
    
    # Keiffer - downwelling
    DownIRPolarNight = Mars_Trough.downwellingPerc *cT.sigma * CO2_FrostPoints**4
    maxIRdown = np.max([IRdown*sky, DownIRPolarNight])
            
    # Total fluxes
    # requires: soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op.orbital_params(ecc, obl, Lsp, dt_orb, flat_Mars_Trough)
    
    Fin = (sf + visScattered*sky + flatVis*(1-sky))*(1-albedo) + (IRdown*sky + flatIR*(1-sky))*emissivity;
    Fin_frost = (sf + visScattered*sky +flatVis*(1-sky))*(1-albedoFrost)+(maxIRdown*sky + flatIR*(1-sky))*emisFrost
    #Fin_frost = (sf + visScattered.*sky + flatVis.*(1-sky)).*(1-s.albedoFrost) + (maxIRDown + flatIR.*(1-sky)).*s.emisFrost;

    Fin_i = (np.roll(sf, -1) + np.roll(visScattered, -1)*sky + np.roll(flatVis, -1)*(1-sky))*(1-albedo) + (np.roll(IRdown, -1)*sky + np.roll(flatIR, -1)*(1-sky))*emissivity
    
    # Calculate a and b's for surface temperature calculation
    aa = (dz[0]/(2*ktherm[0])*(Fin[0] + 3*emissivity*cT.sigma*Tref**4)/(1+(4*emissivity*cT.sigma*Tref**3*dz[0]/(2*ktherm[0]))))
    b = 1/(1+(4*emissivity*cT.sigma*Tref**3*dz[0]/(2*ktherm[0])))
    Tsurf[0] = aa+b*Tref
    
    # Frost mass
    gamma_frost = (-1/cT.Lc_CO2)*(2*ktherm[0]*(dt/dz[0]))
    theta_frost = (dt/cT.Lc_CO2)*(2*ktherm[0]*CO2_FrostPoints/dz[0] - Fin_frost +emisFrost*cT.sigma*CO2_FrostPoints**4)
    theta_frost_i = np.roll(theta_frost, -1)
    
    defrosting_decrease = np.exp(-depthsAtMiddleOfLayers/timestepSkinDepth)

    for yr in range(0, runTime):  # this is the run up time before actually starting.   
        for n in range(0, nStepsInYear):
            Tfrost = CO2_FrostPoints[n]
            if frostMass == 0:
                # Have to recacluate each time  
                # Mssing: if ktherm is temperature dependent.
                b = 1/(1+(4*emissivity*cT.sigma*Tref**3*dz[0]/(2*ktherm[0])))
                a_e = (dz[0]/(2*ktherm[0]))*(Fin[n] + 3*emissivity*cT.sigma*Tref**4/(1+(4*emissivity*cT.sigma*Tref**3*dz[0]/(2*ktherm[0]))))
                a_i = (dz[0]/(2*ktherm[0])*(Fin_i[n] + 3*emissivity*cT.sigma*Tref**4)/(1+(4*emissivity*cT.sigma*Tref**3*dz[0]/(2*ktherm[0]))))
                
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
                
                if n == 10:
                     print("pre-handle")
                     print("Tref: ", Tsurf[n])
                     print("Frost Mass: ", frostMass)
                     print("Tfrost:", Tfrost)           
                
                # uf surface Temp is below Frost Temp, make a frost layer. 
                if Tsurf[n] < Tfrost:
                    deltaTsurf = Tfrost - Tsurf[n]
                    frostMass = deltaTsurf*rhoc[0]*timestepSkinDepth/cT.Lc_CO2
                    Temps[:, n] = Temps[:, n] +deltaTsurf*defrosting_decrease
                    Tsurf[n] = Tfrost
                if n == 10:
                    print("post-handle")
                    print("Tref: ", Tsurf[n])
                    print("Frost Mass: ", frostMass)
                    print("Tfrost:", Tfrost)

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
                
                if n == 10:
                     print("pre-handle")
                     print("Tref: ", Tsurf[n])
                     print("Frost Mass: ", frostMass)
                     print("Tfrost:", Tfrost)     

                if frostMass < 0:
                    shiftedFrostMasses = np.roll(frostMasses, 1)
                    timeDefrosted = np.sqrt((0-frostMass)/shiftedFrostMasses[n] -frostMass)
                    deltaTsurf2 = -frostMass*cT.Lc_CO2/(rhoc[0]*timestepSkinDepth*timeDefrosted)
                    Tsurf[n] = Tfrost+deltaTsurf2
                    Temps[:, n] = Temps[:, n]+deltaTsurf2*defrosting_decrease
                    frostMass = 0
                if n == 10:
                    print("post-handle")
                    print("Tref: ", Tsurf[n])
                    print("Frost Mass: ", frostMass)
                    print("Tfrost:", Tfrost)

            
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
                #print('Converge to %3.7f'%(np.max(np.abs(tempDiffs))))
        if yr > 1:
            tempDiffs = lastTimestepTemps[:,yr] - lastTimestepTemps[:,yr-1]
            #print('Still at least %3.7f K off' %np.max(np.abs(tempDiffs)))
      
    return Temps, windupTemps, lastTimestepTemps, Tsurf, frostMasses






