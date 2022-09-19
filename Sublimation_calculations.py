#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:48:23 2022

@author: laferrierek


Sublimation and such
"""

# ** Import libaries **
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius

# ** Import custom Libraries **
import constants_MT as cT 

# ** Define things needed to run **    
loc = '/Users/laferrierek/Box Sync/Desktop/Mars_Troughs/Project_MCMC/Thermal_model_KLL/'

# Constants - which of these are needed here?

#other 
A_drag_coeff = 0.002
u_wind = 2.5 # Dundas 2010

# Surface conditions
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

#%% functions
def clapyeron(triple_pressure, triple_T, R_bar, Lc, T):
    return triple_pressure*np.exp( (Lc/R_bar) * ((1/triple_T) - (1/T)))


def rho_molecules(MolarMassa, MolarMassb, atmP, esat, T_sub):
    
    return (((MolarMassa/cT.NA) * atmP) + ((MolarMassb-MolarMassa)/cT.NA) * esat) /(cT.k*T_sub)


def diffusion_coeff(T, triple_T, P_atm):
    return (1.387*10**(-5)) * ((T/triple_T)**(3/2)) * (10**5/P_atm)


def gas_density_diff(molarMassa, molarMassb, atmP, T_surf, T_atm, esat, ewatvap):
    # for mars, a is co2 and b is h2o
    num = (molarMassa * atmP * ((T_surf/T_atm) - 1)) + ((molarMassa - molarMassb) * (esat - ((T_surf/T_atm) * ewatvap)))
    denom = 0.5*((molarMassa * atmP * ((T_surf/T_atm) + 1)) - ((molarMassa - molarMassb) * ((T_surf/T_atm) * ewatvap + esat)))
    value = num/denom
    if value < 0:
        value = 0
    return value


def kin_visc(molarMass, T, atmP):
    # Sutherland formula
    refT = 293.15
    sut_con = 240
    return (1.48*10**(-5))*(cT.R_bar * T/(molarMass * atmP)) * ((sut_con+refT)/(sut_con+T)) * (T/refT)**(3/2)


def calc_atmPress(obliquity_rad):
    oblDeg = np.rad2deg(obliquity_rad)
    if 10 < oblDeg < 28:
        atmoPressure = np.exp(cT.SF_a1 + cT.SF_b1*(oblDeg - 28))
    elif 28 < oblDeg < 50:
        atmoPressure = np.exp(cT.SF_a1 + cT.SF_b1*(oblDeg-28) + cT.SF_c1*(oblDeg-28)**2)
    return atmoPressure


def water_scheme_AMB(Ls0, L04obl, atmoPressSF, HCH):
    # Bramson et al. (2019)
    #using phoenix lander, fit
    # uses SF12?
    
    # partial pressure H2O scheme - understand this?
    PPH2O_x = np.r_[0:61:1, 119.5:120.6:0.1, 150:361:1]
    PPH2O_y = 5*np.ones((np.size(PPH2O_x)))
    PPH2O_y[60:70] = 75
    PPH2O_x = PPH2O_x.T
    PPH2O_y = PPH2O_y.T
    PPH2O_f = ius(PPH2O_x, PPH2O_y, k=3)
    
    # values are off
    PPH2Otest_x = Ls0 
    PPH2Otest_y = PPH2O_f(PPH2Otest_x)
    PPH2Otest_y[PPH2Otest_y < 1*10**(-2)]= 0
    
    # Hc/H (height of water vapor condensation level, over scale seight H = RT/44g)
    HcH = 1
    
    # Partial Pressure of water at present day
    # from Schorghofer and Aharonson 2005
    # - for an isthermal atmsophere
    # h is the thickness of the precipitable layer
    # partial pressure of h20 at the surface is equal to:
    # gravity at the surface * ?h * density of the liquid * (44/18) * fracion
    
    # PPH2Otest_y *10**(-6) is a stand in for h?
    
    pho_water = 1000
    PP_H2O_present = cT.Mars_g * PPH2Otest_y * 10**(-6) * pho_water * (44/18) * (1/(1-np.exp(-HcH)))   
    
    obl_present = L04obl[-1]
    
    atmosPressSF_present = calc_atmPress(obl_present)    
    atmosPressSF_present = atmosPressSF_present * cT.atmofactor
    
    # Atmospheric factor: Vary relative to present day and Schorghofer and Forget 2012
    if cT.atmofactor == 0:
        PPH2O = PP_H2O_present * 0 # 0 at all times
    else:
        PPH2O = PP_H2O_present * (atmoPressSF/atmosPressSF_present)
    return PPH2O

def water_scheme_SF12():
    # Schorghofer et al. (2012)
    #
    return 

def water_scheme_CD07():
    # ? Dundas 2007
    return

def water_scheme_MS22(obl):
    # Mellon and Sizemore 2022
    
    # assumes that humidity scales with sublimation
    # returns atmospheric water vapor density
    # at 0 km elevation, to scale for height:
        # e^(-Z/H), h being atmospheric scale height (10.8 km assumed)
    if obl < 25.19:
        a = -1.83
        b = 16.6
        c = -28.7
        d = 9.97 # *10^10 #/cc
        expo = np.exp(a*obl**2 + b*obl + c)
        f = 1
    elif 25.19 <= obl <= 35:
        a = -15.3
        b = 117
        c = -212
        d = 6.47 # *10^10 #/cc
        expo = np.exp(a*obl**2 + b*obl+c)
        f = 1+3*((obl-25.19)/(45-25.19))
    elif 35 < obl:
        a = -4.78
        b = 40.9
        c = -74.9
        d = 10.5 # *10^10 #/cc
        expo = np.exp(a*obl**2 + b*obl + c)   
        f = 1+3*((obl-25.19)/(45-25.19))
    else:
        print('Warning, this scheme no longer applies!')
    d_over_f = d/f
    
    N0 = d_over_f*expo
    return N0


def sublimation_free_forced(Ls0, L04obl, atmofactor, atmoPressSF, SLOPEDaverageDiurnalSurfTemps, REGminDiurnalSurfTemps,REGdiurnalTsurfCurves):
    PPH2O = water_scheme_AMB(Ls0, L04obl, atmoPressSF, 1)
    counter = 1

    # Free and forced
    numDays_sublimation = np.size((SLOPEDaverageDiurnalSurfTemps))
    
    summedForcedDayTotals = np.zeros((numDays_sublimation))
    summedFreeDayTotals = np.zeros((numDays_sublimation))
    summedDayTotals = np.zeros((numDays_sublimation))
   
    # Loop over every day
    

    for day in range(0, numDays_sublimation-1):
        # Calculate surface sublimation  
        #print('Day: %1.0f'%day)
            
        T_surf_subl = SLOPEDaverageDiurnalSurfTemps[day]
        
        reg_Tmin = REGminDiurnalSurfTemps[day]
        reg_all_Tsurfs = np.array(REGdiurnalTsurfCurves[day])
             
        numTimesteps_subl = np.size((reg_all_Tsurfs))

        diurnal_m_free_thickness = np.zeros((numTimesteps_subl))
        diurnal_m_forced_thickness = np.zeros((numTimesteps_subl))
        
        for timestep in range(0, numTimesteps_subl):
            reg_Tsurf = np.float32(reg_all_Tsurfs[timestep])
            
            # Compute Temp
            T_min = reg_Tmin # most recent diurnal minimum surface temp
            T_reg = reg_Tsurf # current surface temp
            b_exp = 0.2     # viking
            T_atm_subl = T_min**b_exp * T_reg**(1-b_exp) # near sirface atmospheric temperature
            T_bl = (T_surf_subl + T_atm_subl)/2 # boundary layer tempearture

            # Sensible heat for forced convection, (Eq 5 Dundas et al. 2010)
            T_subl = T_bl
            # should use: cT.Lc_H2O, Bramson 2019 used: 51058
            e_sat = clapyeron(cT.triple_P_h2o, cT.triple_Temp_h2o, cT.R_bar, 51058, T_surf_subl)    # Saturation vapor pressure given temo
            
            #print("scheme is AMB19")
            e_watvap = PPH2O[counter] # this s where water scheme matters!!
            m_forced = (cT.mass_molecule_h2o/(cT.k*T_subl))*A_drag_coeff * u_wind * (e_sat - e_watvap)
            diurnal_m_forced_thickness[timestep] = (m_forced*dt/cT.density_ice)
            
            # Sensible heat for free convection (Eq 6 from Dundas et al. 2010)
            rho_surf=  rho_molecules(cT.m_gas_co2, cT.m_gas_h2o, cT.P_atm, e_sat, T_surf_subl)
            rho_atm = rho_molecules(cT.m_gas_co2, cT.m_gas_h2o, cT.P_atm, e_watvap, T_atm_subl)
            rho_ave = (rho_atm + rho_surf )/2
            
            # rho_bl_H2O = rho_molecules(0, cT.m_gas_h2o, 0, e_sat, T_bl)
            rho_atm_H2O = rho_molecules(0, cT.m_gas_h2o, 0, e_watvap, T_atm_subl)
            rho_surf_H2O = rho_molecules(0, cT.m_gas_h2o, 0, e_sat, T_surf_subl)
            # difference between atmospheric and surface gas water mass
            deltaEta = (rho_surf_H2O - rho_atm_H2O) /rho_atm
            
            # diffusion coeefient for H2O in Co2 at T_bl
            Diff_H2OinCO2 = diffusion_coeff(T_bl, cT.triple_Temp_h2o, cT.P_atm)
            
            # Atmospheric and surface gas density difference divided by reference density
            delta_rho_over_rho = gas_density_diff(cT.m_gas_co2, cT.m_gas_h2o, cT.P_atm, T_surf_subl, T_atm_subl, e_sat, e_watvap)
            
            # Kinematic viscosity
            nu = kin_visc(cT.m_gas_co2, T_bl, cT.P_atm)    
           
            # free
            x = (delta_rho_over_rho*(cT.Mars_g/nu**2)*(nu/Diff_H2OinCO2)) # check sign
            m_free = 0.14*deltaEta * rho_ave * Diff_H2OinCO2 * x**(1/3) # kg/m2?
            diurnal_m_free_thickness[timestep] = m_free * dt / cT.density_ice
            #print(m_free, dt, cT.density_ice)
        
        summedForcedDayTotals[day] = sum(diurnal_m_forced_thickness)
        summedFreeDayTotals[day] = sum(diurnal_m_free_thickness)
        summedDayTotals[day]= sum(diurnal_m_free_thickness+ diurnal_m_forced_thickness)
        
    TotalFree = sum(summedFreeDayTotals)
    TotalForced = sum(summedForcedDayTotals)
    TotalSublimation = sum(summedDayTotals)
    
    print('Final forced convection sublimation is %10.9f mm \n'%(TotalForced*1000))
    print('Final free convection sublimation is %10.9f mm \n' %(TotalFree*1000))
    print('Final total sublimation is %3.3f mm \n'%(TotalSublimation*1000))
    return TotalFree, TotalForced, TotalSublimation
#%% free and forced convection
'''
things that are equivalent to matlab fit(x), but not fit
from scipy.interpolate import InterpolatedUnivariateSpline as ius
c = ius(PPH2O_x, PPH2O_y, k=3)
PPH2O_f = c(PPH2O_x)

from scipy.interpolate import CubicSpline
test = CubicSpline(PPH2O_x, PPH2O_y, bc_type="clamped")

from scipy import interpolate
tck = interpolate.splrep(PPH2O_x, PPH2O_y, s=0)
c_interp = interpolate.splev(freq, tck, der=0)

PPH2O_f = constants_MT.smooth(PPH2O_y, 5)

from scipy import interpolate
PPH2O_f = interpolate.CubicHermiteSpline(PPH2O_x, PPH2O_y)

'''



'''
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


'''
