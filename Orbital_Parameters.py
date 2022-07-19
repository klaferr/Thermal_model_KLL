# The goal of this file is to essentially recreate Dr. Ali Bramson's matlab function for 
# determining orbital parameters from the Lasker orbital solutions.

# Dependences
import numpy as np
import math 

# Constants
au = 1.4959787061e11    # meters
gc = 6.67259e-11        # Gravitational constal, unit?
sm  = 1.9891e30         # Solar Mass, kg
r2d = np.pi/180
nn=np.int(1e4)

# Set to be zero to start
flatVisSaved = 0 
flatIRSaved = 0 

#%% Functions
    
def surface_properties(slope, s, sloperad, lsrad):
    
    # If sloped and the parameter file is set to first calculate reradiation
    # from nearby terrain based on flat values, then load in those values.
    if s.Slope != 0 and s.Slope_rerad == 1:
        flatVis = flatVisSaved
        flatIR = flatIRSaved
        sky = np.cos(sloperad/2)**2
    else:
        flatVis = np.zeros((nStepsInYear,1))
        flatIR = np.zeros((nStepsInYear,1))
        sky = 1
    
    # Calculate frost point temperatures vs time for a given elevation
    atmPressTerms = np.array([7.97078, -0.539781, 0.468818, 0.368771, -0.392702, 0.0206071, -0.0224410, -0.0326866, -0.00261966, 0.0145776, -0.000184519])
    atmPress = atmPressTerms[0]
    for ii in range(0, 5):
        atmPress = atmPress + atmPressTerms[ii*2]*np.sin(ii*lsrad) + atmPressTerms[ii*2+1]*np.cos(ii*lsrad) # inside the sin/cos could also do 2pi*ii*ls/360 if ls was in degrees
    
    atmPress = atmPress*np.exp(-(s.Elevation+3627)/s.atmScaleHeight) # Scale pressure from Viking Landing site at -3627 m elevation
    CO2FrostPoints = 3148/(23.102 - np.log10(atmPress)) # Sublimation point of CO2 ice
    CO2FrostPoints = CO2FrostPoints * s.frostSwitch # Turns frost off by setting frost point to 0 if inputted in param file
    
    return CO2FrostPoints, atmPress, flatVis, flatIR, sky

    

def low_res_true_anomly(ecc, s):
    '''
    Determine the true anomly value across the whole orbit

    Parameters
    ----------
    nn : float
        number of steps (resolution of orbit)
    ecc : float
        eccentricity of orbit

    Returns
    -------
    t : np.array
        true anomly across whole orbit

    '''
    # Specific constants
    yearLength = 2*np.pi/np.sqrt(gc*sm/(s.Semimajor_Axis*au)**3)     # Length of Mars year in seconds using Kepler's 3rd law
    
    # Evenly spaced
    TA = ((np.arange(0, nn, 1)+0.5)*2*np.pi)/nn
    # Calculate eccentric anomalies
    EA = np.arccos((ecc+np.cos(TA))/(1+ecc*np.cos(TA)))
    # Mean anomalies
    MA = EA - ecc*np.sin(EA)
    # Time along Mars' orbital path (irregular spacing)
    t = MA/np.sqrt(gc*sm/(s.Semimajor_Axis*au)**3)
    t[TA > np.pi] = yearLength - t[TA > np.pi]
    return t, TA, yearLength

def high_res_true_anomly(dt, ecc, s):
    # this is not pythonic, nor do i understand what is happening
    
    t, TA, yearLength = low_res_true_anomly(ecc, s)
    
    # high res
    numTimesteps = math.ceil(yearLength/dt)
    
    timesBeginAndEnd = np.array([dt/2, numTimesteps*dt - (dt/2)])
    lmstBeginAndEnd = ((((timesBeginAndEnd / s.Rotation_rate * 2*np.pi) % (2*np.pi)) - np.pi)*180/(15*np.pi)) + 12  # Local mean solar time from 0 to 24 hours
    
    
    # Make number of timesteps connect back to midnight
    if lmstBeginAndEnd[-1] - lmstBeginAndEnd[0] > 12:
        numTimesteps = numTimesteps + round((((lmstBeginAndEnd[0]-lmstBeginAndEnd[-1] + 24) % 24)/24.0 * s.Rotation_rate/dt))

    if lmstBeginAndEnd[-1] - lmstBeginAndEnd[0] < 12:
        numTimesteps = numTimesteps - round((((lmstBeginAndEnd[-1]-lmstBeginAndEnd[0] + 24) % 24)/24.0 * s.Rotation_rate/dt))
    
    #calculations
    t2 = ((np.arange(0, numTimesteps-2, 1)) + 0.5)*dt
    nStepsInYear = np.size(t2)

    f = PchipInterpolator(t, TA, extrapolate=True)
    TA2 = f(t2)
    EA2 = np.arccos( (ecc+np.cos(TA2))/(1+ecc*np.cos(TA2)))
    
    return TA2, EA2, t2, nStepsInYear, yearLength


def orbital_params(ecc, obl, Lsp, dt, s):
    
    TA2, EA2, t2, nStepsInYear, yearLength = high_res_true_anomly(dt, ecc, s)
    sol_dist = s.Semimajor_Axis*(1-ecc*np.cos(EA2))
    lsrad = (TA2+Lsp) % (2*np.pi)     # Solar longitude in radians
    ls = np.rad2deg(TA2+Lsp)
    lsWrapped = np.rad2deg((TA2+Lsp) % (2*np.pi))
    
    sin_dec = np.sin(obl)*np.sin(lsrad) # nanhere
    cos_dec = np.sqrt(1 - sin_dec**2)
    
    sloperad = np.deg2rad(s.Slope)
    slopeaspectrad = np.deg2rad(s.Slope_Aspect)
    latrad = np.deg2rad(s.Latitude)
    
    lsNew = np.zeros((np.size(ls))) 
    lsOvershoot = ls[nStepsInYear-1] - ls[math.ceil(yearLength/dt)-1]
    for i in range(0, nStepsInYear):
        lsNew[i] = ls[i] - lsOvershoot*(i/nStepsInYear)
    lsWrapped = lsNew % 360
    
    # where the first time, moves from 360 back to zero
    #whereCrossOver360to0 = np.argwhere((lsWrapped[1:]-lsWrapped[0:-1]) <=0)[0]
    
    # Local mean true solar time in hour angle (-pi to pi)- cant be used to compare to ephemeris time because fixed rotation...calculated as if was a mean solar
    hr = ((t2/s.Rotation_rate * 2*np.pi) % (2*np.pi) ) - np.pi
    ltst = hr * (180/(np.pi*15)) +12
    

    cosi = np.sin(latrad) * sin_dec + np.cos(latrad)*cos_dec*np.cos(hr)
    
    cosi[cosi < -1.0] = -1.0
    cosi[cosi > 1.0] = 1.0    
    
    sini = np.sqrt(1-cosi**2)
    
    # Value that goes into calculating solar azimuth- want to make sure it
    # stays within [-1,1] or will get imaginary numbers
    cos_az = (sin_dec - np.sin(latrad)*cosi) / (np.cos(latrad)*sini);
    cos_az[cos_az > 1.0] = 1.0;
    cos_az[cos_az < -1.0] = -1.0;
    az  = np.arccos(cos_az);
    az[np.argwhere(ltst > 12)] = 2*np.pi - az[np.argwhere(ltst > 12)] # flip around since arccos degenerate
    
 
    cosi_slope = cosi*np.cos(sloperad) + sini*np.sin(sloperad)*np.cos(slopeaspectrad - az)
    cosi_slope[np.argwhere(cosi_slope < 0)] = 0  # Set any value less than 0 (shadowed) to 0
    cosi_slope[np.argwhere(cosi < 0)] = 0 
    
    # Solar Flux
    f1au = 1367                                         # Solar Flux at 1AU (W/m^2)
    sf = (f1au/(sol_dist**2)) * cosi_slope

    # No slope
    if s.Slope == 0:
        cosi[np.argwhere(cosi < 0)] = 0 
        sf = f1au/(sol_dist**2) * cosi
    
    annual_sf = sum(sf*dt)   # Total annual energy
    #print('Without atmosphere extinction')
    #print('Solar Flux Min = %8.4f, Max = %8.4f and Mean = %8.4f [W/m^2]' %(min(sf), max(sf), np.average(sf)))
    #print ('Total Annual Solar Flux = %.6e [W/m^2] ' %annual_sf)

    
    # Add in extinction due to atmospheric attenuation, as is in Schorghofer
    # and Edgett 2006 and Aharonson and Schorghofer 2006
    # this breaks because cosi_slope is an array. 
    #maxCoefAtmosAtt = np.array([np.nanmax((np.sin(np.pi/2 - np.arccos(cosi[i])), 0.04)) if s.Slope==0 else np.nanmax((np.sin(np.pi/2 - np.arccos(cosi_slope[i])), 0.04)) for i in range(np.size(cosi))])
    
    maxCoefAtmosAtt = np.zeros((np.size(cosi)))
    
    for i in range(np.size(cosi)):
        if s.Slope == 0:
            maxCoefAtmosAtt[i] = max((np.sin(np.pi/2 - np.arccos(cosi[i])), 0.04))
        else:
            maxCoefAtmosAtt[i] = max((np.sin(np.pi/2 - np.arccos(cosi_slope[i])), 0.04))
    
    atmosAtt = (1-s.scatteredVisPerc-s.downwellingPerc)**(1/maxCoefAtmosAtt)

    sfTOA = sf
    sf = sf*atmosAtt
    
    annual_sf = sum(sf*dt)   # Total annual energy
    #print('Solar Flux Min = %8.4f, Max = %8.4f and Mean = %8.4f [W/m^2]' %(min(sf), max(sf), np.average(sf)))
    #print ('Total Annual Solar Flux = %.6e [W/m^2] ' %annual_sf)
    
    # Daily noontime flux and the value of this to be used for downwelling IR
    # radiation. Gets modified for sloped cases below.
    sf_noon = f1au/(sol_dist**2) * ( np.sin(latrad)*sin_dec + np.cos(latrad)*cos_dec)
    IRdown = s.downwellingPerc * sf_noon
    
    # Scattered Visible Light at each timestep (will get 0 scattered vis light
    # at night so use the sf array).
    # This approximation assumes light scatters isotropically even though in
    # actuality more light is scattered near the disk of the sun in the sky
    # 1/2 factor in front for half scattered towards ground, half scattered
    # towards space
    #visScattered = 0.5 * s.scatteredVisPerc * sfTOA
    visScattered =  s.scatteredVisPerc * sf

    sky = np.cos(sloperad/2)**2
    if s.Slope != 0:
        sky = np.cos(np.deg2rad(s.Slope)/2)**2
        # Ali - load(s.flatSavedFile);
        flatVis = np.loadtxt('flatVis_Saved_Trough1.txt', delimiter=',')
        #flatVis = np.zeros((nStepsInYear))
        flatIR = np.zeros((nStepsInYear))
    else:
        flatVis = np.zeros((nStepsInYear))
        flatIR = np.zeros((nStepsInYear))
        sky = 1

    return sol_dist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR
