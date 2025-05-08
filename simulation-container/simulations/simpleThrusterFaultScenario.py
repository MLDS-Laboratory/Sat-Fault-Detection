import matplotlib.pyplot as plt
import numpy as np
import random
from utilities.thrusterfault import ThrusterFault
import psycopg2
from psycopg2 import sql
import json  
import sys

#utilities?
from Basilisk.architecture import messaging
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import simIncludeThruster
from Basilisk.utilities import simIncludeGravBody



from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav
from Basilisk.simulation import thrusterDynamicEffector
from Basilisk.utilities import SimulationBaseClass
from Basilisk.fswAlgorithms import velocityPoint
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import attTrackingError


def simulate(plot):
    #a bunch of initializations
    simTaskName = "sim city"
    simProcessName = "mr. sim"

    satSim = SimulationBaseClass.SimBaseClass()
    timestep = 0.5
    dynamics = satSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(timestep)
    dynamics.addTask(satSim.CreateNewTask(simTaskName, simulationTimeStep))

    satellite = spacecraft.Spacecraft()
    satellite.ModelTag = "oops"

    satSim.SetProgressBar(True) #completely useless for this sim

    
    #satellite state definitions
    inertia = [1000., 0., 0.,
               0., 1000., 0., 
               0., 0., 1000.]

    #note that all angular orientations (here and all throughout) are in MRPs
    #angular velocities are in rad/s tho
    #satellite mass
    satellite.hub.mHub = 1000.0 
    #distance from body frame origin to COM
    satellite.hub.r_BcB_B= [[0.0], [0.0], [0.0]]
    #adding inertia to the objectsatellite
    satellite.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(inertia) 
    #orientation of body frame relative to inertial
    satellite.hub.sigma_BNInit = rbk.euler3212MRP([0, 0, 0])
    #ang velocity of body frame relative to inertial expressed in body frame coords
    satellite.hub.omega_BN_BInit = [[0.0 * macros.D2R], [0.0], [0.0]]#[2.00 * macros.D2R]] 

    satSim.AddModelToTask(simTaskName, satellite)


    #gravity stuff
    gravity = simIncludeGravBody.gravBodyFactory()
    earth = gravity.createEarth()
    earth.isCentralBody = True

    
    gravity.addBodiesTo(satellite)

    #orbits! (first, for initial orbit)
    oe1 = orbitalMotion.ClassicElements()

    r1 = 7000. * 1000
    oe1.a = r1
    oe1.e = 0.000 #1
    oe1.i = 0.0 #90.0 * macros.D2R

    oe1.Omega = 0 #110 gets permanent illumination at i = 90
    oe1.omega = 0 #90.0 * macros.D2R
    oe1.f = 0 #85.3  * macros.D2R
    rN1, vN1 = orbitalMotion.elem2rv(earth.mu, oe1)
    oe1 = orbitalMotion.rv2elem(earth.mu, rN1, vN1) #yea idk why this exists

    #more satellite initializations
    #for some reason these are relative to planet, but the
    #satellite log's aren't 
    satellite.hub.r_CN_NInit = rN1
    satellite.hub.v_CN_NInit = vN1

    #sim time
    n1 = np.sqrt(earth.mu / oe1.a**3)
    period1 = 2. * np.pi / n1
    maneuverTime1 = macros.sec2nano(period1)
    

    #now, define final circular orbit as needed
    r2 = 10000. * 1000
    a2 = r2

    n2 = np.sqrt(earth.mu / a2**3)
    period2 = 2. * np.pi / n2

    finalTime = macros.sec2nano(period2)

    #finally, define hohmann elliptical transfer orbit as needed
    rp = r1
    ra = r2
    at = (rp + ra) / 2.
    periodTransfer = 2. * np.pi * np.sqrt(at**3 / earth.mu)
    maneuverTime2 = macros.sec2nano(periodTransfer / 2.)

    #define the two impulse maneuvers that make up the hohmann transfer
    v1 = np.sqrt(earth.mu / r1)
    v2 = np.sqrt(earth.mu / r2)
    vtp = np.sqrt(earth.mu * (2 / rp - 1 / at))
    vta = np.sqrt(earth.mu * (2 / ra - 1 / at))
    dv1 = (vtp - v1)
    dv2 = (v2 - vta)
    totalTime = maneuverTime2 + maneuverTime1 + finalTime

    #navigation module - can be done via the satellite's own thing, but
    #i thought this would be good to use for future noise-addition purposes,
    #which the satellite doesn't support (i don't think)
    nav = simpleNav.SimpleNav()
    nav.ModelTag = "navigation"
    satSim.AddModelToTask(simTaskName, nav)
    nav.scStateInMsg.subscribeTo(satellite.scStateOutMsg)

    "Thruster Setup"
    #state effector
    thrusterSet = thrusterDynamicEffector.ThrusterDynamicEffector()
    satSim.AddModelToTask(simTaskName, thrusterSet)

    #creating the actual thrusters
    thFactory = simIncludeThruster.thrusterFactory()
    location = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    direction = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    maxThrust = 50000.
    for (pos, direct) in zip(location, direction):
        thFactory.create('Blank_Thruster', pos, direct, MaxThrust=maxThrust)
    thrTimeData = messaging.THRArrayOnTimeCmdMsgPayload()
    thrTimeData.OnTimeRequest = np.zeros(len(location))
    thrTimeMsg = messaging.THRArrayOnTimeCmdMsg().write(thrTimeData)
    thrusterSet.cmdsInMsg.subscribeTo(thrTimeMsg)
    

    #adding thrusters to satellite
    thrModelTag = "Thrusters"
    thFactory.addToSpacecraft(thrModelTag, thrusterSet, satellite)

    #calculate burn times
    #note that because this uses thrusters capable of finite thrust, this isn't actually an ideal hohmann transfer. 
    t1 = dv1 / (maxThrust / satellite.hub.mHub)
    t2 = dv2 / (maxThrust / satellite.hub.mHub)
    totalTime = totalTime + macros.sec2nano(t1) + macros.sec2nano(t2)


    # setup module that makes satellite point towards its velocity vector
    attGuidance = velocityPoint.velocityPoint()
    attGuidance.ModelTag = "velocityPoint"
    attGuidance.transNavInMsg.subscribeTo(nav.transOutMsg)
    attGuidance.mu = earth.mu
    satSim.AddModelToTask(simTaskName, attGuidance)

    #attitude error from reference
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    attError.attNavInMsg.subscribeTo(nav.attOutMsg)
    attError.attRefInMsg.subscribeTo(attGuidance.attRefOutMsg)
    satSim.AddModelToTask(simTaskName, attError)

    control = mrpFeedback.mrpFeedback()
    
    control.ModelTag = "mrpFeedback"
    satSim.AddModelToTask(simTaskName, control)
    #parameters taken from scenarioAttitudeFeedbackRW
    control.K = 3.5
    control.Ki = -1 #negative turns integral control off
    control.P = 30.0
    control.integralLimit = 2. / control.Ki * 0.1

    #external torque
    ext = extForceTorque.ExtForceTorque()
    satellite.addDynamicEffector(ext)
    satSim.AddModelToTask(simTaskName, ext)
    ext.cmdTorqueInMsg.subscribeTo(control.cmdTorqueOutMsg)

    #some final module subscriptions
    
    #apparently mrpFeedback needs config info for the satellite
    configData = messaging.VehicleConfigMsgPayload()
    configData.ISCPntB_B = inertia
    configDataMsg = messaging.VehicleConfigMsg()
    configDataMsg.write(configData)
    control.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    control.vehConfigInMsg.subscribeTo(configDataMsg)


    """data collection"""

    #how often each logger samples
    samplingTime = unitTestSupport.samplingTime(totalTime , simulationTimeStep,\
                                                totalTime / simulationTimeStep)

    #true satellite states (translational and rotational position/velocity)
    satLog = satellite.scStateOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, satLog)



    #technically a module for adding noise to sensors, but eh i use it for sun pointing
    navLog = nav.attOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, navLog)

    veloLog = nav.transOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, veloLog)

    #thruster recorders
    thrLog = []
    for i in range(thFactory.getNumOfDevices()):
        thrLog.append(thrusterSet.thrusterOutMsgs[i].recorder(samplingTime))
        satSim.AddModelToTask(simTaskName, thrLog[i])
    
    """simulation start"""
    satSim.SetProgressBar(True)
    satSim.InitializeSimulation()
    satSim.ConfigureStopTime(maneuverTime1)
    satSim.ExecuteSimulation()


    thrTimeData.OnTimeRequest = [0, t1, 0]
    thrTimeMsg.write(thrTimeData, time=maneuverTime1)

    satSim.ConfigureStopTime(maneuverTime1 + macros.sec2nano(t1))
    satSim.ExecuteSimulation()

    thrTimeData.OnTimeRequest = [0, 0, 0]
    thrTimeMsg.write(thrTimeData, time=maneuverTime1 + macros.sec2nano(t1))    

    satSim.ConfigureStopTime(maneuverTime1 + maneuverTime2)
    satSim.ExecuteSimulation()


    thrTimeData.OnTimeRequest = [0, t2, 0]
    thrTimeMsg.write(thrTimeData, time=maneuverTime1 + maneuverTime2)

    satSim.ConfigureStopTime(maneuverTime1 + maneuverTime2 + macros.sec2nano(t2))
    satSim.ExecuteSimulation()

    thrTimeData.OnTimeRequest = [0, 0, 0]
    thrTimeMsg.write(thrTimeData, time=maneuverTime1 + maneuverTime2 + macros.sec2nano(t2))

    satSim.ConfigureStopTime(maneuverTime1 + maneuverTime2 + finalTime)
    satSim.ExecuteSimulation()
    #collecting some of the data for plotting and returning

    sat_pos = satLog.r_BN_N

    sat_velo = satLog.v_BN_N


    pos = np.array(sat_pos[:])
    velo = np.array(sat_velo[:])
    #^^ cause i decided to add the sun, which centers the inertial frame of these loggers at the sun

    navVelo = veloLog.v_BN_N

    sigma  = np.array(satLog.sigma_BN)
    omega = np.array(satLog.omega_BN_B)

    thrustLog = [thruster.thrustForce for thruster in thrLog]
    oeEnd = orbitalMotion.rv2elem(earth.mu, pos[-1], velo[-1])
    if plot:
        plt.figure(1)
        for i in range(3):
            plt.plot(satLog.times() * macros.NANO2SEC / period1, pos[:, i] / 1000,
                        color=unitTestSupport.getLineColor(i, 3))
        plt.title("Planet-relative Cartesian Position")
        plt.legend(["x", "y", "z", 'vx', 'vy', 'vz'])
        plt.xlabel("Time [orbits]")
        plt.ylabel("Position [km]")

        plt.figure(2)
        for i in range(thFactory.getNumOfDevices()):
            plt.plot(satLog.times() * macros.NANO2SEC / period1, thrustLog[i], label=f'Thruster {i+1}')
        plt.legend()
        plt.title("Thruster Outputs")
        plt.xlabel("Time [orbits]")
        plt.ylabel("Thrust [N]")


        plt.show()
    return

if __name__ == "__main__":
    simulate(True)