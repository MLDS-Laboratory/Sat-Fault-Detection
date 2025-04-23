import matplotlib.pyplot as plt
import numpy as np
import random
from utilities.CSSfault import CSSfault
import psycopg2
from psycopg2 import sql
import json  

#utilities?
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import RigidBodyKinematics as rbk

from Basilisk.utilities import simIncludeGravBody


#simulation tools

from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav

from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.simulation import coarseSunSensor


#general simulation initialization, i think
from Basilisk.utilities import SimulationBaseClass

class SensorSunPos(sysModel.SysModel):
        def __init__(self, sim, samplingTime, simTaskName, sensors, nav, inertial):
            super().__init__()
            self.sensors = sensors
            self.navLog = nav.attOutMsg.recorder(samplingTime)
            sim.AddModelToTask(simTaskName, self.navLog)
            self.inertial = inertial
            self.UpdateState(-1)
            
        def UpdateState(self, CurrentSimNanos):
            weightedSum = []
            for i in self.sensors:
                weightedSum.append((i.sensedValue - i.senBias) * np.array(i.nHat_B))
            weightedSum = np.sum(weightedSum, axis=0)
            if np.linalg.norm(weightedSum): 
                v_sun_B = weightedSum  / np.linalg.norm(weightedSum)
                C_BN = rbk.MRP2C(self.navLog.sigma_BN[-1])
                v_sun = C_BN.T @ v_sun_B
                theta = np.atan(v_sun[1] / v_sun[0])
                phi = np.atan(v_sun[2] / np.sqrt(v_sun[0]**2 + v_sun[1]**2)) * -1
                psi = 0
                euler = [theta, phi, psi]
                orientation = rbk.euler3212MRP(euler)
                self.sensedSun = [euler[0][0], euler[1][0], euler[2]]
                self.inertial.sigma_R0N = orientation
            else:
                self.sensedSun = self.inertial.sigma_R0N

def simulate(plot, CSSsun):
    #a bunch of initializations
    simTaskName = "sim city"
    simProcessName = "mr. sim"

    satSim = SimulationBaseClass.SimBaseClass()
    timestep = 5
    dynamics = satSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(timestep)
    dynamics.addTask(satSim.CreateNewTask(simTaskName, simulationTimeStep))

    satellite = spacecraft.Spacecraft()
    satellite.ModelTag = "oops"


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
    satellite.hub.sigma_BNInit = rbk.euler3212MRP([0, 1, 0])
    #ang velocity of body frame relative to inertial expressed in body frame coords
    satellite.hub.omega_BN_BInit = [[0.01/ np.sqrt(3)], [-0.01 / np.sqrt(3)], [0.01 / np.sqrt(3)]]

    satSim.AddModelToTask(simTaskName, satellite)

    #gravity stuff - 2BP
    gravity = simIncludeGravBody.gravBodyFactory()
    earth = gravity.createEarth()
    earth.isCentralBody = True
    gravity.createSun()

    #spice log initialization - date is just cause that's what the example used
    UTCInit = "2012 MAY 1 00:28:30.0"
    spice = gravity.createSpiceInterface(time=UTCInit, epochInMsg=True)
    satSim.AddModelToTask(simTaskName, spice)
    
    gravity.addBodiesTo(satellite)

    #orbits!
    oe = orbitalMotion.ClassicElements()

    r = 7000. * 1000
    oe.a = r
    oe.e = 0.000 #1
    oe.i = 0.0 #90.0 * macros.D2R

    oe.Omega = 0 #110 gets permanent illumination at i = 90
    oe.omega = 0 #90.0 * macros.D2R
    oe.f = 0 #85.3  * macros.D2R
    rN, vN = orbitalMotion.elem2rv(earth.mu, oe)
    oe = orbitalMotion.rv2elem(earth.mu, rN, vN) #yea idk why this exists

    #more satellite initializations
    #for some reason these are relative to planet, but the
    #satellite log's aren't 
    satellite.hub.r_CN_NInit = rN
    satellite.hub.v_CN_NInit = vN

    #sim time
    n = np.sqrt(earth.mu / oe.a**3)
    period = 2. * np.pi / n
    simTime = macros.sec2nano(1 * period)

    #navigation module
    nav = simpleNav.SimpleNav()
    nav.ModelTag = "navigation"
    satSim.AddModelToTask(simTaskName, nav)
    nav.scStateInMsg.subscribeTo(satellite.scStateOutMsg)
    nav.sunStateInMsg.subscribeTo(spice.planetStateOutMsgs[1])

    #inertial reference attitude
    inertial = inertial3D.inertial3D()
    inertial.ModelTag = "inertial3D"
    satSim.AddModelToTask(simTaskName, inertial)
    inertial.sigma_R0N = [0., 0.5, 0.]

   

    #attitude error from reference
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    attError.attNavInMsg.subscribeTo(nav.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial.attRefOutMsg)
    satSim.AddModelToTask(simTaskName, attError)

    control = mrpFeedback.mrpFeedback()
    
    control.ModelTag = "mrpFeedback"
    if CSSsun:
        satSim.AddModelToTask(simTaskName, control)
    #parameters taken from scenarioAttitudeFeedbackRW
    control.K = 3.5
    control.Ki = -1 #negative turns integral control off
    control.P = 30.0
    control.integralLimit = 2. / control.Ki * 0.1

    #external torque
    
    if CSSsun:
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


    #CSS stuff
    def setup(CSS):
        CSS.fov = 90. * macros.D2R
        CSS.scaleFactor = 1.0
        CSS.maxOutput = 4.0
        CSS.minOutput = 0.0
        CSS.sunInMsg.subscribeTo(spice.planetStateOutMsgs[1])
        CSS.stateInMsg.subscribeTo(satellite.scStateOutMsg)
        #CSS.sunEclipseInMsg.subscribeTo(eclipses.eclipseOutMsgs[0])
        CSS.nHat_B = np.array([1.0, 0.0, 0.0])

    sensors = []
    loggers = []
    numCSS = 18
    for i in range(numCSS):
        sensors.append(coarseSunSensor.CoarseSunSensor())
        setup(sensors[i])
        #sensors[i].senNoiseStd = i/500
        sensors[i].senBias = 0#i/4.0
        satSim.AddModelToTask(simTaskName, sensors[i])
        loggers.append(sensors[i].cssDataOutMsg.recorder())
        satSim.AddModelToTask(simTaskName, loggers[i])
    sensors[3].nHat_B = np.array([-1.0, 0.0, 0.0])
    sensors[4].nHat_B = np.array([-1.0, 0.0, 0.0])
    sensors[5].nHat_B = np.array([-1.0, 0.0, 0.0])
    sensors[6].nHat_B = np.array([0.0, 1.0, 0.0])
    sensors[7].nHat_B = np.array([0.0, 1.0, 0.0])
    sensors[8].nHat_B = np.array([0.0, 1.0, 0.0])
    sensors[9].nHat_B = np.array([0.0, -1.0, 0.0])
    sensors[10].nHat_B = np.array([0.0, -1.0, 0.0])
    sensors[11].nHat_B = np.array([0.0, -1.0, 0.0])
    sensors[12].nHat_B = np.array([0.0, 0.0, 1.0])
    sensors[13].nHat_B = np.array([0.0, 0.0, 1.0])
    sensors[14].nHat_B = np.array([0.0, 0.0, 1.0])
    sensors[15].nHat_B = np.array([0.0, 0.0, -1.0])
    sensors[16].nHat_B = np.array([0.0, 0.0, -1.0])
    sensors[17].nHat_B = np.array([0.0, 0.0, -1.0])
    for i in range(numCSS):
        sensors[i].r_B = sensors[i].nHat_B

    #how often each logger samples
    samplingTime = unitTestSupport.samplingTime(simTime, simulationTimeStep,\
                                                simTime / simulationTimeStep)

    cssfault = CSSfault(sensors, chance=0.0001)
    satSim.AddModelToTask(simTaskName, cssfault)

    senseSun = SensorSunPos(satSim, samplingTime, simTaskName, sensors, nav, inertial)
    if CSSsun:
        satSim.AddModelToTask(simTaskName, senseSun)

    """data collection"""

    sensedSunLog = senseSun.logger("sensedSun")
    satSim.AddModelToTask(simTaskName, sensedSunLog)

    faults = cssfault.logger("faultState")
    satSim.AddModelToTask(simTaskName, faults)
    
    #true satellite states (translational and rotational position/velocity)
    satLog = satellite.scStateOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, satLog)

    #technically a module for adding noise to sensors, but eh i use it for sun pointing
    navLog = nav.attOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, navLog)

    #planet states (main planet and sun)
    spiceLog = spice.planetStateOutMsgs[0].recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, spiceLog)

    #attitude error (from reference)
    errorLog = attError.attGuidOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, errorLog)

    sunPoint = np.array(navLog.vehSunPntBdy)

    CSSdata = []
    for i in loggers:
        CSSdata.append(i.OutputData)
    CSSdata = np.array(CSSdata)

    sigma  = np.array(satLog.sigma_BN)

    """SIMULATION"""

    satSim.SetProgressBar(True)
    satSim.InitializeSimulation()

    satSim.TotalSim.SingleStepProcesses()

    satSim.ConfigureStopTime(simTime)
    satSim.ExecuteSimulation()

    sensedSun = np.array(sensedSunLog.sensedSun)

    sunPoint = np.array(navLog.vehSunPntBdy)

    CSSdata = []
    for i in loggers:
        CSSdata.append(i.OutputData)
    CSSdata = np.array(CSSdata)

    sigma  = np.array(satLog.sigma_BN)

    faults = np.array(faults.faultState)

    """PLOTTING"""
    if plot:
        plt.close("all")

        #pointing vector to the sun in the body frame
        plt.figure(1)
        timeAxis = satLog.times()
        for i in range(3):
            plt.plot(timeAxis * macros.NANO2SEC / period, sunPoint[:, i],
                     color=unitTestSupport.getLineColor(i, 3),
                     label=rf'$r_{i+1}$')
        plt.title("Sun Direction (Body)")
        plt.legend()
        plt.xlabel("Time [orbits]")
        plt.ylabel("Vector Component")

        #CSS sensor values, biases included
        plt.figure(2, figsize=(20,len(CSSdata) / 2))
        colors = plt.cm.tab20.colors[:len(CSSdata)]
        timeAxis = loggers[0].times()
        for i in range(len(CSSdata)):
            plt.subplot(int(len(CSSdata) / 6), 6, i+1)
            plt.plot(timeAxis * macros.NANO2SEC / period, CSSdata[i],
                     color=colors[i])
            plt.title(f'$CSS_{{{i+1}}}$')
            plt.ylim(-0.5, 1.5)
            plt.xlabel("Time [orbits]")
            plt.ylabel("Sensor Output")

        if CSSsun:
            #where the sensors collectively think the sun is (orientation vector)
            #plt.figure(3)
            #not sure what i was thinking when i made this plot
            plt.figure(3)
            sensedSun = np.array(sensedSun)
            plt.plot(satLog.times () * macros.NANO2SEC / period, sensedSun[:, 0], label=rf"$\sigma_{1}$")
            plt.plot(satLog.times () * macros.NANO2SEC / period, sensedSun[:, 1], label=rf"$\sigma_{2}$")
            plt.plot(satLog.times () * macros.NANO2SEC / period, sensedSun[:, 2], label=rf"$\sigma_{3}$")
            plt.title("Sun Orientation via CSS Data (Inertial, 321 Euler)")
            plt.legend()
            plt.xlabel("Time [orbits]")
            plt.ylabel("Orientation (rad)")

        #satellite orientation relative to inertial
        plt.figure(4)
        for i in range(3):
            plt.plot(satLog.times() / period, sigma[:, i], label=rf"$\sigma_{i+1}$")
        plt.title("Inertial Orientation")
        plt.xlabel("Time [orbits]")
        plt.ylabel("Orientation (MRP)")
        plt.legend()

        plt.figure(5)
        for i in range(numCSS):
            plt.plot(satLog.times() / period, faults[:, i], label=rf"$CSS_{{{i+1}}}$")
        plt.title("CSS Sensors' Fault State")
        plt.xlabel("Time [orbits]")
        plt.ylabel("Fault State")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return np.array(satLog.times() / period), np.array(sigma), np.array(sensedSun), np.array(CSSdata), np.array(faults)

if __name__ == "__main__":
    simulate(False, False)
    