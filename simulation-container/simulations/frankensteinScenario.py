import os
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import copy

#utilities?
from Basilisk.architecture import messaging
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import RigidBodyKinematics as rbk

#simulation tools

from Basilisk.simulation import coarseSunSensor

from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque
from Basilisk.utilities import simIncludeGravBody
from Basilisk.simulation import simpleNav
from Basilisk.simulation import eclipse

from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.fswAlgorithms import mrpFeedback

#general simulation initialization, i think
from Basilisk.utilities import SimulationBaseClass


def run(vfault, afault, sfault, plot):
    #a bunch of initializations
    simTaskName = "sim city"
    simProcessName = "mr. sim"

    satSim = SimulationBaseClass.SimBaseClass()

    dynamics = satSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(5)
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
    satellite.hub.omega_BN_BInit = [[0.0 * macros.D2R], [0.0], [2.00 * macros.D2R]] 

    satSim.AddModelToTask(simTaskName, satellite)


    #gravity stuff
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

    #navigation module - can be done via the satellite's own thing, but
    #i thought this would be good to use for future noise-addition purposes,
    #which the satellite doesn't support (i don't think)
    nav = simpleNav.SimpleNav()
    nav.ModelTag = "navigation"
    satSim.AddModelToTask(simTaskName, nav)
    nav.scStateInMsg.subscribeTo(satellite.scStateOutMsg)
    nav.sunStateInMsg.subscribeTo(spice.planetStateOutMsgs[1])

    #inertial reference attitude
    inertial = inertial3D.inertial3D()
    inertial.ModelTag = "inertial3D"
    satSim.AddModelToTask(simTaskName, inertial)
    inertial.sigma_R0N = [0., 1., 0.] #changed during sim
    
    #attitude error from reference
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    satSim.AddModelToTask(simTaskName, attError)

    #eclipses
    #optionally can be used if desired, just uncomment eclipse line in setup
    #i'm not using it though, to make seeing errors easier
    eclipses = eclipse.Eclipse()
    eclipses.sunInMsg.subscribeTo(spice.planetStateOutMsgs[1])
    eclipses.addSpacecraftToModel(satellite.scStateOutMsg)
    eclipses.addPlanetToModel(gravity.spiceObject.planetStateOutMsgs[0])
    satSim.AddModelToTask(simTaskName, eclipses)
    shadow = eclipses.eclipseOutMsgs[0].recorder()
    satSim.AddModelToTask(simTaskName, shadow)


    #CSS stuff
    def setup(CSS):
        CSS.fov = 90. * macros.D2R
        CSS.scaleFactor = 1.0
        CSS.maxOutput = 4.0
        CSS.minOutput = 0.0
        CSS.r_B = [0., 0., 0.]
        CSS.sunInMsg.subscribeTo(spice.planetStateOutMsgs[1])
        CSS.stateInMsg.subscribeTo(satellite.scStateOutMsg)
        #CSS.sunEclipseInMsg.subscribeTo(eclipses.eclipseOutMsgs[0])
        CSS.nHat_B = np.array([1.0, 0.0, 0.0])

    sensors = []
    loggers = []
    for i in range(6):
        sensors.append(coarseSunSensor.CoarseSunSensor())
        setup(sensors[i])
        #sensors[i].senNoiseStd = i/500
        sensors[i].senBias = 0#i/4.0
        satSim.AddModelToTask(simTaskName, sensors[i])
        loggers.append(sensors[i].cssDataOutMsg.recorder())
        satSim.AddModelToTask(simTaskName, loggers[i])
    sensors[1].nHat_B = np.array([-1.0, 0.0, 0.0])
    sensors[2].nHat_B = np.array([0.0, 1.0, 0.0])
    sensors[3].nHat_B = np.array([0.0, -1.0, 0.0])
    sensors[4].nHat_B = np.array([0.0, 0.0, 1.0])
    sensors[5].nHat_B = np.array([0.0, 0.0, -1.0])

        
   

    #control torque
    control = mrpFeedback.mrpFeedback()
    control.ModelTag = "mrpFeedback"
    satSim.AddModelToTask(simTaskName, control)
    control.K = 1
    control.Ki = 0.001 #negative turns integral control off
    control.P = 5

    #disturbance torque setup
    disturbance = extForceTorque.ExtForceTorque()
    disturbance.ModelTag = "externalDisturbance"
    satellite.addDynamicEffector(disturbance)
    satSim.AddModelToTask(simTaskName, disturbance)
    disturbance.cmdTorqueInMsg.subscribeTo(control.cmdTorqueOutMsg)
    disturbance.extTorquePntB_B = [[0], [0], [0]]

    #some final module subscriptions
    
    #apparently mrpFeedback needs config info for the satellite
    configData = messaging.VehicleConfigMsgPayload()
    configData.ISCPntB_B = inertia
    configDataMsg = messaging.VehicleConfigMsg()
    configDataMsg.write(configData)
    control.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    control.vehConfigInMsg.subscribeTo(configDataMsg)

    #idk why this is here and not above
    attError.attNavInMsg.subscribeTo(nav.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial.attRefOutMsg)
    
    #data collection

    #how often each logger samples
    samplingTime = unitTestSupport.samplingTime(simTime, simulationTimeStep,\
                                                simTime / simulationTimeStep)

    #true satellite states (translational and rotational position/velocity)
    satLog = satellite.scStateOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, satLog)

    #planet states (main planet and sun)
    spiceLog = spice.planetStateOutMsgs[0].recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, spiceLog)

    #reference orientation for control module
    refLog = inertial.attRefOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, refLog)

    #attitude error (from reference)
    errorLog = attError.attGuidOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, errorLog)

    #technically a module for adding noise to sensors, but eh i use it for sun pointing
    navLog = nav.attOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, navLog)

    #torque applied by control module
    torqueLog = control.cmdTorqueOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, torqueLog)

    #external disturbance torques
    disturbanceLog = disturbance.cmdTorqueInMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, disturbanceLog)

    #sun state; completely unnecessary
    sunLog = spice.planetStateOutMsgs[1].recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, sunLog)


    
    #set up the simulation
    satSim.InitializeSimulation()

    #needed to affect the velocity
    velRef = satellite.dynManager.getStateObject(satellite.hub.nameOfHubVelocity)

    #faults will start at this point
    faultTime = macros.sec2nano(0.25 * period)

    #for plotting purposes
    sensedSun = []
    disturbances = [] 
    
    #self-explanatory
    def fault_addition():
        rand = random.random()
        #random delta-v faults
        if vfault and rand < 0.001:
            rand = random.random() - 0.5
            velo = unitTestSupport.EigenVector3d2np(velRef.getState())
            velRef.setState([x + y for x, y in zip(velo, [1 * rand, 1 * rand, 0.25 * rand])])

        #random external torques
        if afault and rand < 0.001:
            factor = 5
            disturbance.extTorquePntB_B = [[(factor * random.random() - factor / 2.0) / 10 * 0],\
                                           [(factor * random.random() - factor / 2.0) / 10 ],\
                                           [(factor * random.random() - factor / 2.0) / 20 * 0]]
        disturbances.append(disturbance.extTorquePntB_B)

        #random CSS faults
        if sfault and rand < 0.01:
            sensorNum = int(rand * 10000) % 5
            srand = random.random() - 0.5
            if srand > 0.2:
                sensors[sensorNum].faultState = coarseSunSensor.CSSFAULT_STUCK_RAND
            else:
                sensors[sensorNum].faultState = coarseSunSensor.NOMINAL
    
    #update sun direction from CSS
    def sensorSunPos():
        weightedSum = []
        for i in sensors:
            weightedSum.append((i.sensedValue - i.senBias) * np.array(i.nHat_B))
        weightedSum = np.sum(weightedSum, axis=0)
        #print(f"Weighted Sum: {weightedSum}")
        if np.linalg.norm(weightedSum): 
            v_sun_B = weightedSum  / np.linalg.norm(weightedSum)
            C_BN = rbk.MRP2C(navLog.sigma_BN[-1])
            v_sun = C_BN.T @ v_sun_B
            theta = np.atan(v_sun[1] / v_sun[0])
            phi = np.atan(v_sun[2] / np.sqrt(v_sun[0]**2 + v_sun[1]**2)) * -1
            psi = 0
            euler = [theta, phi, psi]
            orientation = rbk.euler3212MRP(euler)
            sensedSun.append([euler[0][0], euler[1][0], euler[2]])
            #print(str(euler) + " YES WEIGHT")
            #print(f"Body: {v_sun_B}")
            #print(f"Inertial: {v_sun}")
            #print(f"Orientation: {orientation}")
            #print(f"Attitude: {navLog.sigma_BN[-1]}")
            #print(f"Error: {errorLog.sigma_BR[-1]}")
            inertial.sigma_R0N = orientation
        else:
            sensedSun.append(inertial.sigma_R0N)
        

    #actual execution loop
    while satSim.TotalSim.CurrentNanos < simTime:
        satSim.TotalSim.SingleStepProcesses()
        #msgs.append(stateReader)
        if satSim.TotalSim.CurrentNanos >= faultTime:
            fault_addition()
        else:
            disturbances.append([[0], [0], [0]])
        sensorSunPos()

    #collecting some of the data for plotting and returning

    sat_pos = satLog.r_BN_N
    earth_pos = spiceLog.PositionVector
    sat_velo = satLog.v_BN_N
    earth_velo = spiceLog.VelocityVector

    pos = np.array(sat_pos[:] - earth_pos[:])
    velo = np.array(sat_velo[:] - earth_velo[:])
    #^^ cause i decided to add the sun, which apparently fucks position vectors up

    sunPoint = np.array(navLog.vehSunPntBdy)

    CSSdata = []
    for i in loggers:
        CSSdata.append(i.OutputData)
    CSSdata = np.array(CSSdata)

    sigma  = np.array(satLog.sigma_BN)
    omega = np.array(satLog.omega_BN_B)
    
    #plotting
    if plot:
        
        plt.close("all")

        if vfault:
            #plot position relative to planet
            plt.figure(1)
            for i in range(3):
                plt.plot(timeAxis * macros.NANO2SEC / period, pos[:, i] / 1000,
                         color=unitTestSupport.getLineColor(i, 3))
            plt.title("Planet-relative Cartesian Position")
            plt.legend(["x", "y", "z"])
            plt.xlabel("Time [orbits]")
            plt.ylabel("Position [km]")

            #plot differences from theoretical orbit to numerical solution
            #intended to display impact of delta-v faults
            #note that in long enough sims and orbits far away enough from earth,
            #this will be significant even w/o faults because i've added the sun's influence
            #entirely copied from scenarioBasicOrbit
            plt.figure(2) 
            timeAxis = satLog.times()
            fig = plt.gcf()
            ax = fig.gca()
            ax.ticklabel_format(useOffset=False, style='plain')
            Deltar = np.empty((0, 3))
            E0 = orbitalMotion.f2E(oe.f, oe.e)
            M0 = orbitalMotion.E2M(E0, oe.e)
            n = np.sqrt(earth.mu/(oe.a*oe.a*oe.a))
            oe2 = copy(oe)
            for idx in range(0, len(pos)):
                M = M0 + n * timeAxis[idx] * macros.NANO2SEC
                Et = orbitalMotion.M2E(M, oe.e)
                oe2.f = orbitalMotion.E2f(Et, oe.e)
                rv, vv = orbitalMotion.elem2rv(earth.mu, oe2)
                Deltar = np.append(Deltar, [pos[idx] - rv], axis=0)
            for idx in range(3):
                plt.plot(timeAxis * macros.NANO2SEC / period, Deltar[:, idx] ,
                         color=unitTestSupport.getLineColor(idx, 3),
                         label=r'$\Delta r_{BN,' + str(idx+1) + '}$')
                plt.legend(loc='lower right')
                plt.xlabel('Time [orbits]')
                plt.ylabel('Trajectory Differences [m]')

        if afault:
            #external torque generations
            plt.figure(3)
            for i in range(len(disturbances)):
                pass#times.append(i)
            disturbances = np.array(disturbances)
            for i in range(3):
                plt.plot(satLog.times() * macros.NANO2SEC / period, disturbances[:, i], label = rf"$\tau_{i+1}$")
            plt.title("Disturbance Torques")
            plt.legend()
            plt.xlabel("Time [orbits]")
            plt.ylabel("Torque [N-m]")


        #pointing vector to the sun in the body frame
        plt.figure(4)
        timeAxis = navLog.times()
        for i in range(3):
            plt.plot(timeAxis * macros.NANO2SEC / period, sunPoint[:, i],
                     color=unitTestSupport.getLineColor(i, 3),
                     label=rf'$r_{i+1}$')
        plt.title("Sun Direction (Body)")
        plt.legend()
        plt.xlabel("Time [orbits]")
        plt.ylabel("Vector Component")

        #CSS sensor values, biases included
        plt.figure(5, figsize=(10,6))
        timeAxis = loggers[0].times()
        for i in range(len(CSSdata)):
            plt.subplot(2, 3, i+1)
            plt.plot(timeAxis * macros.NANO2SEC / period, CSSdata[i],
                     color=unitTestSupport.getLineColor(i, len(CSSdata)))
            plt.title(f'CSS$_{i+1}$')
            plt.ylim(-0.5, 1.5)
            plt.xlabel("Time [orbits]")
            plt.ylabel("Sensor Output")


        if sfault:
            #where the sensors collectively think the sun is (orientation vector)
            plt.figure(6)
            #not sure what i was thinking when i made this plot
            sensedSun = np.array(sensedSun)
            plt.plot(satLog.times () * macros.NANO2SEC / period, sensedSun[:, 0], label=rf"$\sigma_{1}$")
            plt.plot(satLog.times () * macros.NANO2SEC / period, sensedSun[:, 1], label=rf"$\sigma_{2}$")
            plt.plot(satLog.times () * macros.NANO2SEC / period, sensedSun[:, 2], label=rf"$\sigma_{3}$")
            plt.title("Sun Orientation via CSS Data (Inertial, 321 Euler)")
            plt.legend()
            plt.xlabel("Time [orbits]")
            plt.ylabel("Orientation (rad)")

        #satellite orientation relative to inertial
        plt.figure(7)
        for i in range(3):
            plt.plot(navLog.times() / period, sigma[:, i], label=rf"$\sigma_{i+1}$")
        plt.title("Inertial Orientation")
        plt.xlabel("Time [orbits]")
        plt.ylabel("Orientation (MRP)")
        plt.legend()

        plt.tight_layout()
        plt.show()

        
    return satLog.times(), pos, velo, sigma, omega, CSSdata, disturbances, sensedSun, sunPoint

if __name__ == "__main__":
    #vfault, afault, sfault, plot
    run(False, False, False, False)
