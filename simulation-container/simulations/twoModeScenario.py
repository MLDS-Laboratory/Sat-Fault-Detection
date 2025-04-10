import matplotlib.pyplot as plt
import numpy as np
from utilities.flightModes import FlightModes
import sys

#utilities and setup
from Basilisk.architecture import messaging
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import fswSetupRW
from Basilisk.utilities import simIncludeRW
from Basilisk.utilities import simIncludeGravBody

#RWs and navigation
from Basilisk.simulation import reactionWheelStateEffector
from Basilisk.simulation import spacecraft
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import inertial3D, attRefCorrection
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import rwMotorTorque
#general sim
from Basilisk.utilities import SimulationBaseClass

def simulate(plot):
    #a bunch of initializations
    simTaskName = "sim city"
    simProcessName = "mr. sim"

    satSim = SimulationBaseClass.SimBaseClass()
    timestep = 10
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
    satellite.hub.sigma_BNInit = rbk.euler3212MRP([0, 0, 0])
    #ang velocity of body frame relative to inertial expressed in body frame coords
    satellite.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]] 

    satSim.AddModelToTask(simTaskName, satellite, 15)

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
    oe.f = 90 * macros.D2R #85.3  * macros.D2R
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
    orbits = 2
    simTime = macros.sec2nano(orbits * period)

    #navigation module
    nav = simpleNav.SimpleNav()
    nav.ModelTag = "navigation"
    satSim.AddModelToTask(simTaskName, nav)
    nav.scStateInMsg.subscribeTo(satellite.scStateOutMsg)
    nav.sunStateInMsg.subscribeTo(spice.planetStateOutMsgs[1])

    samplingTime = unitTestSupport.samplingTime(simTime, simulationTimeStep,\
                                            simTime / simulationTimeStep)

    modes = ["sunPoint", "nadirPoint"]
    interval = 1./4.
    switches = [{"mode":modes[i % 2], "time":i * interval} for i in range(1, int(orbits / interval) + 1)]
    #switches = [{"mode":"nadirPoint", "time":0.25}, {"mode":"sunPoint", "time":0.5}, {"mode":"nadirPoint", "time":0.75}]
    mode = FlightModes(satSim, "sunPoint", switches, samplingTime, simTaskName, period, nav=nav, spice=spice)
    
    satSim.AddModelToTask(simTaskName, mode)

    #extraneous but it works
    correction = attRefCorrection.attRefCorrection()
    correction.ModelTag = "correction"
    correction.sigma_BcB = [0, 0, 0]
    correction.attRefInMsg.subscribeTo(mode.attRef)
    satSim.AddModelToTask(simTaskName, correction)

    #attitude error from reference
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    attError.attNavInMsg.subscribeTo(nav.attOutMsg)
    attError.attRefInMsg.subscribeTo(correction.attRefOutMsg)
    satSim.AddModelToTask(simTaskName, attError)


    #reaction wheels
    rwFactory = simIncludeRW.rwFactory()
    """
    RW Model:

    BalancedWheels: ideal RW, no imperfections or external effects
    SimpleJitter: includes forces and undesirable torques induced by RW as external effects
    JitterFullyCoupled: includes forces and desirable torques induced by RW as internal effects
    """
    rwModel = messaging.BalancedWheels

    """
    Available RW types (afaik) - Honeywell_HR16, Honeywell_HR14, Honeywell_HR12

    Mandatory arguments: RW type, spin axis (body frame)
    Select optional arguments: Omega (initial speed in RPM, Float), maxMomentum (max ang. momentum storage in Nms, Float), 
        useRWfriction (Bool), useMinTorque (Bool), useMaxTorque (saturation point, Bool), 
        rWB_B (RW COM relative to body frame in m, Float(3), label (String), u_max (max motor torque in N-m, Float)) 
    
    The Honeywells require defining maxMomentum, though each has unique acceptable values. 
    """
    maxMomentum = 100.
    defaults = [
         {"axis":[1, 0, 0], "u_max":1., "type":"Honeywell_HR16"},
         {"axis":[0, 1, 0], "u_max":1., "type":"Honeywell_HR16"},
         {"axis":[0, 0, 1], "u_max":1., "type":"Honeywell_HR16"},
    ]
    
    for i in range(len(defaults)):
         rwFactory.create(defaults[i]["type"], defaults[i]["axis"], maxMomentum=maxMomentum, RWModel=rwModel, u_max=defaults[i]["u_max"])

    numRW = rwFactory.getNumOfDevices()
    

    #adding RWs to s/c and sim
    rwEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(satellite.ModelTag, rwEffector, satellite)
    satSim.AddModelToTask(simTaskName, rwEffector, 2) #the 2 ensures it will get updated before the satellite - higher priority

    #control torque
    """note: because of the RW's higher execution priority, 
        this MUST be created before rwMotor in the code to ensure initialization of its cmdTorqueOutMsg"""
    control = mrpFeedback.mrpFeedback()
    
    control.ModelTag = "mrpFeedback"
    satSim.AddModelToTask(simTaskName, control)
    #parameters taken from scenarioAttitudeFeedbackRW
    control.K = 3.5
    control.Ki = -1 #negative turns integral control off
    control.P = 30.0
    control.integralLimit = 2. / control.Ki * 0.1

    #maps desired torque to RW, apparently
    rwMotor = rwMotorTorque.rwMotorTorque()
    rwMotor.ModelTag = "rwMotorTorque"
    satSim.AddModelToTask(simTaskName, rwMotor)

    #defines axes of control available to RWs (all 3, here)
    controlAxes_B = [
         1, 0, 0, 0, 1, 0, 0, 0, 1
    ]
    rwMotor.controlAxes_B = controlAxes_B

    #some final module subscriptions
    
    #apparently mrpFeedback needs config info for the satellite
    configData = messaging.VehicleConfigMsgPayload()
    configData.ISCPntB_B = inertia
    configDataMsg = messaging.VehicleConfigMsg()
    configDataMsg.write(configData)
    control.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    control.vehConfigInMsg.subscribeTo(configDataMsg)

    #connecting RWs to mrpFeedback and motors
    fswSetupRW.clearSetup()
    for _, rw in rwFactory.rwList.items():
            fswSetupRW.create(unitTestSupport.EigenVector3d2np(rw.gsHat_B), rw.Js, uMax=0.2)    
    rwParamMsg = fswSetupRW.writeConfigMessage()
    control.rwParamsInMsg.subscribeTo(rwParamMsg)
    control.rwSpeedsInMsg.subscribeTo(rwEffector.rwSpeedOutMsg)
    rwMotor.vehControlInMsg.subscribeTo(control.cmdTorqueOutMsg)
    rwMotor.rwParamsInMsg.subscribeTo(rwParamMsg)
    rwEffector.rwMotorCmdInMsg.subscribeTo(rwMotor.rwMotorTorqueOutMsg)


    """data collection"""

    #how often each logger samples

    
    #true satellite states (translational and rotational position/velocity)
    satLog = satellite.scStateOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, satLog)

    #planet states (main planet and sun)
    spiceLog = spice.planetStateOutMsgs[0].recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, spiceLog)

    #attitude error (from reference)
    errorLog = attError.attGuidOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, errorLog)

    #RW recorders
    rwMotorLog = rwMotor.rwMotorTorqueOutMsg.recorder(samplingTime)
    satSim.AddModelToTask(simTaskName, rwMotorLog)
    rwTorqueLog = []
    for i in range(numRW):
        rwTorqueLog.append(rwEffector.rwOutMsgs[i].recorder(samplingTime))
        satSim.AddModelToTask(simTaskName, rwTorqueLog[i])
        

    modeLog = mode.logger("currentMode")
    satSim.AddModelToTask(simTaskName, modeLog)

    satSim.SetProgressBar(True)

    satSim.InitializeSimulation()

    satSim.TotalSim.SingleStepProcesses()
    satSim.ConfigureStopTime(simTime)
    satSim.ExecuteSimulation()

    
    motorTorque = [rw.u_current for rw in rwTorqueLog]
    sigma  = np.array(satLog.sigma_BN)

    buffer = 30
    pseudofaults = []
    for i in range(len(modeLog.currentMode)):
        if modeLog.currentMode[i] != modeLog.currentMode[i-buffer]:
            pseudofaults.append(1)
        else:
            pseudofaults.append(0)
    
    if plot:
        plt.figure(1)
        
        for i in range(3):
            plt.plot(satLog.times() / period, sigma[:, i], label=rf"$\sigma_{i+1}$")
        plt.title("Inertial Orientation")
        plt.xlabel("Time [orbits]")
        plt.ylabel("Orientation (MRP)")
        plt.legend()

        #mrpFeedback Desired Torque Outputs
        plt.figure(2)
        for i in range(numRW):
            plt.plot(satLog.times() * macros.NANO2SEC / period, rwMotorLog.motorTorque[:, i], label=f'RW {i+1}')
        plt.title("mrpFeedback Desired Torques")
        plt.legend()
        plt.xlabel("Time [orbits]")
        plt.ylabel("Torque [N-m]")

        #RW motor actual torques
        plt.figure(3)
        for i in range(numRW):
            plt.plot(satLog.times() / period, motorTorque[i], label=f'RW {i+1}')
        plt.title("RW Motor - Torque Applied")
        plt.legend()
        plt.xlabel("Time [orbits]")
        plt.ylabel("Torque [N-m]")
        plt.ylim(-0.22, 0.22)
        
        plt.figure(4)
        plt.title("Pseudo-Faults - Mode Changes")
        plt.xlabel("Time [orbits]")
        plt.ylabel("Fault Status")
        plt.plot(satLog.times() / period, pseudofaults)
        
        plt.tight_layout()
        plt.show()
    return satLog.times() / period, sigma, rwMotorLog.motorTorque[:, :3], motorTorque, pseudofaults

if __name__ == "__main__":
    #plot?
    simulate(True)