from Basilisk.utilities import RigidBodyKinematics as rbk, macros
from Basilisk.architecture import sysModel, messaging
import sys
import numpy as np

"""
This module handles switching between different pointing flight modes. Currently, 
sun-pointing and modified nadir-pointing have been implemented. It is not autonomous, as
switching times and modes have to be predefined and specific criteria can't yet be specified
to cause switches. 

It is a Basilisk module, so it will run as part of the simulation provided it is added to the task. 
The module outputs one message - attRef - that specifies a reference attitude determined by the mode. 
"""
class PointingModes(sysModel.SysModel):
    """
    Requires, in order:
      - the simulation in which this is being run
      - the starting mode
      - a list telling the module which modes to switch to at which timesteps (as a fraction of the orbital period):
        - the list must be a list of dicts, with the fields "time" and "mode"
      - the sampling time of all data loggers in the greater simulation
      - the task name of the task in the greater sim in which this is to be embedded
      - the nominal orbital period - or the factor by which the switching times relate to the simulation's nanosecond times
    
    Optional parameters, required for certain modes (all):
      - a simpleNav module that has already been attached to the greater sim. required for both currently implemented modes
      - a spice module that has already been attached to the greater sim. required for nadir pointing. 
    """
    def __init__(self, sim, startMode, switches : list[dict], samplingTime, taskName, period, **kwargs):
        super().__init__()
        self.period = macros.sec2nano(period)
        self.switches = switches
        self.currentMode = startMode
        nav = kwargs.get("nav")
        if nav:
            #set up readers for messages to be able to access data w/o the pain of messages
            self.navAttLog = nav.attOutMsg.recorder(samplingTime)
            self.navTransLog = nav.transOutMsg.recorder(samplingTime)
            sim.AddModelToTask(taskName, self.navAttLog)
            sim.AddModelToTask(taskName, self.navTransLog)
        spice = kwargs.get("spice")
        if spice:
            #set up readers for messages to be able to access data w/o the pain of messages
            self.spiceLog = spice.planetStateOutMsgs[0].recorder(samplingTime)
            sim.AddModelToTask(taskName, self.spiceLog)
        #set up output message
        self.attRef = messaging.AttRefMsg()
        buffer = messaging.AttRefMsgPayload()
        buffer.sigma_RN = [0, 0, 0]
        buffer.omega_RN_N = [0, 0, 0]
        buffer.domega_RN_N = [0, 0, 0]
        self.attRef.write(buffer, 0, self.moduleID)
    
    """
    Update method. Runs every timestep. 

    Essentially just calls the mode manager, which handles all the actual decisions, 
    and updates the output message. 
    """
    def UpdateState(self, CurrentSimNanos):
        buffer = messaging.AttRefMsgPayload()
        buffer.sigma_RN = self.modeManager(CurrentSimNanos)
        buffer.omega_RN_N = [0, 0, 0]
        buffer.domega_RN_N = [0, 0, 0]
        self.attRef.write(buffer, CurrentSimNanos, self.moduleID)
    
    """
    This method decides, based on the given switching times, which mode to execute at all timesteps. 
    It calls the required method, and returns the reference orientation it specifies. 
    """
    def modeManager(self, currentTime):
        #check if it's time to switch modes
        if self.switches and self.switches[0]["time"] < currentTime / self.period:
            self.currentMode = self.switches[0]["mode"]
            self.switches = self.switches[1:]
        #make sure the desired mode exists
        try:
            orientation = getattr(self, self.currentMode, None)()
        except:
            raise NameError(self.currentMode + " is not yet a supported flight mode.")
        return orientation

    """
    Sun-pointing mechanism. 

    This takes the sun-pointing vector from simpleNav and converts it to an orientation
    vector via 321 Euler angles. It returns that as the reference orientation. 
    """
    def sunPoint(self):
        #Direction Cosine Matrix of the current orientation relative to inertial    
        try:
            DCM = rbk.MRP2C(self.navAttLog.sigma_BN[-1])
        except:
            raise ValueError("simpleNav must be passed and correctly initialized to run a pointing mode. ")
        #sun pointing vector
        sun = self.navAttLog.vehSunPntBdy[-1]
        sun = sun / np.linalg.norm(sun)
        sunInertial = DCM.T @ sun

        #euler angle math
        theta = np.atan2(sunInertial[1], sunInertial[0])
        phi = np.atan2(sunInertial[2], np.sqrt(sunInertial[0]**2 + sunInertial[1]**2)) * -1
        psi = 0
        euler = [theta, phi, psi]
        orientation = rbk.euler3212MRP(euler)
        return orientation
    """
    Modified nadir-pointing mechanism. 

    Not a conventional nadir-pointing, because I felt like it - it points the body x-axis 
    towards the central body (or whichever body is in the first position on the Spice module 
    passed in, really). 

    Takes the current s/c position, takes the celestial body position, subtracts them, 
    and uses that as a pointing vector. That vector is then converted into an orientation
    using 321 Euler angles. It returns that as the reference orientation. 
    """
    def nadirPoint(self):
        #find relative positions
        try:
            scPos = self.navTransLog.r_BN_N[-1]
        except:
            raise ValueError("simpleNav must be passed and correctly initialized to run a pointing mode. ")
        try:
            earthPos = self.spiceLog.PositionVector[-1]
        except:
            raise ValueError("SPICE must be passed and correctly initialized to run a nadir pointing mode. ")
        point = (earthPos - scPos) / np.linalg.norm(earthPos - scPos)
        
        #euler angle math
        theta = np.atan2(point[1], point[0])
        phi = np.atan2(point[2], np.sqrt(point[0]**2 + point[1]**2)) * -1
        psi = 0
        euler = [theta, phi, psi]
        orientation = rbk.euler3212MRP(euler)

        return orientation
