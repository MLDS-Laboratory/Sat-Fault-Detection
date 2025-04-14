from Basilisk.utilities import RigidBodyKinematics as rbk, macros
from Basilisk.architecture import sysModel, messaging
import sys
import numpy as np

class PointingModes(sysModel.SysModel):
    def __init__(self, sim, startMode, switches : list[dict], samplingTime, taskName, period, **kwargs):
        super().__init__()
        self.period = macros.sec2nano(period)
        self.switches = switches
        self.currentMode = startMode
        nav = kwargs.get("nav")
        if nav:
            self.navAttLog = nav.attOutMsg.recorder(samplingTime)
            self.navTransLog = nav.transOutMsg.recorder(samplingTime)
            sim.AddModelToTask(taskName, self.navAttLog)
            sim.AddModelToTask(taskName, self.navTransLog)
        spice = kwargs.get("spice")
        if spice:
            self.spiceLog = spice.planetStateOutMsgs[0].recorder(samplingTime)
            sim.AddModelToTask(taskName, self.spiceLog)
        self.attRef = messaging.AttRefMsg()
        buffer = messaging.AttRefMsgPayload()
        buffer.sigma_RN = [0, 0, 0]
        buffer.omega_RN_N = [0, 0, 0]
        buffer.domega_RN_N = [0, 0, 0]
        self.attRef.write(buffer, 0, self.moduleID)
    def UpdateState(self, CurrentSimNanos):
        buffer = messaging.AttRefMsgPayload()
        buffer.sigma_RN = self.modeManager(CurrentSimNanos)
        buffer.omega_RN_N = [0, 0, 0]
        buffer.domega_RN_N = [0, 0, 0]
        self.attRef.write(buffer, CurrentSimNanos, self.moduleID)
    def modeManager(self, currentTime):
        if self.switches and self.switches[0]["time"] < currentTime / self.period:
            self.currentMode = self.switches[0]["mode"]
            self.switches = self.switches[1:]
        try:
            orientation = eval("self." + self.currentMode)()
        except:
            raise NameError(self.currentMode + " is not yet a supported flight mode.")
        return orientation
    def sunPoint(self):
        try:
            DCM = rbk.MRP2C(self.navAttLog.sigma_BN[-1])
        except:
            raise ValueError("simpleNav must be passed and correctly initialized to run a pointing mode. ")
        sun = self.navAttLog.vehSunPntBdy[-1]
        sun = sun / np.linalg.norm(sun)
        sunInertial = DCM.T @ sun
        theta = np.atan2(sunInertial[1], sunInertial[0])
        phi = np.atan2(sunInertial[2], np.sqrt(sunInertial[0]**2 + sunInertial[1]**2)) * -1
        psi = 0
        euler = [theta, phi, psi]
        orientation = rbk.euler3212MRP(euler)
        return orientation
    def nadirPoint(self):
        try:
            scPos = self.navTransLog.r_BN_N[-1]
        except:
            raise ValueError("simpleNav must be passed and correctly initialized to run a pointing mode. ")
        try:
            earthPos = self.spiceLog.PositionVector[-1]
        except:
            raise ValueError("SPICE must be passed and correctly initialized to run a nadir pointing mode. ")
        point = (earthPos - scPos) / np.linalg.norm(earthPos - scPos)
        
        theta = np.atan2(point[1], point[0])
        phi = np.atan2(point[2], np.sqrt(point[0]**2 + point[1]**2)) * -1
        psi = 0
        euler = [theta, phi, psi]
        orientation = rbk.euler3212MRP(euler)

        return orientation
